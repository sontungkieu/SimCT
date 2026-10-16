#!/usr/bin/env python3
"""Freeze sb-main-v2 main/dev/calibration prompt manifests (P-freeze, CPU-only).

Method (matches structural_behavioral_plan_contract.json partitioning):
  eligible = all train row indices minus 100 pilot indices
  rank key = sha256(f"{source}-train-{idx}:sb-main-v2:42"), sort ascending
  main = top 1000/source; dev = next 128/source; cal = next 256/source
  shards = round-robin over hash-sorted main order -> 10 x 200
Manifest stores IDs + indices only; prompt text resolves from pinned parquets
(hashes in run_matrix / pilot datasets receipt).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "structural-behavioral-main-runs"
PILOT_POOL = Path(
    r"D:\dev\dsh\research_vai\SimCT\remote_artifacts"
    r"\h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1\received\word-prompts.json"
)
OUT = ART / "main_pool_manifest.json"

SOURCES = {
    "gsm8k": {"n_rows": 7473, "id_prefix": "gsm8k-train-"},
    "math": {"n_rows": 7500, "id_prefix": "math-train-"},
}
STUDY = "sb-main-v2"
SEED = 42
N_MAIN, N_DEV, N_CAL, N_SHARDS = 1000, 128, 256, 10


def rank(source: str, idx: int) -> str:
    return hashlib.sha256(f"{source}-train-{idx}:{STUDY}:{SEED}".encode()).hexdigest()


def main() -> None:
    pilot = json.loads(PILOT_POOL.read_text())
    pilot_ids = {p["id"] for p in pilot["pool"]}
    assert len(pilot_ids) == 100, len(pilot_ids)

    manifest = {
        "study_version": STUDY,
        "seed": SEED,
        "method": "sha256_rank_topN_per_source_then_round_robin_shards",
        "datasets": pilot["datasets"],
        "pilot_excluded_ids": sorted(pilot_ids),
        "splits": {},
    }
    for source, spec in SOURCES.items():
        eligible = [
            i for i in range(spec["n_rows"])
            if f"{spec['id_prefix']}{i}" not in pilot_ids
        ]
        ranked = sorted(eligible, key=lambda i: rank(source, i))
        main_idx = ranked[:N_MAIN]
        dev_idx = ranked[N_MAIN:N_MAIN + N_DEV]
        cal_idx = ranked[N_MAIN + N_DEV:N_MAIN + N_DEV + N_CAL]
        assert len(main_idx) == N_MAIN and len(dev_idx) == N_DEV and len(cal_idx) == N_CAL
        assert not (set(main_idx) & set(dev_idx) or set(main_idx) & set(cal_idx) or set(dev_idx) & set(cal_idx))
        manifest["splits"][source] = {
            "main": [f"{spec['id_prefix']}{i}" for i in main_idx],
            "dev": [f"{spec['id_prefix']}{i}" for i in dev_idx],
            "calibration": [f"{spec['id_prefix']}{i}" for i in cal_idx],
            "row_index": {"main": main_idx, "dev": dev_idx, "calibration": cal_idx},
        }

    # Shards: round-robin over combined hash-sorted main (stable across runs).
    combined = sorted(
        ([("gsm8k", i) for i in manifest["splits"]["gsm8k"]["row_index"]["main"]]
         + [("math", i) for i in manifest["splits"]["math"]["row_index"]["main"]]),
        key=lambda t: rank(*t),
    )
    assert len(combined) == 2000
    shards = {f"{s:02d}": [] for s in range(N_SHARDS)}
    for pos, (source, idx) in enumerate(combined):
        shards[f"{pos % N_SHARDS:02d}"].append(f"{SOURCES[source]['id_prefix']}{idx}")
    manifest["shards"] = shards

    OUT.write_text(json.dumps(manifest, indent=1))
    # Self-validation.
    m = json.loads(OUT.read_text())
    all_main = m["splits"]["gsm8k"]["main"] + m["splits"]["math"]["main"]
    assert len(all_main) == 2000 and len(set(all_main)) == 2000
    assert not (set(all_main) & set(m["pilot_excluded_ids"]))
    assert all(len(v) == 200 for v in m["shards"].values())
    flat = [x for v in m["shards"].values() for x in v]
    assert sorted(flat) == sorted(all_main)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); main=2000 dev=256/source cal=256/source shards=10x200 OK")


if __name__ == "__main__":
    main()
