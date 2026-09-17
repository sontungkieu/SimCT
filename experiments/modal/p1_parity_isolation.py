"""P1 parity isolation — offline first-divergence and sampler/serving audit.

CPU-only by default. The paid rollout-only A/A lives behind its own gate and is
never triggered implicitly. Reads r5 artifacts read-only; writes new receipts
under /runs/p1-parity-20260917/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

import modal

LOCAL_ROOT = Path("/home/tung/simct-b200-portable")
REMOTE_ROOT = Path("/opt/repo")
IMAGE_REF = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME = "simct-qwen7b-gemma2-assets-20260916"
RUN_VOLUME = "simct-qwen7b-gemma2-runs-20260916-main"
APP_NAME = "simct-p1-parity-isolation-20260917"
OUT_DIR = "/runs/p1-parity-20260917"

RUN_TAG = "p1-hostmask-ab-20260916-r5"
CONTROL = f"/runs/{RUN_TAG}-control"
CANDIDATE = f"/runs/{RUN_TAG}-candidate"
RESUME_ATTEMPT3 = f"/runs/{RUN_TAG}-resume-control-attempt3"

image = (
    modal.Image.from_registry(IMAGE_REF)
    .entrypoint([])
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/runai"), "/opt/repo/experiments/runai", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/modal/vendor"),
                   "/opt/repo/experiments/modal/vendor", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "tests"), "/opt/repo/tests", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/mp_opd_phi_gemma_50.py"),
                    "/opt/repo/experiments/modal/mp_opd_phi_gemma_50.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/provenance_prepare.py"),
                    "/opt/repo/experiments/modal/provenance_prepare.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/_sampler_audit_probe.py"),
                    "/opt/repo/experiments/modal/_sampler_audit_probe.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/_rollout_aa_probe.py"),
                    "/opt/repo/experiments/modal/_rollout_aa_probe.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/p1_parity_isolation.py"),
                    "/opt/repo/experiments/modal/p1_parity_isolation.py", copy=True)
)
app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=False)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value):
    """Make NaN/Inf safe for JSON without pretending they are numbers."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(sanitize(value), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def bounded(value, limit: int = 24):
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, str):
        return value[:limit]
    return value


def compare_sequence(left, right, *, numeric: bool = False) -> dict:
    report: dict = {
        "left_present": left is not None,
        "right_present": right is not None,
    }
    if left is None or right is None:
        report["equal"] = left is None and right is None
        return report
    report["length_left"] = len(left)
    report["length_right"] = len(right)
    shared = min(len(left), len(right))
    first_diff = None
    max_abs = 0.0
    nonfinite_left = nonfinite_right = 0
    for index in range(shared):
        x, y = left[index], right[index]
        if numeric:
            try:
                fx, fy = float(x), float(y)
            except (TypeError, ValueError):
                fx = fy = None
            if fx is not None:
                if not math.isfinite(fx):
                    nonfinite_left += 1
                if not math.isfinite(fy):
                    nonfinite_right += 1
                if math.isfinite(fx) and math.isfinite(fy):
                    max_abs = max(max_abs, abs(fx - fy))
        if x != y and first_diff is None:
            first_diff = index
    report["first_diff"] = first_diff
    report["common_prefix"] = shared if first_diff is None else first_diff
    report["equal"] = (
        first_diff is None and len(left) == len(right) and len(left) <= shared
    )
    if first_diff is not None:
        report["left_at_first_diff"] = bounded(left[first_diff], 8)
        report["right_at_first_diff"] = bounded(right[first_diff], 8)
    if numeric:
        report["max_abs_diff_on_prefix"] = max_abs
        report["nonfinite_left"] = nonfinite_left
        report["nonfinite_right"] = nonfinite_right
    return report


def compare_text(left, right) -> dict:
    if left is None or right is None:
        return {"left_present": left is not None, "right_present": right is not None,
                "equal": left == right}
    if left == right:
        return {"equal": True, "length_left": len(left), "length_right": len(right)}
    shared = min(len(left), len(right))
    first = next((i for i in range(shared) if left[i] != right[i]), shared)
    return {
        "equal": False,
        "length_left": len(left),
        "length_right": len(right),
        "first_diff": first,
        "common_prefix": first,
        "left_sha256": hashlib.sha256(left.encode()).hexdigest(),
        "right_sha256": hashlib.sha256(right.encode()).hexdigest(),
    }


def is_logprob_payload(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, (list, tuple)) and row for row in value)
    )


def logprob_series(value) -> list:
    series = []
    for row in value:
        head = row[0]
        series.append(float(head) if isinstance(head, (int, float)) else None)
    return series


def compare_meta(left_meta: dict, right_meta: dict) -> dict:
    left_keys, right_keys = set(left_meta), set(right_meta)
    report: dict = {
        "keys_only_left": sorted(left_keys - right_keys),
        "keys_only_right": sorted(right_keys - left_keys),
        "logprob_fields": {},
        "scalar_mismatch": [],
        "other_mismatch": [],
    }
    for key in sorted(left_keys & right_keys):
        a, b = left_meta[key], right_meta[key]
        if is_logprob_payload(a) and is_logprob_payload(b):
            report["logprob_fields"][key] = compare_sequence(
                logprob_series(a), logprob_series(b), numeric=True
            )
        elif isinstance(a, (int, float, str, bool)) or a is None:
            if a != b:
                report["scalar_mismatch"].append({"field": key, "left": a, "right": b})
        elif a != b:
            report["other_mismatch"].append({"field": key, "left_type": type(a).__name__,
                                             "right_type": type(b).__name__})
    return report


def row_report(left_row: dict, right_row: dict) -> dict:
    left_meta = left_row.get("meta_info") or {}
    right_meta = right_row.get("meta_info") or {}
    prompt_ids = compare_sequence(left_row.get("prompt_ids"), right_row.get("prompt_ids"))
    prompt_text = compare_text(left_row.get("prompt"), right_row.get("prompt"))
    output_ids = compare_sequence(left_row.get("output_ids"), right_row.get("output_ids"))
    output_text = compare_text(left_row.get("output"), right_row.get("output"))
    meta = compare_meta(left_meta, right_meta)
    logprob_bad = sorted(
        name for name, info in meta["logprob_fields"].items() if not info.get("equal")
    )
    weight_left = left_row.get("behavior_weight_version")
    weight_right = right_row.get("behavior_weight_version")

    kind = "identical"
    if not prompt_ids.get("equal") or not prompt_text.get("equal"):
        kind = "request_identity"
    elif not output_ids.get("equal"):
        kind = "output_token"
    elif logprob_bad:
        kind = "logprob_numeric"
    elif meta["scalar_mismatch"] or meta["keys_only_left"] or meta["keys_only_right"]:
        kind = "metadata"
    elif meta["other_mismatch"]:
        kind = "metadata_other"
    elif not output_text.get("equal"):
        kind = "detokenized_text"
    elif weight_left != weight_right:
        kind = "behavior_weight_version"

    return {
        "kind": kind,
        "prompt_ids": prompt_ids,
        "prompt_text": prompt_text,
        "output_ids": output_ids,
        "output_text": output_text,
        "meta": meta,
        "logprob_fields_mismatched": logprob_bad,
        "behavior_weight_version": {"left": weight_left, "right": weight_right},
        "meta_has_sampling_seed": ("sampling_seed" in left_meta) or ("sampling_seed" in right_meta),
        "meta_has_seed": ("seed" in left_meta) or ("seed" in right_meta),
    }


def read_rows(path: Path) -> list:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def permutation_report(left_rows: list, right_rows: list) -> dict:
    def fingerprint(ids):
        return hashlib.sha256(
            json.dumps(ids, sort_keys=True).encode()
        ).hexdigest()

    left_ids = [row.get("prompt_ids") for row in left_rows]
    right_ids = [row.get("prompt_ids") for row in right_rows]
    left_sorted = sorted(fingerprint(i) for i in left_ids)
    right_sorted = sorted(fingerprint(i) for i in right_ids)
    return {
        "ordered_stream_equal": left_ids == right_ids,
        "multiset_equal": left_sorted == right_sorted,
        "rows_left": len(left_ids),
        "rows_right": len(right_ids),
    }


def request_seed(train_seed: int, step: int, index: int) -> int:
    digest = hashlib.sha256(f"kdflow-rollout-v1:{train_seed}:{step}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def compare_pair(label: str, left_root: str, right_root: str, file_name: str) -> dict:
    left_path = Path(left_root) / "checkpoint/rollout_data" / file_name
    right_path = Path(right_root) / "checkpoint/rollout_data" / file_name
    report: dict = {
        "label": label,
        "left_root": left_root,
        "right_root": right_root,
        "file": file_name,
        "left_path": str(left_path),
        "right_path": str(right_path),
    }
    if not left_path.is_file() or not right_path.is_file():
        report["status"] = "missing_artifact"
        report["left_present"] = left_path.is_file()
        report["right_present"] = right_path.is_file()
        return report
    left_rows, right_rows = read_rows(left_path), read_rows(right_path)
    report.update({
        "status": "compared",
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
        "rows_left": len(left_rows),
        "rows_right": len(right_rows),
        "permutation": permutation_report(left_rows, right_rows),
    })
    step = int(Path(file_name).stem)
    rows = []
    kind_counts: dict = {}
    first_divergence = None
    shared = min(len(left_rows), len(right_rows))
    for index in range(shared):
        entry = row_report(left_rows[index], right_rows[index])
        kind_counts[entry["kind"]] = kind_counts.get(entry["kind"], 0) + 1
        if entry["kind"] != "identical":
            rows.append({
                "row_index": index,
                "kind": entry["kind"],
                "recomputed_sampling_seed": request_seed(42, step, index),
                "prompt_ids_len": entry["prompt_ids"].get("length_left"),
                "prompt_ids_first_diff": entry["prompt_ids"].get("first_diff"),
                "output_ids_len": entry["output_ids"].get("length_left"),
                "output_ids_first_diff": entry["output_ids"].get("first_diff"),
                "output_ids_prefix_equal": entry["output_ids"].get("common_prefix"),
                "logprob_fields_mismatched": entry["logprob_fields_mismatched"],
                "logprob_max_abs_diff": {
                    name: entry["meta"]["logprob_fields"][name].get("max_abs_diff_on_prefix")
                    for name in entry["logprob_fields_mismatched"]
                },
                "scalar_mismatch": entry["meta"]["scalar_mismatch"],
                "keys_only_left": entry["meta"]["keys_only_left"],
                "keys_only_right": entry["meta"]["keys_only_right"],
                "behavior_weight_version": entry["behavior_weight_version"],
                "prompt_text": entry["prompt_text"],
                "output_text_equal": entry["output_text"].get("equal"),
                "meta_has_sampling_seed": entry["meta_has_sampling_seed"],
            })
            if first_divergence is None:
                first_divergence = {
                    "row_index": index,
                    "kind": entry["kind"],
                    "recomputed_sampling_seed": request_seed(42, step, index),
                    "prompt_ids_first_diff": entry["prompt_ids"].get("first_diff"),
                    "output_ids_first_diff": entry["output_ids"].get("first_diff"),
                    "logprob_fields_mismatched": entry["logprob_fields_mismatched"],
                    "scalar_mismatch": entry["meta"]["scalar_mismatch"],
                    "keys_only_left": entry["meta"]["keys_only_left"],
                    "keys_only_right": entry["meta"]["keys_only_right"],
                    "prompt_ids_left_len": entry["prompt_ids"].get("length_left"),
                    "prompt_ids_right_len": entry["prompt_ids"].get("length_right"),
                    "output_ids_left_len": entry["output_ids"].get("length_left"),
                    "output_ids_right_len": entry["output_ids"].get("length_right"),
                    "output_ids_common_prefix": entry["output_ids"].get("common_prefix"),
                }
    report["kind_counts"] = kind_counts
    report["first_divergence"] = first_divergence
    report["divergent_rows"] = rows[:16]
    report["divergent_rows_truncated"] = max(0, len(rows) - 16)
    report["both_identical"] = first_divergence is None and len(left_rows) == len(right_rows)
    # Row 0 meta schema is enough to prove whether a request seed field exists at all.
    if left_rows:
        report["meta_keys_row0_left"] = sorted((left_rows[0].get("meta_info") or {}).keys())
    if right_rows:
        report["meta_keys_row0_right"] = sorted((right_rows[0].get("meta_info") or {}).keys())
    return report


# --------------------------------------------------------------------------- #
# CPU gate 1: first divergence
# --------------------------------------------------------------------------- #
PAIRS = (
    ("ab_step1", CONTROL, CANDIDATE, "1.jsonl"),
    ("ab_step2", CONTROL, CANDIDATE, "2.jsonl"),
    ("resume_vs_continuous_step2", CONTROL, RESUME_ATTEMPT3, "2.jsonl"),
)


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/runs": runs})
def first_divergence_remote() -> dict:
    result: dict = {"status": "completed", "pairs": [], "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        for label, left, right, name in PAIRS:
            print(f"DIVERGENCE_STAGE_START {label} pid={os.getpid()}", flush=True)
            result["pairs"].append(compare_pair(label, left, right, name))
            print(f"DIVERGENCE_STAGE_END {label} pid={os.getpid()}", flush=True)
        atomic_json(Path(OUT_DIR) / "FIRST_DIVERGENCE.json", result)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        runs.commit()
    return result


# --------------------------------------------------------------------------- #
# CPU gate 2: sampler / serving audit in the pinned image
# --------------------------------------------------------------------------- #
@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/runs": runs})
def sampler_audit_remote() -> dict:
    """Persist the audit to the runs volume so a dead client cannot lose it."""
    py = "/opt/venvs/simct-b200/bin/python"
    probe = "/opt/repo/experiments/modal/_sampler_audit_probe.py"
    proc = subprocess.run([py, probe], cwd=str(REMOTE_ROOT), text=True, capture_output=True)
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SAMPLER_AUDIT_JSON="):
            payload = json.loads(line.split("=", 1)[1])
    result = {
        "status": "completed" if proc.returncode == 0 and payload else "failed",
        "returncode": proc.returncode,
        "report": payload,
        "stderr_tail": (proc.stderr or "")[-4000:],
        "executable": py,
    }
    atomic_json(Path(OUT_DIR) / "sampler-audit.json", result)
    runs.commit()
    print("SAMPLER_AUDIT_STATUS=" + result["status"], flush=True)
    print("SAMPLER_AUDIT_JSON=" + json.dumps(payload, sort_keys=True)[:20000], flush=True)
    return result


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/runs": runs})
def pinned_tests_remote(paths: list) -> dict:
    """Run CPU-eligible regression with the pinned interpreter inside the pinned image."""
    py = "/opt/venvs/simct-b200/bin/python"
    proc = subprocess.run(
        [py, "-m", "pytest", "-q", *paths, "--disable-warnings", "-p", "no:cacheprovider"],
        cwd=str(REMOTE_ROOT), text=True, capture_output=True,
    )
    result = {
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "executable": py,
        "tests": list(paths),
        "output_tail": (proc.stdout + proc.stderr)[-9000:],
    }
    atomic_json(Path(OUT_DIR) / "pinned-tests.json", result)
    runs.commit()
    print("PINNED_TESTS_STATUS=" + result["status"], flush=True)
    print(result["output_tail"][-5000:], flush=True)
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0)
def list_dir_remote(path: str) -> dict:
    """Diagnostic: prove what the deployed image actually contains."""
    root = Path(path)
    if not root.exists():
        return {"path": path, "exists": False}
    entries = sorted(p.as_posix() for p in root.rglob("*"))
    return {"path": path, "exists": True, "count": len(entries), "entries": entries[:400],
            "tests_mp_opd": sorted(
                p.as_posix() for p in (Path("/opt/repo/tests/mp_opd").rglob("*")
                                       if Path("/opt/repo/tests/mp_opd").exists() else [])
            )[:80]}


@app.local_entrypoint()
def main() -> None:
    gate = os.environ.get("P1_GATE", "divergence")
    if gate == "spawn":
        # Fire-and-forget: the client only has to live for the spawn call, so a
        # dropped heartbeat cannot lose the result. Receipts land on the volume.
        target = os.environ.get("P1_SPAWN_FN", "sampler_audit")
        if target == "pinned_tests":
            paths = [item for item in os.environ.get(
                "P1_TEST_PATHS",
                "/opt/repo/tests/mp_opd/test_rollout_deterministic_args.py,/opt/repo/tests/test_runai_contract.py",
            ).split(",") if item]
            call = pinned_tests_remote.spawn(paths)
        elif target == "sampler_audit":
            call = sampler_audit_remote.spawn()
        elif target == "divergence":
            call = first_divergence_remote.spawn()
        elif target == "rollout_aa":
            call = rollout_aa_remote.spawn(os.environ.get("P1_AA_MODE", "baseline"))
        else:
            raise SystemExit("unknown P1_SPAWN_FN=" + target)
        print("SPAWNED_CALL_ID=" + str(call.object_id), flush=True)
        print("SPAWNED_FN=" + target, flush=True)
        return
    out_dir = Path("/mnt/d/dev/codex/research_vdt/remote_artifacts/p1-parity-20260917")
    out_dir.mkdir(parents=True, exist_ok=True)
    if gate == "divergence":
        result = first_divergence_remote.remote()
        atomic_json(out_dir / "first-divergence.receipt.json", result)
        print("FIRST_DIVERGENCE_STATUS=" + str(result.get("status")), flush=True)
        for pair in result.get("pairs", []):
            print(f"PAIR {pair['label']} status={pair.get('status')} first={pair.get('first_divergence')}", flush=True)
        return
    if gate == "rollout_aa":
        mode = os.environ.get("P1_AA_MODE", "baseline")
        result = rollout_aa_remote.remote(mode)
        atomic_json(out_dir / f"rollout-aa-{mode}.receipt.json", result)
        print("ROLLOUT_AA_GATE_STATUS=" + str(result.get("status")), flush=True)
        print("ROLLOUT_AA_GATE_MARKER=" + json.dumps(sanitize(result.get("marker")), sort_keys=True)[:4000], flush=True)
        return
    if gate == "pinned_tests":
        paths = [item for item in os.environ.get(
            "P1_TEST_PATHS",
            "/opt/repo/tests/mp_opd/test_rollout_deterministic_args.py,/opt/repo/tests/test_runai_contract.py",
        ).split(",") if item]
        result = pinned_tests_remote.remote(paths)
        atomic_json(out_dir / "pinned-tests.receipt.json", result)
        print("PINNED_TESTS_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "sampler_audit":
        result = sampler_audit_remote.remote()
        atomic_json(out_dir / "sampler-audit.receipt.json", result)
        print("SAMPLER_AUDIT_STATUS=" + str(result.get("status")), flush=True)
        return
    raise SystemExit("unknown P1_GATE=" + gate)
# --------------------------------------------------------------------------- #
# GPU gate 3: rollout-only A/A through the production RolloutActorGroup path
# --------------------------------------------------------------------------- #
DRIVER = "/opt/repo/experiments/modal/_rollout_aa_probe.py"


def driver_environment(mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update({
        "PATH": "/opt/venvs/simct-b200/bin:/usr/local/cuda-13.0/bin:" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": "/opt/repo/experiments/modal/vendor:/opt/repo",
        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "KDFLOW_TRUST_REMOTE_CODE": "0",
        "RAY_USAGE_STATS_ENABLED": "0", "NCCL_CUMEM_HOST_ENABLE": "0", "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": "/usr/local/cuda-13.0",
        "CPATH": "/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include",
        "LD_LIBRARY_PATH": "/tmp/runtime-host-libs:/opt/simct-portable-libs:/usr/local/cuda-13.0/lib64",
        "MP_RUNTIME_DIR": "/tmp/runtime", "MP_RAY_TMP": "/tmp/ray",
        "XDG_CACHE_HOME": "/tmp/cache", "TRITON_CACHE_DIR": "/tmp/cache/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/cache/torch", "FLASHINFER_WORKSPACE_BASE": "/tmp/cache",
        "KDFLOW_ROLLOUT_PORT_BASE": "15000", "KDFLOW_ROUTER_PORT_BASE": "16000",
        "MP_SHARED_ROOT": "/assets", "MP_STUDENT_PATH": "/assets/student",
        "P1_AA_MODE": mode,
        "P1_AA_ROWS": os.environ.get("P1_AA_ROWS", "64"),
        "P1_AA_REPLAYS": os.environ.get("P1_AA_REPLAYS", "2"),
        "P1_AA_RESTART": os.environ.get("P1_AA_RESTART", "1"),
        "WANDB_MODE": "offline", "WANDB_DISABLED": "true",
    })
    return env


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=3600, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs})
def rollout_aa_remote(mode: str) -> dict:
    """One arm of the rollout-only A/A. No teacher, no full-meta, no optimizer."""
    for directory in ("/tmp/runtime/runtime-host-libs", "/tmp/ray", "/tmp/cache/triton",
                      "/tmp/cache/torch"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"rollout-aa-{mode}.log"
    result: dict = {"status": "starting", "mode": mode, "driver": DRIVER,
                    "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        env = driver_environment(mode)
        started = time.time()
        with log_path.open("w") as sink:
            sink.write(f"ROLLOUT_AA_STAGE_START mode={mode} rows={env['P1_AA_ROWS']} "
                       f"replays={env['P1_AA_REPLAYS']} restart={env['P1_AA_RESTART']}\n")
            sink.flush()
            proc = subprocess.run(["/opt/venvs/simct-b200/bin/python", DRIVER],
                                  cwd=str(REMOTE_ROOT), env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, text=True, timeout=3300)
        result["driver_seconds"] = time.time() - started
        result["returncode"] = proc.returncode
        tail = log_path.read_text()[-20000:]
        marker = None
        for line in tail.splitlines():
            if line.startswith("ROLLOUT_AA_JSON="):
                marker = json.loads(line.split("=", 1)[1])
        result["marker"] = marker
        result["log_path"] = str(log_path)
        result["log_tail"] = tail[-6000:]
        result["status"] = "completed" if proc.returncode == 0 and marker else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(out_dir / f"rollout-aa-{mode}.receipt.json", result)
        runs.commit()
    print("ROLLOUT_AA_STATUS=" + str(result.get("status")), flush=True)
    print("ROLLOUT_AA_LOG=" + str(result.get("log_path")), flush=True)
    return result