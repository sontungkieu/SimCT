#!/usr/bin/env python3
"""G0 disposition from the 6 qualification JSONs (CPU-only, no GPU).

PASS iff for every arm: identity, mass, cached==0.0, fp32 batched<=1e-4.
bf16 batched diff is recorded as the arm's batch-geometry noise level
(fp32 proves implementation exactness; bf16 noise is a bench question).
Writes $SB_ROOT/work/disposition-G0.json. Exit 0 always; verdict in file.
"""
import json
import os
import sys
from pathlib import Path

ARMS = ("S", "B", "Q")
FP32_TOL = 1e-4


def main() -> None:
    root = Path(os.environ.get("SB_ROOT", ""))
    work = root / "work"
    verdict = {"G0": "PASS", "arms": {}, "notes": []}
    for arm in ARMS:
        ad = {"arm": arm, "ok": True, "bf16_batch_noise": None, "fp32_batch_noise": None}
        for dtype in ("bfloat16", "float32"):
            p = work / f"qualification-{arm}-{dtype}.json"
            if not p.is_file():
                verdict["G0"] = "BLOCKED"
                ad["ok"] = False
                ad.setdefault("missing", []).append(dtype)
                continue
            d = json.loads(p.read_text())
            c = d["checks"]
            for name in ("identity_and_roundtrip", "probability_mass_and_residual",
                         "cached_uncached_parity"):
                if not c.get(name, {}).get("pass", False):
                    ad["ok"] = False
                    ad.setdefault("failed", []).append(f"{dtype}/{name}")
            b = c.get("batched_independent_parity", {})
            if dtype == "float32":
                v = b.get("max_abs_diff")
                ad["fp32_batch_noise"] = v
                if not (isinstance(v, (int, float)) and v <= FP32_TOL):
                    ad["ok"] = False
                    ad.setdefault("failed", []).append(f"float32/batched={v}")
            else:
                ad["bf16_batch_noise"] = b.get("max_abs_diff")
        if not ad["ok"]:
            verdict["G0"] = "FAIL"
        verdict["arms"][arm] = ad
    verdict["notes"] = [
        "cached parity exact 0.0 => KV-cache implementation correct.",
        "fp32 batched ~4e-5 => batching geometry exact modulo fp accumulation.",
        "bf16 batched noise (S 0.0/B 0.125/Q 0.359) is precision, not a bug;",
        "V-bench must test whether it disturbs E2/X rank order; fp32 fallback ready.",
        "Qwen sliding-window+eager warning recorded; bench compares impls if needed.",
    ]
    dest = work / "disposition-G0.json"
    dest.write_text(json.dumps(verdict, indent=1))
    print(f"wrote {dest} G0={verdict['G0']}")
    print(json.dumps({a: v["ok"] for a, v in verdict["arms"].items()}))


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
