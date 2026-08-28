#!/usr/bin/env python3
"""Build fail-closed 4K/8K TPU resource-ladder configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.config import RunConfig


PROTOCOLS = {
    "paper4k": {
        "sequence": 4096,
        "prompt": 256,
        "completion": 3840,
        "minimum_actual": 3968,
        "teacher_buckets": [4096, 6144, 8192],
        "alignment_bucket": 8192,
    },
    "public8k": {
        "sequence": 8192,
        "prompt": 4096,
        "completion": 4096,
        "minimum_actual": 7680,
        "teacher_buckets": [8192, 12288, 16384],
        "alignment_bucket": 16384,
    },
}
BATCH_LADDER = (1, 2, 4, 8)


def build_matrix(baseline: RunConfig) -> dict[str, dict]:
    if baseline.simct.algorithm not in {"simple_opd", "simct"}:
        raise ValueError("resource canaries require an OPD baseline")
    result: dict[str, dict] = {}
    for protocol, contract in PROTOCOLS.items():
        for batch_size in BATCH_LADDER:
            name = f"{protocol}-fsdp8-b{batch_size}"
            payload = copy.deepcopy(baseline.to_dict())
            if payload["training"].get("seed") is None:
                payload["training"].pop("seed", None)
            payload["run_id"] = f"vdt-resource-{baseline.simct.algorithm}-{name}"
            payload["rollout"].update(
                {
                    "prompt_batch_size": batch_size,
                    "samples_per_prompt": 1,
                    "max_prompt_tokens": contract["prompt"],
                    "max_completion_tokens": contract["completion"],
                    "max_sequence_tokens": contract["sequence"],
                    "minimum_actual_sequence_tokens": contract["minimum_actual"],
                    "force_max_completion": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                }
            )
            payload["training"].update(
                {
                    "max_steps": 1,
                    "max_steps_unit": "optimizer_update",
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 64 // batch_size,
                    "learning_rate": 1e-6,
                    "synchronize_phase_timings": True,
                    "teacher_sequence_buckets": contract["teacher_buckets"],
                    "alignment_bucket_size": contract["alignment_bucket"],
                    "teacher_scoring_mode": "cached_teacher_forcing",
                }
            )
            payload["tpu"].update(
                {
                    "fsdp_parallelism": 8,
                    "tensor_parallelism": 1,
                    "pipeline_parallelism": 1,
                }
            )
            payload["checkpoint"]["root"] = (
                f"/kaggle/working/{payload['run_id']}/checkpoints"
            )
            payload["checkpoint"]["save_every_steps"] = 1
            configured = RunConfig.from_mapping(payload)
            serialized = json.loads(json.dumps(configured.to_dict()))
            if serialized["training"].get("seed") is None:
                serialized["training"].pop("seed", None)
            for optional_buckets in (
                "student_sequence_buckets",
                "student_completion_buckets",
                "alignment_unit_buckets",
            ):
                if not serialized["training"].get(optional_buckets):
                    serialized["training"].pop(optional_buckets, None)
            if serialized["training"].get("teacher_scoring_mode") == "dense":
                serialized["training"].pop("teacher_scoring_mode", None)
            result[name] = serialized
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline = RunConfig.from_mapping(json.loads(args.baseline.read_text()))
    matrix = build_matrix(baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, payload in matrix.items():
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        protocol = "paper4k" if name.startswith("paper4k") else "public8k"
        batch_size = payload["rollout"]["prompt_batch_size"]
        manifest.append(
            {
                "name": name,
                "protocol": protocol,
                "batch_size": batch_size,
                "effective_global_batch": 64,
                "gradient_accumulation_steps": 64 // batch_size,
                "path": path.name,
                "config_sha256": RunConfig.from_mapping(payload).digest(),
            }
        )
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "status": "staged",
                "scientific_evidence": False,
                "probe_mode": "forced-full-length-native-rollout",
                "profile_micro_step": 1,
                "submit_policy": (
                    "B=1,2,4,8 may run independently on distinct healthy owners; "
                    "audit and diagnose every configuration separately"
                ),
                "canaries": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
