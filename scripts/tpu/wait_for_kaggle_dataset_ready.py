#!/usr/bin/env python3
"""Wait until a Kaggle dataset is ready and its file listing is stable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.kaggle_dataset_readiness import (  # noqa: E402
    DatasetReadinessError,
    read_dataset_snapshot,
    wait_for_dataset_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--kaggle-bin", required=True)
    parser.add_argument("--kaggle-config-dir", type=Path, required=True)
    parser.add_argument("--expected-file", action="append", default=[])
    parser.add_argument("--min-total-bytes", type=int, default=1)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--interval-s", type=float, default=10.0)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        snapshot = wait_for_dataset_ready(
            snapshot_reader=lambda: read_dataset_snapshot(
                kaggle_bin=args.kaggle_bin,
                config_dir=args.kaggle_config_dir,
                dataset_source=args.dataset_source,
            ),
            expected_files=args.expected_file,
            min_total_bytes=args.min_total_bytes,
            stable_checks=args.stable_checks,
            timeout_s=args.timeout_s,
            interval_s=args.interval_s,
        )
        payload = {
            "ok": True,
            "dataset_source": args.dataset_source,
            "status": snapshot.status,
            "file_count": len(snapshot.files),
            "total_bytes": snapshot.total_bytes,
            "files_fingerprint": snapshot.fingerprint,
            "stable_checks": args.stable_checks,
        }
        exit_code = 0
    except (DatasetReadinessError, ValueError) as exc:
        payload = {
            "ok": False,
            "dataset_source": args.dataset_source,
            "error": str(exc),
        }
        exit_code = 1
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_name(args.out_json.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out_json)
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
