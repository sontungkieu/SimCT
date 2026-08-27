#!/usr/bin/env python3
"""Build immutable fixed-prompt manifests for 4K/8K resource probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _write_protocol(root: Path, protocol: str, prompt: str, count: int) -> None:
    protocol_root = root / protocol
    protocol_root.mkdir(parents=True, exist_ok=True)
    records = protocol_root / "records.jsonl"
    rows = [
        {
            "prompt_id": f"{protocol}-{index:02d}",
            "student_prompt": prompt,
            "teacher_prompt": prompt,
        }
        for index in range(count)
    ]
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    records.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "contract_version": 1,
        "dataset_id": f"vdt/resource-probe/{protocol}",
        "dataset_revision": f"forced-length-v1-{digest[:12]}",
        "split": "resource-probe",
        "records_path": records.name,
        "records_sha256": digest,
        "record_count": count,
    }
    (protocol_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--records", type=int, default=64)
    args = parser.parse_args()
    if args.records < 64:
        raise ValueError("resource probe dataset needs at least 64 records")
    _write_protocol(
        args.output_dir,
        "paper4k",
        # Gemma tokenizes the leading-space word close to one token per repeat.
        # Leave headroom under the 256-token prompt cap while ensuring the
        # realized prompt plus the forced 3840-token completion clears the
        # checked-in 3968-token fail-closed floor.
        " probe" * 192,
        args.records,
    )
    # Gemma tokenizes the leading-space word close to one token per repeat.
    # Runtime config still fails closed unless actual total length reaches 7680.
    _write_protocol(
        args.output_dir,
        "public8k",
        " probe" * 4000,
        args.records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
