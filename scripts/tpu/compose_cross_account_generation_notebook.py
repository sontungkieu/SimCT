#!/usr/bin/env python3
"""Insert KJO cross-account checkpoint retrieval into a generation notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.kaggle_cross_account_generation import (
    compose_cross_account_generation_notebook,
)
from vdt_tunix.kaggle_model_sources import KaggleModelSourceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", required=True, type=Path)
    parser.add_argument("--cross-account-cell", required=True, type=Path)
    parser.add_argument("--source-kernel-id", required=True)
    parser.add_argument("--runtime-owner", required=True)
    parser.add_argument("--evaluation-dataset-source", required=True)
    parser.add_argument("--source-config-dir", required=True)
    parser.add_argument("--cross-account-output-dir", required=True)
    parser.add_argument("--overlay-input-root", required=True)
    parser.add_argument("--source-key-placeholder", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        base = json.loads(args.base_notebook.read_text(encoding="utf-8"))
        notebook = compose_cross_account_generation_notebook(
            base_notebook=base,
            cross_account_output_source=args.cross_account_cell.read_text(
                encoding="utf-8"
            ),
            source_kernel_id=args.source_kernel_id,
            runtime_owner=args.runtime_owner,
            evaluation_dataset_source=args.evaluation_dataset_source,
            source_config_dir=args.source_config_dir,
            cross_account_output_dir=args.cross_account_output_dir,
            overlay_input_root=args.overlay_input_root,
            source_key_placeholder=args.source_key_placeholder,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.out)
    except (KaggleModelSourceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 78
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "cell_count": len(notebook["cells"]),
                "scientific_evidence": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
