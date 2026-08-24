#!/usr/bin/env python3
"""Run the deterministic CPU reference canary and optionally save JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vdt_span.canary import run_reference_canary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_reference_canary()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
