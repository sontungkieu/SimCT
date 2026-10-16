#!/usr/bin/env python3
"""P-positions: select the 5000 joint-strict positions (CPU-only).

Reads fixtures-*.json from $SB_ROOT/work, global rank-hash order, top 5000
(not forced 500/shard). Writes positions.json with shard + rank per entry.
Usage: SB_ROOT=... sb_positions.py [--n 5000]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    root = Path(args.root)
    work = root / "work"
    fichs = sorted(work.glob("fixtures-*.json"))
    if len(fichs) < 10:
        print(f"REFUSE: only {len(fichs)}/10 fixtures present", file=sys.stderr)
        sys.exit(2)
    all_pos = []
    for fp in fichs:
        shard = fp.stem.split("-")[1]
        for r in json.loads(fp.read_text())["positions"]:
            r = dict(r)
            r["shard"] = shard
            all_pos.append(r)
    all_pos.sort(key=lambda r: r["rank"])
    sel = all_pos[:args.n]
    if len(sel) < args.n:
        print(f"REFUSE: only {len(sel)}/{args.n} positions available", file=sys.stderr)
        sys.exit(2)
    dest = work / "positions.json"
    dest.write_text(json.dumps({"n": args.n, "positions": sel,
                                "per_shard": dict(Counter(r["shard"] for r in sel)),
                                "per_category": dict(Counter(
                                    r["category"] for r in sel))}))
    print(f"wrote {dest} n={len(sel)} per_shard={dict(Counter(r['shard'] for r in sel))}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
