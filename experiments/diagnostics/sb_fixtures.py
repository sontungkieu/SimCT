#!/usr/bin/env python3
"""A3.2: per-shard joint-strict fixtures (CPU-only, no GPU).

Joint positions = S-indices present in both stored SB and SQ strict maps,
filtered (j>=5, non-space, word continuation). Writes fixtures-{sh}.json
with prefix_ids per arm for the proposal stage.

Usage: SB_ROOT=... sb_fixtures.py --shard 09
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from h2_measurements import vocabulary_bytes, byte_spans, category
    from h2_word_events import next_word

    root = Path(args.root)
    work = root / "work"
    pool_path = work / f"pool-{args.shard}.json"
    audit_path = work / f"audit-{args.shard}.json"
    if not pool_path.is_file() or not audit_path.is_file():
        print(f"REFUSE: pool/audit-{args.shard} not ready", file=sys.stderr)
        sys.exit(2)
    pool = {r["id"]: r for r in json.loads(pool_path.read_text())["rows"]}
    audit = {r["id"]: r for r in json.loads(audit_path.read_text())["rows"]}
    assets = json.loads((root / "assets.json").read_text())
    toks, vocabs = {}, {}
    for arm in ("S", "B", "Q"):
        toks[arm] = AutoTokenizer.from_pretrained(
            assets["models"][arm]["path"], local_files_only=True)
        vocabs[arm] = vocabulary_bytes(toks[arm], "bpe")

    positions = []
    for rid, arow in audit.items():
        if arow["errors"] or "strict" not in arow:
            continue
        prow = pool[rid]
        sb = dict(arow["strict"].get("SB", []))
        sq = dict(arow["strict"].get("SQ", []))
        joint = sorted(set(sb) & set(sq))
        if not joint:
            continue
        rtext = toks["S"].decode(prow["response_ids"],
                                 skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        raw = rtext.encode()
        s_spans = byte_spans(prow["response_ids"], vocabs["S"], rtext)
        full_text = toks["S"].decode(prow["prompt_ids"] + prow["response_ids"],
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        prompt_text = toks["S"].decode(prow["prompt_ids"],
                                       skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        enc, pre_len = {}, {}
        for arm in ("B", "Q"):
            full = toks[arm].encode(full_text, add_special_tokens=False)
            pre = toks[arm].encode(prompt_text, add_special_tokens=False)
            enc[arm] = full
            pre_len[arm] = len(pre)
        enc["S"] = prow["prompt_ids"] + prow["response_ids"]
        pre_len["S"] = len(prow["prompt_ids"])
        for j in joint:
            if j < 5:
                continue
            left, right, _ = s_spans[j]
            piece = raw[left:right]
            if piece.decode("utf-8", errors="ignore").isspace():
                continue
            try:
                cont = raw[left:].decode("utf-8")
                raw[:left].decode("utf-8")
            except UnicodeDecodeError:
                continue
            word = next_word(cont, terminated=True)
            if not word:
                continue
            i_b, i_q = sb[j], sq[j]
            prefix = {arm: enc[arm][:pre_len[arm] + (j if arm == "S"
                                                     else i_b if arm == "B"
                                                     else i_q)]
                      for arm in ("S", "B", "Q")}
            cat = category(piece)
            initial = (left == 0 or raw[left - 1:left] == b" ")
            positions.append({
                "key": f"{rid}/{j}", "prompt_id": arow["prompt_id"],
                "response_id": rid, "position": j,
                "byte_start": left, "byte_end": right,
                "category": cat, "initial": initial,
                "stratum": f"{arow['prompt_id']}/{cat}/{'init' if initial else 'cont'}",
                "reference_word": word, "prefix_ids": prefix,
                "rank": hashlib.sha256(
                    f"sb-main-v2/{rid}/{j}".encode()).hexdigest()})
    dest = work / f"fixtures-{args.shard}.json"
    dest.write_text(json.dumps({"shard": args.shard, "positions": positions}))
    print(f"wrote {dest} positions={len(positions)}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
