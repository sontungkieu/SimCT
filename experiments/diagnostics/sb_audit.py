#!/usr/bin/env python3
"""A3.1: per-shard tokenizer audit (CPU-only, no GPU).

For each pool response: decode S-native ids to text, re-encode in S/B/Q
with boundary + roundtrip checks (pilot encode_response contract), then
pairwise audits (S<->B, S<->Q, B<->Q). Writes audit-{shard}.json.

Usage: SB_ROOT=... sb_audit.py --shard 09   (tokenizers from assets.json)
"""
import argparse
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
    from h2_measurements import vocabulary_bytes, byte_spans
    from h2_tokenizer_audit import audit_pair, aggregate_audits

    root = Path(args.root)
    pool_path = root / "work" / f"pool-{args.shard}.json"
    if not pool_path.is_file():
        print(f"REFUSE: pool-{args.shard}.json not ready; run pool first",
              file=sys.stderr)
        sys.exit(2)
    assets = json.loads((root / "assets.json").read_text())
    pool = json.loads(pool_path.read_text())
    toks, vocabs = {}, {}
    for arm in ("S", "B", "Q"):
        toks[arm] = AutoTokenizer.from_pretrained(
            assets["models"][arm]["path"], local_files_only=True)
        vocabs[arm] = vocabulary_bytes(toks[arm], "bpe")

    rows, failures = [], {"S": 0, "B": 0, "Q": 0}
    for row in pool["rows"]:
        full_text = toks["S"].decode(row["prompt_ids"] + row["response_ids"],
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        prompt_text = toks["S"].decode(row["prompt_ids"],
                                       skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        rtext = toks["S"].decode(row["response_ids"],
                                 skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        spans, errors, ok = {}, {}, True
        # S native: ids are already S encodings of this exact response.
        try:
            spans["S"] = byte_spans(row["response_ids"], vocabs["S"], rtext)
        except Exception as ex:  # noqa: BLE001
            errors["S"] = str(ex)[:200]
            ok = False
        for arm in ("B", "Q"):
            try:
                full = toks[arm].encode(full_text, add_special_tokens=False)
                back = toks[arm].decode(full, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                if back != full_text:
                    raise ValueError("roundtrip mismatch")
                pre = toks[arm].encode(prompt_text, add_special_tokens=False)
                if full[:len(pre)] != pre:
                    raise ValueError("prompt boundary changed")
                spans[arm] = byte_spans(full[len(pre):], vocabs[arm], rtext)
            except Exception as ex:  # noqa: BLE001
                errors[arm] = str(ex)[:200]
                ok = False
        audits = {}
        if ok:
            for x, y in (("S", "B"), ("S", "Q"), ("B", "Q")):
                try:
                    audits[f"{x}{y}"] = audit_pair(spans[x], spans[y], rtext)
                except Exception as ex:  # noqa: BLE001
                    audits[f"{x}{y}"] = {"error": str(ex)[:200]}
        else:
            for a in errors:
                failures[a] += 1
        rows.append({"id": row["id"], "prompt_id": row["prompt_id"],
                     "rollout_id": row["id"],
                     "n_response_tokens": len(row["response_ids"]),
                     "errors": errors, "audits": audits})
    per_pair = {}
    for pair in ("SB", "SQ", "BQ"):
        prows = [{"prompt_id": r["prompt_id"], "rollout_id": r["id"],
                  "audit": r["audits"][pair]}
                 for r in rows
                 if not r["errors"] and pair in r["audits"]
                 and "error" not in r["audits"][pair]]
        per_pair[pair] = aggregate_audits(prows) if prows else {}
    dest = root / "work" / f"audit-{args.shard}.json"
    dest.write_text(json.dumps({"shard": args.shard, "rows": rows,
                                "encode_failures": failures,
                                "aggregate": per_pair}))
    print(f"wrote {dest} rows={len(rows)} failures={failures}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
