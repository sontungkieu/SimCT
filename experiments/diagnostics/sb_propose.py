#!/usr/bin/env python3
"""A3.3: per-shard per-arm candidate proposal (single GPU).

For each fixture position: top-k next-token seeds extended greedily to word
boundaries (pilot proposal contract). Writes proposals-{sh}-{arm}.json.

Operating: top_k=10, max_tokens=4, dtype S/B bf16 + Q fp32.
Usage: SB_ROOT=... sb_propose.py --shard 09 --arm S
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

TOP_K = 10
MAX_TOKENS = 4
DTYPES = {"S": "bfloat16", "B": "bfloat16", "Q": "float32"}


def canonical_head(model, hidden):
    import torch
    logits = model.get_output_embeddings()(hidden)
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--arm", required=True, choices=("S", "B", "Q"))
    ap.add_argument("--max-positions", type=int, default=0,
                    help="smoke cap (0 = all)")
    ap.add_argument("--positions-file", default="",
                    help="P-positions manifest (default: fixtures-{shard}.json)")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids
    from h2_measurements import vocabulary_bytes
    from h2_word_events import next_word

    root = Path(args.root)
    work = root / "work"
    fx_path = work / f"fixtures-{args.shard}.json"
    if not fx_path.is_file():
        print(f"REFUSE: fixtures-{args.shard} not ready", file=sys.stderr)
        sys.exit(2)
    rows = json.loads(fx_path.read_text())["positions"]
    if args.positions_file:
        rows = json.loads(Path(args.positions_file).read_text())["positions"]
        rows = [r for r in rows if r.get("shard", args.shard) == args.shard]
    if args.max_positions > 0:
        rows = rows[:args.max_positions]
    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"][args.arm]
    dtype = DTYPES[args.arm]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True, dtype=dtype,
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    vocab = vocabulary_bytes(tok, "bpe")
    stop = set(terminal_token_ids(tok))
    special = set(tok.all_special_ids)

    dest = work / f"proposals-{args.shard}-{args.arm}.json"
    result = json.loads(dest.read_text())["rows"] if dest.is_file() else []
    done = {r["key"] for r in result}
    if [r["key"] for r in result] != [r["key"] for r in rows][:len(result)]:
        print("resume manifest differs; inspect before rerun", file=sys.stderr)
        sys.exit(3)
    began = time.monotonic()
    with torch.inference_mode():
        for n, row in enumerate(rows):
            if row["key"] in done:
                continue
            tick = time.monotonic()
            ids = row["prefix_ids"][args.arm]
            base = model.model(input_ids=torch.tensor([ids], device="cuda"),
                               use_cache=True)
            logits = canonical_head(model, base.last_hidden_state[0, -1:])[0]
            top = [i for i in logits.topk(32).indices.tolist()
                   if i < len(tok) and i not in special][:TOP_K]
            prefix = tok.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
            candidates, incomplete = [], 0
            for tid in top:
                tail = [tid]
                cache = copy.deepcopy(base.past_key_values)
                for depth in range(MAX_TOKENS):
                    ended = tail[-1] in stop
                    whole = tok.decode(
                        ids + (tail[:-1] if ended else tail),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False)
                    if not whole.startswith(prefix):
                        raise RuntimeError("proposal changed decoded prefix")
                    lexical = tail[:-1] if ended else tail
                    if any(vocab[i] is None for i in lexical):
                        candidate, ended = None, True
                    else:
                        try:
                            tail_text = b"".join(vocab[i] for i in lexical).decode("utf-8")
                            candidate = next_word(tail_text, terminated=ended)
                        except UnicodeDecodeError:
                            candidate = None
                    if candidate or ended or depth == MAX_TOKENS - 1:
                        break
                    ext = model.model(
                        input_ids=torch.tensor([[tail[-1]]], device="cuda"),
                        past_key_values=cache, use_cache=True)
                    cache = ext.past_key_values
                    tail.append(int(canonical_head(
                        model, ext.last_hidden_state[0, -1:]).argmax(-1)))
                    del ext
                if candidate:
                    candidates.append(candidate)
                else:
                    incomplete += 1
                del cache
            torch.cuda.synchronize()
            result.append({"key": row["key"], "candidates": sorted(set(candidates)),
                           "incomplete_proposals": incomplete,
                           "wall_seconds": round(time.monotonic() - tick, 2)})
            del base
            if len(result) % 50 == 0 or n == len(rows) - 1:
                dest.write_text(json.dumps(
                    {"arm": args.arm, "shard": args.shard, "rows": result,
                     "elapsed_seconds": round(time.monotonic() - began, 1),
                     "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                     "dtype": dtype}))
                print(f"propose {args.shard}/{args.arm}: {len(result)}/{len(rows)}",
                      flush=True)
    print(f"wrote {dest} rows={len(result)}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
