#!/usr/bin/env python3
"""F-forks: counterfactual student continuations (single GPU).

Per position: force branch word (reference vs top alternative) as next event,
then free continuation with paired seeds (same seed across branches) or
independent seeds for agreement-null. Writes forks.json rows.
No quality judgment here (F-analysis, later).

Usage: SB_ROOT=... sb_forks.py --positions-file work/positions.json
       --proposals-file work/proposals-09-S.json --max-positions 20
       --continuations 4 --cap 1024 [--independent-seeds]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--positions-file", required=True)
    ap.add_argument("--proposals-file", required=True)
    ap.add_argument("--tag", default="screen")
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--continuations", type=int, default=4)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--independent-seeds", action="store_true")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids

    root = Path(args.root)
    work = root / "work"
    positions = json.loads(Path(args.positions_file).read_text())["positions"]
    if args.max_positions > 0:
        positions = positions[:args.max_positions]
    prop = {r["key"]: r["candidates"]
            for r in json.loads(Path(args.proposals_file).read_text())["rows"]}
    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"]["S"]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    stop = set(terminal_token_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    dest = work / f"forks-{args.tag}.json"
    result = json.loads(dest.read_text())["rows"] if dest.is_file() else []
    done = {(r["key"], r["branch"], r["seed"]) for r in result}
    t0 = time.monotonic()
    with torch.inference_mode():
        for row in positions:
            cands = prop.get(row["key"], [])
            alts = [c for c in cands if c != row["reference_word"]]
            if not alts:
                continue
            branches = [("ref", row["reference_word"]), ("alt", alts[0])]
            for branch, word in branches:
                suffix = tok.encode(word, add_special_tokens=False)
                base_ids = row["prefix_ids"]["S"]
                full = base_ids + suffix
                if not suffix or tok.decode(
                        full, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != tok.decode(
                        base_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) + word:
                    continue
                width = len(full)
                ids = torch.tensor([full] * args.continuations, device="cuda")
                for j in range(args.continuations):
                    # Paired (default): same seed across branches isolates the
                    # forced event. Independent: unique seed per branch (null).
                    seed = (95000 + j) if not args.independent_seeds else (
                        96000 + j if branch == "ref" else 97000 + j)
                    if (row["key"], branch, seed) in done:
                        continue
                    torch.manual_seed(seed)
                    out = model.generate(
                        input_ids=ids[j:j + 1], do_sample=True,
                        temperature=1.0, top_p=1.0, top_k=0,
                        repetition_penalty=1.0, max_new_tokens=args.cap,
                        pad_token_id=pad, eos_token_id=sorted(stop),
                        use_cache=True)
                    gen = out[0, width:].tolist()
                    end = next((i for i, v in enumerate(gen) if v in stop), len(gen))
                    result.append({"key": row["key"], "branch": branch,
                                   "forced_word": word, "seed": seed,
                                   "continuation_ids": gen[:end],
                                   "truncated": end == len(gen)})
                dest.write_text(json.dumps({"tag": args.tag, "rows": result}))
    print(f"wrote {dest} rows={len(result)} "
          f"({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
