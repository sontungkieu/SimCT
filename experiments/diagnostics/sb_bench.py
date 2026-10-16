#!/usr/bin/env python3
"""V-bench: one arm + dtype fit measurement (single GPU).

Measures: load time, VRAM peak, generation throughput on 8 main prompts,
cap-hit count, top-20 snapshot on 4 fixed probe prefixes (for later
bf16/fp32 rank-concordance in bench disposition).
Writes $SB_ROOT/work/bench-{arm}-{dtype}.json.

Usage: SB_ROOT=... SB_NEW=... sb_gpu_run.sh bench --arm S --dtype bfloat16 --gpu 1
Prompts resolve from pinned parquets + repo main_pool_manifest.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

N_GEN_PROMPTS = 8
GEN_TOKENS = 64
PROBES = ["2+2=4.\n", "A fraction: \\frac{1}{2}.",
          "Summarize: hello world.", "If x+5=12, x="]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--arm", required=True, choices=("S", "B", "Q"))
    ap.add_argument("--dtype", required=True, choices=("bfloat16", "float32"))
    args = ap.parse_args()
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids
    from h2_pilot import render

    root = Path(args.root)
    new = Path(os.environ.get("SB_NEW", ""))
    manifest = json.loads((new / "experiments" / "diagnostics" / "artifacts"
                           / "structural-behavioral-main-runs"
                           / "main_pool_manifest.json").read_text())
    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"][args.arm]
    out = {"arm": args.arm, "dtype": args.dtype}

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True, dtype=args.dtype,
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="eager", device_map={"": "cuda"}).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    out["load_s"] = round(time.monotonic() - t0, 1)
    out["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    dsets = root / "assets" / "datasets"
    gsm = pd.read_parquet(dsets / "gsm8k-train.parquet")
    math = pd.read_parquet(dsets / "math-train.parquet")

    def text_of(pid: str) -> str:
        if pid.startswith("gsm8k-train-"):
            return gsm.iloc[int(pid.split("-")[-1])]["question"]
        return math.iloc[int(pid.split("-")[-1])]["problem"]

    main_ids = (manifest["splits"]["gsm8k"]["main"][:4]
                + manifest["splits"]["math"]["main"][:4])
    stop = set(terminal_token_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prefixes = [render(tok, text_of(pid)) for pid in main_ids]
    width = max(map(len, prefixes))
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in prefixes], device="cuda")
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes],
                        device="cuda")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    torch.manual_seed(42)
    t1 = time.monotonic()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=ids, attention_mask=mask, position_ids=pos,
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=GEN_TOKENS,
            pad_token_id=pad, eos_token_id=sorted(stop), use_cache=True)
    dt = time.monotonic() - t1
    new_toks = int((gen[:, width:].ne(pad)).sum())
    out["gen_tok_per_s"] = round(new_toks / dt, 1)
    out["gen_s"] = round(dt, 1)
    capped = sum(1 for r in gen[:, width:].tolist()
                 if not any(v in stop for v in r))
    out["cap_hit_64"] = capped
    out["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    topk = {}
    with torch.inference_mode():
        for i, text in enumerate(PROBES):
            x = torch.tensor([tok.encode(text, add_special_tokens=False)], device="cuda")
            lp = model(input_ids=x, use_cache=False).logits.double()[0, -1]
            topk[f"p{i}"] = torch.topk(lp, 20).indices.tolist()
    out["top20"] = topk
    dest = root / "work" / f"bench-{args.arm}-{args.dtype}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    print(json.dumps({k: v for k, v in out.items() if k != "top20"}, indent=1))


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
