#!/usr/bin/env python3
"""D-pool: production student generation for one main-pool shard (single GPU).

One shard = 200 prompts x 4 responses (seed42, T1/top-p1/top-k0, cap1024),
resume-safe, logical task ids shared with quality tasks.
Writes $SB_ROOT/work/pool-{shard}.json.

Usage: SB_ROOT=... SB_NEW=... sb_gpu_run.sh pool --arm S --shard 00 --gpu 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

RESPONSES_PER_PROMPT = 4
SEED = 42
MAX_NEW_TOKENS = 1024
BATCH = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids

    root = Path(args.root)
    new = Path(os.environ.get("SB_NEW", ""))
    manifest = json.loads((new / "experiments" / "diagnostics" / "artifacts"
                           / "structural-behavioral-main-runs"
                           / "main_pool_manifest.json").read_text())
    shard_ids = manifest["shards"].get(args.shard)
    if not shard_ids or len(shard_ids) != 200:
        print(f"bad shard {args.shard}", file=sys.stderr)
        sys.exit(2)
    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"]["S"]

    dsets = root / "assets" / "datasets"
    gsm = pd.read_parquet(dsets / "gsm8k-train.parquet")
    math = pd.read_parquet(dsets / "math-train.parquet")

    def text_of(pid: str) -> str:
        if pid.startswith("gsm8k-train-"):
            return gsm.iloc[int(pid.split("-")[-1])]["question"]
        return math.iloc[int(pid.split("-")[-1])]["problem"]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        entry["path"], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda"}).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    stop = set(terminal_token_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def render(prompt: str):
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        try:
            ids = out["input_ids"]
        except TypeError:
            ids = out
        return [int(x) for x in ids]

    tasks = [(pid, j) for pid in shard_ids for j in range(RESPONSES_PER_PROMPT)]
    dest = root / "work" / f"pool-{args.shard}.json"
    rows = json.loads(dest.read_text())["rows"] if dest.is_file() else []
    expected = [f"{pid}-s{j}" for pid, j in tasks]
    if [r["id"] for r in rows] != expected[:len(rows)]:
        print("resume manifest differs; inspect before rerun", file=sys.stderr)
        sys.exit(3)
    t0 = time.monotonic()
    for off in range(len(rows), len(tasks), BATCH):
        batch = tasks[off:off + BATCH]
        prefixes = [render(text_of(pid)) for pid, _ in batch]
        width = max(map(len, prefixes))
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prefixes],
                           device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes],
                            device="cuda")
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        torch.manual_seed(SEED + off)
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids, attention_mask=mask, position_ids=pos,
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad, eos_token_id=sorted(stop), use_cache=True)
        for (pid, j), prefix, gen in zip(batch, prefixes, out[:, width:].tolist()):
            end = next((i for i, v in enumerate(gen) if v in stop), len(gen))
            rows.append({"id": f"{pid}-s{j}", "prompt_id": pid, "sample": j,
                         "prompt_ids": prefix, "response_ids": gen[:end],
                         "truncated": end == len(gen)})
        dest.write_text(json.dumps({"shard": args.shard, "rows": rows}))
        print(f"shard {args.shard}: {len(rows)}/{len(tasks)} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"wrote {dest} rows={len(rows)} truncated="
          f"{sum(1 for r in rows if r['truncated'])}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
