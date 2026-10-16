#!/usr/bin/env python3
"""C-quality: 8-samples generation with pool reuse (single GPU).

Screen: first 128 gsm8k-main + 128 math-main prompts x 8 samples.
Full: 2000 train + GSM8K-test 1319 + MATH-500 x 8 samples.
Reuse rule: for main-train prompts, samples 0-3 are COPIED from pool files
(same logical sample ids); only samples 4-7 are generated. Test prompts and
extra samples are always generated. Writes quality-{screen,full}-{arm}.json.

Arms: S (+B/Q full only when needed). Smoke: --max-prompts N.
Usage: SB_ROOT=... sb_quality.py --kind screen --arm S
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

CAP = 1024
BATCH = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--kind", required=True, choices=("screen", "full"))
    ap.add_argument("--arm", required=True, choices=("S", "B", "Q"))
    ap.add_argument("--max-prompts", type=int, default=0)
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
    dsets = root / "assets" / "datasets"
    gsm_train = pd.read_parquet(dsets / "gsm8k-train.parquet")
    math_train = pd.read_parquet(dsets / "math-train.parquet")

    prompts = []  # (pid, text, from_pool?)
    if args.kind == "screen":
        for pid in manifest["splits"]["gsm8k"]["main"][:128]:
            prompts.append((pid, gsm_train.iloc[int(pid.split("-")[-1])]["question"], True))
        for pid in manifest["splits"]["math"]["main"][:128]:
            prompts.append((pid, math_train.iloc[int(pid.split("-")[-1])]["problem"], True))
    else:
        for pid in manifest["splits"]["gsm8k"]["main"] + manifest["splits"]["math"]["main"]:
            src = gsm_train if pid.startswith("gsm8k-") else math_train
            col = "question" if pid.startswith("gsm8k-") else "problem"
            prompts.append((pid, src.iloc[int(pid.split("-")[-1])][col], True))
        gsm_test = pd.read_parquet(dsets / "gsm8k-test.parquet")
        for i in range(len(gsm_test)):
            prompts.append((f"gsm8k-test-{i}", gsm_test.iloc[i]["question"], False))
        import json as _j
        math500 = [ _j.loads(l) for l in
                    (dsets / "math500-test.jsonl").read_text().splitlines()]
        for i, row in enumerate(math500):
            prompts.append((f"math500-test-{i}",
                            row.get("problem", row.get("question", "")), False))
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    # Pool reuse index (samples 0-3) for main-train prompts.
    pool_idx = {}
    if args.arm == "S":
        for shard, ids in manifest["shards"].items():
            pp = root / "work" / f"pool-{shard}.json"
            if not pp.is_file():
                continue
            for r in json.loads(pp.read_text())["rows"]:
                pool_idx.setdefault(r["prompt_id"], {})[r["sample"]] = r
    else:
        print(f"REFUSE: quality-full {args.arm} needs teacher policy freeze",
              file=sys.stderr)
        sys.exit(2)

    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"][args.arm]
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

    def render(prompt):
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        try:
            ids = out["input_ids"]
        except TypeError:
            ids = out
        return [int(x) for x in ids]

    dest = root / "work" / f"quality-{args.kind}-{args.arm}.json"
    rows = json.loads(dest.read_text())["rows"] if dest.is_file() else []
    done = {(r["prompt_id"], r["sample"]) for r in rows}
    t0 = time.monotonic()
    gen_jobs = []  # (pid, text, sample) needing generation
    for pid, text, from_pool in prompts:
        for s in range(8):
            if (pid, s) in done:
                continue
            if from_pool and s < 4 and s in pool_idx.get(pid, {}):
                pr = pool_idx[pid][s]
                rows.append({"prompt_id": pid, "sample": s,
                             "prompt_ids": pr["prompt_ids"],
                             "response_ids": pr["response_ids"],
                             "truncated": pr["truncated"], "reused_from_pool": True})
                continue
            gen_jobs.append((pid, text, s))
    with torch.inference_mode():
        for off in range(0, len(gen_jobs), BATCH):
            batch = gen_jobs[off:off + BATCH]
            prefixes = [render(t) for _, t, _ in batch]
            width = max(map(len, prefixes))
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in prefixes],
                               device="cuda")
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p)
                                 for p in prefixes], device="cuda")
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            torch.manual_seed(4200 + off)
            out = model.generate(
                input_ids=ids, attention_mask=mask, position_ids=pos,
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=CAP,
                pad_token_id=pad, eos_token_id=sorted(stop), use_cache=True)
            for (pid, _, s), pre, gen in zip(batch, prefixes, out[:, width:].tolist()):
                end = next((i for i, v in enumerate(gen) if v in stop), len(gen))
                rows.append({"prompt_id": pid, "sample": s, "prompt_ids": pre,
                             "response_ids": gen[:end],
                             "truncated": end == len(gen),
                             "reused_from_pool": False})
            dest.write_text(json.dumps({"kind": args.kind, "arm": args.arm,
                                        "rows": rows}))
            print(f"quality {args.kind}/{args.arm}: {len(rows)} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"wrote {dest} rows={len(rows)}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
