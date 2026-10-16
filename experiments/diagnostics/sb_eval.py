#!/usr/bin/env python3
"""Dev/test accuracy eval for study checkpoints (single GPU).

Generates n samples per problem with the student (T1/top-p1/top-k0, cap1024)
and verifies with the frozen answer verifier. Writes eval-{tag}.json.

Problems: dev split from main_pool_manifest (128 gsm8k + 128 math) by default;
--test uses GSM8K-test + MATH-500 pins. --checkpoint: sb_train .pt (state_dict
under 'model') or 'initial' for the base student.
Usage: SB_ROOT=... sb_eval.py --tag L-B-5e7-step30 --split dev --samples 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SAMPLES_DEFAULT = 4
CAP = 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--samples", type=int, default=SAMPLES_DEFAULT)
    ap.add_argument("--checkpoint", default="initial")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
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
    if args.split == "dev":
        problems = ([("gsm8k", f"gsm8k-train-{i}")
                     for i in manifest["splits"]["gsm8k"]["dev"]]
                    + [("math", f"math-train-{i}")
                       for i in manifest["splits"]["math"]["dev"]])
        gsm = pd.read_parquet(dsets / "gsm8k-train.parquet")
        math = pd.read_parquet(dsets / "math-train.parquet")

        def qa(pid):
            if pid.startswith("gsm8k-"):
                r = gsm.iloc[int(pid.split("-")[-1])]
                return r["question"], r["answer"], "gsm8k"
            r = math.iloc[int(pid.split("-")[-1])]
            return r["problem"], r["solution"], "math"
    else:
        gsm = pd.read_parquet(dsets / "gsm8k-test.parquet")
        math = pd.read_parquet(dsets / "math-train.parquet")  # placeholder
        raise SystemExit("test split wired at eval-freeze; use dev for LR screens")
    if args.max_problems > 0:
        problems = problems[args.offset:args.offset + args.max_problems]

    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"]["S"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        entry["path"], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda"} if torch.cuda.is_available() else None).eval()
    if args.checkpoint != "initial":
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    stop = set(terminal_token_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Verifier ships with the repo (scripts/sft) — import by path.
    sys.path.insert(0, str(new / "scripts" / "sft"))
    from answer_verifier import verify_response

    def render(prompt):
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        try:
            ids = out["input_ids"]
        except TypeError:
            ids = out
        return [int(x) for x in ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    t0 = time.monotonic()
    with torch.inference_mode():
        for pi, (source, pid) in enumerate(problems):
            question, label, vsource = qa(pid)
            prefix = render(question)
            ids = torch.tensor([prefix] * args.samples, device=device)
            mask = torch.ones_like(ids)
            torch.manual_seed(42000 + pi)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=CAP,
                pad_token_id=pad, eos_token_id=sorted(stop), use_cache=True)
            hits = 0
            for gen in out[:, len(prefix):].tolist():
                end = next((i for i, v in enumerate(gen) if v in stop), len(gen))
                text = tok.decode(gen[:end], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
                try:
                    ok = bool(verify_response(response=text, label=str(label),
                                             source=vsource))
                except Exception:  # noqa: BLE001
                    ok = False
                hits += ok
            rows.append({"id": pid, "source": source, "hits": hits,
                         "n": args.samples})
            if (pi + 1) % 32 == 0:
                print(f"eval {args.tag}: {pi + 1}/{len(problems)}", flush=True)
    acc = sum(r["hits"] for r in rows) / max(sum(r["n"] for r in rows), 1)
    dest = root / "work" / f"eval-{args.tag}.json"
    dest.write_text(json.dumps({"tag": args.tag, "split": args.split,
                                "samples": args.samples,
                                "checkpoint": args.checkpoint,
                                "accuracy": acc,
                                "n_problems": len(rows),
                                "seconds": round(time.monotonic() - t0, 1),
                                "rows": rows}))
    print(f"wrote {dest} accuracy={acc:.4f} n={len(rows)}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
