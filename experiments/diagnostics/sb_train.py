#!/usr/bin/env python3
"""A5: on-policy reverse-KL study training, single GPU, eager stack.

Recipe (sb-main-v2, frozen): seed42; full-param AdamW(lr, 0.9/0.999, 1e-8, wd 0);
cosine horizon150 warmup8; clip 1.0; reverse KL on audited support, normalized
by total pre-mask response tokens; 64 prompts/update x4 rollouts, T1/top-p1/
top-k0, cap1024; checkpoints 25/50/75/100/125/150 (+step_latest for resume).

Run table (policy): B-all | B-strict | Q-strict | B-shift15/05/30 |
Q-random25 | B-random25. Top25 mask arms refuse (need X estimator gate).

Usage: SB_ROOT=... SB_NEW=... sb_train.py --run T-B-all --updates 150
Smoke: --smoke (2 prompts x1 rollout x16 tokens x2 updates, same code path).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROMPTS_PER_UPDATE = 64
ROLLOUTS = 4
CAP = 1024
HORIZON = 150
WARMUP = 8
CLIP = 1.0
MICRO_ROLL = 8
MICRO_FWD = 4

RUNS = {
    # run: (teacher, policy, lr, updates)
    "L-B-5e7": ("B", "all", 5e-7, 30),
    "L-B-1e6": ("B", "all", 1e-6, 30),
    "L-B-2e6": ("B", "all", 2e-6, 30),
    "T-B-all": ("B", "all", None, 150),
    "T-B-strict": ("B", "strict", None, 150),
    "T-Q-strict": ("Q", "strict", None, 150),
    "T-B-shift15": ("B", "shift15", None, 150),
    "T-B-shift05": ("B", "shift05", None, 150),
    "T-B-shift30": ("B", "shift30", None, 150),
    "T-Q-random25": ("Q", "random25", None, 150),
    "T-B-random25": ("B", "random25", None, 150),
    # Smoke-only: self-distill validates the loop (rollout/loss/update/
    # checkpoint/resume) without a second large teacher. REFUSED unless --smoke.
    "T-SS-self": ("S", "strict", 1e-6, 2),
}
DTYPES = {"S": "bfloat16", "B": "bfloat16", "Q": "float32"}


def detkey(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def load_lm(path, dtype, trainable):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype,
            attn_implementation="eager",
            device_map={"": "cuda"} if torch.cuda.is_available() else None).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",
            device_map={"": "cuda"} if torch.cuda.is_available() else None).eval()
    model.requires_grad_(trainable)
    if trainable:
        model.train()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer  # noqa: F401
    from hf_frozen_sampler import terminal_token_ids
    from h2_measurements import byte_spans, strict_pairs

    teacher_arm, policy, base_lr, base_updates = RUNS[args.run]
    if args.run == "T-SS-self" and not args.smoke:
        print("REFUSE: T-SS-self is smoke-only (use --smoke)", file=sys.stderr)
        sys.exit(2)
    lr = args.lr if args.lr is not None else base_lr
    if lr is None:
        print(f"REFUSE: {args.run} needs --lr (P-LR-freeze first)", file=sys.stderr)
        sys.exit(2)
    n_updates = args.updates if args.updates is not None else base_updates
    if policy.startswith("shift"):
        rate = {"shift05": 0.05, "shift15": 0.15, "shift30": 0.30}[policy]
    if "top25" in policy:
        print("REFUSE: top25 needs X-estimator gate", file=sys.stderr)
        sys.exit(2)

    root = Path(args.root)
    new = Path(os.environ.get("SB_NEW", ""))
    manifest = json.loads((new / "experiments" / "diagnostics" / "artifacts"
                           / "structural-behavioral-main-runs"
                           / "main_pool_manifest.json").read_text())
    import pandas as pd
    dsets = root / "assets" / "datasets"
    gsm = pd.read_parquet(dsets / "gsm8k-train.parquet")
    math = pd.read_parquet(dsets / "math-train.parquet")

    def text_of(pid):
        if pid.startswith("gsm8k-train-"):
            return gsm.iloc[int(pid.split("-")[-1])]["question"]
        return math.iloc[int(pid.split("-")[-1])]["problem"]

    train_ids = (manifest["splits"]["gsm8k"]["main"]
                 + manifest["splits"]["math"]["main"])
    assets = json.loads((root / "assets.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student, s_tok = load_lm(assets["models"]["S"]["path"], DTYPES["S"], True)
    teacher, t_tok = load_lm(assets["models"][teacher_arm]["path"],
                             DTYPES[teacher_arm], False)
    stop = set(terminal_token_ids(s_tok))
    pad = s_tok.pad_token_id if s_tok.pad_token_id is not None else s_tok.eos_token_id
    s_vocab = __import__("h2_measurements").vocabulary_bytes(s_tok, "bpe")
    t_vocab = __import__("h2_measurements").vocabulary_bytes(t_tok, "bpe")

    def render(prompt):
        out = s_tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        try:
            ids = out["input_ids"]
        except TypeError:
            ids = out
        return [int(x) for x in ids]

    run_dir = root / "work" / "train" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    def sched_lr(u):
        # u = completed updates; cosine horizon with linear warmup.
        if u < WARMUP:
            return lr * (u + 1) / WARMUP
        t = min(1.0, (u + 1 - WARMUP) / (HORIZON - WARMUP))
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))

    start_u = 0
    latest = run_dir / "step_latest.pt"
    if latest.is_file():
        ck = torch.load(str(latest), map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_u = ck["update"]
        print(f"resumed {args.run} at update {start_u}", flush=True)

    n_prompts = 2 if args.smoke else PROMPTS_PER_UPDATE
    n_roll = 1 if args.smoke else ROLLOUTS
    cap = 16 if args.smoke else CAP
    n_updates = 2 if args.smoke else n_updates
    logf = open(run_dir / "updates.jsonl", "a")

    for u in range(start_u, n_updates):
        for pg in opt.param_groups:
            pg["lr"] = sched_lr(u)
        # Deterministic prompt rotation over manifest order.
        order = train_ids if not args.smoke else train_ids[:n_prompts]
        if not args.smoke:
            off = (u * PROMPTS_PER_UPDATE) % len(train_ids)
            order = [train_ids[(off + i) % len(train_ids)]
                     for i in range(PROMPTS_PER_UPDATE)]
        prompts = [text_of(pid) for pid in order]
        # --- on-policy rollout (no grad) ---
        seqs = []  # (prompt_ids, response_ids)
        student.eval()
        with torch.inference_mode():
            for off in range(0, len(prompts) * n_roll, MICRO_ROLL):
                jobs = []
                for k in range(off, min(off + MICRO_ROLL, len(prompts) * n_roll)):
                    pi, r = divmod(k, n_roll)
                    jobs.append((pi, r))
                prefixes = [render(prompts[pi]) for pi, _ in jobs]
                width = max(map(len, prefixes))
                ids = torch.tensor([[pad] * (width - len(p)) + p for p in prefixes],
                                   device=device)
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p)
                                     for p in prefixes], device=device)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                torch.manual_seed(42 + u * 100003 + off)
                out = student.generate(
                    input_ids=ids, attention_mask=mask, position_ids=pos,
                    do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, max_new_tokens=cap,
                    pad_token_id=pad, eos_token_id=sorted(stop), use_cache=True)
                for (pi, _), pre, gen in zip(jobs, prefixes, out[:, width:].tolist()):
                    end = next((i for i, v in enumerate(gen) if v in stop), len(gen))
                    seqs.append((pre, gen[:end]))
                del out
        # --- loss over micro-batches ---
        student.train()
        opt.zero_grad()
        tot_loss_num, tot_tok, t0 = 0.0, 0, time.monotonic()
        for moff in range(0, len(seqs), MICRO_FWD):
            chunk = seqs[moff:moff + MICRO_FWD]
            full = [p + r for p, r in chunk]
            width = max(map(len, full))
            ids = torch.tensor([[pad] * (width - len(f)) + f for f in full],
                               device=device)
            attn = torch.tensor([[0] * (width - len(f)) + [1] * len(f)
                                 for f in full], device=device)
            with torch.inference_mode():
                t_lp_all = teacher(input_ids=ids, attention_mask=attn).logits.double().log_softmax(-1)
            s_logits = student(input_ids=ids, attention_mask=attn).logits.double()
            s_lp = s_logits.log_softmax(-1)
            s_pr = s_lp.exp()
            # Position mask: response region, teacher support, policy rule.
            loss_num = s_lp.new_zeros(())
            ntok = 0
            for bi, (pre, resp) in enumerate(chunk):
                plen = len(pre)
                rlen = len(resp)
                if rlen == 0:
                    continue
                s_ids = list(range(plen, plen + rlen))
                # Teacher support via re-encode (joint strict).
                keep = set(range(rlen))
                if policy != "all":
                    text = s_tok.decode(pre + resp, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                    ptext = s_tok.decode(pre, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
                    rtext = s_tok.decode(resp, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
                    try:
                        tfull = t_tok.encode(text, add_special_tokens=False)
                        tpre = t_tok.encode(ptext, add_special_tokens=False)
                        assert tfull[:len(tpre)] == tpre
                        s_sp = byte_spans(resp, s_vocab, rtext)
                        t_sp = byte_spans(tfull[len(tpre):], t_vocab, rtext)
                        smap = strict_pairs(s_sp, t_sp)
                        keep = set(smap)
                    except Exception:  # noqa: BLE001
                        keep = set()
                if not keep:
                    continue
                t_row = t_lp_all[bi, plen - 1:plen + rlen - 1]
                s_row = s_lp[bi, plen - 1:plen + rlen - 1]
                s_pr_row = s_pr[bi, plen - 1:plen + rlen - 1]
                idx = torch.tensor(sorted(keep), device=device)
                if policy.startswith("shift"):
                    n_cor = max(1, int(round(rate * len(idx)))) if len(idx) else 0
                    h = sorted(idx.tolist(),
                               key=lambda i: detkey(args.run, u, bi, i))[:n_cor]
                    for ci in h:
                        d = 1 if int(detkey(args.run, bi, ci), 16) % 2 == 0 else -1
                        nb = min(max(ci + d, 0), rlen - 1)
                        t_row = t_row.clone()
                        t_row[ci] = t_lp_all[bi, plen - 1 + nb]
                if policy.endswith("random25"):
                    h = sorted(idx.tolist(),
                               key=lambda i: detkey(args.run, u, bi, i))
                    drop = set(h[:len(h) // 4]) if h else set()
                    idx = torch.tensor([i for i in idx.tolist() if i not in drop],
                                       device=device)
                    if len(idx) == 0:
                        continue
                kl = (s_pr_row[idx] * (s_row[idx] - t_row[idx])).sum()
                loss_num = loss_num + kl
                ntok += len(resp)
            tot_loss_num += float(loss_num.detach())
            tot_tok += ntok
            if ntok:
                # Exact contract normalizer: backward raw sums now, divide
                # the accumulated grads by TOTAL pre-mask tokens below.
                loss_num.backward()
        if tot_tok == 0:
            print(f"update {u}: no eligible tokens, skip", flush=True)
            continue
        with torch.no_grad():
            for p in student.parameters():
                if p.grad is not None:
                    p.grad.div_(tot_tok)
        torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP)
        opt.step()
        rec = {"update": u + 1, "loss": tot_loss_num / tot_tok,
               "tokens": tot_tok, "lr": sched_lr(u),
               "seconds": round(time.monotonic() - t0, 1)}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"{args.run} update {u + 1}/{n_updates} " + json.dumps(rec), flush=True)
        torch.save({"update": u + 1,
                    "model": student.state_dict(),
                    "optim": opt.state_dict()}, str(latest))
        if (u + 1) in (25, 50, 75, 100, 125, 150):
            torch.save({"update": u + 1, "model": student.state_dict()},
                       str(run_dir / f"checkpoint-step{u + 1}.pt"))
    logf.close()
    print(f"DONE {args.run} updates={n_updates}")


if __name__ == "__main__":
    main()
