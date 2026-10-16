#!/usr/bin/env python3
"""A3.4: per-shard per-arm event scoring (single GPU).

For each fixture position: union candidates (reference + S/B/Q proposals),
KV-cache path mass per candidate, byte-event distribution, top-64 telemetry,
realized next-token logprob. Writes scores-{sh}-{arm}.json.

Operating dtype S/B bf16 + Q fp32. E-space labels (E1/E2/E3) are assigned in
D-H2 analysis from these raw records + calibration (not here).
Usage: SB_ROOT=... sb_score.py --shard 09 --arm S
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

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
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--positions-file", default="",
                    help="P-positions manifest (default: fixtures-{shard}.json)")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids
    from h2_measurements import vocabulary_bytes
    from h2_word_events import byte_event_map, boundary_flags, is_ascii_atom

    root = Path(args.root)
    work = root / "work"
    for name in (f"fixtures-{args.shard}.json", f"audit-{args.shard}.json",
                 f"pool-{args.shard}.json"):
        if not (work / name).is_file():
            print(f"REFUSE: {name} not ready", file=sys.stderr)
            sys.exit(2)
    fixtures = json.loads((work / f"fixtures-{args.shard}.json").read_text())["positions"]
    if args.positions_file:
        fixtures = [r for r in
                    json.loads(Path(args.positions_file).read_text())["positions"]
                    if r.get("shard", args.shard) == args.shard]
    if args.max_positions > 0:
        fixtures = fixtures[:args.max_positions]
    audit = {r["id"]: r for r in
             json.loads((work / f"audit-{args.shard}.json").read_text())["rows"]}
    pool = {r["id"]: r for r in
            json.loads((work / f"pool-{args.shard}.json").read_text())["rows"]}
    prop = {}
    for a in ("S", "B", "Q"):
        p = work / f"proposals-{args.shard}-{a}.json"
        if not p.is_file():
            print(f"REFUSE: proposals-{args.shard}-{a} not ready", file=sys.stderr)
            sys.exit(2)
        for r in json.loads(p.read_text())["rows"]:
            prop.setdefault(r["key"], set()).update(r["candidates"])
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
    s_tok = (tok if args.arm == "S" else AutoTokenizer.from_pretrained(
        assets["models"]["S"]["path"], local_files_only=True))

    def s_text(prow, resp_only=False):
        ids = prow["response_ids"] if resp_only else (
            prow["prompt_ids"] + prow["response_ids"])
        return s_tok.decode(ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)

    def s_prompt(prow):
        return s_tok.decode(prow["prompt_ids"], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    vocab = vocabulary_bytes(tok, "bpe")
    stop = set(terminal_token_ids(tok))
    head_size = model.get_output_embeddings().weight.shape[0]
    byte_map = torch.tensor(byte_event_map(vocab, head_size, stop),
                            device="cuda", dtype=torch.long)

    dest = work / f"scores-{args.shard}-{args.arm}.json"
    result = json.loads(dest.read_text())["rows"] if dest.is_file() else []
    done = {r["key"] for r in result}
    if [r["key"] for r in result] != [r["key"] for r in fixtures][:len(result)]:
        print("resume manifest differs; inspect before rerun", file=sys.stderr)
        sys.exit(3)
    began = time.monotonic()
    with torch.inference_mode():
        for n, row in enumerate(fixtures):
            if row["key"] in done:
                continue
            ids = row["prefix_ids"][args.arm]
            base = model.model(input_ids=torch.tensor([ids], device="cuda"),
                               use_cache=True)
            first_lp = canonical_head(
                model, base.last_hidden_state[0, -1:])[0].double().log_softmax(-1)
            first_probs = first_lp.exp()
            byte_probs = torch.zeros(258, device="cuda", dtype=torch.float64)
            byte_probs.scatter_add_(0, byte_map, first_probs)
            top_values, top_ids = first_lp.topk(64)
            # Realized next token: S index j; B/Q via stored strict maps over
            # re-encoded full text (cached per response).
            prow = pool[row["response_id"]]
            if args.arm == "S":
                realized = prow["response_ids"][row["position"]]
            else:
                smap = dict(audit[row["response_id"]]["strict"]
                            [f"S{args.arm}"])[row["position"]]
                full_ids = tok.encode(s_text(prow), add_special_tokens=False)
                pre = tok.encode(s_prompt(prow), add_special_tokens=False)
                realized = full_ids[len(pre) + smap]
            prefix = tok.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
            masses, reasons = [], []
            cands = sorted(set([row["reference_word"]]) | prop.get(row["key"], set()))
            for candidate in cands:
                suffix = tok.encode(candidate, add_special_tokens=False)
                full = ids + suffix
                if not suffix:
                    masses.append(None)
                    reasons.append("empty_candidate_encoding")
                    continue
                if tok.decode(full, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False) != prefix + candidate:
                    masses.append(None)
                    reasons.append("contextual_roundtrip_failed")
                    continue
                if is_ascii_atom(candidate):
                    masses.append(float(byte_probs[ord(candidate)]))
                    reasons.append(None)
                    continue
                cache = copy.deepcopy(base.past_key_values)
                ext = model.model(input_ids=torch.tensor([suffix], device="cuda"),
                                  past_key_values=cache, use_cache=True)
                h = torch.cat([base.last_hidden_state[0, -1:],
                               ext.last_hidden_state[0]], 0)
                lp = canonical_head(model, h).double().log_softmax(-1)
                labels = torch.tensor(suffix, device="cuda")
                masses.append(float(lp[:-1].gather(1, labels.unsqueeze(1)).sum()))
                reasons.append(None)
                del cache, ext
            rec = {"key": row["key"], "candidates": cands,
                   "masses": masses, "reasons": reasons,
                   "realized_token_id": realized,
                   "realized_logprob": (float(first_lp[realized])
                                        if realized is not None else None),
                   "byte_probs": byte_probs.cpu().tolist(),
                   "top64_ids": top_ids.cpu().tolist(),
                   "top64_logprobs": top_values.cpu().tolist()}
            result.append(rec)
            del base
            if len(result) % 20 == 0 or n == len(fixtures) - 1:
                dest.write_text(json.dumps(
                    {"arm": args.arm, "shard": args.shard, "rows": result,
                     "elapsed_seconds": round(time.monotonic() - began, 1),
                     "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                     "dtype": dtype}))
                print(f"score {args.shard}/{args.arm}: {len(result)}/{len(fixtures)}",
                      flush=True)
    print(f"wrote {dest} rows={len(result)}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
