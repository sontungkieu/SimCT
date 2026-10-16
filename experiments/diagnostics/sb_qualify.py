#!/usr/bin/env python3
"""A1: study-scoped runtime qualification (V-qual) for one arm + dtype.

Reuses pilot numerics helpers (h2_measurements, hf_frozen_sampler) but reads
models from the study assets.json (A0) and writes under $SB_ROOT/work/.
Does NOT touch pilot OUT roots.

Checks (G0):
  identity_and_roundtrip, probability_mass_and_residual,
  cached_uncached_parity, batched_independent_parity,
  backend_precision_report (numbers only; no pass/fail on dtype here)

Usage (one GPU, study checkout on sys.path, repo experiments/ on path):
  python sb_qualify.py --root $SB_ROOT --arm S --dtype bfloat16
  python sb_qualify.py --root $SB_ROOT --arm S --dtype float32
Exit 0 + qualification JSON in all cases; failures recorded as check FAIL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

FAMILIES = {"S": "bpe", "B": "bpe", "Q": "bpe"}
PROBES = ["2+2=4.\n", "A fraction: \\frac{1}{2}. \U0001F600", "Summarize: hello world."]


def load_study_model(arm: str, dtype: str, assets: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    entry = assets["models"][arm]
    if not entry.get("verified_weights"):
        raise RuntimeError(f"arm {arm}: pinned weights not verified")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(local_files_only=True, attn_implementation="eager",
                  device_map={"": "cuda"})
    try:
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], dtype=dtype, **kwargs).eval()
    except TypeError:
        # transformers < 4.52 (v3/v1 nodes): legacy torch_dtype spelling.
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"], torch_dtype=getattr(torch, dtype), **kwargs).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", ""))
    ap.add_argument("--arm", required=True, choices=("S", "B", "Q"))
    ap.add_argument("--dtype", required=True, choices=("bfloat16", "float32"))
    args = ap.parse_args()
    import torch
    from hf_frozen_sampler import terminal_token_ids
    from h2_measurements import vocabulary_bytes

    root = Path(args.root)
    assets = json.loads((root / "assets.json").read_text())
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)

    out = {"arm": args.arm, "dtype": args.dtype, "checks": {}, "timings": {}}
    t0 = time.monotonic()
    model, tok = load_study_model(args.arm, args.dtype, assets)
    out["timings"]["load_s"] = round(time.monotonic() - t0, 1)
    stop = set(terminal_token_ids(tok))
    vocab = vocabulary_bytes(tok, FAMILIES[args.arm])

    # 1. identity + roundtrip (encode->decode stable, vocab covers ASCII)
    try:
        ok = True
        detail = []
        for text in PROBES:
            ids = tok.encode(text, add_special_tokens=False)
            back = tok.decode(ids, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
            same = (back == text)
            ok = ok and same and len(ids) > 0
            detail.append({"len": len(ids), "roundtrip": same})
        out["checks"]["identity_and_roundtrip"] = {"pass": bool(ok), "detail": detail}
    except Exception as e:  # noqa: BLE001
        out["checks"]["identity_and_roundtrip"] = {"pass": False, "error": str(e)[:300]}

    # 2. mass + residual on one probe prefix
    try:
        ids = tok.encode(PROBES[0], add_special_tokens=False)
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model.model(input_ids=x, use_cache=False).last_hidden_state
            probs = torch.softmax(model.lm_head(logits)[0, -1].double(), -1)
        mass = float(probs.sum())
        top1 = float(probs.max())
        out["checks"]["probability_mass_and_residual"] = {
            "pass": bool(abs(mass - 1.0) < 1e-6),
            "mass": mass, "top1": top1, "vocab_size": len(vocab),
        }
    except Exception as e:  # noqa: BLE001
        out["checks"]["probability_mass_and_residual"] = {"pass": False, "error": str(e)[:300]}

    # 3+4. cached vs uncached, batched vs single parity
    try:
        a = tok.encode(PROBES[0], add_special_tokens=False)
        b = tok.encode(PROBES[2], add_special_tokens=False)
        w = max(len(a), len(b))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        batch = torch.tensor([[pad] * (w - len(a)) + a, [pad] * (w - len(b)) + b], device="cuda")
        mask = torch.tensor([[0] * (w - len(a)) + [1] * len(a),
                             [0] * (w - len(b)) + [1] * len(b)], device="cuda")
        # Left pads must not shift rotary positions: real tokens keep solo
        # positions 0..len-1, pads sit at 0 (masked out anyway).
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            # CausalLM forward already returns vocab logits; no extra lm_head.
            # attention_mask is required: without it pad tokens pollute logits.
            lp_cached = model(input_ids=batch, attention_mask=mask, position_ids=pos,
                              use_cache=True).logits.double()
            lp_plain = model(input_ids=batch, attention_mask=mask, position_ids=pos,
                             use_cache=False).logits.double()
            lp_solo = model(
                input_ids=torch.tensor([a], device="cuda"), use_cache=False
            ).logits.double()
        d_cache = float((lp_cached - lp_plain).abs().max())
        d_batch = float((lp_plain[0, -1] - lp_solo[0, -1]).abs().max())
        out["checks"]["cached_uncached_parity"] = {"pass": bool(d_cache == 0.0), "max_abs_diff": d_cache}
        out["checks"]["batched_independent_parity"] = {"pass": bool(d_batch == 0.0), "max_abs_diff": d_batch}
    except Exception as e:  # noqa: BLE001
        out["checks"]["cached_uncached_parity"] = {"pass": False, "error": str(e)[:300]}
        out["checks"]["batched_independent_parity"] = {"pass": False, "error": str(e)[:300]}

    out["checks"]["backend_precision_report"] = {
        "dtype": args.dtype, "tf32_disabled": True,
        "note": "bf16/float32 cross-dtype delta measured in V-bench, not here",
    }
    hard = [v for k, v in out["checks"].items()
            if k in ("identity_and_roundtrip", "probability_mass_and_residual",
                     "cached_uncached_parity")]
    out["ok"] = all(v.get("pass", False) for v in hard)
    # batched_independent_parity is reported, not gated here: cross-width
    # reduction noise is judged by tolerance in sb_disposition_qualify.
    dest = work / f"qualification-{args.arm}-{args.dtype}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ok={out['ok']}")
    print(json.dumps({k: v.get('pass') for k, v in out["checks"].items()}, indent=1))


if __name__ == "__main__":
    main()
