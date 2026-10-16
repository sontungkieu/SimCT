#!/usr/bin/env python3
"""V-B-bridge: official-B vs archived mirror-B on pilot positions (single GPU).

Condition native: score pilot candidates from stored prefix_text with the
official tokenizer/model. Condition controlled: re-render the original prompt
with the official template and compare ids. Reports per-candidate mass
max/mean delta vs mirror + template/prefix audit. No regeneration.

Inputs (shipped): pilot word-candidates.json (positions[pREFIX_TEXT]),
word-scores-B.json (mirror masses), word-prompts.json (pool prompt text).
Writes bridge.json. Usage: SB_ROOT=... sb_bridge.py [--n 200]
"""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path


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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pilot-dir", default=os.environ.get("SB_PILOT_DIR", ""))
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(args.root)
    pilot = Path(args.pilot_dir)
    for name in ("word-candidates.json", "word-scores-B.json", "word-prompts.json"):
        if not (pilot / name).is_file():
            print(f"REFUSE: pilot {name} missing", file=sys.stderr)
            sys.exit(2)
    cand = json.loads((pilot / "word-candidates.json").read_text())["positions"]
    mirror = {r["key"]: r for r in
              json.loads((pilot / "word-scores-B.json").read_text())["rows"]}
    pool = {p["id"]: p["prompt"] for p in
            json.loads((pilot / "word-prompts.json").read_text())["pool"]}
    cand.sort(key=lambda r: r.get("rank", ""))
    cand = cand[:args.n]
    assets = json.loads((root / "assets.json").read_text())
    entry = assets["models"]["B"]
    if "MIRROR_B_PATH" in os.environ:
        mirror_entry = {"path": os.environ["MIRROR_B_PATH"]}
    else:
        mirror_entry = None

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
    mirror_model, vocab_match = None, None
    if mirror_entry is not None:
        from transformers import AutoTokenizer as _AT
        mtok = _AT.from_pretrained(mirror_entry["path"], local_files_only=True)
        vocab_match = (set(tok.get_vocab()) == set(mtok.get_vocab()))
        try:
            mirror_model = AutoModelForCausalLM.from_pretrained(
                mirror_entry["path"], local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="eager", device_map={"": "cuda"}).eval()
        except TypeError:
            mirror_model = AutoModelForCausalLM.from_pretrained(
                mirror_entry["path"], local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager", device_map={"": "cuda"}).eval()
        mirror_model.requires_grad_(False)

    def path_masses(mdl, ids, prefix_text, candidates):
        """Log-mass per candidate under one model (official ids for both)."""
        base = mdl.model(input_ids=torch.tensor([ids], device="cuda"),
                         use_cache=True)
        out, why = [], []
        for candidate in candidates:
            suffix = tok.encode(candidate, add_special_tokens=False)
            full = ids + suffix
            if not suffix or tok.decode(
                    full, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) != prefix_text + candidate:
                out.append(None)
                why.append("roundtrip_failed")
                continue
            cache = copy.deepcopy(base.past_key_values)
            ext = mdl.model(input_ids=torch.tensor([suffix], device="cuda"),
                            past_key_values=cache, use_cache=True)
            h = torch.cat([base.last_hidden_state[0, -1:],
                           ext.last_hidden_state[0]], 0)
            lp = canonical_head(mdl, h).double().log_softmax(-1)
            labels = torch.tensor(suffix, device="cuda")
            out.append(float(lp[:-1].gather(1, labels.unsqueeze(1)).sum()))
            why.append(None)
            del cache, ext
        del base
        return out, why

    out_rows, tmpl_match, tmpl_total = [], 0, 0
    with torch.inference_mode():
        for row in cand:
            prefix_text = row["prefix_text"]
            ids = tok.encode(prefix_text, add_special_tokens=False)
            back = tok.decode(ids, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
            if back != prefix_text:
                out_rows.append({"key": row["key"], "error": "prefix_roundtrip_failed"})
                continue
            base = model.model(input_ids=torch.tensor([ids], device="cuda"),
                               use_cache=True)
            first_lp = canonical_head(
                model, base.last_hidden_state[0, -1:])[0].double().log_softmax(-1)
            masses, reasons = path_masses(model, ids, prefix_text, row["candidates"])
            del base
            mmasses, wdelta = None, []
            if mirror_model is not None:
                mmasses, _ = path_masses(mirror_model, ids, prefix_text,
                                         row["candidates"])
                wdelta = [abs(a - b) for a, b in zip(masses, mmasses)
                          if a is not None and b is not None]
            mrow = mirror.get(row["key"], {})
            mm = mrow.get("masses", [])
            # Mirror stores probabilities; official side is log-mass: compare
            # in log space (guard mirror zeros from smoothing/absent paths).
            deltas = []
            for a, bstr in zip(masses, mm):
                try:
                    b = float(bstr)
                except (TypeError, ValueError):
                    continue
                if a is not None and b is not None and b > 0:
                    deltas.append(abs(a - math.log(b)))
            # Controlled-rendering audit: compare rendered STRINGS (id-level
            # comparison is confounded by BOS/special handling).
            prompt = pool.get(row["prompt_id"], "")
            match, off_render, first_diff = None, None, None
            if prompt:
                tmpl_total += 1
                off_render = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
                match = (off_render == prefix_text)
                if match:
                    tmpl_match += 1
                else:
                    first_diff = next(
                        (i for i, (x, y) in enumerate(zip(off_render, prefix_text))
                         if x != y), min(len(off_render), len(prefix_text)))
            out_rows.append({"key": row["key"], "masses": masses, "reasons": reasons,
                             "n_delta": len(deltas),
                             "max_abs_delta": max(deltas) if deltas else None,
                             "mean_abs_delta": (sum(deltas) / len(deltas)
                                                if deltas else None),
                             "weight_max_abs_delta": (max(wdelta) if wdelta else None),
                             "weight_mean_abs_delta": (sum(wdelta) / len(wdelta)
                                                       if wdelta else None),
                             "controlled_render_match": match,
                             "controlled_first_diff": first_diff,
                             "official_render_head": (off_render[:200]
                                                      if off_render else None)})
    ok = [r for r in out_rows if r.get("max_abs_delta") is not None]
    wok = [r for r in out_rows if r.get("weight_max_abs_delta") is not None]
    wmax = [r["weight_max_abs_delta"] for r in wok]
    wmean = [r["weight_mean_abs_delta"] for r in wok]
    dest = root / "work" / "bridge.json"
    dest.write_text(json.dumps({
        "n": len(out_rows), "n_scored": len(ok),
        "max_abs_delta": max(r["max_abs_delta"] for r in ok) if ok else None,
        "mean_abs_delta": (sum(r["mean_abs_delta"] for r in ok) / len(ok)
                           if ok else None),
        "weight_max_abs_delta": max(wmax) if wmax else None,
        "weight_mean_abs_delta": (sum(wmean) / len(wmean) if wmean else None),
        "weight_vocab_match": vocab_match,
        "controlled_prefix_match": f"{tmpl_match}/{tmpl_total}",
        "tokenizer_config_sha256": __import__("hashlib").sha256(
            (entry["path"] + "/tokenizer_config.json").encode()).hexdigest()[:16],
        "rows": out_rows}))
    print(f"wrote {dest} scored={len(ok)}/{len(out_rows)} "
          f"maxd={max((r['max_abs_delta'] for r in ok), default=None)} "
          f"wmaxd={max(wmax, default=None)} tmpl={tmpl_match}/{tmpl_total}")


if __name__ == "__main__":
    if not os.environ.get("SB_ROOT"):
        print("need SB_ROOT env", file=sys.stderr)
        sys.exit(2)
    main()
