#!/usr/bin/env python3
"""Offline preparation and explicit single-device HF oracle diagnostics.

Not a training launcher, not an SGLang parity test, not full-parameter Adam.
Preparation needs no GPU; run loads local models only and never executes code
from generated responses. Reference responses come only from the supplied data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time


def token_ids(value, *, allow_empty=False):
    """Normalize one tokenized sequence; never iterate BatchEncoding as IDs."""
    from collections.abc import Mapping
    from numbers import Integral
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("tokenizer must return one flat sequence of integer token IDs")
    if (not value and not allow_empty) or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value):
        raise ValueError("empty, batched or non-integer token IDs")
    return [int(x) for x in value]


def chat_prompt_ids(tok, messages, max_tokens):
    ids = token_ids(tok.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True, return_dict=False))
    if len(ids) > max_tokens:
        raise ValueError("prompt exceeds diagnostic cap; no silent truncation")
    return ids


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_groups(groups):
    if not groups:
        raise ValueError("empty diagnostic groups")
    ids, prompts = set(), set()
    for group in groups:
        for role in ("rollout", "select", "eval"):
            rows = reference_rows(group[role])
            if role == "rollout" and len(rows) != 1:
                raise ValueError("one rollout prompt per group required")
            for row in rows:
                messages = row["messages"]
                if not messages or not all(isinstance(m,dict) and m.get("role") in {"user","system","assistant"} and isinstance(m.get("content"),str) for m in messages):
                    raise ValueError("messages must be text chat messages")
                if messages[-1]["role"] != "user": raise ValueError("prompt must end with user, without reference answer")
                key=identity(messages)
                if str(row["id"]) in ids or key in prompts:
                    raise ValueError("duplicate/overlapping IDs or prompt content across groups/splits")
                ids.add(str(row["id"]));prompts.add(key)
                if role != "rollout" and (not isinstance(row.get("reference"),str) or not row["reference"].strip()):
                    raise ValueError("select/eval need explicit nonempty reference text")
    return {"rows":len(ids),"prompt_id_sha256":hashlib.sha256("\n".join(sorted(prompts)).encode()).hexdigest()}


def reference_rows(value):
    rows=value if isinstance(value,list) else [value]
    if not rows or not all(isinstance(x,dict) for x in rows): raise ValueError("nonempty reference rows required")
    return rows


def reference_mean(callbacks):
    """Equal weight per reference, each callback already averages its tokens."""
    if not callbacks: raise ValueError("no references")
    return lambda params: sum(fn(params) for fn in callbacks)/len(callbacks)


def select_counts(groups, requested):
    sizes={len(reference_rows(g["select"])) for g in groups}
    if len(sizes)!=1: raise ValueError("select pool sizes must match across groups")
    size=next(iter(sizes));counts=list(requested) if requested is not None else [size]
    if not counts or counts!=sorted(set(counts)) or any(x<1 or x>size for x in counts):
        raise ValueError("select counts must be increasing, unique and within prepared pool")
    return counts


def read_rows(path):
    if path.suffix == ".parquet":
        from datasets import load_dataset
        return load_dataset("parquet", data_files=str(path), split="train")
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and payload.get("schema") in {"mp-oracle-data-v1","mp-oracle-data-v2"}:
            return [row for group in payload["groups"] for role in ("rollout","select","eval") for row in reference_rows(group[role])]
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON input must be rows or a prepared oracle data file")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(args):
    rows = read_rows(args.input)
    excluded = set()
    excluded_files = {}
    for path in getattr(args, "exclude_prompts", []):
        for row in read_rows(path):
            messages = row[args.messages_key]
            # SFT conversations include the target final assistant message.
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            excluded.add(identity(messages))
        excluded_files[str(path)] = digest(path)
    unique = {}
    conflicting = set()
    conflict_policy = getattr(args, "conflicting_references", "error")
    for row in rows:
        messages = row[args.messages_key]
        if args.reference_key == "@last-assistant":
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError("@last-assistant requires a final assistant response")
            reference = messages[-1]["content"]
            messages = messages[:-1]
        else:
            reference = row[args.reference_key]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("reference must be explicit nonempty text; no label-to-answer conversion")
        key = identity(messages)
        if key in excluded:
            continue
        record = {"id": key, "messages": messages, "reference": reference}
        if key in conflicting:
            continue
        if key in unique and unique[key]["reference"] != reference:
            if conflict_policy == "error":
                raise ValueError("same prompt has conflicting references; use --conflicting-references exclude to exclude the entire prompt group")
            conflicting.add(key)
            del unique[key]
            continue
        unique[key] = record
    candidates = sorted(unique.values(), key=lambda x: x["id"])
    random.Random(args.seed).shuffle(candidates)
    ns=getattr(args,"select_references",1);ne=getattr(args,"eval_references",1)
    if min(ns,ne)<=0: raise ValueError("reference counts must be positive")
    stride=1+ns+ne
    if len(candidates)<stride*args.groups:
        raise ValueError(f"need at least {stride*args.groups} unique prompts")
    groups=[]
    for start in range(0,stride*args.groups,stride):
        chunk=candidates[start:start+stride];chunk[0].pop("reference")
        groups.append({"rollout":chunk[0],"select":chunk[1] if ns==1 else chunk[1:1+ns],
                       "eval":chunk[1+ns] if ne==1 else chunk[1+ns:]})
    checked = validate_groups(groups)
    payload = {"schema": "mp-oracle-data-v2" if ns>1 or ne>1 else "mp-oracle-data-v1",
               "select_references":ns,"eval_references":ne, "source_sha256": digest(args.input),
               "seed": args.seed, "reference_key": args.reference_key,
               "excluded_files_sha256": excluded_files, "excluded_prompt_count": len(excluded),
               "reference_provenance": getattr(args, "reference_provenance", "unspecified"),
               "split_audit": checked, "groups": groups,
               "conflicting_reference_policy": conflict_policy,
               "conflicting_prompt_count": len(conflicting),
               "conflicting_prompt_ids": sorted(conflicting),
               "reference_policy": "user-supplied references; quality not automatically established"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"prepared": str(args.output), "sha256": digest(args.output),
                      "conflicting_prompts_excluded": len(conflicting), **checked}))


def preflight_lengths(groups, tokenizer, teacher_tokenizer, prompt_cap, reference_cap):
    """Check every role before model allocation, with actionable failure identities."""
    maxima={'student_prompt':0,'teacher_prompt':0,'reference':0}
    for index,group in enumerate(groups):
        for role in ('rollout','select','eval'):
            for row in reference_rows(group[role]):
                for label,tok in [('student',tokenizer)]+([('teacher',teacher_tokenizer)] if role=='rollout' else []):
                    ids=token_ids(tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False))
                    maxima[label+'_prompt']=max(maxima[label+'_prompt'],len(ids))
                    if len(ids)>prompt_cap:
                        raise ValueError(f'group={index} role={role} id={row.get("id")} {label} prompt={len(ids)} cap={prompt_cap}; no truncation')
                if role!='rollout':
                    n=len(token_ids(tokenizer.encode(row['reference'],add_special_tokens=False)))+int(tokenizer.eos_token_id is not None)
                    maxima['reference']=max(maxima['reference'],n)
                    if n>reference_cap:
                        raise ValueError(f'group={index} role={role} id={row.get("id")} reference={n} cap={reference_cap}; no truncation')
    return maxima


def run(args):
    # The run subcommand is the explicit GPU boundary. Imports below are lazy.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"
    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer, mp_content_ids
    from kdflow.algorithms._mp_opd_credit import realized_token_log_probs
    from kdflow.algorithms._mp_opd_diagnostic import AtomWeighting, diagnostic, train_weighting_step
    from torch.func import functional_call
    data = json.loads(args.data.read_text())
    if data.get("schema") not in {"mp-oracle-data-v1","mp-oracle-data-v2"}:
        raise ValueError("unsupported data contract")
    groups = data["groups"]
    audit = validate_groups(groups)
    counts = select_counts(groups,getattr(args,"select_counts",None))
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dtype_name = getattr(args,"model_dtype","auto")
    dtype = torch.float32 if dtype_name == "float32" or device.type != "cuda" else torch.bfloat16
    for path in (args.student, args.teacher):
        if not path.is_dir():
            raise ValueError("models must be existing local directories")
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True, trust_remote_code=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=False)
    lengths=preflight_lengths(groups,tokenizer,teacher_tokenizer,args.max_prompt_tokens,args.max_reference_tokens)
    (args.output/'length-preflight.json').write_text(json.dumps(lengths,indent=2))
    print('LENGTH_PREFLIGHT',json.dumps(lengths),flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, local_files_only=True, trust_remote_code=False,
                torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=False,
                torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    student.requires_grad_(False); teacher.requires_grad_(False)
    # Explicit selected module; zero B leaves the original function unchanged.
    original = student.get_submodule(args.adapter_module)
    if not isinstance(original, torch.nn.Linear):
        raise ValueError("adapter module must name a Linear")
    class ProbeAdapter(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.base = linear
            self.register_buffer("a", torch.randn(args.rank, linear.in_features, device=device) / math.sqrt(linear.in_features))
            self.b = torch.nn.Parameter(torch.zeros(linear.out_features, args.rank, device=device))
        def forward(self, x):
            residual = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
            return self.base(x) + (residual / args.rank).to(x.dtype)
    parent_name, child_name = args.adapter_module.rsplit(".", 1)
    parent = student.get_submodule(parent_name)
    setattr(parent, child_name, ProbeAdapter(original))
    adapter = student.get_submodule(args.adapter_module)
    param_name = args.adapter_module + ".b"
    params = (adapter.b,)
    weighting = AtomWeighting().to(device)
    import copy
    weightings={n:copy.deepcopy(weighting) for n in counts}
    optimizers={n:torch.optim.AdamW(weightings[n].parameters(),lr=args.weighting_lr,weight_decay=0.) for n in counts}
    atomizer = SimCTAtomizer(tokenizer, teacher_tokenizer)

    def prompt(tok, messages):
        return chat_prompt_ids(tok, messages, args.max_prompt_tokens)

    def scores(model, prefix, response, virtual=None):
        ids = torch.tensor([prefix + response], device=device, dtype=torch.long)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        if virtual is None:
            output = model(**kwargs)
        else:
            output = functional_call(model, {param_name: virtual[0]}, (), kwargs)
        logits = output.logits[0, len(prefix)-1:len(prefix)+len(response)-1]
        return realized_token_log_probs(logits, torch.tensor(response, device=device, dtype=torch.long))

    def outer(row):
        prefix = prompt(tokenizer, row["messages"])
        response = token_ids(tokenizer.encode(row["reference"], add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            response.append(tokenizer.eos_token_id)
        if not response or len(response) > args.max_reference_tokens:
            raise ValueError("reference empty/too long; no silent truncation")
        return lambda ps: -scores(student, prefix, response, ps).mean()

    model_files = {}
    for label, root in (("student", args.student), ("teacher", args.teacher)):
        print(f"Hashing {label} model files for provenance", flush=True)
        model_files[label] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
                              if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}}
    manifest = {"schema": "mp-real-oracle-v1", "data_sha256": digest(args.data), "split_audit": audit,
                "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != "func"},
                "models": model_files, "torch": torch.__version__,
                "source_commit": getattr(args,"source_commit",None) or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                "source_diff_sha256": None if getattr(args,"source_commit",None) else hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=Path(__file__).resolve().parents[2])).hexdigest(),
                "effective_model_dtype": str(dtype),
                "oracle_script_sha256": digest(Path(__file__)),
                "scope": "frozen checkpoint, zero-initialized last-module low-rank B probe, virtual SGD",
                "rollout_backend": "HF diagnostic; not SGLang parity validation", "credit_temperature": 1.,
                "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": 0, "num_beams": 1, "repetition_penalty": 1.0},
                "weighting_protocol": "prequential: evaluate before learning current select; never train on eval",
                "benchmark_exclusion": "caller must supply non-benchmark references; source content not automatically classified"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results, invalid = [], 0
    for index, group in enumerate(groups):
        started = time.perf_counter()
        prefix = prompt(tokenizer, group["rollout"]["messages"])
        with torch.no_grad():
            inp = torch.tensor([prefix], device=device, dtype=torch.long)
            generated = student.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                        num_beams=1, repetition_penalty=1.0,
                        max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        sampled_ids = generated[0, len(prefix):].tolist()
        response, terminal_count = mp_content_ids(sampled_ids, tokenizer)
        ended = terminal_count > 0
        text = tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        tea_prefix = prompt(teacher_tokenizer, group["rollout"]["messages"])
        tea_response = token_ids(teacher_tokenizer.encode(text, add_special_tokens=False), allow_empty=True)
        atomized = atomizer.atomize(response, tea_response, sample_id=group["rollout"]["id"])
        trace = {"index": index, "rollout_id": group["rollout"]["id"], "student_prompt_ids": prefix,
                 "student_response_ids": response, "sampled_student_ids": sampled_ids, "teacher_prompt_ids": tea_prefix,
                 "teacher_response_ids": tea_response, "sampled_eos": ended}
        if not atomized.valid or not atomized.atoms:
            invalid += 1
            result = {"index": index, "invalid": True, "reason": atomized.failure_reason}
        else:
            atoms = atomized.atoms
            with torch.no_grad():
                old = scores(student, prefix, response)
                tea = scores(teacher, tea_prefix, tea_response)
                base = torch.stack([tea[a.teacher_start:a.teacher_end].sum() - old[a.student_start:a.student_end].sum() for a in atoms]).detach()
            weight = torch.tensor([a.student_token_count for a in atoms], device=device, dtype=torch.float32)
            def atom_nll(ps):
                lp = scores(student, prefix, response, ps)
                return torch.stack([-lp[a.student_start:a.student_end].sum() for a in atoms])
            select_callbacks=[outer(row) for row in reference_rows(group["select"])]
            eval_loss=reference_mean([outer(row) for row in reference_rows(group["eval"])])
            sweep={}
            for count in counts:
                select_loss=reference_mean(select_callbacks[:count])
                report=diagnostic(params,atom_nll,base,weight,select_loss,eval_loss,
                    lr=args.virtual_lr,max_span=args.max_span,seed=args.seed+index,weighting=weightings[count])
                report.update(select_count=count,select_ids=[r["id"] for r in reference_rows(group["select"])[:count]],
                              eval_ids=[r["id"] for r in reference_rows(group["eval"])])
                sweep[str(count)]=report
            result=dict(sweep[str(counts[-1])])
            if len(counts)>1: result["select_count_sweep"]=sweep
            trace.update(base_credit=base.tolist(),atom_weights=weight.tolist())
            result.update(index=index,invalid=False,weighting_prior_groups=index-invalid,
                          sampled_eos=ended,seconds=time.perf_counter()-started)
            # Independent weighting states per select count; evaluation never trains them.
            for count in counts:
                select_loss=reference_mean(select_callbacks[:count])
                for _ in range(args.weighting_steps):
                    train_weighting_step(weightings[count],optimizers[count],params,atom_nll,base,weight,select_loss,args.virtual_lr)
        results.append(result)
        for name, value in (("results.jsonl", result), ("trajectories.jsonl", trace)):
            with (args.output/name).open("a") as f:
                f.write(json.dumps(value, allow_nan=False)+"\n")
        print(json.dumps({"group": index, "invalid": result["invalid"], "seconds": time.perf_counter()-started}), flush=True)
    valid_results = [x for x in results if not x["invalid"]]
    if not valid_results:
        raise RuntimeError("no valid groups; oracle gate unavailable")
    if len(counts)>1:
        paired={str(n):[{"group":x["index"],"atomic_minus_oracle_eval_nll":x["select_count_sweep"][str(n)]["controls"]["atomic"]["eval_nll"]-x["select_count_sweep"][str(n)]["controls"]["oracle"]["eval_nll"]} for x in valid_results] for n in counts}
        (args.output/"paired-select-counts.json").write_text(json.dumps({"counts":counts,"independent_groups":len(valid_results),"paired":paired,"unit":"groups; counts are repeated measurements, not independent samples"},indent=2))
    gains = [x["controls"]["atomic"]["eval_nll"]-x["controls"]["oracle"]["eval_nll"] for x in valid_results]
    comparisons = {}
    rng = random.Random(args.seed)
    for name in valid_results[0]["controls"]:
        values = [x["controls"][name]["eval_nll"] - x["controls"]["oracle"]["eval_nll"] for x in valid_results]
        ci = None
        if len(values) > 1:
            means = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(1000))
            ci = [means[25], means[974]]
        comparisons[name] = {"mean_oracle_gain": sum(values)/len(values),
                             "positive_fraction": sum(v>0 for v in values)/len(values),
                             "exploratory_group_bootstrap_95pct": ci}
    summary = {"reported_select_count":counts[-1],"select_counts":counts,"comparisons": comparisons, "completed_groups": len(results), "valid_groups": len(valid_results), "invalid_groups": invalid,
               "oracle_vs_atomic_mean_eval_gain": sum(gains)/len(gains),
               "oracle_vs_atomic_positive_fraction": sum(g>0 for g in gains)/len(gains),
               "evidence": "adapter-only diagnostic; no efficacy pass/fail from mean alone"}
    torch.save({"network": weightings[counts[-1]].state_dict(), "optimizer": optimizers[counts[-1]].state_dict(),
                "data_sha256": manifest["data_sha256"], "protocol": manifest["weighting_protocol"]}, args.output/"weighting.pt")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    for count in counts:
        torch.save({"network":weightings[count].state_dict(),"optimizer":optimizers[count].state_dict(),"select_count":count,"data_sha256":manifest["data_sha256"]},args.output/f"weighting-select-{count}.pt")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--messages-key", default="messages")
    p.add_argument("--reference-key", required=True)
    p.add_argument("--conflicting-references", choices=("error", "exclude"), default="error")
    p.add_argument("--exclude-prompts", type=Path, action="append", default=[],
                   help="Repeat for SFT/train/benchmark prompts or previous diagnostic groups")
    p.add_argument("--reference-provenance", default="unspecified",
                   help="Teacher revision / generation artifact identifier; never credentials")
    p.add_argument("--select-references",type=int,default=1)
    p.add_argument("--eval-references",type=int,default=1)
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    for key in ("student", "teacher", "data", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--adapter-module", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--select-counts",type=int,nargs="+",help="Nested select subsets, e.g. 1 4 8; same rollout and eval")
    p.add_argument("--model-dtype", choices=("auto","bfloat16","float32"),default="auto")
    p.add_argument("--source-commit",help="Audited source overlay commit for runners without .git")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-span", type=int, default=4)
    p.add_argument("--virtual-lr", type=float, required=True)
    p.add_argument("--weighting-lr", type=float, default=1e-3)
    p.add_argument("--weighting-steps", type=int, default=1)
    p.add_argument("--temperature", type=float, default=.6)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-reference-tokens", type=int, default=1024)
    p.set_defaults(func=run)
    args = parser.parse_args()
    if args.command == "prepare" and args.groups <= 0:
        parser.error("groups must be positive")
    if args.command == "run" and (not all(math.isfinite(x) for x in (args.virtual_lr, args.weighting_lr, args.temperature, args.top_p)) or args.rank <= 0 or args.max_span <= 0 or args.virtual_lr <= 0
            or args.weighting_lr <= 0 or args.weighting_steps <= 0 or args.temperature <= 0
            or not 0 < args.top_p <= 1 or min(args.max_new_tokens,args.max_prompt_tokens,args.max_reference_tokens)<=0):
        parser.error("invalid diagnostic settings")
    args.func(args)


if __name__ == "__main__":
    main()
