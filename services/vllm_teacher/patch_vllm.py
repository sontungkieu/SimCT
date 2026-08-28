#!/usr/bin/env python3
"""Version-gated patch for vLLM's prompt-logprob IPC channel.

The reserved ``prompt_logprobs=-1`` value carries BF16 final hidden states plus
FP32 full-vocabulary log-normalizers and realized-token log probabilities. All
ordinary non-negative prompt-logprob requests retain upstream behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import py_compile
import shutil
from pathlib import Path


EXPECTED_VERSION = "0.27.1"
PATCH_MARKER = "VDT_TEACHER_HIDDEN_STATS_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_root() -> Path:
    import vllm

    if vllm.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"expected vLLM {EXPECTED_VERSION}, observed {vllm.__version__}"
        )
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not resolve installed vLLM package")
    return Path(spec.origin).resolve().parent


def patch_prompt(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    old = '''        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)
'''
    new = '''        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        if num_prompt_logprobs == -1:
            # VDT_TEACHER_HIDDEN_STATS_V1
            target_ids = prompt_token_ids[start_idx:end_idx]
            float_logits = prompt_logits.to(torch.float32)
            log_normalizer = torch.logsumexp(float_logits, dim=-1)
            selected_logits = torch.take_along_dim(
                float_logits, target_ids[:, None], dim=-1
            )[:, 0]
            selected_log_probs = selected_logits - log_normalizer
            metadata = torch.zeros(
                (target_ids.shape[0], 3),
                dtype=torch.int64,
                device=target_ids.device,
            )
            metadata[:, 0] = target_ids
            metadata[:, 1] = log_normalizer.contiguous().view(torch.int32).to(torch.int64)
            metadata[:, 2] = selected_log_probs.contiguous().view(torch.int32).to(torch.int64)
            token_ids.append(metadata)
            scores.append(prompt_hidden_states[start_idx:end_idx].to(torch.bfloat16))
            ranks.append(torch.zeros_like(target_ids, dtype=torch.int64))
            continue
        requested_num = num_prompt_logprobs
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)
'''
    if source.count(old) != 1:
        raise RuntimeError("vLLM prompt-logprob patch target drifted")
    source = source.replace(old, new)
    if source.count("    CHUNK_SIZE = 1024\n") != 1:
        raise RuntimeError("vLLM prompt-logprob chunk constant drifted")
    source = source.replace("    CHUNK_SIZE = 1024\n", "    CHUNK_SIZE = 256\n", 1)
    path.write_text(source, encoding="utf-8")
    return True


def patch_engine(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    if "\nimport torch\n" not in source:
        source = source.replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\n\nimport torch\n",
            1,
        )
    old = '''        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes.
'''
    new = '''        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        if (
            self.num_prompt_logprobs == -1
            and token_ids.ndim == 2
            and token_ids.shape[1] == 3
            and logprobs.ndim == 2
        ):
            # VDT_TEACHER_HIDDEN_STATS_V1
            current = self.prompt_logprobs
            if not isinstance(current, dict) or not current.get("__vdt_hidden_stats__"):
                current = {
                    "__vdt_hidden_stats__": True,
                    "hidden_chunks": [],
                    "metadata_chunks": [],
                }
                self.prompt_logprobs = current  # type: ignore[assignment]
            current["hidden_chunks"].append(
                logprobs.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            )
            current["metadata_chunks"].append(
                token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
            )
            return

        # Recover shapes.
'''
    if source.count(old) != 1:
        raise RuntimeError("vLLM engine-logprobs update patch target drifted")
    source = source.replace(old, new)
    old_pop = '''        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp
'''
    new_pop = '''        plp = self.prompt_logprobs
        if isinstance(plp, dict) and plp.get("__vdt_hidden_stats__"):
            result = {
                "__vdt_hidden_stats__": True,
                "hidden_states": torch.cat(plp["hidden_chunks"], dim=0),
                "metadata": torch.cat(plp["metadata_chunks"], dim=0),
            }
            self.prompt_logprobs = []
            return result  # type: ignore[return-value]
        if plp:
            self.prompt_logprobs = []
        return plp
'''
    if source.count(old_pop) != 1:
        raise RuntimeError("vLLM engine-logprobs pop patch target drifted")
    path.write_text(source.replace(old_pop, new_pop), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = package_root()
    prompt = root / "v1/worker/gpu/sample/prompt_logprob.py"
    engine = root / "v1/engine/logprobs.py"
    args.backup_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    for source in (prompt, engine):
        backup = args.backup_dir / source.name
        if not backup.exists():
            shutil.copy2(source, backup)
    changed = {"prompt": patch_prompt(prompt), "engine": patch_engine(engine)}
    py_compile.compile(str(prompt), doraise=True)
    py_compile.compile(str(engine), doraise=True)
    manifest = {
        "contract_version": 1,
        "vllm_version": EXPECTED_VERSION,
        "patch_marker": PATCH_MARKER,
        "changed": changed,
        "prompt_sha256": sha256(prompt),
        "engine_sha256": sha256(engine),
        "backup_prompt_sha256": sha256(args.backup_dir / prompt.name),
        "backup_engine_sha256": sha256(args.backup_dir / engine.name),
    }
    args.manifest.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
