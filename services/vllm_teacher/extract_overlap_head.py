#!/usr/bin/env python3
"""Extract immutable Qwen LM-head rows for the teacher overlap profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_VOCAB_SIZE = 151665


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--teacher-ids", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--profile-id", default="gemma2-qwen25-paper-v1")
    args = parser.parse_args()

    index_path = args.snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index.get("weight_map", {}).get("lm_head.weight")
    if not shard_name:
        raise RuntimeError("lm_head.weight is absent from the checkpoint index")
    shard = args.snapshot / shard_name
    teacher_ids = np.fromfile(args.teacher_ids, dtype="<i4")
    if not len(teacher_ids) or len(np.unique(teacher_ids)) != len(teacher_ids):
        raise RuntimeError("teacher overlap IDs must be non-empty and unique")

    with safe_open(shard, framework="pt", device="cpu") as handle:
        full_head = handle.get_tensor("lm_head.weight")
    if full_head.ndim != 2 or full_head.shape[1] != 3584:
        raise RuntimeError(f"unexpected Qwen LM-head shape {list(full_head.shape)}")
    if teacher_ids.min() < 0 or teacher_ids.max() >= TOKENIZER_VOCAB_SIZE:
        raise RuntimeError("overlap ID lies outside the teacher tokenizer vocabulary")

    selected = full_head.index_select(
        0, torch.from_numpy(teacher_ids.astype(np.int64, copy=False))
    ).to(torch.bfloat16).contiguous()
    args.profile_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    output = args.profile_dir / "teacher_overlap_lm_head.bf16le"
    temporary = output.with_suffix(output.suffix + ".tmp")
    selected.view(torch.uint16).numpy().astype("<u2", copy=False).tofile(temporary)
    temporary.replace(output)
    ids_output = args.profile_dir / "teacher_ids.i32le"
    ids_output.write_bytes(args.teacher_ids.read_bytes())
    manifest = {
        "contract_version": 1,
        "profile_id": args.profile_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tensor_name": "lm_head.weight",
        "source_shard": shard.name,
        "source_shard_sha256": sha256(shard),
        "index_sha256": sha256(index_path),
        "teacher_ids_sha256": sha256(ids_output),
        "teacher_id_count": int(len(teacher_ids)),
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "checkpoint_vocab_rows": int(full_head.shape[0]),
        "shape": [int(selected.shape[0]), int(selected.shape[1])],
        "dtype": "bfloat16_le",
        "byte_size": output.stat().st_size,
        "sha256": sha256(output),
    }
    (args.profile_dir / "teacher_overlap_lm_head.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
