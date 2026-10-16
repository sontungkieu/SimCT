#!/usr/bin/env python3
"""A0: bootstrap sb-main-v2 study OUT root on shared storage (CPU-only).

Reads the verified staging manifest, writes:
  $SB_ROOT/study_config.json  - models, revisions, recipe, manifest pointers
  $SB_ROOT/assets.json        - per-arm local paths + verified_weights gate
                                (same schema h2_word_pilot.load_hf expects)

No download, no GPU, no HF access. Idempotent: refuses to overwrite
existing files unless --force.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = "/workspace/storage-shared/nlp/tungks/tmp/structural-behavioral/sb-main-v2"

RECIPE = {
    "seed": 42,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 1024,
    "backbone_dtype_candidates": ["bfloat16", "float32"],
    "tf32_allowed": False,
    "lr_screens": [5e-7, 1e-6, 2e-6],
    "lr_screen_updates": 30,
    "main_updates": 150,
    "prompts_per_update": 64,
    "rollouts_per_prompt": 4,
    "checkpoints": [25, 50, 75, 100, 125, 150],
    "primary_checkpoint": 150,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SB_ROOT", DEFAULT_ROOT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    staging = root / "assets" / "staging-manifest.json"
    if not staging.is_file():
        print(f"missing staging manifest: {staging}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(staging.read_text())
    for key in ("S_rev", "B_rev", "Q_rev"):
        if not manifest.get(key):
            print(f"staging manifest lacks {key}", file=sys.stderr)
            sys.exit(1)

    models_cfg = {
        "S": {"repo": "meta-llama/Llama-3.2-3B-Instruct",
              "revision": manifest["S_rev"], "role": "student"},
        "B": {"repo": "meta-llama/Llama-3.1-8B-Instruct",
              "revision": manifest["B_rev"], "role": "same_tokenizer_reference"},
        "Q": {"repo": "Qwen/Qwen2-7B-Instruct",
              "revision": manifest["Q_rev"], "role": "cross_family_teacher"},
    }
    assets = {"models": {}, "access_errors": {}}
    for arm in ("S", "B", "Q"):
        model_dir = root / "assets" / "models" / arm
        required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                    "model.safetensors.index.json"]
        missing = [f for f in required if not (model_dir / f).is_file()]
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if missing or not shards:
            print(f"staged model {arm} incomplete: missing={missing} shards={len(shards)}",
                  file=sys.stderr)
            sys.exit(1)
        entry = dict(models_cfg[arm], path=str(model_dir),
                     verified_weights=True,
                     weight_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in shards},
                     tokenizer_files_sha256={
                         p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in model_dir.iterdir() if p.is_file()
                         and p.suffix in (".json", ".txt")})
        assets["models"][arm] = entry

    study_config = {
        "study_version": "sb-main-v2",
        "models": models_cfg,
        "recipe": RECIPE,
        "pool_manifest": "main_pool_manifest.json (repo artifacts/structural-behavioral-main-runs)",
        "physical_node_gpu_uuid_mapping": None,
    }
    for name, payload in (("study_config.json", study_config), ("assets.json", assets)):
        out = root / name
        if out.exists() and not args.force:
            print(f"exists, skip (use --force to overwrite): {out}")
            continue
        out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {out} ({out.stat().st_size} bytes)")
    print("A0 bootstrap OK")


if __name__ == "__main__":
    main()
