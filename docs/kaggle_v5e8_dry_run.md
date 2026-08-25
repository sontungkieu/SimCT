# Kaggle TPU v5e-8 bounded dry run

This package stages the existing `scripts/tpu/kaggle_v5e8_canary.py` as a
private Kaggle notebook for **TPU VM v5-8**. It is a contract canary, not a
training reproduction: the expected successful report keeps
`scientific_evidence=false` and `simct_update_executed=false`.

## Local contract

The spec schema is shown in
`scripts/tpu/kaggle_v5e8_dry_run.example.json`. Staging fails before any
Kaggle operation unless all of these are present and consistent:

- an explicit, non-placeholder Kaggle owner and slug; live TPU eligibility is
  checked later by the KJO capacity reservation rather than hard-coded here;
- a KJO `repo_dataset_manifest.json` snapshot whose Git commit is
  `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` and whose payload contains the
  canary, `vdt_tunix`, `requirements-tpu.txt`, and the complete
  `environments/kaggle-tpu` uv contract;
- one local student checkpoint and one local teacher checkpoint below
  `/mnt/d/dev/codex/vdt-dynamic-span`, each with a caller-supplied manifest and
  exact manifest SHA-256;
- 40-hex immutable model and tokenizer revisions plus both exact tokenizer
  snapshots in a local HF cache below the same D-drive boundary;
- `environments/kaggle-tpu/uv.lock` pinning the complete Linux/Python 3.12
  userspace graph, including Tunix commit
  `50f5752a17edec56e2aa30aabfc03859949adf6f`, plus an exact
  `provider-constraints.json` for the Kaggle-owned Python/JAX/JAXLIB stack.

The spec contains paths and public provenance only. Do not put Kaggle tokens,
API keys, `.env` paths, or credential filenames in it.

## Stage locally

First create the required source snapshot locally through Kaggle Job Ops. This
does not upload it:

```bash
python3 /mnt/c/Users/Tung/.codex/skills/kaggle-job-ops/scripts/kaggle_job_ops.py \
  stage-repo-dataset \
  --repo-root /home/tung/vdt-dynamic-span \
  --out /mnt/d/dev/codex/vdt-dynamic-span/source/vdt-simct-source \
  --owner OWNER \
  --dataset-slug vdt-simct-source \
  --allow-dirty \
  --clean
```

After materializing the checkpoint datasets, tokenizer-cache dataset, and a
real spec, run:

```bash
cd /home/tung/vdt-dynamic-span
python3 scripts/tpu/stage_kaggle_v5e8.py validate \
  --spec /mnt/d/dev/codex/vdt-dynamic-span/specs/kaggle_v5e8.json
python3 scripts/tpu/stage_kaggle_v5e8.py stage \
  --spec /mnt/d/dev/codex/vdt-dynamic-span/specs/kaggle_v5e8.json \
  --output-dir /mnt/d/dev/codex/vdt-dynamic-span/kaggle_jobs/vdt-simct-single-teacher-v5e8-dry-run
```

The wrapper generates a readable source notebook, then uses KJO staging to add
the repo-copy cell, structured cell logs, and the shape-aware TPU probe. It
requires metadata `machine_shape=TpuV5E8`, `enable_tpu=true`,
`enable_gpu=false`, a private exact `owner/slug`, and the four validated dataset
sources. It never calls a submit, status, dataset-push, or credential command.

## One future submit command

Only after the source, checkpoint, and tokenizer datasets have successful
same-owner upload evidence, after an exact-slug absence check, and after KJO
returns a TPU reservation token, the exact generated command is:

```bash
RESERVATION_TOKEN='<token returned by reserve-owners>' \
  bash /mnt/d/dev/codex/vdt-dynamic-span/kaggle_jobs/vdt-simct-single-teacher-v5e8-dry-run/future_submit_command.sh
```

The generated script calls KJO `submit-kernel` with `TpuV5E8`, requires the
repo-dataset push, requires every other dataset source, requires notebook
logging and accelerator-probe contracts, uses `secret_mode=none`, and records
the run in the project registry. Do not run it as part of local staging.

## Success evidence gates

A future remote canary is operationally successful only when all gates pass:

1. Kaggle stdout contains `successfully pushed`; a zero return code alone is
   insufficient.
2. Terminal status is reached and downloaded diagnostics include complete KJO
   split cell logs, `cells.jsonl`, and `run_summary.json`.
3. `KJO_ACCELERATOR_SUMMARY` reports requested `TpuV5E8`, eight visible TPU
   devices, and `runtime_matches_requested=true`.
4. `KJO_REPO_DATASET_COPY_SUMMARY` matches the staged source dataset and tree
   hash.
5. `VDT_LOCKED_ENVIRONMENT_PROVENANCE` reports a checked `uv.lock`, successful
   `uv pip check`, exact Tunix commit, W&B 0.19.11, and unchanged provider JAX
   and JAXLIB outside the uv virtual environment.
6. `VDT_CANARY_SUMMARY` reports `status=passed`, real student and teacher model
   integration, and observed cross-tokenization.
7. The same summary must still report `scientific_evidence=false` and
   `simct_update_executed=false`. A scientific baseline remains blocked until
   an optimizer update, metrics, checkpoint evidence, and run audit exist.

Download diagnostics only; do not download model weights by default.
