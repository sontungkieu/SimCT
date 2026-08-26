# Tunix/MaxText TPU scaffold for single-teacher SimCT

This directory is a bounded real-backend layer, not a complete TPU trainer. It
keeps the CPU-testable boundary separate from the existing KDFlow and
`vdt_span` code.

| Capability | Status | What is actually established |
|---|---:|---|
| Strict JSON configuration | implemented | one student, exactly one teacher, distinct tokenizers, explicit SimCT/SimpleOPD support, v5e-8 layout |
| Student rollout interface | implemented | prompt/completion ids, exact UTF-8 token pieces, rollout log-probability shape and provenance |
| Teacher score interface | implemented | the same completion text retokenized by one teacher plus full-vocabulary logit shape/provenance |
| CPU contract pipeline | implemented | identity, sample count, exact text bridge and genuinely different tokenization are checked with both original mocks and dependency-injected adapter fakes |
| Prompt dataset contract | implemented | strict JSONL schema, immutable revision, file and manifest SHA-256, unique prompt IDs, deterministic cursor batches |
| Provenance-checked SFT dataset | implemented | strict teacher-response rows with source, source ID, license, immutable revision and content digest |
| Checkpoint/resume manifest | implemented | atomic latest pointer, config/dataset/model/tokenizer identity, data cursor, RNG and content-addressed parameter/optimizer references |
| TPU v5e-8 hardware preflight | implemented | lazily checks JAX backend, exactly eight devices and v5e/v5-lite device kinds |
| Tunix/MaxText model adapters | implemented, execution pending | lazy local-checkpoint `AutoModel` restore contract, full-sequence causal forward, trainable student and stop-gradient teacher; TP8/PP1 only |
| Real student rollout canary | implemented, execution pending | native Tunix KV-cached sampler, deterministic rollout coordinate, rollout log-probabilities and exact decoded-byte bridge |
| Real frozen-teacher scoring | implemented, execution pending | exact completion retokenization and full-vocabulary causal rows from the local MaxText teacher |
| JAX paper-math SimCT update | implemented, TPU execution pending | Eq. (7) mean log-probability scores, finite-candidate softmax, reverse KL, NNX gradient and AdamW update |
| JAX paper-control SimpleOPD update | implemented, TPU execution pending | reverse KL on normalized overlap vocabulary at exact one-to-one byte-aligned units; no span credit |
| Tunix/Orbax array save/restore | implemented, TPU execution pending | synchronous model plus optimizer persistence, directory digest, custom metadata cross-check, movable resume root |
| Native Tunix SFT warm start | implemented, TPU execution pending | completion-plus-EOS causal loss, configurable AdamW/cosine schedule, deterministic cursor and identical student parameter representation |
| Model-only SFT to OPD warm start | implemented, TPU execution pending | verifies student model/tokenizer and checkpoint digest, restores no SFT optimizer/cursor into the new OPD phase |
| Resume-safe TPU train entrypoint | implemented, TPU execution pending | verified prompt manifest, deterministic cursor, finite metric gate, periodic full checkpoints and training summary |
| Kaggle/TPU evidence | submitted, terminal evidence pending | a v5e-8 canary is operational work only; no scientific metric claim is made |

## Contract boundaries

`TokenSequence.pieces` excludes padding and terminal special tokens. Concatenated
UTF-8 bytes must exactly equal the shared completion text. This lets a later
SimCT implementation align tokenizer boundaries without equating teacher tokens
with student tokens. Prompt token IDs stored in the rollout/score contracts are
likewise text-only. Native rollout, SFT, teacher scoring, and OPD forwards add
the tokenizer's BOS model prefix separately, so every loss is conditioned on
the same model state as Tunix's sampler without introducing a zero-byte special
token into the byte-alignment coordinate.

The configuration contains a singular `teacher` object. Unknown keys, a
`teachers` list, mutable `main`/`latest` revisions, same-tokenizer configs, and
non-v5e-8 layouts fail validation. `algorithm=simct` requires shared tokens plus
realized spans; `algorithm=simple_opd` requires overlap-only support, paper-math
mode, and no post-paper span safeguard.

Checkpoint manifests are immutable at a completed-step coordinate. A resume
loads only through `latest.json`, verifies the manifest hash, then matches the
training-identity digest, dataset-manifest digest, and all model/tokenizer
revisions. Storage paths are deliberately excluded from training identity so a
checkpoint downloaded from one Kaggle output can resume under a new input
mount. Tunix writes student parameters and optimizer state synchronously before
the manifest publishes their content digest. Teacher weights are immutable
configuration inputs, not mutable checkpoint payloads in this single-teacher
contract.

## Local CPU checks

The mock implementations are importable only by explicit choice; the Kaggle
entrypoint never falls back to them:

```bash
python3 -m pytest tests/tunix tests/tunix_real
```

`tests/tunix_real/test_jax_kernels.py` is a true JAX CPU parity suite against
the pure-Python `vdt_span` reference. It skips rather than installing JAX when
the local study environment has no JAX runtime.

## Kaggle v5e-8 canary

Replace every placeholder in the example config, then invoke:

```bash
python scripts/tpu/kaggle_v5e8_canary.py \
  --config scripts/tpu/kaggle_v5e8_canary.example.json \
  --output /kaggle/working/vdt_simct_canary/canary.json
```

`vdt_tunix.real_backend.build_backends(config)` now returns production backends
only when all of the following are true: Tunix/MaxText/JAX/Transformers are
importable, both Tunix model specs are supported, both pinned tokenizers are
available locally, both checkpoint paths are absolute local directories, and
the layout is TP8/PP1. It never downloads weights, reads an access token, or
falls back to CPU mocks. Missing conditions still produce exit code 69 with
`phase=real_model_integration`; valid adapters advance to the v5e-8 hardware
preflight, and exact weight-restore/forward failures then fail the canary closed.

The current example intentionally contains placeholders and therefore remains
blocked at `real_model_integration` until replaced with supported local assets.

The local-only Kaggle package wrapper is documented in
[`docs/kaggle_v5e8_dry_run.md`](../docs/kaggle_v5e8_dry_run.md). It validates
checkpoint manifests, exact tokenizer revisions/cache, a pinned KJO source
snapshot, owner/slug, `TpuV5E8` metadata, and provenance before generating a
staged notebook below `/mnt/d/dev/codex/vdt-dynamic-span`. It has no submit
subcommand and the checked-in example spec is intentionally fail-closed.

The executable Kaggle userspace dependency contract lives in
`environments/kaggle-tpu/uv.lock`; `requirements-tpu.txt` is only a readable
list of direct inputs. The notebook installs pinned `uv==0.10.2`, verifies the
lock, exports the complete transitive graph, and installs that export with
`--no-deps` into a fresh `--system-site-packages` virtual environment. JAX,
JAXLIB, and libtpu are excluded from the export because Kaggle owns that
accelerator-coupled stack. Their exact versions, plus the Kaggle Python patch
version and TPU topology, are fail-closed in
`environments/kaggle-tpu/provider-constraints.json`.

All training, generation, and scoring subprocesses use the virtual
environment's Python. The bootstrap checks the locked dependency closure from
that interpreter so inherited provider packages such as JAX are visible,
verifies that provider packages were not copied into the virtual environment,
and emits `VDT_LOCKED_ENVIRONMENT_PROVENANCE` plus
`locked_environment_summary.json`. The virtual environment itself lives under
`/tmp` on Kaggle, while only its attestation is retained as a run artifact.
Refresh the lock deliberately with:

```bash
uv lock --project environments/kaggle-tpu --python 3.12
uv lock --check --project environments/kaggle-tpu
```

Do not update the provider constraints from dependency resolution; update them
only after observing and approving a new Kaggle TPU image.
The project package discovery includes `vdt_tunix`; the CPU canaries can also be
executed directly from the repository checkout.

## Public-data pipeline substitute

The exact paper 10K corpus and filtered teacher trajectories remain
unavailable. For an explicitly non-paper pipeline screen, materialize a
content-addressed, balanced GSM8K-train/MBPP-train subset with exact test-prompt
decontamination:

```bash
python scripts/data/materialize_public_substitute.py \
  --output-root /path/to/public-substitute \
  --per-source 128 \
  --seed 42
```

This writes separate strict `sft/` and `opd/` views plus `provenance.json`.
The provenance pins all four train/decontamination source files, licenses,
selection algorithm and hashes, and fixes
`paper_training_corpus_reproduced=false`. It is suitable for exercising the
shared-checkpoint comparison path, not for claiming the paper's reported
numbers.

## Resume-safe training

After materializing a strict prompt manifest, either SimCT or SimpleOPD uses:

```bash
python scripts/tpu/kaggle_v5e8_train.py \
  --config configs/reproduction/<run>.json \
  --dataset-manifest /kaggle/input/<dataset>/manifest.json \
  --metrics /kaggle/working/<run>/train_metrics.jsonl \
  --output /kaggle/working/<run>/train_summary.json
```

The trainer dispatches from `simct.algorithm` and records the objective in every
metric row and summary. Both objectives must point `checkpoint.warm_start_from`
to the identical SFT checkpoint for a fair control. Their output explicitly
remains `scientific_evidence=false` until downstream evaluation under the shared
comparison contract is complete.

The SFT phase consumes the stricter teacher-response schema and writes the same
native student state used by OPD:

```bash
python scripts/tpu/kaggle_v5e8_sft.py \
  --config configs/reproduction/<sft-run>.json \
  --dataset-manifest /kaggle/input/<sft-dataset>/manifest.json \
  --metrics /kaggle/working/<sft-run>/train_metrics.jsonl \
  --output /kaggle/working/<sft-run>/train_summary.json
```

Set `checkpoint.warm_start_from` in a new OPD config to the completed SFT
checkpoint root. This restores only the verified student model arrays. A true
resume instead sets `checkpoint.resume_from` and restores model, optimizer,
data cursor, and RNG metadata; the two fields are mutually exclusive.

Every completed training summary records the final student-parameter SHA-256.
OPD summaries additionally record the source SFT run ID, step, dataset-manifest
digest, and student-parameter SHA-256. This proves that SimpleOPD and SimCT
actually loaded the same SFT tensors rather than merely using similarly named
checkpoint directories.

The checked-in public screen configs deliberately run only ten batch-one,
non-accumulated optimizer updates. SFT keeps the public-script learning rate
`2e-6`; SimpleOPD and SimCT both use `5e-7`, the same 256 prompt manifest,
sampling temperature `0.6`, and a 256-token completion cap. The public paper
scripts instead use a 10K selected corpus, larger effective batches, one full
OPD epoch, and up to 4096 generated tokens. Therefore this bounded run is a
pipeline/one-seed screen and must not be reported as a paper-number
reproduction.

Render a Kaggle notebook from those immutable inputs with:

```bash
python scripts/tpu/render_training_notebook.py \
  --phase sft \
  --config-relative-path \
    configs/reproduction/qwen25_7b_to_gemma2_2b_public_sft_screen.json \
  --repo-dataset-source <owner>/<repo-snapshot> \
  --training-dataset-source <owner>/<public-substitute> \
  --training-manifest-relative-path sft/manifest.json \
  --student-model-source google/gemma-2/flax/gemma2-2b-it/1 \
  --teacher-model-source qwen-lm/qwen2.5/transformers/7b-instruct/1 \
  --output /path/to/source_notebook.ipynb
```

For `simple_opd` or `simct`, also pass the completed same-owner SFT notebook as
`--warm-start-kernel-source`, pin the committed output with
`--warm-start-kernel-version`, and pass its output-relative checkpoint
directory as `--warm-start-relative-path`. The renderer first resolves
Kaggle's static legacy, `notebooks/<owner>/<slug>`, and owner/version mount
layouts. If the accepted
`kernel_sources` metadata has not materialized a mount, it uses
`kagglehub.notebook_output_download()` with the exact versioned handle inside
the Kaggle runtime; the checkpoint never transits through the submitting
machine. It also handles the zip representation used by dataset directory
uploads, patches only storage paths (which are excluded from the training
identity digest), and fails if the terminal training summary does not match the
expected run, step count, warm-start lineage, and non-scientific status.

Attached Kaggle Models are resolved at runtime with
`kagglehub.model_download()` using the exact versioned model handle. The
renderer then writes the returned local model and tokenizer paths into a
runtime-only config. Do not derive model locations from a presumed
`/kaggle/input` directory layout: Kaggle may change or normalize that mount
layout independently of the stable model handle.

Before spending an accelerator session on an unverified Kaggle account, a
CPU-only private probe can test whether the exact attached model licenses are
available without loading model tensors:

```bash
python scripts/tpu/render_model_source_mount_probe.py \
  --out /path/to/model_source_probe.ipynb \
  --model-source google/gemma-2/flax/gemma2-2b-it/1 \
  --model-source qwen-lm/qwen2.5/transformers/7b-instruct/1
```

Its `VDT_MODEL_SOURCE_MOUNT_PROBE` marker is operational evidence only and
always records `scientific_evidence=false`.

The library observability adapter is fail-open by default, but staged
scientific Kaggle notebooks set `VDT_REQUIRE_WANDB=1`. They fail before model
loading if the native online run cannot start and stop at the first logging
failure instead of spending the remaining TPU session without a dashboard.
The multi-seed orchestration also treats terminal W&B evidence as a separate
completion gate. A staged private notebook
may replace `__KJO_SECRET_WANDB_API_KEY__` immediately before submission; the
source notebook remains secret-free. The uv lock pins W&B 0.19.11 and its
transitive dependencies, so observability does not depend on the ambient Kaggle
image.
Secret literals are rendered in dedicated assignment-only notebook cells.
Failure-prone model resolution, checkpoint transfer, and environment bootstrap
run in later cells that reference the assigned variables. This prevents a
Papermill traceback from echoing an embedded key as part of a failed cell's
`_kjo_source` string.
W&B receives finite numeric training metrics, elapsed time,
gradient/parameter norms, and span/token counts. Generation logs one progress
row per resumable batch, and scoring logs the final score/counts for all four
benchmarks. W&B import, initialization, network, logging, or finish failures
are recorded in `VDT_WANDB_STATUS`. In required native runs, initialization
and step-log failures block the run; generic local callers remain fail-open.
A run with incomplete terminal evidence does not pass the monitoring gate
until an audited backfill or logger retry succeeds. The staged
and archived notebook copies must be scrubbed back to the placeholder after
submission and checked with the sensitive-artifact audit.

For a native online run, validate the three distinct W&B URLs and exact log
counts with:

```bash
python scripts/evaluation/audit_wandb_evidence.py \
  --training-summary /path/to/train_summary.json \
  --generation-summary /path/to/generation_summary.json \
  --scoring-summary /path/to/scoring_summary.json \
  --output /path/to/wandb_evidence.json
```

Runs produced before native logging was enabled must not be relabeled as live
observability. After their training, generation, and scoring artifacts pass the
normal lineage audits, replay those immutable metrics into three explicitly
tagged historical W&B runs:

```bash
python scripts/evaluation/backfill_wandb_evidence.py \
  --training-summary /path/to/original/train_summary.json \
  --training-metrics /path/to/original/train_metrics.jsonl \
  --generation-summary /path/to/original/generation_summary.json \
  --scoring-summary /path/to/original/scoring_summary.json \
  --output-dir /path/to/derived/wandb_backfill

python scripts/evaluation/audit_wandb_evidence.py \
  --training-summary /path/to/derived/wandb_backfill/training_summary.json \
  --generation-summary /path/to/derived/wandb_backfill/generation_summary.json \
  --scoring-summary /path/to/derived/wandb_backfill/scoring_summary.json \
  --allow-backfill \
  --output /path/to/derived/wandb_backfill/wandb_evidence.json
```

The backfill never edits source summaries. Its manifest hashes all four source
artifacts and all three derived summaries. The default audit continues to
reject backfill evidence unless `--allow-backfill` is explicit; this prevents a
historical replay from being mistaken for live monitoring.

Multi-seed training configs may set `training.seed` explicitly. Legacy configs
without that field retain their original digest and run-id-derived rollout
RNG. Explicit seeds decouple rollout randomness from the unique run id. The
evaluation protocol seed stays fixed across training replicates so measured
variation is not confounded with a second generation-sampling change. SFT is
otherwise deterministic and therefore acts as a useful TPU determinism
control.

The executable policy is
`configs/evaluation/simct_public_multiseed_policy.json`: the current legacy
replicate is followed by explicit training seed 43, and seed 44 is permitted
only after the first two pass the consistency gate. Renderers accept
`--expected-run-id`, `--training-seed`, and `--wandb-group`; they materialize a
content-hashed `runtime_config.json` while keeping the pinned base config
unchanged. The evaluation seed remains 42.

Before spending quota on the third replicate, compare per-instance correctness
from the two scoring roots:

```bash
python scripts/evaluation/audit_two_seed_consistency.py \
  --first-sft-scoring-root /path/to/seed1/sft \
  --first-simple-opd-scoring-root /path/to/seed1/simple_opd \
  --first-simct-scoring-root /path/to/seed1/simct \
  --second-sft-scoring-root /path/to/seed2/sft \
  --second-simple-opd-scoring-root /path/to/seed2/simple_opd \
  --second-simct-scoring-root /path/to/seed2/simct \
  --output /path/to/two_seed_consistency.json
```

The gate stops before seed 44 if any benchmark's absolute paired score gap is
greater than `max(0.05, 3 * paired_SE)`, or if a SimCT-minus-SimpleOPD effect
changes sign with at least two percentage points on both sides. A stopped run
is an instruction to audit seed propagation, warm-start lineage, data order,
runtime locks, TPU topology, and step metrics; it is not evidence that a third
sample should average the discrepancy away.

## Shared evaluation contract

`vdt_tunix.evaluation_contract` validates a machine-readable comparison before
scores may be combined. It requires exactly SFT, SimpleOPD, and SimCT; the two
OPD variants must share the same prompt-manifest digest, optimizer-step count,
student/tokenizer revision, and exact SFT parameter hash. One protocol fixes
all four paper benchmarks, immutable dataset identities and record hashes,
zero-shot prompt/evaluator revisions, temperature `0.6`, top-p `0.95`, one
completion per instance, and explicit run seeds.

The current economical gate is labeled `one_seed_screen`. It cannot be parsed
as `paper_five_run`; the latter requires five distinct seeds. Passing training
or the one-seed screen still does not, by itself, reproduce the paper's
five-run mean and standard deviation.

After all three terminal generation jobs have produced both
`generation_summary.json` and `scoring_summary.json`, assemble the result with
`scripts/evaluation/assemble_one_seed_comparison.py`. The assembler also reads
each terminal training summary/config and the pinned benchmark manifests. It
fails closed on checkpoint lineage, config/protocol hashes, TPU shape, prompt
preflight, evaluator identity/environment, record counts, and score arithmetic:

```bash
python scripts/evaluation/assemble_one_seed_comparison.py \
  --comparison-id qwen25-gemma2-public-one-seed-v1 \
  --generation-protocol configs/evaluation/simct_paper_one_seed_generation.json \
  --evaluation-root /path/to/pinned-evaluation-bundle \
  --output-dir /path/to/comparison \
  --sft-training-config /path/to/sft-config.json \
  --sft-training-summary /path/to/sft/train_summary.json \
  --sft-generation-summary /path/to/sft/generation_summary.json \
  --sft-scoring-summary /path/to/sft/scoring_summary.json \
  --simple-opd-training-config /path/to/simple-opd-config.json \
  --simple-opd-training-summary /path/to/simple-opd/train_summary.json \
  --simple-opd-generation-summary /path/to/simple-opd/generation_summary.json \
  --simple-opd-scoring-summary /path/to/simple-opd/scoring_summary.json \
  --simct-training-config /path/to/simct-config.json \
  --simct-training-summary /path/to/simct/train_summary.json \
  --simct-generation-summary /path/to/simct/generation_summary.json \
  --simct-scoring-summary /path/to/simct/scoring_summary.json
```

The outputs are a machine-readable `comparison_summary.json` plus a compact
score table. A passing report is bounded one-seed public-substitute evidence
and deliberately retains `paper_reproduction=false`.

### Native generation and paper-released scoring

`configs/evaluation/simct_paper_one_seed_generation.json` pins the four
materialized benchmark hashes, seed 42, temperature `0.6`, top-p `0.95`, one
sample per instance, and the released evaluator revision. The generation path
loads only the Gemma student, verifies the full Orbax checkpoint lineage, and
restores model arrays through the model-only warm-start contract; it does not
materialize Qwen teacher weights. Batch artifacts are atomic and resumable, but
every resumed row must match its prompt hash, checkpoint hash, seed, sampling
parameters, token counts, and truncation flag.

Render the checkpoint-attached Kaggle notebook with:

```bash
python scripts/tpu/render_generation_notebook.py \
  --variant sft \
  --training-config-relative-path \
    configs/reproduction/qwen25_7b_to_gemma2_2b_public_sft_screen.json \
  --generation-protocol-relative-path \
    configs/evaluation/simct_paper_one_seed_generation.json \
  --repo-dataset-source <owner>/<repo-snapshot> \
  --evaluation-dataset-source <owner>/<pinned-evaluation-bundle> \
  --checkpoint-kernel-source <owner>/<completed-training-kernel> \
  --checkpoint-relative-path vdt_public_sft_screen/checkpoints \
  --student-model-source google/gemma-2/flax/gemma2-2b-it/1 \
  --output /path/to/generation_notebook.ipynb
```

If the completed checkpoint is private and the generation notebook runs under
a different Kaggle owner, do not attach that private kernel as a cross-owner
`kernel_source`. Render KJO's isolated source-owner download cell first, then
compose it into the ordinary generation notebook:

```bash
python /home/tung/.codex/skills/kaggle-job-ops/scripts/kaggle_job_ops.py \
  render-cross-account-output-cell \
  --out /path/to/cross_account_checkpoint.py \
  --kernel-id <source-owner>/<completed-training-kernel> \
  --runtime-owner <generation-owner> \
  --kaggle-config-dir /tmp/.kaggle_source_owner \
  --output-dir \
    /tmp/vdt_cross_account_inputs/kernels/<source-owner>/<completed-training-kernel> \
  --file-pattern '(?i)^vdt_public_sft_screen/checkpoints/.*'

python scripts/tpu/compose_cross_account_generation_notebook.py \
  --base-notebook /path/to/generation_notebook.ipynb \
  --cross-account-cell /path/to/cross_account_checkpoint.py \
  --source-kernel-id <source-owner>/<completed-training-kernel> \
  --runtime-owner <generation-owner> \
  --evaluation-dataset-source <generation-owner>/<pinned-evaluation-bundle> \
  --source-config-dir /tmp/.kaggle_source_owner \
  --cross-account-output-dir \
    /tmp/vdt_cross_account_inputs/kernels/<source-owner>/<completed-training-kernel> \
  --overlay-input-root /tmp/vdt_cross_account_inputs \
  --source-key-placeholder __KJO_SECRET_KAGGLE_SOURCE_KEY__ \
  --out /path/to/cross_account_generation_notebook.ipynb
```

After KJO stages the composed notebook, attach the exact student model to the
staged Kaggle metadata before secret injection or submission:

```bash
python scripts/tpu/attach_model_sources.py \
  --metadata <run-dir>/stage/kernel-metadata.json \
  --stage-manifest <run-dir>/stage/stage_package_manifest.json \
  --model-source google/gemma-2/flax/gemma2-2b-it/1

python scripts/tpu/attach_model_sources.py \
  --metadata <run-dir>/stage/kernel-metadata.json \
  --stage-manifest <run-dir>/stage/stage_package_manifest.json \
  --model-source google/gemma-2/flax/gemma2-2b-it/1 \
  --verify-only
```

This step is mandatory for batch generation. Kaggle does not allow a
non-interactive session to attach a new model dynamically, even when the
destination account has accepted the model license. The helper writes
`model_sources` into both the metadata and stage manifest and refreshes the
metadata fingerprint. Verify that exact field in the final pre-submit audit;
access probes alone do not prove that the batch notebook has the model attached.
The fail-closed ``--verify-only`` call is intended to run immediately before
secret injection, so a restage that dropped ``model_sources`` cannot reach a
non-interactive Kaggle session.

Treat a successful evaluation-dataset upload as accepted work, not as proof
that Kaggle can mount every file yet. Before submitting a dependent notebook,
run `scripts/tpu/wait_for_kaggle_dataset_ready.py` with the destination owner's
temporary `KAGGLE_CONFIG_DIR`, all eight benchmark manifest/record paths, and
at least two stable file-list checks. The byte threshold should cover the eight
remote data files; Kaggle consumes `dataset-metadata.json` during publication
and does not expose it in `datasets files`. The gate fails closed on a
non-`ready` status, a missing file, a shrinking/incomplete listing, or timeout.

The composer fails closed if the KJO cell, owner, config directory, output
directory, or source slug drift. Its first inserted runtime cell also verifies
that a Kaggle CLI is usable and, when the ambient image has neither the
executable nor `kaggle.cli`, installs the pinned `kaggle==2.2.3` client before
emitting `KJO_KAGGLE_CLI_BOOTSTRAP`. Inject only the source owner's Kaggle key
into the private staged copy immediately before submit. KJO must scrub the staged
and archived notebook afterward; the remote notebook remains in the
embedded-secret retention lifecycle until its scored outputs are local and it
is actually deleted. The checkpoint itself stays Kaggle-to-Kaggle.
The transfer overlay lives under `/tmp`, so Kaggle does not publish a second
copy of the checkpoint as destination-notebook output.

The notebook runs generation first and then
`scripts/evaluation/score_generated_predictions.py` before removing the
expanded 4.5 GB evaluation input. The scorer verifies that
`scripts/evaluation/evaluation.py` has SHA-256
`21378cfb1aa1d2f3ddab684a1bcb671fd588919c76fd410fac424bd062db2839`,
then AST-extracts the exact GSM8K, MATH-500, MBPP, and LCB scoring functions
released at `cf0f33a`. It records Python, multiprocessing, SymPy, and LaTeX
parser availability because the upstream requirements do not pin the latter
two behaviors.

These metrics are classified as **paper-released evaluator** results, not
official Math-Verify or LiveCodeBench results. A completed one-seed score is
scientific evidence about the bounded public-substitute screen, but
`paper_reproduction=false`: it cannot recover the unavailable 10K training
corpus or replace the paper's five-run mean and standard deviation.

## Still pending

- a provenance-complete reconstruction of the unavailable paper 10K corpus and
  warm-start checkpoint (the implemented public-data substitute is only a
  pipeline screen);
- terminal TPU execution evidence for SimpleOPD from the identical warm start;
- downstream GSM8K/MATH-500/MBPP/LCB evaluation artifacts under one decoding
  and scoring contract;
- terminal proof that the native Orbax/Tunix restore, generation, and
  paper-released scoring path works on the pinned Kaggle TPU runtime;
- a separately pinned official/reference scorer for robustness against quirks
  in the released paper evaluator;
- terminal TPU canary evidence and any scientific comparison metric.
