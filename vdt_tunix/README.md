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

`requirements-tpu.txt` mirrors the pinned Tunix/MaxText infrastructure versions
from the read-only reference. It deliberately does not pin or install JAX.
Because Kaggle installs this file with `--no-deps`, compatibility-sensitive
leaf dependencies such as `huggingface-hub` are pinned explicitly and reported
by the canary provenance cell.
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

Training observability is optional and fail-open. A staged private notebook
may replace `__KJO_SECRET_WANDB_API_KEY__` immediately before submission; the
source notebook remains secret-free. `requirements-tpu.txt` pins the W&B
client explicitly so observability does not depend on the ambient Kaggle image.
When present, W&B receives finite numeric training metrics, elapsed time,
gradient/parameter norms, and span/token counts. W&B import, initialization,
network, logging, or finish failures are recorded in `VDT_WANDB_STATUS` and
never alter the optimizer path or exit code. The staged and archived notebook
copies must be scrubbed back to the placeholder after submission and checked
with the sensitive-artifact audit.

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
