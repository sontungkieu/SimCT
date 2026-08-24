# Tunix/MaxText TPU scaffold for single-teacher SimCT

This directory is a bounded real-backend layer, not a complete TPU trainer. It
keeps the CPU-testable boundary separate from the existing KDFlow and
`vdt_span` code.

| Capability | Status | What is actually established |
|---|---:|---|
| Strict JSON configuration | implemented | one student, exactly one teacher, distinct tokenizers, SimCT constants, v5e-8 layout |
| Student rollout interface | implemented | prompt/completion ids, exact UTF-8 token pieces, rollout log-probability shape and provenance |
| Teacher score interface | implemented | the same completion text retokenized by one teacher plus full-vocabulary logit shape/provenance |
| CPU contract pipeline | implemented | identity, sample count, exact text bridge and genuinely different tokenization are checked with both original mocks and dependency-injected adapter fakes |
| Prompt dataset contract | implemented | strict JSONL schema, immutable revision, file and manifest SHA-256, unique prompt IDs, deterministic cursor batches |
| Checkpoint/resume manifest | implemented | atomic latest pointer, config/dataset/model/tokenizer identity, data cursor, RNG and content-addressed parameter/optimizer references |
| TPU v5e-8 hardware preflight | implemented | lazily checks JAX backend, exactly eight devices and v5e/v5-lite device kinds |
| Tunix/MaxText model adapters | implemented, execution pending | lazy local-checkpoint `AutoModel` restore contract, full-sequence causal forward, trainable student and stop-gradient teacher; TP8/PP1 only |
| Real student rollout canary | implemented, execution pending | native Tunix KV-cached sampler, deterministic rollout coordinate, rollout log-probabilities and exact decoded-byte bridge |
| Real frozen-teacher scoring | implemented, execution pending | exact completion retokenization and full-vocabulary causal rows from the local MaxText teacher |
| JAX paper-math SimCT update | implemented, TPU execution pending | Eq. (7) mean log-probability scores, finite-candidate softmax, reverse KL, NNX gradient and AdamW update |
| Tunix/Orbax array save/restore | implemented, TPU execution pending | synchronous model plus optimizer persistence, directory digest, custom metadata cross-check, movable resume root |
| Resume-safe TPU train entrypoint | implemented, TPU execution pending | verified prompt manifest, deterministic cursor, finite metric gate, periodic full checkpoints and training summary |
| Kaggle/TPU evidence | submitted, terminal evidence pending | a v5e-8 canary is operational work only; no scientific metric claim is made |

## Contract boundaries

`TokenSequence.pieces` excludes padding and terminal special tokens. Concatenated
UTF-8 bytes must exactly equal the shared completion text. This lets a later
SimCT implementation align tokenizer boundaries without equating teacher tokens
with student tokens.

The configuration contains a singular `teacher` object. Unknown keys, a
`teachers` list, mutable `main`/`latest` revisions, same-tokenizer configs, and
non-v5e-8 layouts fail validation.

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
The project package discovery includes `vdt_tunix`; the CPU canaries can also be
executed directly from the repository checkout.

## Resume-safe training

After materializing a strict prompt manifest, a full training process uses:

```bash
python scripts/tpu/kaggle_v5e8_train.py \
  --config configs/reproduction/<run>.json \
  --dataset-manifest /kaggle/input/<dataset>/manifest.json \
  --metrics /kaggle/working/<run>/train_metrics.jsonl \
  --output /kaggle/working/<run>/train_summary.json
```

The output explicitly remains `scientific_evidence=false` until a downstream
evaluation under the shared comparison contract is complete.

## Still pending

- a provenance-complete reconstruction of the unavailable paper 10K corpus and
  warm-start checkpoint, or an explicitly labeled public-data substitute;
- SFT warm-start training in the same native Tunix model representation;
- SimpleOPD control training from the identical warm start;
- downstream GSM8K/MATH-500/MBPP/LCB evaluation artifacts under one decoding
  and scoring contract;
- terminal TPU canary evidence and any scientific comparison metric.
