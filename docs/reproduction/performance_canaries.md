# TPU resource and performance canaries

These canaries answer a resource question before full training: what is the
largest prompt batch that preserves the intended cross-tokenizer OPD workload
at a fixed sequence budget on one TPU v5e-8? They are not benchmark or paper
reproduction evidence.

## Two protocols that must stay separate

`paper4k` is the operational interpretation of Table 4 in the SimCT paper:

- maximum student prompt-plus-completion sequence: 4096 tokens;
- rollout completion cap: 3840 tokens with a static prompt cap of 256;
- temperature 0.6, top-p 0.95, one response per prompt;
- the runtime caps completion by the remaining sequence capacity.

The paper reports both maximum sequence length 4096 and rollout maximum length
4096, but does not specify the exact truncation interaction for a non-empty
prompt. Therefore the rule above is an explicit reproduction interpretation,
not an author-confirmed fact.

The native Tunix sampler pads every prompt to the configured static prompt
width before appending generated tokens. Therefore the executable generation
budget is
`min(max_completion_tokens, max_sequence_tokens - max_prompt_tokens)`, not the
sequence budget minus the observed prompt length. With the current static
prompt cap of 256, `paper4k` explicitly requests at most 3840 generated tokens.
This keeps the sampler cache shape within 4096 tokens while preserving a
fixed-shape worst-case resource probe.

`public8k` follows the released shell script instead:

- model sequence length: 8192 tokens;
- rollout completion cap: 4096 tokens;
- prompt cap: 4096 for the worst-case probe;
- temperature 0.6 and top-p 0.95 in the controlled comparison.

The released script itself leaves top-p at the code default 1.0. The controlled
canary pins 0.95 so memory and throughput comparisons differ only in length and
batch. This is public-code configuration evidence, not Table 4 evidence. Never
pool 4K and 8K results.

## Resource ladder

For each protocol, test prompt batch `B = 1, 2, 4, 8` on the unchanged
`FSDP8 x TP1 x PP1` topology. The four batch sizes may be submitted as
independent speculative canaries on different healthy owners to reduce queue
wall time. Each run keeps a unique slug, fixed prompt/seed/checkpoint and its
own audit trail; a larger-B result never conceals or retroactively invalidates
a smaller-B failure. Diagnose every OOM, non-finite metric, missing W&B
evidence, or runtime-contract failure separately before retrying that exact
configuration.

Every canary performs one logical optimizer update. The trainer call consumes
the whole prompt batch and Optax accumulates across calls:

| Prompt batch B | Gradient accumulation | Effective responses/update |
|---:|---:|---:|
| 1 | 64 | 64 |
| 2 | 32 | 64 |
| 4 | 16 | 64 |
| 8 | 8 | 64 |

`FSDP8`, `TP1`, and `PP1` describe model topology, not batch replication.
W&B `step` advances only after accumulation completes; raw trainer calls are
written separately to `*.micro.jsonl`.

## A real worst-case rollout

The probe uses the native Tunix sampler. In probe mode, normal EOS tokens are
forbidden and the pad token is supplied as the unreachable EOS sentinel, so
the sampler executes the requested completion length instead of ending early.
The run fails closed unless the observed total token length reaches the
protocol-specific floor. This prevents a short/EOS-terminated output from
being misreported as proof that 4K or 8K fits.

Build fixed prompt datasets and the config matrix:

```bash
python3 scripts/tpu/build_resource_probe_dataset.py \
  --output-dir /tmp/vdt-resource-probes

python3 scripts/tpu/build_performance_canary_matrix.py \
  --baseline configs/reproduction/qwen25_7b_to_gemma2_2b_public_simple_opd_screen.json \
  --output-dir /tmp/vdt-performance-canaries
```

## Required W&B and local evidence

Each accepted canary must have a native finished W&B run and matching local
summary. Required fields include:

- requested and actual prompt, completion, and total token lengths;
- prompt batch, accumulation count, effective global batch;
- `compile_s`, rollout, teacher tokenize/forward/score, alignment, and student
  forward-backward timings plus token/s rates (`compile_s` is a first-shape
  compile-attributed upper bound, not a compiler-internal timer);
- finite loss, gradient norm, and parameter norm;
- truncation count, TPU device count/topology, shape signature and JIT-cache
  observations;
- best-effort JAX memory-in-use, peak, and limit values when exposed by the
  runtime.

Pass `--profile-step 1` with `--profile-dir` to profile one micro-call. The
first call deliberately includes compile/startup costs. Later optimization
canaries may profile a warm call, but must label it separately.

## Optimization order after the ladder

If `paper4k` cannot run at B>=4 or `public8k` cannot run at B>=2, profile before
changing topology. The original implementation returned teacher
full-vocabulary tensors of shape approximately `B x L x V`, which can dominate
weights or KV-cache memory. The production teacher forward now computes the
exact full-vocabulary log-normalizer inside its JIT and returns only shared-token
and realized selected-token log-probabilities. A small-tensor regression test
checks numerical parity with the full-logit kernel. This is local correctness
evidence only; a fresh remote canary must still verify peak HBM and throughput.

For both protocols, `training.teacher_scoring_mode=cached_teacher_forcing`
replaces the dense teacher call with exact Qwen prefill plus teacher-forced
KV-cache decoding. Prompt prefill skips the vocabulary head and projects only
the final prompt state. A `jax.lax.scan` carries hidden state and scores each
realized completion token from a barrier-bracketed one-step `B x 1 x V`
projection, immediately reducing it to shared-token and selected-token
log-probabilities. The selected-token path flattens the compile-time leading
dimensions and uses statically unrolled scalar dynamic slices instead of a
batched dynamic gather, vocabulary reduction, or batch dot inside the scan.
The barriers keep XLA from lifting the LM head out of the loop and recreating a
retained `T x B x V` tensor. This also avoids dense
completion-length attention while preserving the dense causal objective. The
small-tensor test compares every cached token score with dense causal
log-softmax. This remains local design/parity evidence until fresh remote
canaries verify HBM, finite metrics, lineage and native W&B evidence.

The dense BF16 `paper4k` B2 canary reached teacher scoring but failed while XLA
tried to reserve 13.95 GiB with only 6.21 GiB reservable. That terminal evidence
is the reason the 4K ladder now uses the already parity-tested exact cached
path; it is not a scientific or W&B pass.

The first cached `paper4k` B2 canary then exposed an independent compiler
failure: XLA lifted the one-step LM head across all 5,888 teacher completion
positions and attempted an FP32 `[5888, 2, 64272]` allocation (about 193.8 GB).
The hidden-state carry and optimization barriers above address that exact
failure; this failed canary also logged zero optimizer steps and is not a
scientific or W&B pass.

The first bounded cached retry reached that scan after its private checkpoint
transfer succeeded, but TPU XLA then reported an internal post-optimization
shape mismatch while fusing the dynamic selected-token gather. A gather-free
retry replaced it with compare/select `reduce_sum`; TPU XLA reached the same
training phase but failed with a different post-optimization fusion shape
mismatch at that reduction. A third dot-selected retry reached the same phase
after the private checkpoint relay and warm-start both passed, but TPU XLA then
reported the corresponding post-optimization shape mismatch in its fused batch
dot. The scalar-slice selector above addresses the newly isolated batch-fusion
failure and is still pending a fresh remote canary. All three failed retries
logged zero optimizer steps and are not scientific or W&B passes.

The allowed optimization order is:

1. use the exact cached teacher-forcing path for both protocols, then verify
   remote HBM separately for `paper4k` and `public8k`;
2. if necessary, chunk prompt prefill or exact scoring by example/sequence;
3. bucket lengths or dynamically microbatch under a token budget;
4. rematerialize/checkpoint activations;
5. only then test HSDP or rollout replicas as a separate topology experiment.

Any loss-path optimization must pass numerical parity tests before a remote
canary. The fastest safe `paper4k` batch becomes the candidate for the full
two-epoch, effective-batch-64 scientific run. `public8k` remains a separate
ablation.
