# SimCT paper-versus-code reproduction contract

Status: static audit plus CPU-only contract fixtures. No model inference,
training, evaluation, Kaggle submission, or metric reproduction was performed.

## 1. Immutable source pins

| Artifact | Pin | Evidence status |
|---|---|---|
| Paper | arXiv `2605.07711v2`, revised 2026-05-21 | Primary source |
| Paper PDF | SHA-256 `99348c25912d39dc2ee679bc736a9547a82eb645deb71208f8a7f30b7e59d590`; `pdf_read_preflight/1` PASS, 30/30/30 pages | Read-integrity check only |
| Public code | `sunjie279/SimCT-` commit `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` (2026-08-03) | Primary source, audited locally |
| Pre-safeguard parent | `4dae6fcd374290f718ac0a950e57947e292be927` | Primary source, used only to isolate the safeguard diff |

The PDF preflight establishes that the downloaded paper was readable; it does
not validate the scientific claims. At audit time the working-tree copies of
the upstream implementation, arguments, loss, README, and public scripts were
byte-identical to commit `cf0f33a` (`git diff --quiet <commit> -- <paths>`
returned zero).

Evidence labels used below:

- **PAPER**: stated by arXiv v2.
- **CODE**: directly observed at the pinned commit.
- **INFERENCE**: mathematical or execution consequence of PAPER/CODE.
- **UNVERIFIED**: requires unavailable data, environment, or a full run.

## 2. Paper-faithful scoring contract

For model `M`, prefix `h`, and an aligned unit whose tokenizer realization is
`(v_1, ..., v_k)`, paper Eq. (7) defines

```text
s_paper(u | h)
  = (1/k) * sum_j log p_M(v_j | h, v_<j)
  = (1/k) * sum_j [z_j[v_j] - logsumexp(z_j)].
```

For a shared one-token candidate, the score is its original next-token
log-probability. Paper Eq. (8) softmax-normalizes these scores over the finite
candidate set, and Eq. (9) applies the OPD divergence. Appendix B specifies
reverse KL for SimCT and SimpleOPD. The paper explicitly says this is an
operational candidate-scoring interface, not a mass-preserving marginalization
of the full next-token distribution.

Paper-faithful parity therefore requires all of the following:

1. Each selected token score includes its position-specific
   `-logsumexp(z_j)` term.
2. Multi-token units are scored along their autoregressive realization and
   length-normalized before candidate softmax.
3. Shared-token and span scores are expressed in the same log-probability
   coordinate system before normalization.
4. Reverse KL is computed as `KL(q_student || q_teacher)` on the resulting
   finite candidate distributions.

The public reverse-KL implementation satisfies item 4. The public span scoring
does not satisfy items 1--3 in general.

## 3. Pinned public-code behavior

### 3.1 Alignment and virtual rows

**CODE.** `SpanCrossTokenizerKD`:

- aligns next-token label IDs after applying the loss masks;
- decodes each token independently and scans cumulative decoded strings;
- allocates one virtual matrix per sample with shape
  `[number_of_segments, overlap_vocab_size + number_of_span_segments]`;
- uses the first position's raw logits for every shared-vocabulary coordinate;
- activates only the current span column in a row and fills other span columns
  with `-1e9`;
- applies reverse KL row-wise, sums segment losses, then divides by
  `avg_micro_batch_token_num`.

**Observed alignment edge contract.** The scan records a boundary only when
cumulative histories are already equal and the *next* decoded tokens match.
Consequently, a mismatched region is merged with its following shared-token
anchor. A trailing mismatched region with no following shared token/EOS is not
emitted. This is coarser than the paper's common-boundary partition for these
fixtures. `tests/upstream_parity/test_simct_alignment_parity.py` locks both the
paper reference partition and the current scan behavior without importing
PyTorch.

This finding is about the released scan, not the minimal-unit theorem. Whether
real training examples always contain a loss-masked EOS anchor is dataset and
collator dependent and was not established by the static audit.

### 3.2 Mean raw logit is not mean log-probability

**CODE.** For a realized span, `_build_virtual_vocab_logits` assigns

```text
s_code(u | h) = (1/k) * sum_j z_j[v_j].
```

It does not call `log_softmax` for the span or shared coordinates. The paper
instead assigns

```text
s_paper(u | h)
  = s_code(u | h) - (1/k) * sum_j logsumexp(z_j).
```

**INFERENCE.** Candidate softmax is invariant only to a single constant added
to every candidate in a row. Therefore the two distributions coincide only in
special cases, such as when every span position has the same log-normalizer as
the first position used by the shared-token coordinates. They differ in
general. The deterministic fixtures in
`tests/upstream_parity/test_simct_scoring_parity.py` exercise both the invariant
special case and a counterexample.

The code comment that averaging raw logits "corresponds to geometric mean after
softmax" needs this qualification: exponentiating a mean raw logit gives a
geometric mean of unnormalized exponentiated logits. It is not the geometric
mean of full-vocabulary token probabilities unless the omitted per-position
normalizers cancel.

### 3.3 `G(h)` safeguard at `cf0f33a`

**CODE.** Commit `cf0f33a` adds a teacher-only mask with default
`span_gh_mask_threshold = 2.0`. The public scripts do not override it, so the
snapshot's default behavior is:

```text
m(h)        = sum_{w in overlap vocabulary} p_T(w | first position)
span_score  = exp(mean_j log p_T(v_j | h, v_<j))  # span rows only
Z_T(h)      = clamp(m(h) + span_score, max=1)      # span rows
Z_T(h)      = clamp(m(h), max=1)                   # 1:1 rows
keep        = 1[Z_T(h) >= 1/2]                     # equality is kept
loss_row    = loss_row * keep
```

It also reports `gh_mask_ratio`; setting the threshold to `0` disables the
mask. The commit parent has neither this argument nor this loss mask.

This safeguard must be reported separately from paper-faithful SimCT:

- **PAPER:** arXiv v2 predates the safeguard and does not specify it in Eq.
  (7)--(9), Algorithm 1, or Table 4.
- **CODE:** the default at `cf0f33a` is active even when the public OPD scripts
  omit the flag.
- **INFERENCE:** the mask changes which aligned rows contribute gradients, so a
  default run at `cf0f33a` is not a literal reproduction of the paper objective.
- **INFERENCE:** for a multi-token span, `exp(mean log p)` is a
  length-normalized score rather than the sequence-event probability `prod p`.
  Adding it to shared-token mass and clamping can be a useful safeguard, but
  `Z_T` is not generally an exact captured probability mass. It may also overlap
  a shared first-token event. Thus `G=1/Z_T` is a code-defined heuristic
  multiplier, not a proven mass-preserving renormalization constant.
- **CODE consistency caveat:** `Z_T` uses mean *log-probability*, while the
  virtual distribution being masked uses mean *raw logit*. The safeguard's
  span score therefore does not exactly reconstruct the score used by that
  distribution.

## 4. Public training-script audit

### 4.1 Teacher-response generation and SFT wrappers

| Item | Paper v2 | Public code at `cf0f33a` | Contract verdict |
|---|---|---|---|
| Teacher candidates | 8, temperature `0.6`, top-p `0.95`, max 4096 | Generation scripts pass all four values | Matches declared sampling values |
| Generation entry point | Released curation/generation pipeline | Both wrappers set `SCRIPT_DIR` to `scripts/` but call `scripts/generate_teacher_responses.py`; the file is actually under `scripts/sft/` | Wrapper path is unresolved from the documented invocation |
| SFT dataset builder | Select one response through task checks/filters | Both wrappers similarly call nonexistent `scripts/build_sft_warmup_dataset.py` instead of `scripts/sft/...` | Wrapper path is unresolved |
| SFT LR / epochs | `2e-6` / 2 | `2e-6` / 2 | Matches |
| SFT per-device / accumulation / effective batch | 2 / 4 / 64 on 8 GPUs | YAML uses 4 / 4, implying 128 on 8 GPUs | Does not match Table 4 |
| SFT max sequence length | 4096 | YAML `cutoff_len: 2048` | Does not match Table 4 |
| SFT model path | Not environment-specific | Wrappers copy to `/root/workspace/models/...`, but YAML points to relative `models/...` | Copied path is not consumed by the YAML |
| LLaMA-Factory version | Not pinned in Table 4 | README names LLaMA-Factory; upstream requirements do not pin it | Environment blocker |

The SFT wrappers' destructive process management and GPU execution were not run.
The path findings come from deterministic path resolution only.

### 4.2 On-policy distillation scripts

All three `scripts/ctopd/*span*.sh` files select `span_ctkd`, reverse KL, global
batch 64, micro-batch 1, warmup 0.05, bf16, 8 GPUs, temperature 0.6, and rollout
max length 4096. The following paper/script gaps are common to all three:

| Field | Paper Table 4 | Public scripts / effective default | Verdict |
|---|---:|---:|---|
| Peak learning rate | `1e-6` | `5e-7` | Mismatch |
| Training length | 2 epochs | 1 epoch | Mismatch |
| Max sequence length | 4096 | `--max_len 8192` | Mismatch |
| Rollout top-p | `0.95` | flag omitted; `RolloutArguments.top_p` defaults to `1.0` | Mismatch |
| `G(h)` mask | Not specified | threshold omitted; default `2.0` is active | Post-paper behavior |
| Warm-start checkpoint | Completed shared warm-start is described | hard-coded `checkpoint-80`, `checkpoint-40`, or `checkpoint-20`; selection rationale/manifests absent | Provenance blocker |

Additional code defaults relevant to exact reruns include seed 42, Adam betas
`(0.9, 0.98)`, weight decay 0, and a `cosine_with_min_lr` scheduler with minimum
LR `1e-8`. Only weight decay and a generic cosine schedule are stated in paper
Table 4; the other exact optimizer/scheduler details are code-observed and must
not be back-attributed to the paper without author confirmation.

### 4.3 Sequence-budget interpretation used by the Tunix port

The paper reports both maximum sequence length 4096 and rollout maximum length
4096, but does not specify whether the latter is an independent completion cap
or whether prompts reduce the available rollout length. The Tunix `paper4k`
protocol therefore makes this reproduction interpretation explicit:

```text
student prompt tokens + generated completion tokens <= 4096
generated completion tokens <= 4096
```

This is an **INFERENCE**, not an author-confirmed detail. The runtime caps the
completion by the remaining student sequence capacity and logs requested versus
actual lengths. Because the native Tunix sampler statically pads prompts to
`max_prompt_tokens`, the executable completion cap is
`min(max_completion_tokens, max_sequence_tokens - max_prompt_tokens)`; it is
3840 for the current `paper4k` prompt cap of 256. The corresponding two-epoch
10K-data template uses 314 logical
optimizer updates at effective batch 64 (20,096 response consumptions), so the
small final-epoch overshoot is explicit rather than silently calling 314 updates
an exact 20,000-example traversal.

The separate `public8k` protocol freezes the released launch semantics:
`max_len=8192`, rollout cap 4096, LR `5e-7`, one epoch (157 effective-batch-64
updates), and the omitted top-p flag's default 1.0. It is a public-code
ablation, not Table 4 reproduction. Results from `paper4k` and `public8k` must
not be pooled.

`tests/upstream_parity/test_public_training_contract.py` checks these public
values and the unresolved wrapper paths without launching a process.

## 5. Reproduction modes that must stay distinct

| Mode | Required implementation | Meaning |
|---|---|---|
| `paper_math` | mean log-probability Eq. (7), finite candidate softmax Eq. (8), reverse KL, no undocumented `G(h)` mask | Tests the written objective |
| `public_code_pre_safeguard` | commit `4dae6fc`, mean raw logits, released alignment/virtual-row behavior | Tests the initial public implementation |
| `public_code_cf0f33a_default` | commit `cf0f33a`, mean raw logits, default `G(h)>2` mask | Tests current pinned public defaults |
| `public_scripts_cf0f33a` | preceding code mode plus the released shell/YAML values | Tests what the public launch artifacts request, once path blockers are fixed externally |

Results from one mode must not be labeled as results from another. In
particular, disabling the safeguard makes `cf0f33a` closer to its parent but
does not repair mean-raw-logit scoring or the alignment edge behavior.

## 6. CPU-light parity checks

Run only the scoped tests:

```bash
python -m pytest -q tests/upstream_parity
```

The tests are standard-library-only apart from pytest. They deliberately avoid
importing KDFlow/PyTorch and never launch SGLang, Ray, CUDA, or training.

These are contract tests, not empirical reproduction. A passing result means
the checked source snapshot still exhibits the documented behavior and the
paper/reference fixtures still demonstrate the stated mathematical relation.

## 7. Remaining blockers before scientific reproduction

1. Decide which of the four modes in Section 5 is the experimental target.
2. For a paper-math run, implement Eq. (7) in a separate authorized path and
   validate gradients; this audit did not edit `span_ctkd.py`.
3. Obtain the exact 10K corpus or reproducible source identifiers, filtering
   manifests, teacher generations, and selected-response checksums. The public
   repo does not contain those artifacts.
4. Obtain exact warm-start checkpoint hashes and explain the three hard-coded
   checkpoint selections.
5. Pin LLaMA-Factory, Transformers, SGLang, CUDA, driver, and hardware versions;
   the repository requirements alone do not reconstruct the paper environment.
6. Resolve wrapper paths and reconcile SFT/OPD hyperparameters with Table 4.
7. Record tokenizer revisions, chat templates, special-token behavior, and
   whether loss-masked responses always include EOS.
8. Only then run a cheap one-batch parity canary before any full 8-GPU job.
9. The paper's Pass@1 means/standard deviations, recovered-supervision ratios,
   KL-gap ablations, and compute claims remain **UNVERIFIED**.

## 8. Primary-source claim map

### Paper

- arXiv metadata and version history:
  <https://arxiv.org/abs/2605.07711>
- Methodology, Eq. (6)--(9), and explicit non-marginal interpretation:
  <https://arxiv.org/html/2605.07711v2#S3.SS3>
- OPD loop, reverse KL statement, Algorithm 1, and Table 4:
  <https://arxiv.org/html/2605.07711v2#A2>
- Boundary graph and minimal aligned units:
  <https://arxiv.org/html/2605.07711v2#A4.SS1>

### Pinned public code

- Span alignment and virtual scoring:
  <https://github.com/sunjie279/SimCT-/blob/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e/kdflow/algorithms/span_ctkd.py>
- `G(h)` default argument:
  <https://github.com/sunjie279/SimCT-/blob/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e/kdflow/arguments/distillation_args.py>
- Reverse KL:
  <https://github.com/sunjie279/SimCT-/blob/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e/kdflow/loss/reverse_kl_div.py>
- Public reproduction pipeline and script inventory:
  <https://github.com/sunjie279/SimCT-/blob/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e/README.md>
- Safeguard-only commit diff:
  <https://github.com/sunjie279/SimCT-/commit/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e>
