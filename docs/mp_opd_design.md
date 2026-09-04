# MP-OPD v0 design and implementation contract

## Status and evidence boundary

MP-OPD is exposed as a **KDFlow algorithm mode**, not as a long-lived server
branch:

```text
--kd_algorithm mp_opd
--mp_opd_mode atomic|fixed|random|oracle|soft
```

The isolated worktree used during development protects the completed SimCT and
X-Token evidence. It is not part of the runtime interface. Unit tests and the
toy oracle are labelled `implementation_validation` or `oracle_diagnostic`.
They do not establish an LLM improvement, paper reproduction, or novelty.

Distributional `span_ctkd` and projection-based `xtoken` remain unchanged.
`mp_opd atomic` is a scalar score-credit baseline that uses SimCT boundaries;
it must not be reported as the original distributional SimCT baseline.

## Formal correction to the proposal

The proposal defines `h_i` as a **sum** of token NLLs but its span-loss equation
multiplies those sums by token counts again. That double-counts length. MP-OPD
v0 uses the following single convention throughout.

For atom `a_i`, student prediction range `I_i`, teacher prediction range `J_i`:

```text
lT_i     = sum_(j in J_i) log p_T(t_j | teacher prefix)
lS_i_old = sum_(t in I_i) log p_theta_old(y_t | student prefix)
b_i      = stopgrad(lT_i - lS_i_old)
w_i      = |I_i| > 0
r_i      = b_i / w_i
h_i      = -sum_(t in I_i) log p_theta(y_t | student prefix)
```

For a candidate contiguous span `c=[i,j)`:

```text
B_c = sum_(k=i)^(j-1) b_k
W_c = sum_(k=i)^(j-1) w_k
r_c = B_c / W_c
ell_c = r_c * sum_(k=i)^(j-1) h_k
L_P = sum_(c in P) ell_c
```

There is no extra `w_i` inside `ell_c`. For span marginals `mu(c)`:

```text
rbar_i = sum_(c contains i) mu(c) r_c
L_soft = sum_i rbar_i h_i = sum_c mu(c) ell_c
sum_i w_i rbar_i = sum_i b_i
```

If all rates in a merged span are constant, loss and gradient are identical to
the atomic partition. If maximum span length is one, the exact result is
`sum_i (b_i/w_i) h_i`.

For outer gradient `v=grad F_M(theta)` and realized student path score:

```text
z_i = <v, grad_theta sum_(t in I_i) log p_theta(y_t | ...)>
U_c = <v, grad ell_c> = -r_c * sum_(i in c) z_i
```

The hard oracle maximizes `sum_c U_c`. Tests compare this prefix-sum form with
explicit autograd and check the sign using actual virtual updates. The proposal
needs an erratum in both its span-loss equation and its utility equation: remove
the second token-count factor in each.

## Tensor and detach contract

| Quantity | Shape | Gradient |
|---|---:|---|
| student logits at response predictors | `[S,V_s]` | student |
| teacher logits at response predictors | `[T,V_t]` | none |
| atom `b,w,r` | `[n]` | detached |
| current atom NLL `h` | `[n]` | student |
| atom features | `[n,10]` | detached |
| span energy | `[n,L]` | energy model only |
| span marginal | `[n,L]` | energy model only during meta step |
| rate used by real student loss | `[n]` | detached from energy |
| oracle utility | `[n,L]` | detached target |

Teacher parameters are frozen. The real student optimizer never owns energy
parameters. In `soft` student training, marginals are detached before weighting
`h_i`. A separate energy optimizer is serialized with a separate format,
configuration hash, and step. The one-step claim applies to virtual SGD in the
declared adapter subspace, not to a full Adam trajectory.

## Atomization and failure policy

`SimCTAtomizer` consumes response-only, shifted label IDs after the caller's
loss masks. It creates only minimal synchronized segments whose cumulative
decoded UTF-8 bytes are identical. Each atom records:

- stable sample ID;
- half-open student and teacher prediction ranges;
- half-open response byte interval;
- student/teacher token counts;
- one-to-one or multi-token boundary type;
- validity/failure reason; and
- optional detached scalar credit fields.

Atoms are ordered, non-overlapping and gap-free on covered events. A transient
replacement character from a token prefix that ends inside a multi-byte UTF-8
scalar is skipped as a candidate boundary; the complete decode must still be
replacement-free and byte-identical across tokenizers. The v0 atomizer fails
closed on a replacement character in the complete decode, normalization
mismatch, empty decode, unsupported added-token semantics, empty responses, or
an unaligned suffix. Padding never reaches the atomizer. Terminal EOS is masked
and counted: v0 does not pretend two unrelated EOS IDs decode to the same
response bytes.

An invalid sample contributes a differentiable zero loss and explicit failure
telemetry. This matters when `micro_train_batch_size=1`: an unrepresentable
teacher decode must be excluded without aborting the other valid samples in the
same gradient-accumulation window. The effective valid-sample count and each
failure reason are logged, so exclusion cannot be mistaken for training signal.

Prompt/completion ambiguity is prevented structurally: the API accepts only
labels selected by response loss masks. A caller that cannot prove that
boundary must not call MP-OPD.

## Partition modes

- `atomic`: every atom is a length-one span.
- `fixed`: deterministic length `mp_opd_fixed_span_length`; the tail may be
  shorter.
- `random`: a full-cover contiguous partition drawn from the documented
  seeded uniform-next-length procedure. This is a matched-capacity control.
- `oracle`: hard max-sum DP over detached utilities. Normal training fails
  closed unless instrumentation supplies per-atom directional scores.
- `soft`: a learned semi-Markov distribution. Normal training requires an
  audited energy checkpoint; random initialization is rejected.

No mode splits a SimCT atom.

## Semi-Markov dynamic program

Candidate span `(i,j)` is valid when `1 <= j-i <= L`. Energies are laid out as
`energy[i,j-i-1]`. Invalid cells are `-inf`. The forward and backward recurrences
run in float32 log space:

```text
alpha[0] = 0
alpha[j] = logsumexp_i(alpha[i] + energy[i,j]/tau)
beta[n] = 0
beta[i] = logsumexp_j(energy[i,j]/tau + beta[j])
mu(i,j) = exp(alpha[i] + energy[i,j]/tau + beta[j] - alpha[n])
```

Time is `O(nL)` and marginal storage is `O(nL)`. Every atom must have marginal
coverage one. Partition entropy is `logZ - sum_c mu(c)s(c)/tau`; expected span
count is `sum_c mu(c)` and expected span length is total marginal length divided
by expected count. `n=0` returns the neutral empty partition. `n=1` and `L>n`
are handled geometrically. An all-invalid full-cover graph raises an error.

## Energy model and bilevel order

The v0 energy network is a two-layer bidirectional GRU followed by a span
scorer over start, end, prefix-pooled interior and normalized span length.
Inputs are detached atom features: rate, total credit, student count, teacher
count, byte length, teacher/student average log probability, boundary-type
indicators and validity. Reference/meta answers are never features.

The reusable model, checkpoint and semi-Markov mechanics are implemented. The
current KDFlow rollout batch does not carry an independent stable-ID meta batch,
so distributed `soft` training is deliberately fail-closed unless a separately
trained energy checkpoint is supplied. The next integration must add a B/M
loader with disjoint stable IDs, then perform:

1. freeze current adapter state;
2. compute detached atoms/features on rollout batch B;
3. compute `mu_phi` and expected virtual SGD update;
4. evaluate reference NLL on independent meta batch M;
5. form exact one-step hypergradient or
   `-eta sum_c mu(c) stopgrad(U_c)`;
6. step only the energy optimizer;
7. recompute/fix marginals and step the real student optimizer exactly once.

The toy runner validates this surrogate's directional derivative but is not a
substitute for the missing real-data B/M loader.

## Oracle falsification gate and data splits

`experiments/mp_opd/toy_oracle.py` uses disjoint stable fixture IDs for rollout,
meta-train, validation and test. It computes every candidate utility in an
adapter-only linear model, runs hard DP, enumerates all small partitions,
performs actual mutation-free virtual updates for atomic and oracle branches,
and reports predicted/actual headroom plus Spearman rank correlation.

The real gate must pin dataset and model revisions and record exact ID hashes.
Rollout remains student generated; M uses high-quality reference responses.
Benchmark test data cannot select partitions. Adapter-only headroom does not
prove full-parameter headroom.

## Metrics and evidence labels

Training emits scalar, finite, W&B-safe values for atom counts, valid/invalid
sample counts and ratio, per-reason exclusions, covered student/teacher events,
one-to-one/multi-token ratio, candidate span count, masked EOS count and `b/r`
distribution. Soft mode additionally
emits logZ, partition entropy, expected span count/length, marginal coverage
error and credit-conservation residual. Oracle instrumentation emits predicted
utility. The local JSON also records source identity, config hash, pinned toy
revisions, split IDs, seeds, timing and gate results.

Use exactly these evidence labels:

- `implementation_validation`: unit/integration consistency only;
- `oracle_diagnostic`: local headroom/falsification evidence only;
- `training_evidence`: only a terminal audited training run with finite metrics.

## Known v0 limitations

- Byte equality is established through exact cumulative decode. Tokenizers
  with context-dependent decode that cannot expose stable response bytes fail
  closed.
- EOS is masked, not learned as an endpoint atom.
- The current online trainer has no independent meta batch and stable record
  IDs, so learned-energy training is not yet an end-to-end KDFlow feature.
- Oracle computations are adapter/subspace diagnostics; they make no claim
  about full-parameter FSDP optimization.
- The toy positive headroom is an analytic fixture, not an LLM result.
