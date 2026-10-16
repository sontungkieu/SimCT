# H2 word pilot and tokenizer audit

User-authorized scope: 100 prompts (50 GSM8K train, 50 MATH train), two frozen
student responses per prompt, seed42, temperature1/top-p1/top-k0, cap1024;
1000 exactly aligned response positions. No training or H1/H3. USD10 maximum
total Modal spend, profile `lhtu05`. No company jobs are authorized yet.

Run: `h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1`.
Config: `h2_word_pilot_config.json`. Roles:

- S: `meta-llama/Llama-3.2-3B-Instruct`.
- B: `NousResearch/Meta-Llama-3.1-8B-Instruct`, the user-authorized mirror of
  gated `meta-llama/Llama-3.1-8B-Instruct` (official access returned403).
- Q: `Qwen/Qwen2-7B-Instruct`, explicitly Qwen2, not Qwen2.5.

Model revisions, tokenizer file/vocabulary hashes and verified downloaded weight
SHA256s are in each run's `assets.json`. Mirror and official public LFS shard
metadata are compared; small-file Git blobs are compared when available.
Do not assume byte-identical tokenizer/config from a model name alone.
Models/cache remain on Modal volumes; local source overlays reuse the SimCT
runtime digest from `h2_pilot_config.json`. No local packages/models are installed.

## Precision qualification

HF eager BF16 on S failed the initial all-vocabulary generation/cache test.
Single-row versus full-sequence LM-head GEMMs also used different shapes.
The repaired check compares the full native-forward head shape; the pilot uses
float32 backbone/head, TF32 disabled, L40S, generation batch4. It validates
padded generation versus independent forward, cached versus uncached logits,
and real same-model word distributions through both scoring paths. Old failed
logs and receipts remain; thresholds have not been relaxed. This is a new HF
pilot and does not qualify the old SGLang study or its failed parity gate.

Generation stores native prefix/content/output IDs and strips only the first
selected terminal and everything after it. Llama stop IDs are the union of
tokenizer EOS, model/generation EOS, eot/eom/end-of-text controls. The stable
batch offset determines CUDA RNG seed. Resume requires an exact task prefix.

## Tokenizer audit: all responses, before filtering

The audit runs on all100 neutral reference solutions and all200 student
responses, independently of the 1000-position word subset. Exact native
template/content byte roundtrips are asserted; failures and excluded terminal
IDs are retained rather than silently counted as mismatches.

Every pair reports:

- strict1:1 span occurrence coverage, with separate student-token,
  teacher-token and byte denominators; whitespace included/excluded;
- interior-boundary precision/recall/Jaccard, excluding source start/end;
- minimal common-boundary interval decomposition into1:1,1:N,N:1,N:M,
  including interval counts, token counts, byte counts and fractions;
- word-initial/internal, digits, punctuation, LaTeX marker, whitespace,
  other categories; numeric-run length, boundary offsets and segmentation;
- Unicode/newline totals, split UTF8 boundaries, token expansion ratio
  recoverable from exact counts, dropped tokens under strict filtering;
- global micro, rollout macro, prompt macro, per-prompt and per-rollout
  statistics plus p0/p10/p25/p50/p75/p90/p100 coverage quantiles;
- full source text and per-student-token trace with all overlapping teacher
  IDs/indices, byte offsets, same head/end, category and minimal interval.

The primary new pairs preserve S's generated native content IDs. B consumes
those same IDs only after identical vocabulary hashes are verified, making its
100% span match a property of this control. Separate canonical-B/canonical-Q
audits tokenize the raw text independently, and native-versus-canonical S is
reported. Compare historical and new tokenizer families through the canonical
controls on the common successful texts. Occurrence-weighted availability of
each token's byte string in the other vocabulary is separate from strict spans.

Vocabulary overlap is a separate unique decoded-byte-string metric. Numeric
token-ID equality across vocabularies is only an incidental diagnostic.
Historical Gemma2-2B/Gemma2-9B/Qwen2.5-7B cached tokenizers are applied to the
same raw text without running those models. Old/new comparisons use the common
successful response intersection; independent success summaries remain visible.

No source/denominator for the user's historical90% claim has been provided.
The audit can test that explicitly defined metric on these corpora; it cannot
declare an unknown report fabricated. Neutral-reference and on-policy numbers
are always separate.

## Word measurement

Only positions strict-aligned for S/B/Q are sampled, after excluding response
positions0–4, whitespace-only and incomplete UTF8 contexts. Deterministic hash
ordering and round-robin prompt/category strata spread1000 positions; each
position has weight N_stratum/n_stratum. Bootstrap resamples prompts1000 times;
the estimand is an equal-prompt mean of weighted available position means.

Candidates are the S/B/Q top10 union extended greedily up to4 tokens, plus the
realized event. A shared grammar defines Unicode letter runs, ASCII-letter
LaTeX commands, atomic digits/symbols and leading whitespace. Letter/command
termination uses a consistent next-token boundary mask; single digits and
symbols do not need it. This deliberately repairs the literal guide's
inconsistent single-digit-plus-boundary rule. Truncated proposals are counted.

The sampled native prefix IDs are held fixed; only each candidate suffix is
canonically encoded. Single ASCII non-letter atoms use exact next-byte
marginal probability, including native tokens crossing the atom boundary
(for example, `123` contributes to the first-digit event `1`). Other events
use the canonical suffix path and consistent boundary factor, with independent
KV-cache clones and contextual roundtrip checks. Candidates invalid
under any model are removed jointly; counts are reported. OTHER is1-minus
candidate mass. Masses materially exceeding1 cause a failure; only <=1e-6
roundoff is tolerated. These hybrid measurements include exact ASCII atomic
events but are not exhaustive word text marginals: alternative word paths and
tokens crossing a letter/command boundary are not exhaustively marginalized.
Log-softmax and mass normalization use float64 over float32 model logits.

Outputs include candidate probabilities/OTHER, JS raw, top1 tie overlap,
Kendall tau-b, entropy differences, X_raw and X_shape. Shape matching applies
power-temperature to word-plus-OTHER cells to match student entropy; unreachable
targets retain explicit flags. It is not token-logit temperature calibration.
Global reference calibration is deferred. Report OTHER dominance, failures,
missing shape positions and highest-X examples alongside aggregate results.
This exploratory pilot cannot grant the original paper's Gate1 or isolate
teacher quality/capacity effects.

Sensitivity diagnostics include a shared 258-cell next-byte distribution
(256 bytes, EOS, and nontext/padded residual), its raw/shape excess JS, and
candidate-conditional JS with OTHER removed. They are distinct measurements;
word JS minus byte JS is not a causal decomposition. Per-position telemetry
retains realized-token log-prob, top64 IDs/log-probs and their retained mass,
native-token entropy, probability methods and timing. The first two eligible
sampled prefixes also compare cached versus uncached word probabilities;
the stored JS must remain below1e-5.

## Stage order and artifacts

Use the explicit profile and run ID; every CLI call saves its own numbered log.

```bash
cd /mnt/d/dev/dsh/research_vai/SimCT
H2_RUN_ID=h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1 \
uvx --offline --from modal modal run --profile lhtu05 --timestamps \
  experiments/modal/h2_pilot_modal.py --stage word-access --arm ALL --attempt 3
```

That access command is a reference, not a request to repeat a failed attempt.
Actual order: access -> prepare weights per S/B/Q -> prompts/reference audit ->
qualify S/B/Q -> student rollout -> full tokenizer audit -> fixtures -> proposal
per model -> merge -> scoring per model -> report -> export. CPU stages use ALL. GPU
stages use S/B/Q and L40S; only S generates. Qualification is pinned to model
revision/config hash. Per-stage caps and retries0 stop only the owned process
group; checkpointed proposal/scoring rows preserve work on failure.

Local tests use existing Python/stdlib:

```powershell
python -B -m unittest discover -s experiments/diagnostics -p 'test_h2*.py' -v
```

The result directory on `sb-opd-results-v1` has `word-rollouts.json`,
`tokenizer-audit-{reference,rollout}.json`, per-response detailed audit JSONs,
`vocabulary-overlap.json`, `word-fixtures.json`, `word-candidates.json`,
`word-proposals-{S,B,Q}.json`, `word-scores-{S,B,Q}.json`, `word-metrics.json`,
`word-summary.json`, and all numbered receipts/logs. No raw artifacts are
committed. Scientific completion requires all200 responses, exactly1000 unique
position scores per model, reported failures, and validated probability vectors.
`word-export` emits a ZIP and per-file SHA256/size manifest without model
caches or credentials. Run `h2_word_report_export.py --run-dir <extracted-run>`
with existing Python to produce a Vietnamese HTML/Markdown report and detailed
CSV exports. A company handoff should reuse the frozen pilot responses and
candidates first, qualify the chosen B200/backend, and only then increase the
position count; generation tokens/s is not a word-scoring throughput estimate.
