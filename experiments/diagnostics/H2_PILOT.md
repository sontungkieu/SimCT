# H2 pilot: gated inference scaffold

This is work in progress, not a completed H2 experiment or reproduction.

The user-approved pilot is 512 training prompts (256 GSM8K, 256 MATH), two
student rollouts per prompt, seed 42, temperature 1, top-p 1, cap 1024.
E3 is limited to 1000 sampled positions. No OPD training, H1 or H3 is enabled.
The config is `h2_pilot_config.json`. The fixed prompt manifest is a subset of
the preselected 2000-prompt full pool; test items occur only in quality controls.

The Modal wrapper reuses the immutable SimCT runtime digest and existing
`sb-opd-assets-v1` / `sb-opd-results-v1` volumes on profile `lhtu05`. All downloads
and model execution are remote. The local source overlay is identified by its
per-file SHA256 in each receipt, with separate logs for numbered attempts.
No failed attempt is overwritten. GPU stages have a process runtime cap and
retries=0; only the stage's own child process group can be terminated.

## Verified remote state, 2026-10-15 local time

- CPU preparation completed: 512 pool prompts, 256 quality problems, pinned
  dataset revisions/file hashes, calibration IDs, S/G9/Q pinned cached weights.
- Llama-3.1-8B-Instruct access check returns HTTP403. There is no substitute.
- Canonical Gemma BF16 LM-head reconstruction matches eager model.forward
  exactly on the two short fixed probes (max logit error0).
- SGLang FlashInfer versus HF eager parity FAILED. Raw FP32 log-softmax
  comparison: mean absolute logprob error0.04669036281605562,
  max0.2551841735839844 over39 input tokens.
- A scoped canonical-softcap adapter was tested, but did not resolve parity:
  mean0.04513782556526936, max0.30100440979003906. This is a failed diagnostic,
  not a qualified runtime fix. No threshold was relaxed.
- No student scientific rollout pool has been generated. No GPU inference,
  scoring, E2/E3, quality or H2 efficacy result may be claimed from preparation.
- All apps from these qualification attempts are stopped with tasks0.

`remote_artifacts/h2-pilot-s42-n2-20261015-r1/` contains local submit logs;
the same run directory on `sb-opd-results-v1` contains worker logs, receipts
and numbered parity artifacts. These raw artifacts must not be committed.

## Implemented but not runtime-qualified

The scaffold includes native-ID student generation, teacher-forced eager
hidden-state extraction, full-vocabulary E1/E2 projections (EOS/residual and
padded head IDs explicit), sparse top64 telemetry, finite E3 candidate scoring,
reference-solution temperature calibration, template sensitivity, and
prompt-clustered summaries. None of these downstream stages has been executed.
Five dependency-free measurement tests pass; AST/config checks pass. These
checks do not establish GPU/backend correctness or end-to-end completion.

The later user-authorized Llama3.2-3B / Llama3.1-8B mirror / Qwen2-7B pilot
uses a separate run and HF qualification. See `H2_WORD_PILOT.md`; its outputs
must not be attributed to this original Gemma/SGLang scaffold.

## Bounded performance benchmark (separate from scientific stages)

The subsequent user request authorizes a100-position Modal cost benchmark;
larger work belongs on company hardware. Current requested teachers are G9
and Q, with S as student. L is deferred, and no L job is launched by the
benchmark. `h2_e3_benchmark.py` runs under a fresh run ID and writes only
performance-proxy artifacts; it never relaxes `require_parity` for the study.

Because there is no qualified student pool yet, its fixtures are100 aligned
positions from63 GSM8K/MATH training reference problems. Their native prefix
lengths range36–1176 for S/G9 and57–1214 for Q. Selection is by seed42 hash,
stratified by category/prefix-length bucket; weights refer to this reference
fixture population, not to an unobserved student-rollout population.

It measures top10 greedy candidate proposals (at most4 tokens), the deduplicated
S/G9/Q union plus realized reference word, and scoring with independent KV
cache clones, canonical prefix/roundtrip checks, full-vocabulary boundary
softmax, raw temperature1 and representative calibrated temperature1.25.
No global temperature is fitted by this timing test. It uses HF eager BF16
throughout; the failed SGLang softcap adapter is disabled for benchmark stages.
Each stage has a900-second worker cap. Rate estimates exclude the first5
positions, retain warmup telemetry, and extrapolate8k with20% reserve.
Canonical-path rejections and reference-versus-student differences must be
reported alongside cost. Benchmark data cannot be used as H2 evidence.

Stage order: `bench-prepare` (CPU), `bench-proposal` per S/G9/Q,
`bench-merge` (CPU), `bench-scoring` per S/G9/Q, `bench-report` (CPU).
Use `H2_RUN_ID=h2-e3-bench100-20261015-r1` and the explicit lhtu05 profile.
S uses L4+4CPU+24GiB; G9/Q use L40S+4CPU+32GiB. Cache assets are reused;
no new dependency runtime is installed or rebuilt.

The 100-position run completed successfully for all three models. Measured
population-weighted seconds per position (first five excluded) are:

| Model / GPU | Candidate proposal | Word scoring | Projected 8k GPU hours | Projected 8k USD |
| --- | ---: | ---: | ---: | ---: |
| S / L4 | 1.965878 | 0.815440 | 6.190885 | 7.303065 |
| G9 / L40S | 2.098019 | 0.820735 | 6.498546 | 15.567813 |
| Q / L40S | 0.851697 | 0.455157 | 2.917908 | 6.990093 |

The observed union has 12.68 candidates/position (range 3–25); its weighted
population estimate is 14.126788. Canonical-prefix/roundtrip checks reject
42/1268 candidate attempts each for S/G9 and 40/1268 for Q. The four-token
proposal cap leaves 7/1000 proposals incomplete for S and 9/1000 each for
G9/Q; this cap is part of the measured workload. Extrapolation is
15.607339 aggregate GPU hours and USD29.860971, or USD35.833165 with 20%
reserve. A sensitivity calculation charging every candidate at the observed
valid-candidate rate gives USD36.261969 including reserve. These estimates
exclude rollout generation, full-pool scoring, calibration fitting, parity
repairs and Modal startup/output IO. The 8k workload has not been launched.

At 2026-10-14T18:53:40Z, Modal billing reports USD0.49660090 for this benchmark
description; billing may still lag. The local machine-readable summary and
per-model artifacts are in `remote_artifacts/h2-e3-bench100-20261015-r1/`.
Proposal/scoring receipts identify the executed source hashes; later dispatch
guards prevent L jobs and require a separate benchmark run ID. These guards
were checked locally, and the final CPU report used the guarded wrapper.
All six model artifacts contain 100 matching unique positions, with finite
positive timings and valid/rejected candidate counts accounting for every
attempt. Five measurement tests and AST checks pass.

The finite E3 implementation is explicitly an approximate canonical-path,
candidate-conditional view. It does NOT implement the new PDF's explicit
word-boundary factor plus OTHER distribution. The guide is new source material
for review; it has not automatically replaced the user-approved protocol.
Before scientific continuation, resolve backend parity and settle the event
definition, canonical prefix checks, boundary predicate, OTHER validity,
category priority and sampling-weight estimator.

## Running a stage

Use the cached WSL Modal CLI, the Windows checkout mounted under `/mnt/d`,
and an explicit profile. Refresh and evaluate the billing guard before each
paid stage. The active profile default is unrelated and must not be relied on.

```bash
cd /mnt/d/dev/dsh/research_vai/SimCT
uvx --offline --from modal modal run --profile lhtu05 --timestamps \
  experiments/modal/h2_pilot_modal.py --stage parity --arm S --gpu L4 --attempt 4
```

That command is an example, not an authorization to repeat an unchanged
failed probe. Use a new attempt only after a specific repair and preserve the
prior evidence. `require_parity` blocks downstream model stages after failure.

If the pipeline is qualified, the stage order is prepare -> parity for each
model -> student rollout -> per-model score -> joint events -> per-model
E3 candidates -> CPU merge -> per-model E3 score -> CPU report. Quality
controls are separate per-model generation stages. `report` explicitly labels
missing L/capacity/placebo controls and cannot grant the original Gate1.

For dependency-free local checks:

```powershell
python -m unittest discover -s experiments/diagnostics -p test_h2_measurements.py -v
```
