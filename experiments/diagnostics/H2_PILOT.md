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
