# 20-hour bounded company campaign

This release is an executable **baseline + diagnostic campaign**, not a full learned-partition trainer. No new guide source has been supplied. Default references are selected.parquet already seen by SFT. Never label these probes unseen or a validation pass.

## Frozen data contract

- Baselines keep the original prompts.parquet, SFT initialization, 312 updates, seed43 and fixed length2; no test-based hyperparameter changes.
- Probe preparation normalizes unique prompt identities, excludes conflicting references, and assigns disjoint rollout/select/eval prompts: 32 groups x (1+4+4)=288 prompts, seed20260910.
- Select is guide-train: participates in virtual-update selection and weighting learning. Eval is guide-val: never receives gradient but informs exploratory decisions. These roles are disjoint internally, **not disjoint from historical SFT by default**.
- Existing four benchmark datasets are reused only for endpoint reporting, seeds42/43/44. They are exploratory test results, not an independent confirmation set.
- External reference rows may be supplied with `--references FILE --reference-key KEY --exclude SFT_ROWS --exclude TRAIN_ROWS`. Rows require messages ending in user and textual reference. Exclusion is exact normalized message identity, not semantic near-duplicate detection; external source still needs provenance audit. The workflow does not invent a new test set or call generated teacher text ground truth.
- Teacher generation for new guides is pending selection of an authorized prompt source. It is deliberately not replaced by silently reusing train data as unseen data.

## DAG and budgets

GPU0: fixed5-update canary (30min cap), fixed seed43 full (6.5h cap), endpoint generation128 (2h cap).
GPU1: atomic canary (30min), three paired oracle/weighting diagnostics (.01/.1/1 virtual LR, 45min each), adaptive diagnostic (90min), atomic seed43 full (6.5h), endpoint generation128 (2h).
CPU: preparation, decisions, one scoring coordinator at a time (16 workers) independently of generation. Scoring starts when an endpoint plan is ready and consumes the spool while generation runs.
20h is a maximum elapsed time from first manager start; resume retains the original clock. Do not fill spare time with an unqualified model. Jobs without their entire reserved budget remaining are not admitted. Training is not auto-retried after interruption; preserve checkpoints and inspect before restarting. Completed jobs are skipped.

## Decision frozen before probing

Only LR=.1 is the primary signal. Other LRs are sensitivity measurements. Require >=24 valid groups, <=8 invalid, oracle-vs-atomic bootstrap lower bound >0, positive fraction >=.6. This is an exploratory operational gate, not a scientific significance claim. If met: repeat probe with four weighting optimization steps/group. Otherwise: random partition 50-update pilot. Repeated guide-val use is explicitly exploratory. Neither branch produces a qualified learned-partition energy checkpoint or authorizes long learned training.

## Run

Update source via the existing HF updater, then:

```bash
bash /workspace/storage-shared/nlp/tungks/SimCT-git/experiments/runai/transfer/start-campaign.sh /workspace/storage-shared/nlp/tungks/simct-eval-queue-qjG2gMc2/trial256-20260910-083920-466282/plan.json
```

Logs, immutable plan, config, campaign-state.json, decision.json, per-run checkpoints and endpoint summaries live in the printed WORK directory. No existing training directory is overwritten. Training logs remain local; there is no automatic W&B backfill in this release.

Manager holds a single campaign lock, shares GPU leases with eval, checks <1GiB used before training, tracks owned descendant processes and their start identities for cleanup, and never issues global pkill/ray stop. Do not concurrently update its source checkout. Eval data and checkpoint hashes are revalidated by the existing eval harness. Linux/company runtime required; local CPU tests do not qualify B200 execution.
