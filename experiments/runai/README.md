# RunAI single-GPU MP-OPD launcher

Evaluation preparation (protocol, known paper/code mismatches, read-only inventory):
[SimCT evaluation preparation](../../docs/simct_eval_preparation_20260909.md).
This does not schedule evaluation or start GPU work.
Implemented runner: [evaluation contracts](../../scripts/evaluation/CONTRACT_README.md)
(`author-code` and explicitly reconstructed `paper-spec`, offline preparation,
five-seed generation, sandboxed scoring and complete-result aggregation).

Adapted from the verified server scripts collected on 2026-09-09.
Run one atomic job on slot 0 and fixed-2 on slot 1. Each job creates its own
Ray runtime/cache, rollout port range and Prometheus listener.

From a Git checkout:

```bash
bash experiments/runai/run_single_gpu.sh 0 atomic 5
bash experiments/runai/run_single_gpu.sh 1 fixed 5
```

Limit 5 is a canary, 50 is a pilot; limit 0 requests the configured full 312 updates.
All retain a 312-update scheduler horizon. Atomic/fixed/random are supported.
See [A/B/C contract and oracle guide](../../docs/mp_opd_abc_20260909.md) for seed/span
overrides, CPU preflight, input hashes, baseline algorithm selection and controls.
Microbatch is 4, eager attention, 2 epochs, train batch 64 (16 accumulation steps).
Microbatch 4 is user-requested and has not yet been GPU-validated with MP-OPD;
the prior microbatch-2 canary does not validate its memory use or parity.
Keep runtime libraries outside the checkout:
MP_RUNTIME_DIR defaults to /workspace/storage-shared/nlp/tungks/MP-OPD/work-QA9lLufL.
MP_RUN_ROOT defaults to simct-runs beside the checkout. Model/data defaults
match the original RunAI job; override MP_SHARED_ROOT, MP_STUDENT_PATH,
MP_TEACHER_PATH and MP_DATASET_PATH when paths differ.

The Python wrapper retains CUDA 13, NVRTC include/CPATH and per-job FlashInfer
workspace fixes. The launcher sets PYTHONPATH to this Git checkout and records
source commit and tracked dirty status in launch-config.json.

KDFLOW_ROLLOUT_PORT_BASE and KDFLOW_ROUTER_PROMETHEUS_PORT explicitly isolate
ports. Generic KDFlow callers keep defaults when these variables are absent;
no parsing of CUDA_VISIBLE_DEVICES is forced on multi-GPU or UUID callers.
Use one active run per numeric slot (0..7). Identical slot/port configurations
on the same host are not supported.

The updated MP-OPD temperature/quantile parity checks are retained. The old
server's unconditional finite-outlier bypass is not restored.

Validation: stdlib port regression tests plus Python AST and bash syntax checks.
A GPU canary with the combined Git launcher is still required on the server.
Full server source reconciled against simct-server-source-20260909-041655.tar.gz
(SHA-256 8bda6cea97d65395befa4b13d1e8a29a584db4a105ea9a68cb1740938de62bde).
Atomizer differences are whitespace-only. The newer trainer compatibility fix
and temperature/mean/p99/NaN/Inf parity checks are retained. Historical backup
files are kept in the local evidence archive and not installed as source.
The server's port isolation and launcher/NVRTC changes are integrated here.
No combined GPU training canary has been run after this reconciliation.

Distribution uses the same source commit on GitHub and inside the public HF
Git bundle at https://huggingface.co/codemaivanngu/simct. HF's repository commit
identifies the bundle upload and differs from the source commit recorded in
source-manifest.json. Download bundle and manifest at the same HF revision,
verify SHA-256, then clone into a new directory or pull --ff-only from the bundle.
Keep local model paths/runtime paths in environment variables and outputs
outside the Git checkout. Git may still report conflicts for future source edits;
this integration removes the need to reapply the known server patches manually.


### Separate generation and CPU scoring

`eval_queue.py worker --phase generate` writes durable response journals and a
hash-bound `generation-complete.json` per cell. It never invokes the scorer.
The two GPU workers use `generation-state.json`, independent of scoring failures;
completed legacy metrics are reused. Generation reserves at least 20 GiB of free
disk (`--min-free-gib`) and retains the existing admission/deadline budget.

`eval_queue.py score-spool --plan PLAN --internal-code-execution` uses one
coordinator and at most 16 CPU scorers. It consumes completed generation cells,
reuses saved scores, never launches a model server, and records item IDs plus
subprocess return codes/stdout/stderr on failures. Failed cells are skipped for
that coordinator invocation; other cells continue. Restarting the coordinator
retries unfinished scoring. No infrastructure failure is converted into an
incorrect answer. Existing 120-second timeout semantics remain unchanged.

Scoring starts after each full benchmark/seed cell is generated, not after the
entire checkpoint. Both phases retain the original deadline. A generation job's
completed status does not imply scoring completion; use `summarize` for metrics.
GPU workers release their owned servers at checkpoint transitions and on exit.

The transfer package's `start-separated-eval.sh` validates and forks an existing
25-checkpoint queue, preserves journals and the original clock, then launches two
generation workers and one scoring coordinator. Source the old queue environment
first. It prints the new `queue.env` and log paths. Old queue files remain intact.

The 128-request trial enables up to 128 only for generation phase; combined
workers remain capped at 64. The transfer launcher sets GPU0=128 and GPU1=64.
Migration stops the exact scoring coordinator as well as generation workers,
rebinds generation markers, and preserves completed generation jobs and clocks.
The launcher writes `launched.env` alongside itself for telemetry commands.

The next trial uses GPU0=256 and GPU1=128 generation requests; only generation
accepts up to 256. CPU scoring remains 16 workers with buffer 128. The worker
CLI buffer 256 is validation-only in generation mode: responses spool to disk.
Both phases resume after migration; scorer is briefly stopped for consistent
copying, then restarted alongside generation. Original deadlines remain.

### Append completed evaluation to W&B

`scripts/evaluation/publish_wandb_eval.py --summary FILE --out DIR --publish`
validates the 25-checkpoint aggregate and four exact finished run IDs before
appending eval-only history, a checkpoint table, endpoint summaries and an
evaluation artifact. Existing training history is read back and checked as an
unchanged prefix. Successful runs receive the `evaluated` tag with existing tags
preserved. A source SHA marker prevents re-upload; partial imports fail closed.
No training logs are replayed. Aggregate provenance is not a per-item audit.

`publish_wandb_uncertainty.py --summary FILE --out DIR` appends seed dispersion
to those already-evaluated runs. It uses sample variance (ddof=1), computes macro
per seed before dispersion, and adds spread lines (mean/min/max/mean +/- std)
and seed charts. These are evaluation-seed variability, not confidence intervals.
History-prefix and endpoint readback checks follow each upload.

## Oracle recovery from prepared guide

`recover_oracle.py --original MAIN_WORK --guide GUIDE_WORK --out FRESH_WORK`
uses existing local tokenizers (run through `python-b200-host.sh`) before model
allocation. It preserves every prepared guide row, verifies split identities,
uses explicit 4096 prompt/reference caps with no truncation, and checks context
capacity. It prepares GPU0 primary LR .1 and GPU1 sequential .01/1.0 diagnostics.
Start the returned plan with `python3.12 experiments/runai/campaign.py FRESH_WORK/plan.json`.
The original start/deadline is retained; sensitivity jobs share the remaining
budget, timeouts preserve partial artifacts but are not completed results.
No full training is promoted automatically. CPU endpoint scoring may continue.
