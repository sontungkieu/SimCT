# RunAI single-GPU MP-OPD launcher

Adapted from the verified server scripts collected on 2026-09-09.
Run one atomic job on slot 0 and fixed-2 on slot 1. Each job creates its own
Ray runtime/cache, rollout port range and Prometheus listener.

From a Git checkout:

```bash
bash experiments/runai/run_single_gpu.sh 0 atomic 5
bash experiments/runai/run_single_gpu.sh 1 fixed 5
```

Limit 5 is a canary; limit 0 requests the configured full 312 updates.
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
