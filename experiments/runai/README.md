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
Microbatch remains 2, eager attention, 2 epochs, train batch 64.
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
Full server source reconciliation is pending the requested archive.
