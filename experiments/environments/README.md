# Two independent native CUDA 12.8 environments

These **checked-in `pyproject.toml` + `uv.lock` pairs** extend the historical
[X-Token harness](../xtoken/README.md). Run them instead of that harness's
CUDA 13 upstream install on a driver-570 RTX 3090 host. The root SimCT
`pyproject.toml` and both algorithms remain unchanged.

The recovered **single-GPU CUDA 13 setup** is preserved separately in the
[historical archive](archive/single-3090-cu130-20260903/README.md), including
original workspace manifests/locks, installed package snapshots and scripts.
It must not be mixed into these CUDA 12.8 projects. Old SimCT was incomplete;
the archive does not invent a successful install or a missing native lockfile.

| Project | Python | PyTorch | Transformers | Scope |
|---|---|---|---|---|
| `xtoken/` | 3.13.15 | 2.11.0+cu128 | 5.12.1 | NeMo dense HF/SDPA, off-policy smoke |
| `simct/` | 3.12.3 | 2.9.1+cu128 | 4.57.1 | KDFlow + SGLang 0.5.9 + FlashAttention 2.8.3 |

Workload uv is pinned to **0.12.7**. Source archives use exact commits and
SHA-256 in the lock. FlashAttention uses the upstream official CPython 3.12,
CUDA 12, Torch 2.9, CXX11-ABI wheel, also hash-locked. No driver changes,
Docker daemon, default-venv changes, cloud rental or credential storage occur.

## Native variant limitations

NeMo's upstream metadata mandates five dependencies that are deliberately
excluded here: `nvidia-nvshmem-cu13`, `nvidia-cudnn-cu13`,
`mooncake-transfer-engine-cuda13`, `tilelang`, and `nixl-cu13`.
Consequently **X-Token `uv pip check` returns nonzero with those five missing
requirements**. This is recorded, not suppressed or called a clean health pass.
Torch supplies its CUDA 12 cuDNN runtime. MoE, NIXL/Mooncake/NVSHMEM data planes,
TE, FP8, context parallelism and optional generation workers are not supported
by this dense smoke environment. Do not use it as a drop-in full NeMo install.

The dense import/CUDA/NCCL canaries have passed; trainer success is a separate
gate. Consult the [current report](reports/2026-09-03-2x3090.md), not a successful
resolver or import message, before claiming training or scientific results.

## Rebuild on an authorized Linux GPU machine

Use a dedicated runtime directory outside the worktree. The example assumes
the branch checkout is `/workspace/SimCT`; replace this with its absolute path.
The actual two-GPU test host has driver 570.181, two 24 GiB RTX 3090 cards over
PCIe (no NVLink), and roughly 171 GiB container RAM. Projection construction
materializes a large CPU tensor: a low-RAM host is not equivalent.
The observed dual envs plus weights occupy about 26 GiB. Prefer >=50 GiB disk;
the tested 32 GiB disk leaves only about 6 GiB, inadequate for optimizer
checkpoints or paper-size output. Check free space before any larger run.

```bash
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken sync --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct sync --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/prepare_source.py --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken check --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct check --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken canary --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct canary --root /workspace/vdt-runtime
```

The expected nonzero X-Token metadata check above requires explicit review;
do not blindly chain through arbitrary dependency failures. Each phase saves
its real exit code, wall time and GPU telemetry in a unique `evidence/` folder.
Independent envs may be synced concurrently; never mutate one env while its
workload runs. `lock` is a deliberate maintainer action, not part of rebuild.

`prepare_source.py` verifies **all** original archive members against an existing
source tree and refuses modification. It does not overwrite/reset the tree.
Runtime bytecode/build extras are not treated as upstream source changes.

## Exact model and smoke setup

`download_models.py --root /workspace/vdt-runtime` receives the authorized HF
token on **stdin**, never argv. Use a protected task-local credential relay;
do not paste a token or commit a relay containing machine/account details.
Only the gated student download receives it. Weights remain outside Git;
the script verifies all shards and records file SHA-256, then exits without
persisting auth. An existing `models-evidence/` is not silently retried.

- Student: Meta Llama-3.2-1B `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Teacher: Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Data: first 32 Formal Logic rows from NVIDIA's public Specialized v1.1
  dataset, Viewer revision `13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`.
  Both revision and frozen Parquet hash are checked; change is a hard failure.

```bash
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py data --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py projection --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py config --root /workspace/vdt-runtime
# Execute only with an explicit authorized training attempt:
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py train --root /workspace/vdt-runtime
```

Smoke follows NVIDIA's functional test: exact Llama-1B/Qwen-1.7B, two GPUs,
BF16, batch 8, microbatch 1, sequence 256, three updates, top-4 tokenizer
projection, original P-KL+CE loss/scheduler. Changes are the pinned 32-row
subset and dense HF/SDPA backend. **This is off-policy, not OPD.** Checkpointing
stays disabled as in the functional test; no quality or persistence claim is
possible from this smoke. Student/teacher revisions are checked before use.

The launcher passes `uv --project` explicitly while keeping the **NeMo source
root as cwd** for upstream prompt files. New Hydra fields use `+`. Registered
Automodel Ray workers reuse the same native venv via the supported
`NEMO_RL_VENV_DIR/<actor-class>/bin/python` lookup, preventing automatic
CUDA 13 worker installation. Model operations are offline after download.

No automatic training retry occurs. A second attempt needs both a distinct
`--train-attempt` number and `--retry-reason`, after diagnosis and authorization.
The runner watches process liveness and enforces a 20-minute training deadline;
it terminates only its own process group. This foreground runner is **not** a
durable service across host failure or lost SSH. Export small diagnostics and
projection before relinquishing a rented machine. Never assume disk persists.

## Local packaging tests (no network/GPU/model download)

```bash
python3 -m unittest discover -s /workspace/SimCT/experiments/environments/tests -v
python3 -m unittest discover -s /workspace/SimCT/experiments/xtoken/tests -v
```
