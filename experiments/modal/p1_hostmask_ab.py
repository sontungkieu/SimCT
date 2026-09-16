
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import traceback
import time
from pathlib import Path

import modal

LOCAL_ROOT = Path("/home/tung/simct-b200-portable")
REMOTE_ROOT = Path("/opt/repo")
IMAGE_REF = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME = "simct-qwen7b-gemma2-assets-20260916"
RUN_VOLUME = "simct-qwen7b-gemma2-runs-20260916-main"
APP_NAME = "simct-p1-hostmask-ab-20260916"
RUN_TAG = "p1-hostmask-ab-20260916-r5"


image = (
    modal.Image.from_registry(IMAGE_REF)
    .entrypoint([])
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/runai"), "/opt/repo/experiments/runai", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/mp_opd_phi_gemma_50.py"),
                    "/opt/repo/experiments/modal/mp_opd_phi_gemma_50.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/provenance_prepare.py"),
                    "/opt/repo/experiments/modal/provenance_prepare.py", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/modal/vendor"),
                   "/opt/repo/experiments/modal/vendor", copy=True)
)
app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=False)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=False)
prepvol = modal.Volume.from_name("simct-p1-startup-prep-20260916", create_if_missing=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_logged(cmd, *, cwd, env, log, timeout, stage):
    """Drain child output while retaining bounded PID/I/O progress evidence."""
    started = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    (Path(log.name).parent / f"{stage}.pid").write_text(str(p.pid) + "\n")
    while True:
        rc = p.poll()
        if rc is not None:
            log.write(f"STAGE_END {stage} pid={p.pid} rc={rc} seconds={time.time()-started:.3f}\n")
            log.flush()
            return rc
        elapsed = time.time() - started
        if elapsed > timeout:
            log.write(f"STAGE_TIMEOUT {stage} pid={p.pid} seconds={elapsed:.3f}\n")
            try:
                status = Path(f"/proc/{p.pid}/status").read_text()
                io = Path(f"/proc/{p.pid}/io").read_text()
                log.write("PROC_STATUS_BEGIN\n" + status + "PROC_STATUS_END\n")
                log.write("PROC_IO_BEGIN\n" + io + "PROC_IO_END\n")
            except OSError as exc:
                log.write(f"PROC_EVIDENCE_ERROR {exc!r}\n")
            log.flush()
            p.terminate()
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            return 124
        log.write(f"STAGE_PROGRESS {stage} pid={p.pid} seconds={elapsed:.3f}\n")
        log.flush()
        time.sleep(5)


def environment(arm: str, host_mask: bool, run_root: str, commit: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update({
        "PATH": "/opt/venvs/simct-b200/bin:/usr/local/cuda/bin:" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": "/opt/repo/experiments/modal/vendor:/opt/repo",
        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "RAY_USAGE_STATS_ENABLED": "0", "NCCL_CUMEM_HOST_ENABLE": "0", "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": "/usr/local/cuda-13.0",
        "LD_LIBRARY_PATH": "/tmp/runtime-host-libs:/opt/simct-portable-libs:/usr/local/cuda-13.0/lib64",
        "MP_RUNTIME_DIR": "/tmp/runtime", "MP_RAY_TMP": "/tmp/ray",
        "XDG_CACHE_HOME": "/tmp/cache", "TRITON_CACHE_DIR": "/tmp/cache/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/cache/torch", "FLASHINFER_WORKSPACE_BASE": "/tmp/cache",
        "KDFLOW_ROLLOUT_PORT_BASE": "15000", "KDFLOW_ROUTER_PORT_BASE": "16000",
        "KDFLOW_ROUTER_PROMETHEUS_PORT": "20000",
        "MP_SHARED_ROOT": "/assets", "MP_STUDENT_PATH": "/assets/student",
        "MP_TEACHER_PATH": "/assets/teacher", "MP_DATASET_PATH": "/assets/prompts.parquet",
        "MP_META_PATH": "/assets/selected.parquet", "MP_ENERGY_CHECKPOINT": "/assets/energy-select-4.pt",
        "MP_ALGORITHM": "mp_opd", "MP_ALTERNATING": "1", "MP_OFFLOAD_ADAM_MOMENTS": "1",
        "MP_MICRO_TRAIN_BATCH_SIZE": "1", "MP_META_MICRO_BATCH_SIZE": "4",
        "MP_SEED": "42", "MP_PARTITION_SEED": "43", "MP_MAX_SPAN_LENGTH": "2",
        "MP_FIXED_SPAN_LENGTH": "2", "MP_ATTN_IMPLEMENTATION": "eager",
        "MP_CHECKPOINT_STEPS": "1,2", "MP_SOURCE_COMMIT": commit, "MP_SOURCE_DIRTY": "",
        "MP_QUALIFICATION_POLICY": "p1-matched-ab", "MP_QUALIFICATION_STATUS": "diagnostic",
        "MP_ENERGY_LR": "0.001", "MP_ENERGY_EVERY": "1", "MP_RUN_ROOT": run_root,
        "MP_OPD_HOST_MASK": "1" if host_mask else "0", "MP_OPD_TIMING": "1",
        "MP_PREPARED_RECEIPT": "/prep/startup-provenance.json",
        "MP_PREPARED_RECEIPT_SHA256": os.environ.get("MP_PREPARED_RECEIPT_SHA256", ""),
        "MP_SNAPSHOT_ID": "assets-20260916",
        "WANDB_MODE": "offline", "WANDB_DISABLED": "true",
    })
    return env


@app.function(
    image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
    timeout=3600, retries=0, max_containers=1, single_use_containers=True,
    volumes={"/assets": assets, "/runs": runs, "/prep": prepvol},
)
def run_arm(arm: str, host_mask: bool, commit: str, receipt_sha256: str, *, continuous: bool = True) -> dict[str, object]:
    run_root = "/runs"
    run_dir = Path(run_root) / f"{RUN_TAG}-{arm}"
    result: dict[str, object] = {
        "run_id": f"simct-p1-{arm}-2update-20260916",
        "arm": arm, "host_mask": host_mask, "source_commit": commit,
        "image": IMAGE_REF, "status": "starting",
        "contract": {
            "student": "google/gemma-2-2b-it", "teacher": "Qwen/Qwen2.5-7B-Instruct",
            "updates": 2, "scheduler_horizon": 312, "batch": 64, "meta_batch": 16,
            "micro": 1, "meta_micro": 4, "max_len": 4096, "attn": "eager",
            "full_meta": True, "offload_adam_moments": True, "energy_every": 1,
            "seed": 42, "partition_seed": 43,
        },
    }
    try:
        for d in ("/tmp/runtime/runtime-host-libs", "/tmp/ray", "/tmp/cache/triton", "/tmp/cache/torch"):
            Path(d).mkdir(parents=True, exist_ok=True)
        cmd = ["bash", "/opt/repo/experiments/runai/python-b200-host.sh",
               "/opt/repo/experiments/runai/run_single_gpu.py", "soft", "2", str(run_dir)]
        env = environment(arm, host_mask, run_root, commit)
        env["MP_PREPARED_RECEIPT_SHA256"] = receipt_sha256
        env["MP_PAUSE_AFTER_UPDATES"] = "0" if continuous else "1"
        p1log = run_dir.parent / f"{arm}.phase1.log"
        with p1log.open("w") as out:
            out.write(f"STAGE_START child_spawn pid=pending continuous={continuous}\n"); out.flush()
            p1rc = run_logged(cmd, cwd=str(REMOTE_ROOT), env=env, log=out, timeout=1500, stage="continuous" if continuous else "resume_phase1")
        result["phase1_exit"] = p1rc
        summary = run_dir / "checkpoint" / "run-summary.json"
        result["phase1_summary_present"] = summary.is_file()
        if p1rc != 0 or not summary.is_file():
            raise RuntimeError(f"phase1 failed rc={p1rc} summary={summary.is_file()}")
        result["phase1_summary"] = json.loads(summary.read_text())
        if continuous:
            final = result["phase1_summary"]
            if final.get("status") != "completed" or final.get("optimizer_updates") != 2:
                raise RuntimeError(f"continuous arm incomplete: {final}")
            result["actual_updates"] = final.get("optimizer_updates")
            result["cleanup"] = {"run_process_exit": True, "run_dir": str(run_dir)}
            result["status"] = "completed"
            result["timing_logs"] = {"continuous": str(p1log)}
            return result
        env["MP_RESUME"] = "1"
        env["MP_PAUSE_AFTER_UPDATES"] = "0"
        p2log = run_dir.parent / f"{arm}.phase2.log"
        with p2log.open("w") as out:
            p2rc = run_logged(cmd, cwd=str(REMOTE_ROOT), env=env, log=out, timeout=1500, stage="resume_phase2")
        result["phase2_exit"] = p2rc
        if p2rc != 0 or not summary.is_file():
            raise RuntimeError(f"phase2 failed rc={p2rc}")
        result["final_summary"] = json.loads(summary.read_text())
        result["status"] = "completed"
        result["timing_logs"] = {"phase1": str(p1log), "phase2": str(p2log)}
    except BaseException as ex:
        result.update(status="stopped", error_type=type(ex).__name__,
                      error=str(ex), traceback="".join(traceback.format_exception_only(type(ex), ex)).strip())
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(run_root) / f"{RUN_TAG}-{arm}.result.json", result)
        runs.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/runs": runs})
def resume_check_remote() -> dict[str, object]:
    """CPU-only transactional validation of the pinned control step1."""
    import json
    import sys
    sys.path.insert(0, "/opt/repo")
    from kdflow.training_checkpoint import inspect
    root = Path("/runs/p1-hostmask-ab-20260916-r3-control/checkpoints")
    latest = json.loads((root / "latest.json").read_text())
    manifest_path = root / latest["directory"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    print(f"RESUME_STAGE_START checkpoint_contract_compare pid={os.getpid()}", flush=True)
    checked, value = inspect(root, manifest["contract"], manifest["world_size"])
    print(f"RESUME_STAGE_END checkpoint_payload_verify pid={os.getpid()} step={value['step']}", flush=True)
    return {"status": "validated", "step": value["step"], "world_size": value["world_size"],
            "manifest_sha256": sha256(manifest_path), "directory": str(checked)}


@app.local_entrypoint()
def main() -> None:
    commit = subprocess.check_output(["git", "-C", str(LOCAL_ROOT), "rev-parse", "HEAD"], text=True).strip()
    gate = os.environ.get("P1_GATE", "continuous_ab")
    if gate == "resume_check":
        result = resume_check_remote.remote()
        print("RESUME_CHECK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return
    prep = json.loads(Path("/mnt/d/dev/codex/research_vdt/remote_artifacts/p1-startup-prep-20260916/prep.receipt.json").read_text())
    if prep.get("status") != "ready" or prep.get("source_commit") != commit:
        raise SystemExit("P1_PREP_NOT_READY_OR_SOURCE_MISMATCH")
    receipt_sha256 = prep["receipt_sha256"]
    if gate != "continuous_ab":
        raise SystemExit("unknown P1_GATE=" + gate)
    for arm, host_mask in (("control", False), ("candidate", True)):
        print("START_ARM=" + arm, flush=True)
        result = run_arm.remote(arm, host_mask, commit, receipt_sha256, continuous=True)
        print("P1_ARM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        if result.get("status") != "completed":
            raise SystemExit(1)
