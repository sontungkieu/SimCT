"""Modal debug loop for sb-main-v2 qualify/bench (official B + S + Q).

Reuses the pilot runtime image, model/results volumes and HF secret.
Models (pinned revs) live in /assets/sb-models/<arm>; per-run SB_ROOT under
/results with a generated assets.json. Run dir layout mirrors the node study
root so drivers run unmodified.
"""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import modal

RUN = os.environ.get("SB_MODAL_RUN", "sb-debug-20261016-r5")
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/sb")
if modal.is_local():
    with open(ROOT / "experiments/diagnostics/h2_pilot_config.json") as f:
        IMAGE = json.load(f)["runtime_image"]
else:
    with open("/opt/sb/sb_runtime_config.json") as f:
        IMAGE = json.load(f)["runtime_image"]

app = modal.App(RUN + "-dbg")
assets = modal.Volume.from_name("sb-opd-assets-v1")
results = modal.Volume.from_name("sb-opd-results-v1")
volumes = {"/assets": assets, "/results": results}
image = modal.Image.from_registry(IMAGE).entrypoint([])

SOURCES = {
    "experiments/diagnostics/sb_qualify.py": "sb_qualify.py",
    "experiments/diagnostics/sb_audit.py": "sb_audit.py",
    "experiments/diagnostics/h2_tokenizer_audit.py": "h2_tokenizer_audit.py",
    "experiments/diagnostics/sb_pool.py": "sb_pool.py",
    "experiments/diagnostics/sb_bench.py": "sb_bench.py",
    "experiments/diagnostics/sb_disposition_qualify.py": "sb_disposition_qualify.py",
    "experiments/diagnostics/h2_measurements.py": "h2_measurements.py",
    "experiments/mp_opd/hf_frozen_sampler.py": "hf_frozen_sampler.py",
    "experiments/diagnostics/artifacts/structural-behavioral-main-runs/main_pool_manifest.json":
        "experiments/diagnostics/artifacts/structural-behavioral-main-runs/main_pool_manifest.json",
    "experiments/diagnostics/h2_pilot_config.json": "sb_runtime_config.json",
}
# Pinned train pools (few MB) ship in-image; copied per-run to assets/datasets.
# Expected hashes are the pilot-pinned file hashes (verified on node 2026-10-16).
SEED_HASHES = {
    "gsm8k-train.parquet":
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    "math-train.parquet":
        "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933",
}
if modal.is_local():
    for local, remote in SOURCES.items():
        image = image.add_local_file(str(ROOT / local), "/opt/sb/" + remote)
    for _ref, _name in (("gsm8k-train-ref.parquet", "gsm8k-train.parquet"),
                        ("math-train-ref.parquet", "math-train.parquet")):
        _p = ROOT.parent / "tmp" / _ref
        image = image.add_local_file(str(_p), "/opt/sb/seed-data/" + _name)
PY = "/opt/venvs/simct-b200/bin/python"

MODELS = {
    "S": ("meta-llama/Llama-3.2-3B-Instruct",
          "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "B": ("meta-llama/Llama-3.1-8B-Instruct",
          "0e9e39f249a16976918f6564b8830bc894c89659"),
    "Q": ("Qwen/Qwen2-7B-Instruct",
          "f2826a00ceef68f0f2b946d945ecc0477ce4450c"),
}


def _run(cmd, env, log, seconds):
    start = time.monotonic()
    position = 0
    with log.open("x") as output:
        proc = subprocess.Popen(cmd, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > seconds:
                    raise TimeoutError("stage cap reached")
                with log.open() as reader:
                    reader.seek(position)
                    chunk = reader.read()
                    position = reader.tell()
                if chunk:
                    print(chunk, flush=True)
                time.sleep(5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return proc.returncode


def _study_env(root: Path):
    env = dict(os.environ, HF_HOME="/assets/hf-cache", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               PYTHONPATH="/opt/sb", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",
               SB_ROOT=str(root), SB_NEW="/opt/sb")
    libs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    libs += [str(p) for p in
             Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env


def _write_assets(root: Path):
    models = {}
    for arm, (repo, rev) in MODELS.items():
        path = Path("/assets/sb-models") / arm
        if not (path / "config.json").is_file():
            raise RuntimeError(f"model {arm} not fetched at {path}")
        models[arm] = {"repo": repo, "revision": rev, "path": str(path),
                       "verified_weights": True}
    (root / "assets.json").write_text(json.dumps({"models": models}, indent=1))
    data = root / "assets" / "datasets"
    data.mkdir(parents=True, exist_ok=True)
    for name, want in SEED_HASHES.items():
        src = Path("/opt/sb/seed-data") / name
        dst = data / name
        if not dst.exists():
            blob = src.read_bytes()
            got = hashlib.sha256(blob).hexdigest()
            if got != want:
                raise RuntimeError(f"seed {name} hash mismatch")
            dst.write_bytes(blob)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2, volumes=volumes,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def fetch(arm: str):
    """Download one pinned model to the shared volume (idempotent)."""
    from huggingface_hub import snapshot_download, get_token
    import huggingface_hub.constants as _hf_const
    # Base image bakes HF_HUB_OFFLINE=1; the flag is ALSO frozen at import
    # time, so popping env alone is not enough.
    _hf_const.HF_HUB_OFFLINE = False
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(v, None)
    token = get_token()
    if not token:
        raise RuntimeError("huggingface-secret has no readable token")
    assets.reload()
    repo, rev = MODELS[arm]
    dest = Path("/assets/sb-models") / arm
    if (dest / "config.json").is_file() and (dest / ".rev").is_file() \
            and (dest / ".rev").read_text().strip() == rev:
        print(f"SB_FETCH cached {arm} {rev}", flush=True)
        return {"arm": arm, "cached": True}
    snapshot_download(repo, revision=rev, local_dir=str(dest), token=token)
    (dest / ".rev").write_text(rev)
    assets.commit()
    print(f"SB_FETCH done {arm} {rev}", flush=True)
    return {"arm": arm, "cached": False}


def _gpu_job(kind: str, arm: str, dtype: str, seconds: int, shard: str = ""):
    assets.reload()
    results.reload()
    root = Path("/results") / RUN
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    _write_assets(root)
    script = {"qualify": "sb_qualify.py", "bench": "sb_bench.py",
              "pool": "sb_pool.py"}[kind]
    log = work / (f"{kind}-{arm}-{dtype}.log" if kind != "pool"
                  else f"pool-{shard}.log")
    if log.exists():
        raise RuntimeError("output exists; use a new run id for repairs")
    env = _study_env(root)
    if kind == "pool":
        cmd = [PY, "-u", "/opt/sb/sb_pool.py", "--shard", shard]
    else:
        cmd = [PY, "-u", f"/opt/sb/{script}", "--arm", arm, "--dtype", dtype]
    rc = _run(cmd, env, log, seconds)
    results.commit()
    tail = "\n".join(log.read_text().splitlines()[-8:])
    print(f"SB_{kind.upper()} arm={arm} dtype={dtype} rc={rc}\n{tail}", flush=True)
    print(f"SB_SUMMARY_JSON={json.dumps({'kind': kind, 'arm': arm, 'dtype': dtype, 'rc': rc})}",
          flush=True)
    if rc != 0:
        raise RuntimeError(f"{kind} {arm} {dtype} failed; log authoritative")
    return {"kind": kind, "arm": arm, "dtype": dtype}


@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def qualify_s(arm: str, dtype: str):
    return _gpu_job("qualify", arm, dtype, 1500)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def qualify_big(arm: str, dtype: str):
    return _gpu_job("qualify", arm, dtype, 1500)


@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def bench_s(arm: str, dtype: str):
    return _gpu_job("bench", arm, dtype, 1500)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def bench_big(arm: str, dtype: str):
    return _gpu_job("bench", arm, dtype, 1500)


@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=3300,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def pool_s(shard: str):
    return _gpu_job("pool", "S", "bfloat16", 3200, shard)


@app.function(image=image, cpu=4, memory=16384, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def audit_cpu(shard: str, pool_run: str):
    """CPU audit of one pool shard (pool file read from another run dir)."""
    import shutil
    assets.reload()
    results.reload()
    root = Path("/results") / RUN
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    _write_assets(root)
    src = Path("/results") / pool_run / "work" / f"pool-{shard}.json"
    dst = work / f"pool-{shard}.json"
    if not dst.exists():
        shutil.copy(src, dst)
    log = work / f"audit-{shard}.log"
    if log.exists():
        raise RuntimeError("output exists; use a new run id for repairs")
    env = _study_env(root)
    rc = _run([PY, "-u", "/opt/sb/sb_audit.py", "--shard", shard],
              env, log, 1500)
    results.commit()
    tail = "\n".join(log.read_text().splitlines()[-5:])
    print(f"SB_AUDIT shard={shard} rc={rc}\n{tail}", flush=True)
    print(f"SB_SUMMARY_JSON={json.dumps({'kind': 'audit', 'shard': shard, 'rc': rc})}",
          flush=True)
    if rc != 0:
        raise RuntimeError(f"audit {shard} failed; log authoritative")
    return {"kind": "audit", "shard": shard}


@app.local_entrypoint()
def main(job: str = "fetch", arm: str = "S", dtype: str = "bfloat16",
         shard: str = "", pool_run: str = "sb-debug-20261016-r3"):
    if job == "fetch":
        if arm not in MODELS:
            raise ValueError(arm)
        print("SB_LOCAL_JSON=" + json.dumps(fetch.remote(arm)))
    elif job == "qualify":
        fn = qualify_s if arm == "S" and dtype == "bfloat16" else qualify_big
        if arm == "S" and dtype == "float32":
            fn = qualify_big  # fp32 S ~13GB: L40S headroom
        print("SB_LOCAL_JSON=" + json.dumps(fn.remote(arm, dtype)))
    elif job == "bench":
        fn = bench_s if (arm == "S" and dtype == "bfloat16") else bench_big
        print("SB_LOCAL_JSON=" + json.dumps(fn.remote(arm, dtype)))
    elif job == "pool":
        if arm != "S" or not shard:
            raise ValueError("pool needs arm=S and --shard NN")
        print("SB_LOCAL_JSON=" + json.dumps(pool_s.remote(shard)))
    elif job == "audit":
        if not shard:
            raise ValueError("audit needs --shard NN")
        print("SB_LOCAL_JSON=" + json.dumps(audit_cpu.remote(shard, pool_run)))
    else:
        raise ValueError(job)
