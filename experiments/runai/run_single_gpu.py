import ast
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

mode, limit, output = sys.argv[1:]
assert mode in {"atomic", "fixed", "random", "soft"}
limit = int(limit)
assert 0 <= limit <= 312

root = Path(__file__).resolve().parents[2]
run_dir = Path(output).resolve()
run_dir.mkdir(parents=True, exist_ok=False)

# Đọc dictionary cấu hình từ runner đã đóng gói, không chạy code Modal.
runner = root / "experiments/modal/mp_opd_phi_gemma_50.py"
tree = ast.parse(runner.read_text())
candidates = [
    n.value
    for n in ast.walk(tree)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "opts" for t in n.targets)
]
assert len(candidates) == 1
call = candidates[0]
assert isinstance(call, ast.Call)
assert isinstance(call.func, ast.Name) and call.func.id == "dict"

opts = {}
for kw in call.keywords:
    assert kw.arg is not None
    if kw.arg.startswith("wandb_"):
        continue
    opts[kw.arg] = ast.literal_eval(kw.value)

shared = Path(os.environ.get("MP_SHARED_ROOT", "/workspace/storage-shared/nlp/tungks/SimCT"))
opts.update(
    num_nodes=1,
    num_gpus_per_node=1,
    student_name_or_path=os.environ.get("MP_STUDENT_PATH", str(
        shared / "runs/qwen-gemma-sft-paper-20260908-045828/checkpoint"
    )),
    teacher_name_or_path=os.environ.get("MP_TEACHER_PATH", "/workspace/storage-shared/models/Qwen2.5-7B-Instruct"),
    train_dataset_path=os.environ.get("MP_DATASET_PATH", str(shared / "data/qwen-author/data/prompts.parquet")),
    num_epochs=2,
    train_batch_size=64,
    micro_train_batch_size=4,
    attn_implementation="eager",
    rollout_num_engines=1,
    rollout_tp_size=1,
    teacher_tp_size=1,
    teacher_dp_size=1,
    mp_opd_mode=mode,
    kd_algorithm=os.environ.get("MP_ALGORITHM", "mp_opd"),
    seed=int(os.environ.get("MP_SEED", "42")),
    lr_scheduler_horizon_steps=312,
    mp_opd_random_seed=int(os.environ.get("MP_PARTITION_SEED", "43")),
    mp_opd_max_span_length=1 if mode == "atomic" else int(os.environ.get("MP_MAX_SPAN_LENGTH", "2")),
    mp_opd_fixed_span_length=int(os.environ.get("MP_FIXED_SPAN_LENGTH", "2")),
    diagnostic_max_updates=limit,
    save_steps=20 if mode == "soft" or not limit else -1,
    save_path=str(run_dir / "checkpoint"),
    ckpt_path=str(run_dir / "checkpoints"),
    use_wandb=False,
)

if mode == "soft":
    energy_path = Path(os.environ['MP_ENERGY_CHECKPOINT'])
    if not energy_path.is_file(): raise ValueError('Missing trained energy checkpoint')
    opts.update(mp_opd_energy_checkpoint=str(energy_path),mp_opd_partition_temperature=1.)

if opts["kd_algorithm"] not in {"mp_opd", "span_ctkd", "xtoken"}:
    raise ValueError("MP_ALGORITHM must be mp_opd, span_ctkd or xtoken")
if opts["kd_algorithm"] != "mp_opd" and mode != "atomic":
    raise ValueError("use atomic as the neutral launcher slot for non-MP algorithms")
if opts["kd_algorithm"] == "xtoken":
    projection = Path(os.environ["MP_XTOKEN_PROJECTION_PATH"])
    expected = os.environ["MP_XTOKEN_PROJECTION_SHA256"]
    actual = hashlib.sha256(projection.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("X-Token projection checksum mismatch")
    opts.update(xtoken_projection_path=str(projection), xtoken_projection_sha256=expected)
if opts["mp_opd_fixed_span_length"] <= 0 or opts["mp_opd_max_span_length"] <= 0:
    raise ValueError("span lengths must be positive")
if mode == "fixed" and opts["mp_opd_fixed_span_length"] > opts["mp_opd_max_span_length"]:
    raise ValueError("fixed length must not exceed MP_MAX_SPAN_LENGTH")

for key in ("student_name_or_path", "teacher_name_or_path", "train_dataset_path"):
    assert Path(opts[key]).exists(), opts[key]

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

model_manifest = {}
for role in ("student", "teacher"):
    path = Path(opts[role + "_name_or_path"])
    print(f"Hashing {role} model files for provenance", flush=True)
    model_manifest[role] = {
        str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
    }
    if not model_manifest[role]:
        raise ValueError("model directory contains no identifiable model files")
versions = {}
for package in ("torch", "transformers", "sglang", "ray"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

(run_dir / "launch-config.json").write_text(json.dumps({
    "options": opts,
    "source_root": str(root),
    "source_commit": os.environ.get("MP_SOURCE_COMMIT"),
    "source_dirty": os.environ.get("MP_SOURCE_DIRTY"),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "variant": mode,
    "energy_sha256": file_hash(energy_path) if mode == "soft" else None,
    "partition_dp_dtype": "float64" if mode == "soft" else None,
    "cooperative_stop_at": os.environ.get("MP_TRAIN_STOP_AT"),
    "checkpoint_reserve_seconds": os.environ.get("MP_CHECKPOINT_RESERVE_SECONDS", "300"),
    "models_sha256": model_manifest,
    "dataset_sha256": file_hash(Path(opts["train_dataset_path"])),
    "runtime_versions": versions,
    "contract": {
        "schema": "mp-runai-v2", "algorithm": opts["kd_algorithm"],
        "execution_updates": limit or 312, "scheduler_horizon": 312,
        "sampling_temperature": opts["temperature"],
        "credit_logprob_temperature": 1.0,
        "parity": {"mean_max": 0.1, "p99_max": 0.5} if opts["kd_algorithm"] == "mp_opd" else None,
        "random_rule": "uniform-next-length; not histogram-matched",
        "evaluation_contract": "external pinned eval manifest required",
    },
    "gpu_mapping": "single visible GPU maps SGLang base_gpu_id to zero",
}, indent=2))

if os.environ.get("MP_PREFLIGHT_ONLY") == "1":
    print(f"PREFLIGHT_READY={run_dir / 'launch-config.json'}")
    raise SystemExit(0)

import ray
ray.init(
    address="local",
    num_gpus=1,
    num_cpus=8,
    include_dashboard=False,
    _temp_dir=os.environ["MP_RAY_TMP"],
)

try:
    import kdflow.cli.train_kd_on_policy as cli

    original = cli.create_placement_group

    def single_gpu_placement(num_gpus):
        assert num_gpus == 1
        pg, indices, gpu_ids = original(num_gpus)
        assert len(gpu_ids) == 1
        print(
            f"SINGLE_GPU_MAPPING: visible={os.environ['CUDA_VISIBLE_DEVICES']}, "
            f"ray_ids={gpu_ids}, sglang_local_id=0",
            flush=True,
        )
        return pg, indices, [0]

    cli.create_placement_group = single_gpu_placement

    sys.argv = ["train_kd_on_policy"]
    for key, value in opts.items():
        sys.argv.extend(["--" + key, str(value)])

    # Diagnostic backend with Gemma attention softcapping.
    args = cli.init_args()
    args.model.attn_implementation = "eager"
    print("MP_CANARY_ATTN_OVERRIDE=eager", flush=True)
    cli.train(args)

    summary = json.loads(
        (run_dir / "checkpoint/run-summary.json").read_text()
    )
    expected = limit or 312
    if summary["status"] == "stopped" and summary.get("stop_reason") == "deadline_checkpoint_reserve":
        updates = summary["optimizer_updates"]
        checkpoint = run_dir / "checkpoint" / f"step{updates}"
        assert updates >= 1 and checkpoint.is_dir(), summary
        print(f"RUN_PARTIAL_SAVED: {mode}, {updates}/{expected} updates; checkpoint={checkpoint}", flush=True)
    else:
        assert summary["status"] == "completed", summary
        assert summary["optimizer_updates"] == expected, summary
        print(f"RUN_VERIFIED: {mode}, {expected} updates", flush=True)
finally:
    ray.shutdown()
