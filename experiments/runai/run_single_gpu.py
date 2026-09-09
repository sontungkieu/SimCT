import ast
import json
import os
import sys
from pathlib import Path

mode, limit, output = sys.argv[1:]
assert mode in {"atomic", "fixed"}
limit = int(limit)
assert limit in {0, 5}

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
    mp_opd_max_span_length=1 if mode == "atomic" else 2,
    mp_opd_fixed_span_length=2,
    diagnostic_max_updates=limit,
    save_steps=-1 if limit else 20,
    save_path=str(run_dir / "checkpoint"),
    ckpt_path=str(run_dir / "checkpoints"),
    use_wandb=False,
)

for key in ("student_name_or_path", "teacher_name_or_path", "train_dataset_path"):
    assert Path(opts[key]).exists(), opts[key]

(run_dir / "launch-config.json").write_text(json.dumps({
    "options": opts,
    "source_root": str(root),
    "source_commit": os.environ.get("MP_SOURCE_COMMIT"),
    "source_dirty": os.environ.get("MP_SOURCE_DIRTY"),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "variant": mode,
    "gpu_mapping": "single visible GPU maps SGLang base_gpu_id to zero",
}, indent=2))

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
    assert summary["status"] == "completed", summary
    assert summary["optimizer_updates"] == expected, summary
    print(f"RUN_VERIFIED: {mode}, {expected} updates", flush=True)
finally:
    ray.shutdown()
