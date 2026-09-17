"""Rollout-only A/A probe — production RolloutActorGroup path, no teacher, no full-meta.

Environment contract (set by the Modal harness):
  P1_AA_MODE      baseline | deterministic
  P1_AA_ROWS      number of r5 step-1 requests to replay (default 64)
  P1_AA_REPLAYS   same-engine replays (default 2)
  P1_AA_RESTART   1 to add a post-restart replay (default 1)

Evidence produced per generation: output_ids/text hashes, finish_reason,
weight_version, cached_tokens, wall time, plus a prefill-only (max_new_tokens=0)
logprob fingerprint taken after every wakeup so weights/kernel drift and
sampling drift can be separated.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

R5_CONTROL = Path("/runs/p1-hostmask-ab-20260916-r5-control")
OUT_DIR = Path("/runs/p1-parity-20260917")
STUDENT = "/assets/student"


def ids_sha(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:32]


def build_requests(rows: int):
    source = R5_CONTROL / "checkpoint/rollout_data/1.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = records[:rows]
    prompt_ids = [record["prompt_ids"] for record in records]
    reference = [record.get("output_ids") for record in records]
    return prompt_ids, reference


def main() -> int:
    mode = os.environ.get("P1_AA_MODE", "baseline")
    rows = int(os.environ.get("P1_AA_ROWS", "64"))
    replays = int(os.environ.get("P1_AA_REPLAYS", "2"))
    restart = os.environ.get("P1_AA_RESTART", "1") == "1"

    sys.path.insert(0, "/opt/repo")
    sys.path.insert(0, "/opt/repo/experiments/modal/vendor")

    result: dict = {
        "mode": mode, "rows": rows, "replays": replays, "restart": restart,
        "status": "starting", "generations": [], "prefill_checks": [],
        "source": str(R5_CONTROL / "checkpoint/rollout_data/1.jsonl"),
    }

    import ray
    from kdflow.ray.placement_group import create_placement_group
    from kdflow.ray.rollout.rollout_group import RolloutActorGroup
    from kdflow.trajectory import bounded_sampling_params
    from kdflow.training_checkpoint import seeded_sampling

    # Model identity of the arm under test (proves which pair is being served).
    student_config = Path(STUDENT) / "config.json"
    result["student_identity"] = {
        "path": STUDENT,
        "config_sha256": hashlib.sha256(student_config.read_bytes()).hexdigest(),
        "config": json.loads(student_config.read_text()),
        "shards": sorted(
            (p.name, p.stat().st_size) for p in Path(STUDENT).glob("*.safetensors")
        ),
    }
    result["teacher_loaded"] = False

    prompt_ids, reference = build_requests(rows)
    base = {"max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95}
    params = seeded_sampling(
        bounded_sampling_params(prompt_ids, base, 4096), seed=42, step=1, count=len(prompt_ids)
    )
    result["request_params"] = [
        {k: v for k, v in item.items()} for item in params[:3]
    ]
    result["prompt_ids_sha"] = [ids_sha(item) for item in prompt_ids]

    # Use the patched production helper so the A/A exercises the real CLI seam.
    from types import SimpleNamespace

    from kdflow.cli.train_kd_on_policy import build_extra_server_args

    rollout_args = SimpleNamespace(
        rollout_disable_piecewise_cuda_graph=True,
        rollout_deterministic_inference=(mode == "deterministic"),
        rollout_random_seed=(42 if mode == "deterministic" else -1),
    )
    extra_server_args = build_extra_server_args(SimpleNamespace(rollout=rollout_args))
    result["driver_rollout_args"] = vars(rollout_args)
    result["extra_server_args"] = extra_server_args

    # Effective serving contract in the same environment that launches the engine.
    try:
        from sglang.srt.server_args import ServerArgs

        launch = {
            "model_path": STUDENT, "trust_remote_code": True, "host": "127.0.0.1",
            "port": 15000, "nccl_port": 15001, "dist_init_addr": "127.0.0.1:15002",
            "tp_size": 1, "base_gpu_id": 0, "gpu_id_step": 1, "node_rank": 0, "nnodes": 1,
            "enable_memory_saver": True, "enable_weights_cpu_backup": True,
            "skip_server_warmup": True, "log_level": "warning", "log_level_http": "warning",
        }
        launch.update(extra_server_args)
        normalized = ServerArgs(**launch)
        result["server_args_normalized"] = {
            name: str(getattr(normalized, name, "<absent>")) for name in (
                "enable_deterministic_inference", "random_seed", "sampling_backend",
                "attention_backend", "prefill_attention_backend", "decode_attention_backend",
                "chunked_prefill_size", "disable_radix_cache", "cuda_graph_max_bs",
                "disable_cuda_graph", "disable_piecewise_cuda_graph", "dtype",
            )
        }
    except Exception as exc:
        result["server_args_normalized_error"] = repr(exc)

    ray.init(
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        include_dashboard=False,
    )
    pg = create_placement_group(1)

    def make_group():
        return RolloutActorGroup(
            model_path=STUDENT,
            extra_server_args=dict(extra_server_args),
            num_actors=1,
            tp_size=1,
            num_gpus_per_node=1,
            enable_memory_saver=True,
            mem_fraction_static=0.25,
            num_gpus_per_actor=0.3,
            pg=pg,
        )

    def prefill_probe(group, tag: str) -> dict:
        # One generated token so SGLang returns prompt logprobs for the whole
        # prompt; the sampled token itself is ignored. This isolates weights and
        # kernel state from the sampling step.
        probe_params = [{"max_new_tokens": 1, "temperature": 0.6, "top_p": 0.95,
                         "sampling_seed": params[0]["sampling_seed"]}]
        outputs = group.generate([""], probe_params, None, [prompt_ids[0]])
        meta = (outputs[0] or {}).get("meta_info") or {}
        series = meta.get("input_token_logprobs") or []
        values = [row[0] for row in series if isinstance(row, (list, tuple)) and row]
        return {
            "tag": tag,
            "entries": len(values),
            "sha256": ids_sha([f"{v:.9g}" for v in values]),
            "head": [round(float(v), 9) for v in values[:4]],
            "input_token_logprobs_length": meta.get("input_token_logprobs_length"),
            "weight_version": meta.get("weight_version"),
            "cached_tokens": meta.get("cached_tokens"),
        }

    def one_generation(group, tag: str) -> dict:
        started = time.time()
        outputs = group.generate(["" for _ in prompt_ids], params, None, prompt_ids)
        elapsed = time.time() - started
        entries = []
        for index, output in enumerate(outputs):
            if not output:
                entries.append({"row_index": index, "missing": True})
                continue
            meta = output.get("meta_info") or {}
            ids = output.get("output_ids") or []
            entries.append({
                "row_index": index,
                "output_ids": ids,
                "output_ids_sha": ids_sha(ids),
                "output_ids_len": len(ids),
                "finish_reason": meta.get("finish_reason"),
                "weight_version": meta.get("weight_version"),
                "cached_tokens": meta.get("cached_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "e2e_latency": meta.get("e2e_latency"),
            })
        return {"tag": tag, "seconds": elapsed, "entries": entries}

    try:
        group = make_group()
        group.sleep()
        result["status"] = "engine_ready"
        for index in range(replays):
            group.wakeup()
            result["prefill_checks"].append(prefill_probe(group, f"replay{index}"))
            generation = one_generation(group, f"replay{index}")
            result["generations"].append(generation)
            print(f"AA_STAGE replay{index} seconds={generation['seconds']:.2f}", flush=True)
            group.sleep()
        if restart:
            group.shutdown()
            group = make_group()
            group.sleep()
            group.wakeup()
            result["prefill_checks"].append(prefill_probe(group, "restart"))
            generation = one_generation(group, "restart")
            result["generations"].append(generation)
            print(f"AA_STAGE restart seconds={generation['seconds']:.2f}", flush=True)
            group.sleep()
        try:
            group.shutdown()
        except Exception:
            pass
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        simulation = []
        for generation in result["generations"]:
            simulation.append({
                "tag": generation["tag"],
                "seconds": generation.get("seconds"),
                "entries": [
                    {k: v for k, v in entry.items() if k != "output_ids"}
                    for entry in generation.get("entries", [])
                ],
                "output_ids": [entry.get("output_ids") for entry in generation.get("entries", [])],
            })
        result["comparison"] = compare_generations(simulation)
        if reference:
            result["reference_sha"] = [ids_sha(item) if item else None for item in reference]
            result["vs_reference"] = compare_to_reference(simulation, reference)
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f"rollout-aa-{mode}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            print(f"AA_WRITE_WARNING {exc!r}", flush=True)
        try:
            ray.shutdown()
        except Exception:
            pass

    print("ROLLOUT_AA_JSON=" + json.dumps(compact(result), sort_keys=True, default=str), flush=True)
    return 0


def compact(result: dict) -> dict:
    """Drop the bulky per-token arrays from the stdout marker."""
    trimmed = dict(result)
    trimmed.pop("generations", None)
    trimmed["generation_tags"] = [
        {"tag": g["tag"], "seconds": g.get("seconds")} for g in result.get("generations", [])
    ]
    return trimmed


def compare_generations(generations: list) -> dict:
    if len(generations) < 2:
        return {"status": "insufficient_generations"}
    baseline = generations[0]
    comparisons = []
    for other in generations[1:]:
        left, right = baseline.get("output_ids") or [], other.get("output_ids") or []
        shared = min(len(left), len(right))
        identical = 0
        first_divergent_row = None
        first_divergent_token = None
        for index in range(shared):
            a, b = left[index], right[index]
            if a == b:
                identical += 1
                continue
            if first_divergent_row is None:
                first_divergent_row = index
                common = 0
                for x, y in zip(a or [], b or []):
                    if x != y:
                        break
                    common += 1
                first_divergent_token = common
        positions = []
        for index in range(shared):
            a, b = left[index], right[index]
            if a == b:
                continue
            common = 0
            for x, y in zip(a or [], b or []):
                if x != y:
                    break
                common += 1
            positions.append(common)
        comparisons.append({
            "divergent_row_count": len(positions),
            "divergence_token_positions": sorted(set(positions))[:20],
            "baseline_tag": baseline["tag"],
            "tag": other["tag"],
            "rows_compared": shared,
            "identical_rows": identical,
            "first_divergent_row": first_divergent_row,
            "first_divergent_token_position": first_divergent_token,
        })
    return {"status": "compared", "comparisons": comparisons}


def compare_to_reference(generations: list, reference: list) -> dict:
    out = []
    for generation in generations:
        left = generation.get("output_ids") or []
        shared = min(len(left), len(reference))
        identical = 0
        first_divergent_row = None
        first_divergent_token = None
        for index in range(shared):
            a, b = left[index], reference[index]
            if a == b:
                identical += 1
                continue
            if first_divergent_row is None:
                first_divergent_row = index
                common = 0
                for x, y in zip(a or [], b or []):
                    if x != y:
                        break
                    common += 1
                first_divergent_token = common
        out.append({
            "tag": generation["tag"],
            "rows_compared": shared,
            "identical_rows": identical,
            "first_divergent_row": first_divergent_row,
            "first_divergent_token_position": first_divergent_token,
        })
    return {"status": "compared", "comparisons": out}


if __name__ == "__main__":
    raise SystemExit(main())