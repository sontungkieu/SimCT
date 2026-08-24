"""Materialize an explicitly non-paper public training-data substitute.

The SimCT paper's exact 10K corpus and selected teacher trajectories are not
publicly content-addressed.  This module creates a small, immutable math/code
substitute for pipeline validation.  Its provenance file deliberately forbids
labeling the resulting run as a paper-corpus reproduction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from vdt_tunix.training_data import PromptDatasetManifest


PUBLIC_SUBSTITUTE_CONTRACT_VERSION = 1
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
GSM8K_TRAIN_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/"
    f"{GSM8K_REVISION}/main/train-00000-of-00001.parquet"
)
GSM8K_TEST_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/"
    f"{GSM8K_REVISION}/main/test-00000-of-00001.parquet"
)
MBPP_TRAIN_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{MBPP_REVISION}/full/train-00000-of-00001.parquet"
)
MBPP_TEST_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{MBPP_REVISION}/full/test-00000-of-00001.parquet"
)


class PublicSubstituteError(RuntimeError):
    """Raised when the substitute cannot be materialized without ambiguity."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _decode_parquet(content: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PublicSubstituteError("materialization requires pyarrow") from exc
    try:
        rows = pq.read_table(BytesIO(content)).to_pylist()
    except Exception as exc:
        raise PublicSubstituteError(f"could not decode {source}: {exc}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise PublicSubstituteError(f"{source} contains no object records")
    return [dict(row) for row in rows]


def _fetch_bytes(url: str) -> bytes:
    from vdt_tunix.evaluation_data import fetch_bytes

    return fetch_bytes(url)


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicSubstituteError(f"{context} must be a non-empty string")
    return value.strip()


def _gemma_prompt(question: str) -> str:
    return (
        "<start_of_turn>user\n"
        + question
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _qwen_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _rank(seed: int, source: str, source_id: str) -> str:
    return _sha256(f"{seed}\0{source}\0{source_id}".encode("utf-8"))


def _select(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    source: str,
    identity: Callable[[dict[str, Any], int], str],
) -> list[tuple[str, dict[str, Any]]]:
    ranked = [
        (_rank(seed, source, identity(row, index)), identity(row, index), row)
        for index, row in enumerate(rows)
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) < count:
        raise PublicSubstituteError(
            f"{source} contains {len(ranked)} rows, fewer than requested {count}"
        )
    return [(source_id, row) for _, source_id, row in ranked[:count]]


def _manifest(
    *,
    dataset_id: str,
    dataset_revision: str,
    records_path: str,
    records: bytes,
    count: int,
) -> PromptDatasetManifest:
    return PromptDatasetManifest(
        contract_version=1,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split="train",
        records_path=records_path,
        records_sha256=_sha256(records),
        record_count=count,
    )


def materialize_public_substitute(
    output_root: str | Path,
    *,
    per_source: int = 128,
    seed: int = 42,
    fetcher: Callable[[str], bytes] = _fetch_bytes,
    parquet_decoder: Callable[..., list[dict[str, Any]]] = _decode_parquet,
) -> dict[str, Any]:
    if (
        isinstance(per_source, bool)
        or not isinstance(per_source, int)
        or per_source < 1
    ):
        raise PublicSubstituteError("per_source must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PublicSubstituteError("seed must be a non-negative integer")

    gsm_content = fetcher(GSM8K_TRAIN_URL)
    gsm_test_content = fetcher(GSM8K_TEST_URL)
    mbpp_content = fetcher(MBPP_TRAIN_URL)
    mbpp_test_content = fetcher(MBPP_TEST_URL)
    gsm_rows = parquet_decoder(gsm_content, source="gsm8k/main/train")
    gsm_test_rows = parquet_decoder(gsm_test_content, source="gsm8k/main/test")
    mbpp_rows = parquet_decoder(mbpp_content, source="mbpp/full/train")
    mbpp_test_rows = parquet_decoder(mbpp_test_content, source="mbpp/full/test")
    if len(gsm_rows) != 7473:
        raise PublicSubstituteError(
            f"GSM8K train expected 7473 rows, got {len(gsm_rows)}"
        )
    if len(mbpp_rows) != 374:
        raise PublicSubstituteError(
            f"MBPP train expected 374 rows, got {len(mbpp_rows)}"
        )
    if len(gsm_test_rows) != 1319:
        raise PublicSubstituteError(
            f"GSM8K test expected 1319 rows, got {len(gsm_test_rows)}"
        )
    if len(mbpp_test_rows) != 500:
        raise PublicSubstituteError(
            f"MBPP test expected 500 rows, got {len(mbpp_test_rows)}"
        )

    gsm_test_questions = {
        _text(row.get("question"), "GSM8K test question")
        for row in gsm_test_rows
    }
    mbpp_test_prompts = {
        _text(row.get("text"), "MBPP test text") for row in mbpp_test_rows
    }
    mbpp_test_ids = {str(row.get("task_id", "")) for row in mbpp_test_rows}
    gsm_candidates = [
        {**row, "_vdt_source_index": index}
        for index, row in enumerate(gsm_rows)
        if _text(row.get("question"), f"GSM8K train {index}.question")
        not in gsm_test_questions
    ]
    mbpp_candidates = [
        row
        for row in mbpp_rows
        if _text(row.get("text"), "MBPP train text") not in mbpp_test_prompts
        and str(row.get("task_id", "")) not in mbpp_test_ids
    ]

    gsm_selected = _select(
        gsm_candidates,
        count=per_source,
        seed=seed,
        source="gsm8k",
        identity=lambda row, index: (
            f"train/{int(row['_vdt_source_index']):05d}"
        ),
    )
    mbpp_selected = _select(
        mbpp_candidates,
        count=per_source,
        seed=seed,
        source="mbpp",
        identity=lambda row, index: str(row.get("task_id", "")),
    )

    normalized: list[dict[str, str]] = []
    for offset in range(per_source):
        gsm_id, gsm = gsm_selected[offset]
        gsm_question = _text(gsm.get("question"), f"GSM8K {gsm_id}.question")
        normalized.append(
            {
                "prompt_id": f"gsm8k-{gsm_id.replace('/', '-')}",
                "question": gsm_question,
                "target_response": _text(
                    gsm.get("answer"), f"GSM8K {gsm_id}.answer"
                ),
                "source": f"openai/gsm8k@{GSM8K_REVISION}/main/train",
                "source_id": gsm_id,
                "source_license": "MIT",
            }
        )
        mbpp_id, mbpp = mbpp_selected[offset]
        mbpp_question = _text(mbpp.get("text"), f"MBPP {mbpp_id}.text")
        normalized.append(
            {
                "prompt_id": f"mbpp-train-{mbpp_id}",
                "question": mbpp_question,
                "target_response": _text(
                    mbpp.get("code"), f"MBPP {mbpp_id}.code"
                ),
                "source": (
                    "google-research-datasets/mbpp@"
                    f"{MBPP_REVISION}/full/train"
                ),
                "source_id": f"train/{mbpp_id}",
                "source_license": "CC-BY-4.0",
            }
        )

    sft_rows = [
        {
            "prompt_id": item["prompt_id"],
            "student_prompt": _gemma_prompt(item["question"]),
            "teacher_prompt": _qwen_prompt(item["question"]),
            "target_response": item["target_response"],
            "source": item["source"],
            "source_id": item["source_id"],
            "source_license": item["source_license"],
        }
        for item in normalized
    ]
    opd_rows = [
        {
            "prompt_id": item["prompt_id"],
            "student_prompt": _gemma_prompt(item["question"]),
            "teacher_prompt": _qwen_prompt(item["question"]),
        }
        for item in normalized
    ]
    sft_records = _canonical_jsonl(sft_rows)
    opd_records = _canonical_jsonl(opd_rows)
    selection_digest = _sha256(
        "\n".join(item["prompt_id"] for item in normalized).encode("utf-8")
    )
    revision = (
        f"public-substitute-v1-seed{seed}-n{len(normalized)}-"
        f"{selection_digest[:16]}"
    )
    sft_manifest = _manifest(
        dataset_id="vdt/public-substitute-gsm8k-mbpp-sft",
        dataset_revision=revision,
        records_path="records.jsonl",
        records=sft_records,
        count=len(sft_rows),
    )
    opd_manifest = _manifest(
        dataset_id="vdt/public-substitute-gsm8k-mbpp-opd",
        dataset_revision=revision,
        records_path="records.jsonl",
        records=opd_records,
        count=len(opd_rows),
    )
    root = Path(output_root)
    _atomic_write(root / "sft" / "records.jsonl", sft_records)
    _atomic_write(
        root / "sft" / "manifest.json",
        (
            json.dumps(sft_manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    _atomic_write(root / "opd" / "records.jsonl", opd_records)
    _atomic_write(
        root / "opd" / "manifest.json",
        (
            json.dumps(opd_manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    provenance = {
        "contract_version": PUBLIC_SUBSTITUTE_CONTRACT_VERSION,
        "public_data_substitute": True,
        "paper_training_corpus_reproduced": False,
        "paper_selected_teacher_trajectories_reproduced": False,
        "selection": {
            "algorithm": "sha256-rank-interleaved-v1",
            "decontamination": "exact-prompt-and-source-id-v1",
            "excluded_train_rows": {
                "gsm8k": len(gsm_rows) - len(gsm_candidates),
                "mbpp": len(mbpp_rows) - len(mbpp_candidates),
            },
            "seed": seed,
            "per_source": per_source,
            "selection_sha256": selection_digest,
        },
        "sources": [
            {
                "dataset_id": "openai/gsm8k",
                "revision": GSM8K_REVISION,
                "config": "main",
                "split": "train",
                "license": "MIT",
                "url": GSM8K_TRAIN_URL,
                "bytes": len(gsm_content),
                "sha256": _sha256(gsm_content),
                "purpose": "training",
            },
            {
                "dataset_id": "openai/gsm8k",
                "revision": GSM8K_REVISION,
                "config": "main",
                "split": "test",
                "license": "MIT",
                "url": GSM8K_TEST_URL,
                "bytes": len(gsm_test_content),
                "sha256": _sha256(gsm_test_content),
                "purpose": "decontamination_only",
            },
            {
                "dataset_id": "google-research-datasets/mbpp",
                "revision": MBPP_REVISION,
                "config": "full",
                "split": "train",
                "license": "CC-BY-4.0",
                "url": MBPP_TRAIN_URL,
                "bytes": len(mbpp_content),
                "sha256": _sha256(mbpp_content),
                "purpose": "training",
            },
            {
                "dataset_id": "google-research-datasets/mbpp",
                "revision": MBPP_REVISION,
                "config": "full",
                "split": "test",
                "license": "CC-BY-4.0",
                "url": MBPP_TEST_URL,
                "bytes": len(mbpp_test_content),
                "sha256": _sha256(mbpp_test_content),
                "purpose": "decontamination_only",
            },
        ],
        "views": {
            "sft": {
                "manifest_sha256": sft_manifest.digest(),
                "records_sha256": sft_manifest.records_sha256,
                "record_count": sft_manifest.record_count,
            },
            "opd": {
                "manifest_sha256": opd_manifest.digest(),
                "records_sha256": opd_manifest.records_sha256,
                "record_count": opd_manifest.record_count,
            },
        },
        "limitations": [
            "This 50/50 GSM8K-train and MBPP-train subset is not the paper's 10K eight-source corpus.",
            "SFT targets are public reference solutions, not the paper's filtered Qwen teacher trajectories.",
            "Any downstream result is a public-data pipeline screen, not a paper-number reproduction.",
        ],
    }
    _atomic_write(
        root / "provenance.json",
        (
            json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ).encode("utf-8"),
    )
    return provenance
