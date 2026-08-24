from __future__ import annotations

import dataclasses
import json

import pytest

from vdt_tunix.evaluation_data import (
    BenchmarkSource,
    EvaluationDataError,
    SourceFile,
    load_verified_materialization,
    materialize_benchmark,
)


def _source(expected_count=2):
    return BenchmarkSource(
        name="gsm8k",
        dataset_id="fixture/gsm8k",
        dataset_revision="a" * 40,
        config="main",
        split="test",
        format="jsonl",
        expected_count=expected_count,
        required_fields=("question", "answer"),
        files=(SourceFile(path="test.jsonl", url="https://fixture/test"),),
    )


def test_materialization_is_canonical_and_content_addressed(tmp_path):
    raw = (
        b'{"answer":"#### 4","question":"2+2"}\n'
        b'{"question":"3+3","answer":"#### 6"}\n'
    )
    manifest = materialize_benchmark(
        _source(), tmp_path, fetcher=lambda url: raw
    )
    records = (tmp_path / "gsm8k" / "records.jsonl").read_bytes()
    assert records.startswith(b'{"answer":"#### 4","question":"2+2"}')
    assert manifest["record_count"] == 2
    assert len(manifest["records_sha256"]) == 64
    loaded = json.loads(
        (tmp_path / "gsm8k" / "manifest.json").read_text(encoding="utf-8")
    )
    assert loaded == manifest


def test_materialization_rejects_missing_schema_field(tmp_path):
    raw = b'{"question":"2+2"}\n{"question":"3+3","answer":"6"}\n'
    with pytest.raises(EvaluationDataError, match="missing fields"):
        materialize_benchmark(_source(), tmp_path, fetcher=lambda url: raw)


def test_materialization_rejects_duplicate_identity(tmp_path):
    raw = (
        b'{"question":"same","answer":"4"}\n'
        b'{"question":"same","answer":"4"}\n'
    )
    with pytest.raises(EvaluationDataError, match="duplicate"):
        materialize_benchmark(_source(), tmp_path, fetcher=lambda url: raw)


def test_materialization_rejects_wrong_count(tmp_path):
    raw = b'{"question":"2+2","answer":"4"}\n'
    with pytest.raises(EvaluationDataError, match="expected 2"):
        materialize_benchmark(_source(), tmp_path, fetcher=lambda url: raw)


def test_materialization_streams_multiple_jsonl_sources(tmp_path):
    source = dataclasses.replace(
        _source(),
        files=(
            SourceFile(path="part1.jsonl", url="https://fixture/part1"),
            SourceFile(path="part2.jsonl", url="https://fixture/part2"),
        ),
    )
    payloads = {
        "https://fixture/part1": b'{"question":"q1","answer":"a1"}\n',
        "https://fixture/part2": b'{"answer":"a2","question":"q2"}\n',
    }
    manifest = materialize_benchmark(
        source, tmp_path, fetcher=lambda url: payloads[url]
    )
    assert manifest["record_count"] == 2
    assert len(manifest["source_files"]) == 2
    assert not list((tmp_path / "gsm8k").glob("*.tmp"))
    assert not list((tmp_path / "gsm8k").glob(".source-*.tmp"))


def test_verified_existing_artifact_skips_network(tmp_path):
    raw = (
        b'{"question":"q1","answer":"a1"}\n'
        b'{"question":"q2","answer":"a2"}\n'
    )
    expected = materialize_benchmark(_source(), tmp_path, fetcher=lambda url: raw)

    def unexpected_fetch(url):
        raise AssertionError(f"unexpected network request: {url}")

    actual = materialize_benchmark(
        _source(), tmp_path, fetcher=unexpected_fetch
    )
    assert actual == expected
    assert load_verified_materialization(_source(), tmp_path) == expected


def test_failed_second_shard_does_not_publish_partial_artifact(tmp_path):
    original = (
        b'{"question":"old1","answer":"a1"}\n'
        b'{"question":"old2","answer":"a2"}\n'
    )
    materialize_benchmark(_source(), tmp_path, fetcher=lambda url: original)
    root = tmp_path / "gsm8k"
    old_records = (root / "records.jsonl").read_bytes()
    old_manifest = (root / "manifest.json").read_bytes()
    source = dataclasses.replace(
        _source(),
        files=(
            SourceFile(path="part1.jsonl", url="https://fixture/part1"),
            SourceFile(path="part2.jsonl", url="https://fixture/part2"),
        ),
    )
    payloads = {
        "https://fixture/part1": b'{"question":"new1","answer":"a1"}\n',
        "https://fixture/part2": b'{not-json}\n',
    }
    with pytest.raises(EvaluationDataError, match="invalid JSON"):
        materialize_benchmark(
            source,
            tmp_path,
            fetcher=lambda url: payloads[url],
            reuse_existing=False,
        )
    assert (root / "records.jsonl").read_bytes() == old_records
    assert (root / "manifest.json").read_bytes() == old_manifest
    assert not list(root.glob("*.tmp"))
    assert not list(root.glob(".source-*.tmp"))
