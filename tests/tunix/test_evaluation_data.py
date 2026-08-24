from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import urllib.error

import pytest

from vdt_tunix.evaluation_data import (
    BenchmarkSource,
    EvaluationDataError,
    PINNED_BENCHMARKS,
    SourceFile,
    fetch_to_path,
    load_verified_materialization,
    materialize_benchmark,
)


def test_lcb_v6_release_count_is_pinned_after_full_materialization():
    source = next(
        item for item in PINNED_BENCHMARKS if item.name == "live-code-bench-v6"
    )
    assert source.expected_count == 1055


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
    assert not list((tmp_path / "gsm8k").glob(".source-*.partial"))


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
    assert not list(root.glob(".source-*.partial"))


def test_fetch_to_path_resumes_with_http_range(tmp_path, monkeypatch):
    destination = tmp_path / "source.partial"
    destination.write_bytes(b"abcd")

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 4-7/8", "Content-Length": "4"}

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        assert request.get_header("Range") == "bytes=4-"
        assert timeout == 180
        return Response(b"efgh")

    monkeypatch.setattr(
        "vdt_tunix.evaluation_data.urllib.request.urlopen", fake_urlopen
    )
    report = fetch_to_path("https://fixture/source", destination)
    assert destination.read_bytes() == b"abcdefgh"
    assert report == {
        "bytes": 8,
        "sha256": hashlib.sha256(b"abcdefgh").hexdigest(),
    }


def test_fetch_to_path_recognizes_complete_file_after_416(
    tmp_path, monkeypatch
):
    destination = tmp_path / "source.partial"
    destination.write_bytes(b"complete")
    calls = []

    class HeadResponse(io.BytesIO):
        status = 200
        headers = {"Content-Length": "8"}

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        calls.append(request.get_method())
        if request.get_method() == "GET":
            raise urllib.error.HTTPError(
                request.full_url, 416, "Range Not Satisfiable", {}, None
            )
        return HeadResponse(b"")

    monkeypatch.setattr(
        "vdt_tunix.evaluation_data.urllib.request.urlopen", fake_urlopen
    )
    report = fetch_to_path("https://fixture/source", destination, attempts=1)
    assert calls == ["GET", "HEAD"]
    assert report == {
        "bytes": 8,
        "sha256": hashlib.sha256(b"complete").hexdigest(),
    }
