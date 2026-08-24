"""Pinned, content-addressed benchmark materialization for SimCT evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any


EVALUATION_DATA_CONTRACT_VERSION = 1


class EvaluationDataError(RuntimeError):
    """Raised when a benchmark source or materialized record set is unsafe."""


@dataclasses.dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    url: str


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkSource:
    name: str
    dataset_id: str
    dataset_revision: str
    config: str
    split: str
    format: str
    expected_count: int | None
    required_fields: tuple[str, ...]
    files: tuple[SourceFile, ...]

    def __post_init__(self) -> None:
        if self.format not in {"jsonl", "parquet"}:
            raise EvaluationDataError(f"unsupported source format: {self.format}")
        if not self.files:
            raise EvaluationDataError("benchmark source must contain files")
        if self.expected_count is not None and self.expected_count < 1:
            raise EvaluationDataError("expected_count must be positive")


PINNED_BENCHMARKS = (
    BenchmarkSource(
        name="gsm8k",
        dataset_id="openai/gsm8k",
        dataset_revision="740312add88f781978c0658806c59bc2815b9866",
        config="main",
        split="test",
        format="parquet",
        expected_count=1319,
        required_fields=("question", "answer"),
        files=(
            SourceFile(
                path="main/test-00000-of-00001.parquet",
                url=(
                    "https://huggingface.co/datasets/openai/gsm8k/resolve/"
                    "740312add88f781978c0658806c59bc2815b9866/"
                    "main/test-00000-of-00001.parquet"
                ),
            ),
        ),
    ),
    BenchmarkSource(
        name="math500",
        dataset_id="HuggingFaceH4/MATH-500",
        dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        config="default",
        split="test",
        format="jsonl",
        expected_count=500,
        required_fields=("problem", "solution", "answer", "unique_id"),
        files=(
            SourceFile(
                path="test.jsonl",
                url=(
                    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
                    "resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/"
                    "test.jsonl"
                ),
            ),
        ),
    ),
    BenchmarkSource(
        name="mbpp",
        dataset_id="google-research-datasets/mbpp",
        dataset_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        config="full",
        split="test",
        format="parquet",
        expected_count=500,
        required_fields=("task_id", "text", "test_list", "test_setup_code"),
        files=(
            SourceFile(
                path="full/test-00000-of-00001.parquet",
                url=(
                    "https://huggingface.co/datasets/google-research-datasets/"
                    "mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/"
                    "full/test-00000-of-00001.parquet"
                ),
            ),
        ),
    ),
    BenchmarkSource(
        name="live-code-bench-v6",
        dataset_id="livecodebench/code_generation_lite",
        dataset_revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        config="release_v6",
        split="test",
        format="jsonl",
        expected_count=None,
        required_fields=(
            "question_id",
            "question_content",
            "public_test_cases",
            "private_test_cases",
            "metadata",
        ),
        files=tuple(
            SourceFile(
                path=("test.jsonl" if index == 1 else f"test{index}.jsonl"),
                url=(
                    "https://huggingface.co/datasets/livecodebench/"
                    "code_generation_lite/resolve/"
                    "0fe84c3912ea0c4d4a78037083943e8f0c4dd505/"
                    + ("test.jsonl" if index == 1 else f"test{index}.jsonl")
                ),
            )
            for index in range(1, 7)
        ),
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_row(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_row(row) for row in rows)


def decode_jsonl(content: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationDataError(f"{source} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(
                f"{source}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise EvaluationDataError(f"{source}:{line_number} must be an object")
        rows.append(dict(value))
    if not rows:
        raise EvaluationDataError(f"{source} contains no records")
    return rows


def decode_parquet(content: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvaluationDataError(
            "Parquet materialization requires pyarrow"
        ) from exc
    try:
        rows = pq.read_table(BytesIO(content)).to_pylist()
    except Exception as exc:
        raise EvaluationDataError(f"could not decode {source}: {exc}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise EvaluationDataError(f"{source} contains no object records")
    return [dict(row) for row in rows]


def fetch_bytes(url: str, *, timeout_s: int = 180, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SimCT-reproduction/1 evaluation-materializer"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                content = response.read()
            if not content:
                raise EvaluationDataError(f"empty response from {url}")
            return content
        except (OSError, urllib.error.URLError, EvaluationDataError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2**attempt, 8))
    raise EvaluationDataError(
        f"failed to fetch {url} after {attempts} attempts: {last_error}"
    )


def fetch_to_path(
    url: str,
    destination: Path,
    *,
    timeout_s: int = 180,
    attempts: int = 3,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream an immutable source to disk and resume an interrupted transfer."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        existing_bytes = destination.stat().st_size if destination.is_file() else 0
        headers = {
            "User-Agent": "SimCT-reproduction/1 evaluation-materializer"
        }
        if existing_bytes:
            headers["Range"] = f"bytes={existing_bytes}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_range = response.headers.get("Content-Range", "")
                if existing_bytes and status == 206:
                    if not content_range.startswith(f"bytes {existing_bytes}-"):
                        destination.unlink(missing_ok=True)
                        raise EvaluationDataError(
                            f"invalid Content-Range while resuming {url}: "
                            f"{content_range!r}"
                        )
                    digest = hashlib.sha256()
                    with destination.open("rb") as existing:
                        while chunk := existing.read(chunk_size):
                            digest.update(chunk)
                    byte_count = existing_bytes
                    mode = "ab"
                else:
                    digest = hashlib.sha256()
                    byte_count = 0
                    mode = "wb"
                response_bytes = 0
                with destination.open(mode) as output:
                    while chunk := response.read(chunk_size):
                        output.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                        response_bytes += len(chunk)
                content_length = response.headers.get("Content-Length")
                if content_length is not None and response_bytes != int(
                    content_length
                ):
                    raise EvaluationDataError(
                        f"incomplete response from {url}: expected "
                        f"{content_length} bytes, received {response_bytes}"
                    )
            if byte_count == 0:
                raise EvaluationDataError(f"empty response from {url}")
            return {"bytes": byte_count, "sha256": digest.hexdigest()}
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and existing_bytes:
                content_range = exc.headers.get("Content-Range", "")
                expected_suffix = f"*/{existing_bytes}"
                remote_bytes: int | None = None
                if content_range.endswith(expected_suffix):
                    remote_bytes = existing_bytes
                else:
                    head_request = urllib.request.Request(
                        url,
                        method="HEAD",
                        headers={
                            "User-Agent": (
                                "SimCT-reproduction/1 evaluation-materializer"
                            )
                        },
                    )
                    try:
                        with urllib.request.urlopen(
                            head_request, timeout=timeout_s
                        ) as response:
                            size_header = response.headers.get(
                                "X-Linked-Size"
                            ) or response.headers.get("Content-Length")
                            if size_header is not None:
                                remote_bytes = int(size_header)
                    except (OSError, urllib.error.URLError, ValueError):
                        remote_bytes = None
                if remote_bytes == existing_bytes:
                    return {
                        "bytes": existing_bytes,
                        "sha256": sha256_file(destination),
                    }
                destination.unlink(missing_ok=True)
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2**attempt, 8))
        except (OSError, urllib.error.URLError, EvaluationDataError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2**attempt, 8))
    raise EvaluationDataError(
        f"failed to fetch {url} after {attempts} attempts: {last_error}"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _identity_field(source: BenchmarkSource) -> str:
    try:
        return {
            "gsm8k": "question",
            "math500": "unique_id",
            "mbpp": "task_id",
            "live-code-bench-v6": "question_id",
        }[source.name]
    except KeyError as exc:
        raise EvaluationDataError(
            f"no record identity field is defined for {source.name}"
        ) from exc


def _iter_jsonl_file(path: Path, *, source: str) -> Iterator[dict[str, Any]]:
    row_count = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvaluationDataError(
                    f"{source}:{line_number} is not UTF-8"
                ) from exc
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    f"{source}:{line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise EvaluationDataError(
                    f"{source}:{line_number} must be an object"
                )
            row_count += 1
            yield dict(value)
    if row_count == 0:
        raise EvaluationDataError(f"{source} contains no records")


def _manifest_matches_source(
    manifest: Mapping[str, Any], source: BenchmarkSource
) -> bool:
    expected = {
        "contract_version": EVALUATION_DATA_CONTRACT_VERSION,
        "name": source.name,
        "dataset_id": source.dataset_id,
        "dataset_revision": source.dataset_revision,
        "config": source.config,
        "split": source.split,
        "records_path": "records.jsonl",
        "required_fields": list(source.required_fields),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != len(source.files):
        return False
    for actual, pinned in zip(source_files, source.files, strict=True):
        if not isinstance(actual, Mapping):
            return False
        if actual.get("path") != pinned.path or actual.get("url") != pinned.url:
            return False
        if not isinstance(actual.get("bytes"), int) or actual["bytes"] < 1:
            return False
        if not isinstance(actual.get("sha256"), str):
            return False
    count = manifest.get("record_count")
    if not isinstance(count, int) or count < 1:
        return False
    return source.expected_count is None or count == source.expected_count


def load_verified_materialization(
    source: BenchmarkSource, output_root: str | Path
) -> dict[str, Any] | None:
    """Return an existing artifact only when its pinned contract and hash match."""

    root = Path(output_root) / source.name
    records_path = root / "records.jsonl"
    manifest_path = root / "manifest.json"
    if not records_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping) or not _manifest_matches_source(
        manifest, source
    ):
        return None
    if manifest.get("records_sha256") != sha256_file(records_path):
        return None
    return dict(manifest)


def materialize_benchmark(
    source: BenchmarkSource,
    output_root: str | Path,
    *,
    fetcher: Callable[[str], bytes] = fetch_bytes,
    stream_fetcher: Callable[[str, Path], Mapping[str, Any]] | None = None,
    parquet_decoder: Callable[..., list[dict[str, Any]]] = decode_parquet,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    root = Path(output_root) / source.name
    records_path = root / "records.jsonl"
    manifest_path = root / "manifest.json"
    if reuse_existing:
        existing = load_verified_materialization(source, output_root)
        if existing is not None:
            return existing

    root.mkdir(parents=True, exist_ok=True)
    records_temporary = records_path.with_name(records_path.name + ".tmp")
    source_files: list[dict[str, Any]] = []
    required = set(source.required_fields)
    identity_field = _identity_field(source)
    identifiers: set[str] = set()
    records_digest = hashlib.sha256()
    record_count = 0

    def append_row(row: Mapping[str, Any], output: Any) -> None:
        nonlocal record_count
        missing = sorted(required - row.keys())
        if missing:
            raise EvaluationDataError(
                f"{source.name} row {record_count} is missing fields {missing}"
            )
        identifier = str(row[identity_field])
        if identifier in identifiers:
            raise EvaluationDataError(
                f"{source.name} contains duplicate record identity {identifier!r}"
            )
        identifiers.add(identifier)
        encoded = _canonical_row(row)
        output.write(encoded)
        records_digest.update(encoded)
        record_count += 1

    source_temporaries = [
        root / f".source-{index:03d}.partial"
        for index in range(len(source.files))
    ]
    records_temporary.unlink(missing_ok=True)
    try:
        with records_temporary.open("wb") as output:
            for file_index, item in enumerate(source.files):
                if source.format == "jsonl":
                    source_temporary = source_temporaries[file_index]
                    fetched_complete = False
                    try:
                        if stream_fetcher is not None:
                            source_temporary.unlink(missing_ok=True)
                            fetched = dict(stream_fetcher(item.url, source_temporary))
                        elif fetcher is not fetch_bytes:
                            source_temporary.unlink(missing_ok=True)
                            content = fetcher(item.url)
                            source_temporary.write_bytes(content)
                            fetched = {
                                "bytes": len(content),
                                "sha256": sha256_bytes(content),
                            }
                        else:
                            fetched = fetch_to_path(item.url, source_temporary)
                        fetched_complete = True
                        if not source_temporary.is_file():
                            raise EvaluationDataError(
                                f"stream fetcher did not create {source_temporary}"
                            )
                        actual_bytes = source_temporary.stat().st_size
                        actual_sha256 = sha256_file(source_temporary)
                        if fetched.get("bytes") != actual_bytes or fetched.get(
                            "sha256"
                        ) != actual_sha256:
                            raise EvaluationDataError(
                                f"stream fetch metadata mismatch for {item.path}"
                            )
                        source_files.append(
                            {
                                "path": item.path,
                                "url": item.url,
                                "bytes": actual_bytes,
                                "sha256": actual_sha256,
                            }
                        )
                        for row in _iter_jsonl_file(
                            source_temporary, source=item.path
                        ):
                            append_row(row, output)
                    finally:
                        if fetched_complete and (
                            stream_fetcher is not None or fetcher is not fetch_bytes
                        ):
                            source_temporary.unlink(missing_ok=True)
                else:
                    content = fetcher(item.url)
                    source_files.append(
                        {
                            "path": item.path,
                            "url": item.url,
                            "bytes": len(content),
                            "sha256": sha256_bytes(content),
                        }
                    )
                    for row in parquet_decoder(content, source=item.path):
                        append_row(row, output)
        if source.expected_count is not None and record_count != source.expected_count:
            raise EvaluationDataError(
                f"{source.name} expected {source.expected_count} rows, got {record_count}"
            )
        if record_count == 0:
            raise EvaluationDataError(f"{source.name} contains no records")
        manifest = {
            "contract_version": EVALUATION_DATA_CONTRACT_VERSION,
            "name": source.name,
            "dataset_id": source.dataset_id,
            "dataset_revision": source.dataset_revision,
            "config": source.config,
            "split": source.split,
            "records_path": records_path.name,
            "records_sha256": records_digest.hexdigest(),
            "record_count": record_count,
            "required_fields": list(source.required_fields),
            "source_files": source_files,
        }
        records_temporary.replace(records_path)
        _atomic_write(
            manifest_path,
            (
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        for source_temporary in source_temporaries:
            source_temporary.unlink(missing_ok=True)
        return manifest
    except BaseException:
        records_temporary.unlink(missing_ok=True)
        raise


def materialize_all(
    output_root: str | Path, *, names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    selected = set(names) if names is not None else None
    known = {source.name for source in PINNED_BENCHMARKS}
    if selected is not None and not selected <= known:
        raise EvaluationDataError(
            f"unknown benchmark names: {sorted(selected - known)}"
        )
    return [
        materialize_benchmark(source, output_root)
        for source in PINNED_BENCHMARKS
        if selected is None or source.name in selected
    ]
