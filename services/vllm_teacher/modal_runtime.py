"""Pure runtime helpers shared by the Modal vLLM teacher deployment."""

from __future__ import annotations

import os
import secrets
from collections.abc import MutableMapping
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SERVED_MODEL_NAME = "qwen2.5-7b-instruct"
PROFILE_ID = "gemma2-qwen25-paper-v1"
PROFILE_TEACHER_IDS_SHA256 = (
    "c5fcbde4bc33c4649d5259e25fa701c0a4bb2c23aaaef98373dd0add6db970c0"
)
TOKENIZER_VOCAB_SIZE = 151665
MAX_MODEL_LEN = 8192


def vllm_server_command(
    *,
    host: str = "0.0.0.0",
    port: int = 18000,
    download_dir: str = "/model-cache",
) -> tuple[str, ...]:
    """Return the exact, pinned vLLM command used on Vast and Modal."""

    return (
        "vllm",
        "serve",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--max-logprobs",
        "-1",
        "--download-dir",
        download_dir,
        "--host",
        host,
        "--port",
        str(port),
    )


def materialize_api_token(
    environment: MutableMapping[str, str],
    target: Path,
) -> None:
    """Move the Modal secret from the environment into an owner-only file."""

    token = environment.pop("VDT_TEACHER_API_TOKEN", "").strip()
    if len(token) < 32:
        raise RuntimeError("VDT teacher API token is missing or too short")

    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if target.stat().st_mode & 0o077:
        raise RuntimeError("VDT teacher API token file must be owner-only")


def runtime_environment(base: MutableMapping[str, str], token_file: Path) -> dict[str, str]:
    """Build the subprocess environment without retaining the bearer value."""

    environment = dict(base)
    environment.update(
        {
            "VLLM_PLUGINS": "vdt_teacher",
            "VDT_TEACHER_API_TOKEN_FILE": str(token_file),
            "VDT_TEACHER_MODEL_ID": MODEL_ID,
            "VDT_TEACHER_MODEL_REVISION": MODEL_REVISION,
            "VDT_TEACHER_PROFILE_ID": PROFILE_ID,
            "VDT_TEACHER_PROFILE_TEACHER_IDS_SHA256": PROFILE_TEACHER_IDS_SHA256,
            "VDT_TEACHER_TOKENIZER_VOCAB_SIZE": str(TOKENIZER_VOCAB_SIZE),
            "VDT_TEACHER_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "VDT_TEACHER_MAX_CONCURRENCY": "4",
            "VDT_TEACHER_PRIVATE_ONLY": "1",
            "HF_HOME": "/model-cache/huggingface",
            "VLLM_CACHE_ROOT": "/vllm-cache",
        }
    )
    environment.pop("VDT_TEACHER_API_TOKEN", None)
    return environment
