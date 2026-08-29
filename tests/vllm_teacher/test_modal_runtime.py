from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "services" / "vllm_teacher" / "modal_runtime.py"
SPEC = importlib.util.spec_from_file_location("vdt_modal_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)

SECRET_MODULE_PATH = ROOT / "services" / "vllm_teacher" / "modal_secret.py"
SECRET_SPEC = importlib.util.spec_from_file_location(
    "vdt_modal_secret", SECRET_MODULE_PATH
)
assert SECRET_SPEC is not None and SECRET_SPEC.loader is not None
modal_secret = importlib.util.module_from_spec(SECRET_SPEC)
SECRET_SPEC.loader.exec_module(modal_secret)


def test_vllm_command_preserves_exact_teacher_contract() -> None:
    command = runtime.vllm_server_command()
    assert command[:3] == ("vllm", "serve", "Qwen/Qwen2.5-7B-Instruct")
    assert command[command.index("--revision") + 1] == runtime.MODEL_REVISION
    assert command[command.index("--tokenizer-revision") + 1] == runtime.MODEL_REVISION
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--max-logprobs") + 1] == "-1"
    assert "--enable-prefix-caching" in command


def test_materialize_api_token_is_owner_only_and_removed_from_env(tmp_path: Path) -> None:
    token = "x" * 48
    environment = {"VDT_TEACHER_API_TOKEN": token, "SAFE": "value"}
    target = tmp_path / "secret" / "api_token"
    runtime.materialize_api_token(environment, target)
    assert target.read_text(encoding="utf-8") == token + "\n"
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert "VDT_TEACHER_API_TOKEN" not in environment


def test_materialize_api_token_rejects_short_value(tmp_path: Path) -> None:
    environment = {"VDT_TEACHER_API_TOKEN": "too-short"}
    with pytest.raises(RuntimeError, match="missing or too short"):
        runtime.materialize_api_token(environment, tmp_path / "api_token")


def test_runtime_environment_pins_identity_without_bearer() -> None:
    environment = runtime.runtime_environment(
        {"VDT_TEACHER_API_TOKEN": "x" * 48, "SAFE": "value"},
        Path("/run/secrets/token"),
    )
    assert "VDT_TEACHER_API_TOKEN" not in environment
    assert environment["VDT_TEACHER_MODEL_REVISION"] == runtime.MODEL_REVISION
    assert environment["VDT_TEACHER_PROFILE_TEACHER_IDS_SHA256"] == (
        runtime.PROFILE_TEACHER_IDS_SHA256
    )
    assert environment["VDT_TEACHER_PRIVATE_ONLY"] == "1"


def test_modal_secret_uses_owner_only_json_without_command_line_bearer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_file = tmp_path / "state" / "teacher-token"
    observed: dict[str, object] = {}

    def fake_run(command, *, env, check):
        secret_path = Path(command[command.index("--from-json") + 1])
        observed["command"] = command
        observed["payload"] = json.loads(secret_path.read_text(encoding="utf-8"))
        observed["profile"] = env["MODAL_PROFILE"]
        observed["check"] = check

    monkeypatch.setattr(modal_secret.subprocess, "run", fake_run)
    modal_secret.create_modal_secret(
        profile="fixture-profile",
        secret_name="fixture-secret",
        token_file=token_file,
    )
    token = token_file.read_text(encoding="utf-8").strip()
    assert token_file.stat().st_mode & 0o777 == 0o600
    assert token_file.parent.stat().st_mode & 0o777 == 0o700
    assert observed["payload"] == {"VDT_TEACHER_API_TOKEN": token}
    assert token not in observed["command"]
    assert observed["profile"] == "fixture-profile"
    assert observed["check"] is True
