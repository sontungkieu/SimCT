from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "services/vllm_teacher/vast_quick_tunnel.py"
)
SPEC = importlib.util.spec_from_file_location("vdt_vast_quick_tunnel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "body",
    (
        b"https://example.trycloudflare.com",
        b'{"tunnelUrl":"https://example.trycloudflare.com"}',
        b'{"tunnel_url":"https://example.trycloudflare.com/"}',
    ),
)
def test_parse_tunnel_url_supports_vast_existing_and_start_responses(body):
    assert MODULE._parse_tunnel_url(body) == "https://example.trycloudflare.com"


def test_parse_tunnel_url_rejects_plain_http():
    with pytest.raises(RuntimeError, match="HTTPS"):
        MODULE._parse_tunnel_url(b"http://example.invalid")
