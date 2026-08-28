#!/usr/bin/env python3
"""Create or recover a Vast portal quick tunnel for the private teacher API."""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _load_owner_token(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError("teacher bearer-token file is unavailable")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError("teacher bearer-token file must be owner-only")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("teacher bearer token is missing or too short")
    return token


def _parse_tunnel_url(body: bytes) -> str:
    text = body.decode("utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    if isinstance(payload, str):
        tunnel_url = payload
    elif isinstance(payload, dict):
        tunnel_url = payload.get("tunnelUrl") or payload.get("tunnel_url")
    else:
        tunnel_url = None
    if not isinstance(tunnel_url, str) or not tunnel_url.startswith("https://"):
        raise RuntimeError("Vast tunnel API did not return an HTTPS URL")
    return tunnel_url.rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portal-url", default="http://127.0.0.1:11111"
    )
    parser.add_argument(
        "--target-url", default="http://127.0.0.1:18000"
    )
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-timeout-s", type=float, default=90.0)
    args = parser.parse_args()
    if not args.portal_url.startswith(
        ("http://127.0.0.1:", "http://localhost:")
    ):
        raise RuntimeError("Vast portal URL must be localhost")
    if not args.target_url.startswith(
        ("http://127.0.0.1:", "http://localhost:")
    ):
        raise RuntimeError("tunnel target must be a localhost HTTP service")
    if args.ready_timeout_s <= 0:
        raise RuntimeError("ready timeout must be positive")

    token = _load_owner_token(args.token_file)
    encoded_target = urllib.parse.quote(args.target_url, safe="")
    existing = (
        args.portal_url.rstrip("/")
        + "/get-existing-quick-tunnel/"
        + encoded_target
    )
    status, body = _request(existing)
    created = False
    if status == 404:
        status, body = _request(
            args.portal_url.rstrip("/")
            + "/start-quick-tunnel/"
            + encoded_target,
            method="POST",
            timeout=args.ready_timeout_s,
        )
        created = True
    if status != 200:
        raise RuntimeError(f"Vast tunnel API returned HTTP {status}")
    tunnel_url = _parse_tunnel_url(body)

    deadline = time.monotonic() + args.ready_timeout_s
    last_statuses: tuple[int, int, int] | None = None
    health: dict[str, object] | None = None
    while time.monotonic() < deadline:
        try:
            authenticated, health_body = _request(
                tunnel_url + "/v1/vdt/teacher/health", token=token
            )
            unauthenticated, _ = _request(
                tunnel_url + "/v1/vdt/teacher/health"
            )
            stock, _ = _request(tunnel_url + "/v1/models")
            last_statuses = (authenticated, unauthenticated, stock)
            if last_statuses == (200, 401, 404):
                health = json.loads(health_body)
                break
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(2)
    if health is None:
        raise RuntimeError(
            f"teacher tunnel did not satisfy 200/401/404 gates: {last_statuses}"
        )

    result = {
        "contract_version": 1,
        "created": created,
        "target_url": args.target_url,
        "tunnel_url": tunnel_url,
        "authenticated_health_status": 200,
        "unauthenticated_custom_status": 401,
        "stock_route_status": 404,
        "health": health,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if os.name == "posix":
        temporary.chmod(0o600)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "created": created,
                "tunnel_url": tunnel_url,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
