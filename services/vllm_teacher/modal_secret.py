#!/usr/bin/env python3
"""Create the Modal teacher secret without exposing its bearer value."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path


def ensure_token(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(48) + "\n")
    os.chmod(path, 0o600)
    if len(path.read_text(encoding="utf-8").strip()) < 32:
        raise RuntimeError("stored VDT teacher token is missing or too short")


def create_modal_secret(*, profile: str, secret_name: str, token_file: Path) -> None:
    ensure_token(token_file)
    with tempfile.TemporaryDirectory(prefix="vdt-modal-secret-") as directory:
        temp_dir = Path(directory)
        os.chmod(temp_dir, 0o700)
        secret_json = temp_dir / "teacher.json"
        descriptor = os.open(
            secret_json, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "VDT_TEACHER_API_TOKEN": token_file.read_text(
                        encoding="utf-8"
                    ).strip()
                },
                handle,
            )
        environment = dict(os.environ)
        environment["MODAL_PROFILE"] = profile
        subprocess.run(
            (
                "uvx",
                "--from",
                "modal",
                "modal",
                "secret",
                "create",
                "--profile",
                profile,
                "--from-json",
                str(secret_json),
                "--force",
                secret_name,
            ),
            env=environment,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--secret-name", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    create_modal_secret(
        profile=args.profile,
        secret_name=args.secret_name,
        token_file=args.token_file,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "profile": args.profile,
                "secret_name": args.secret_name,
                "token_file_mode": oct(args.token_file.stat().st_mode & 0o777),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
