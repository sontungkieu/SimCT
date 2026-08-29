"""Scale-to-zero Modal deployment for the exact VDT vLLM teacher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

IMAGE_SERVICE_DIR = Path("/opt/vdt_teacher_src")
if IMAGE_SERVICE_DIR.is_dir():
    sys.path.insert(0, str(IMAGE_SERVICE_DIR))

from modal_runtime import (
    materialize_api_token,
    runtime_environment,
    vllm_server_command,
)


APP_NAME = "vdt-qwen25-exact-teacher-no1ceboy"
SECRET_NAME = "vdt-teacher-no1ceboy"
SERVER_PORT = 18000
SERVICE_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ca-certificates", "curl")
    .uv_pip_install("vllm==0.27.1")
    .add_local_dir(SERVICE_DIR, remote_path="/opt/vdt_teacher_src", copy=True)
    .run_commands(
        "python /opt/vdt_teacher_src/patch_vllm.py "
        "--backup-dir /opt/vdt_teacher_provenance/vllm-originals "
        "--manifest /opt/vdt_teacher_provenance/vllm-patch.json",
        "pip install --no-deps --no-build-isolation /opt/vdt_teacher_src",
    )
)

model_cache = modal.Volume.from_name("vdt-qwen25-model-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vdt-vllm-compile-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={"/model-cache": model_cache, "/vllm-cache": vllm_cache},
    port=SERVER_PORT,
    unauthenticated=True,
    routing_region="us-east",
    target_concurrency=1,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    startup_timeout=1200,
    exit_grace_period=30,
)
class ExactTeacherServer:
    @modal.enter()
    def start(self) -> None:
        token_file = Path("/run/secrets/vdt_teacher_api_token")
        materialize_api_token(os.environ, token_file)
        environment = runtime_environment(os.environ, token_file)
        self.process = subprocess.Popen(
            vllm_server_command(port=SERVER_PORT),
            env=environment,
            start_new_session=True,
        )

    @modal.exit()
    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
