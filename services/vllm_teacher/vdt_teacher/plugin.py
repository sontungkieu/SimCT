from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import struct
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams


CONTRACT_VERSION = 1
MODEL_ID = os.environ.get("VDT_TEACHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MODEL_REVISION = os.environ.get(
    "VDT_TEACHER_MODEL_REVISION",
    "a09a35458c702b33eeacc393d103063234e8bc28",
)
PROFILE_ID = os.environ.get(
    "VDT_TEACHER_PROFILE_ID", "gemma2-qwen25-paper-v1"
)
PROFILE_TEACHER_IDS_SHA256 = os.environ.get(
    "VDT_TEACHER_PROFILE_TEACHER_IDS_SHA256",
    "c5fcbde4bc33c4649d5259e25fa701c0a4bb2c23aaaef98373dd0add6db970c0",
)
TOKEN_PATH = Path(
    os.environ.get(
        "VDT_TEACHER_API_TOKEN_FILE",
        "/workspace/vdt-teacher/secrets/api_token",
    )
)
TOKENIZER_VOCAB_SIZE = int(
    os.environ.get("VDT_TEACHER_TOKENIZER_VOCAB_SIZE", "151665")
)
MAX_MODEL_LEN = int(os.environ.get("VDT_TEACHER_MAX_MODEL_LEN", "8192"))
MAX_CONCURRENCY = int(os.environ.get("VDT_TEACHER_MAX_CONCURRENCY", "4"))
PRIVATE_ONLY = os.environ.get("VDT_TEACHER_PRIVATE_ONLY", "1") != "0"


class TeacherScoreRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1, max_length=MAX_MODEL_LEN - 1)
    completion_token_ids: list[int] = Field(
        min_length=1, max_length=MAX_MODEL_LEN - 1
    )
    profile_id: str
    profile_teacher_ids_sha256: str


class VDTTeacherPlugin:
    """Expose exact low-rank teacher statistics over an authenticated endpoint."""

    name = "vdt_teacher"
    required_tasks = ("generate",)

    def __init__(self) -> None:
        self.engine_client = None
        self._token = ""
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    def attach_router(self, app: FastAPI) -> None:
        if PRIVATE_ONLY:
            @app.middleware("http")
            async def private_routes_only(request: Request, call_next):
                if request.url.path != "/health" and not request.url.path.startswith(
                    "/v1/vdt/teacher/"
                ):
                    return JSONResponse(status_code=404, content={"detail": "not found"})
                return await call_next(request)

        @app.post("/v1/vdt/teacher/hidden-stats")
        async def hidden_stats(
            payload: TeacherScoreRequest,
            authorization: str | None = Header(default=None),
        ):
            return await self._score(payload, authorization)

        @app.get("/v1/vdt/teacher/health")
        async def health(authorization: str | None = Header(default=None)):
            self._authorize(authorization)
            return {
                "ok": self.engine_client is not None,
                "contract_version": CONTRACT_VERSION,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "profile_id": PROFILE_ID,
                "profile_teacher_ids_sha256": PROFILE_TEACHER_IDS_SHA256,
            }

    async def init_state(self, engine_client, state, args) -> None:
        del state, args
        if engine_client is None:
            raise RuntimeError("VDT teacher endpoint requires an engine")
        mode = TOKEN_PATH.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError("VDT teacher API token file must be owner-only")
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise RuntimeError("VDT teacher API token is missing or too short")
        self._token = token
        self.engine_client = engine_client

    def _authorize(self, authorization: str | None) -> None:
        expected = f"Bearer {self._token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    async def _score(
        self,
        payload: TeacherScoreRequest,
        authorization: str | None,
    ) -> Response:
        self._authorize(authorization)
        if payload.profile_id != PROFILE_ID:
            raise HTTPException(status_code=409, detail="profile id mismatch")
        if payload.profile_teacher_ids_sha256 != PROFILE_TEACHER_IDS_SHA256:
            raise HTTPException(status_code=409, detail="profile hash mismatch")
        full_ids = payload.prompt_token_ids + payload.completion_token_ids
        if len(full_ids) > MAX_MODEL_LEN:
            raise HTTPException(status_code=422, detail="sequence exceeds max length")
        if any(value < 0 or value >= TOKENIZER_VOCAB_SIZE for value in full_ids):
            raise HTTPException(
                status_code=422,
                detail="token id outside teacher tokenizer vocabulary",
            )
        if self.engine_client is None:
            raise HTTPException(status_code=503, detail="engine unavailable")

        started = time.perf_counter()
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=-1,
            flat_logprobs=True,
            detokenize=False,
            skip_reading_prefix_cache=True,
        )
        stats = None
        request_id = "vdt-teacher-" + uuid.uuid4().hex
        async with self._semaphore:
            async for output in self.engine_client.generate(
                TokensPrompt(prompt_token_ids=full_ids),
                params,
                request_id,
            ):
                candidate = output.prompt_logprobs
                if isinstance(candidate, dict) and candidate.get(
                    "__vdt_hidden_stats__"
                ):
                    stats = candidate
        if stats is None:
            raise HTTPException(status_code=500, detail="hidden statistics missing")
        if "hidden_states" in stats and "metadata" in stats:
            hidden = stats["hidden_states"]
            metadata = stats["metadata"]
        elif "hidden_chunks" in stats and "metadata_chunks" in stats:
            hidden = torch.cat(stats["hidden_chunks"], dim=0)
            metadata = torch.cat(stats["metadata_chunks"], dim=0)
        else:
            raise HTTPException(status_code=500, detail="hidden statistics malformed")

        start = len(payload.prompt_token_ids) - 1
        stop = start + len(payload.completion_token_ids)
        if stop > hidden.shape[0]:
            raise HTTPException(status_code=500, detail="causal score rows incomplete")
        hidden = hidden[start:stop].contiguous()
        metadata = metadata[start:stop].contiguous()
        selected_ids = metadata[:, 0].to(torch.int32).contiguous()
        expected_ids = torch.tensor(payload.completion_token_ids, dtype=torch.int32)
        if not torch.equal(selected_ids, expected_ids):
            raise HTTPException(status_code=500, detail="causal alignment mismatch")
        log_normalizer = (
            metadata[:, 1].to(torch.int32).contiguous().view(torch.float32)
        )
        selected_log_probs = (
            metadata[:, 2].to(torch.int32).contiguous().view(torch.float32)
        )

        hidden_bytes = hidden.view(torch.uint16).numpy().tobytes(order="C")
        normalizer_bytes = log_normalizer.numpy().tobytes(order="C")
        selected_bytes = selected_log_probs.numpy().tobytes(order="C")
        selected_id_bytes = selected_ids.numpy().tobytes(order="C")
        binary = hidden_bytes + normalizer_bytes + selected_bytes + selected_id_bytes
        header = {
            "contract_version": CONTRACT_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "profile_id": PROFILE_ID,
            "profile_teacher_ids_sha256": PROFILE_TEACHER_IDS_SHA256,
            "prompt_tokens": len(payload.prompt_token_ids),
            "completion_tokens": len(payload.completion_token_ids),
            "hidden_size": int(hidden.shape[1]),
            "elapsed_s": time.perf_counter() - started,
            "input_ids_sha256": hashlib.sha256(
                struct.pack(f"<{len(full_ids)}i", *full_ids)
            ).hexdigest(),
            "payload_sha256": hashlib.sha256(binary).hexdigest(),
            "arrays": [
                {
                    "name": "hidden_states",
                    "dtype": "bfloat16_bits_le",
                    "shape": list(hidden.shape),
                    "offset": 0,
                    "nbytes": len(hidden_bytes),
                },
                {
                    "name": "log_normalizer",
                    "dtype": "float32_le",
                    "shape": [hidden.shape[0]],
                    "offset": len(hidden_bytes),
                    "nbytes": len(normalizer_bytes),
                },
                {
                    "name": "selected_log_probs",
                    "dtype": "float32_le",
                    "shape": [hidden.shape[0]],
                    "offset": len(hidden_bytes) + len(normalizer_bytes),
                    "nbytes": len(selected_bytes),
                },
                {
                    "name": "selected_token_ids",
                    "dtype": "int32_le",
                    "shape": [hidden.shape[0]],
                    "offset": len(hidden_bytes)
                    + len(normalizer_bytes)
                    + len(selected_bytes),
                    "nbytes": len(selected_id_bytes),
                },
            ],
        }
        header_bytes = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        body = struct.pack("<I", len(header_bytes)) + header_bytes + binary
        return Response(
            content=body,
            media_type="application/vnd.vdt.teacher-hidden-stats-v1",
            headers={
                "Cache-Control": "no-store",
                "X-VDT-Payload-SHA256": header["payload_sha256"],
            },
        )


def create_plugin() -> VDTTeacherPlugin:
    return VDTTeacherPlugin()
