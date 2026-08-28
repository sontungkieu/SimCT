# Remote exact vLLM teacher

This service keeps the trainable Gemma student and its rollout on TPU while a
single pinned Qwen teacher performs teacher-forcing on a remote NVIDIA GPU. It
does not change the paper config, sampling parameters, tokenizer identities,
temperature, reverse-KL objective, or effective batch size.

## Exact representation

The server returns, for every completion position:

- the native BF16 final hidden state consumed by Qwen's LM head;
- the FP32 full-vocabulary log-normalizer;
- the FP32 realized-token log probability; and
- the realized teacher token ID.

The TPU holds only the immutable overlap rows of `lm_head.weight` and
reconstructs `shared_log_probs = hidden @ overlap_head.T - log_normalizer`.
This is a low-rank factorization of the same teacher sufficient statistics,
not top-k distillation, quantization, truncation, or a changed loss.

The current `gemma2-qwen25-paper-v1` profile has 64,272 overlap rows and a
460,701,696-byte BF16 head artifact. Binary identity and causal alignment are
checked on every response.

## Security boundary

- vLLM binds only to `127.0.0.1`.
- `VDT_TEACHER_PRIVATE_ONLY=1` hides every vLLM route except the custom teacher
  routes and the basic local health route.
- Both custom routes require a bearer token read from an owner-only file.
- Expose the service only through a TLS tunnel/proxy; the TPU client rejects
  plain HTTP except an explicitly enabled localhost canary.
- Never put the bearer value in a config, command line, notebook, log, Git, or
  profile manifest.

## Server installation

The implementation is pinned to vLLM 0.27.1. In the vLLM environment:

```bash
python services/vllm_teacher/patch_vllm.py \
  --backup-dir /workspace/vdt-teacher/provenance/vllm-originals \
  --manifest /workspace/vdt-teacher/provenance/vllm-patch.json
pip install --no-deps --no-build-isolation services/vllm_teacher
chmod 600 /workspace/vdt-teacher/secrets/api_token
services/vllm_teacher/run_server.sh
```

`prompt_logprobs=-1` is reserved by this dedicated server for the hidden-state
IPC channel. Non-negative upstream prompt-logprob requests remain unchanged.

On the Vast vLLM image, create or recover its managed TLS quick tunnel only
after the private-route checks pass:

```bash
python services/vllm_teacher/vast_quick_tunnel.py \
  --token-file /workspace/vdt-teacher/secrets/api_token \
  --output /workspace/vdt-teacher/provenance/tls-tunnel.json
```

The quick-tunnel URL is operational and can change after a container/tunnel
restart. Keep the resulting JSON and bearer token outside Git; rerun the tool
to recover the current URL rather than hard-coding it in a paper config.

## TPU runtime activation

Keep the scientific JSON unchanged. Supply operational state through files and
environment variables:

```bash
export VDT_REMOTE_TEACHER_URL=https://teacher.example.invalid
export VDT_REMOTE_TEACHER_TOKEN_FILE=/run/secrets/vdt_teacher_token
export VDT_REMOTE_TEACHER_PROFILE_DIR=/kaggle/input/vdt-teacher-profile
export VDT_REMOTE_TEACHER_TOKENIZER_DIR=/kaggle/input/vdt-teacher-profile/tokenizer
export VDT_REMOTE_TEACHER_MAX_PARALLEL=4
export VDT_REMOTE_TEACHER_MAX_ATTEMPTS=3
export VDT_REMOTE_TEACHER_RETRY_BACKOFF_S=2
```

The client retries only pure teacher health/scoring requests after transport
errors and retryable gateway/rate-limit HTTP statuses. It uses bounded
exponential backoff, reports the safe HTTP status on terminal failure, and
emits per-step request-attempt and retry counts. Identity, payload hashes, and
causal alignment are still validated after every successful response.

The profile directory must contain the exact manifest, `teacher_ids.i32le`,
`teacher_overlap_lm_head.bf16le`, and the pinned Qwen tokenizer files. The
remote backend is selected only when `VDT_REMOTE_TEACHER_URL` is present;
otherwise the existing local frozen-teacher path is unchanged.

## Measured deployment canaries

On the initial RTX 3090 deployment, the authenticated contract passed:

- unauthorized health and score requests: HTTP 401;
- selected-logprob parity against stock vLLM: max absolute difference
  `2.18e-6` on the short canary;
- static overlap-head reconstruction: max absolute logprob difference `0.0187`
  in native BF16 arithmetic;
- paper4k shape (256 prompt + 3,840 completion): 1.145 s end-to-end locally,
  3,354 scored tokens/s, 27,571,200-byte payload, peak sampled VRAM 21,934 MiB.
- two concurrent paper4k requests after deploying the branch build: 2.569 s
  wall time, 2,990 aggregate scored
  tokens/s, two independently hash-validated 27,571,200-byte payloads, peak
  sampled VRAM 21,406 MiB;
- an external TLS canary preserved authenticated health, HTTP 401 without the
  bearer, HTTP 404 for stock vLLM routes, and the exact binary score contract;
  and
- the TPU-side projection kernel matches a native BF16 JAX reference under
  `jax.jit` with JAX/JAXLIB 0.10.2 on CPU.

These are service and CPU-JAX canaries, not end-to-end TPU training speedups.
TLS/network, execution of the projection on TPU, and a full optimizer update
remain separate gates before claiming training acceleration.
