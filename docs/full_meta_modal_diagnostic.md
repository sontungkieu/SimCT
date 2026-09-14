# Full-meta FSDP2 diagnostic, 2026-09-14

Modal app `ap-60bn4aIm8chLLBxpTCfiIr` reproduced disconnected optimizer
parameters on NVIDIA A10, Torch 2.11.0+cu130, CUDA 13.0. B200 launch was
rejected by the profile's payment-method requirement. This is a small linear
model autograd test, not company-model training or throughput qualification.

Using the production `full_meta_step`, plain parameters produced a virtual
gradient norm of 0.649764657. FSDP2 parameters returned all `None` from
`autograd.grad(..., allow_unused=True)` and a reported virtual gradient norm
of zero, while ordinary backward populated gradients in both cases.

The production helper now rejects an entirely disconnected inner/outer
gradient. This is a fail-closed guard, not a working FSDP2 hypergradient fix.
An explicit, verified connection between optimizer shards and unsharded
forward parameters is still required before full alternating training.
Exact resume alone cannot establish that the energy gradient is correct.

Reproduce with the existing Modal SDK, using an authorized profile:

```bash
META_DEBUG_GPU=A10G modal run experiments/modal/full_meta_debug_modal.py --run-id unique-diagnostic-id
```

The runner reuses a pinned image, downloads no models, and limits the GPU
function to 180 seconds with no function retries. Results are stored beneath
`remote_artifacts/<run-id>/result.json`. Company queues are not modified.
