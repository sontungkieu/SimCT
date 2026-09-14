# MP-OPD performance diagnostics

Production defaults remain full student, B64/micro4/accumulation16 and
M16/meta-micro4. Do not use synthetic capacity results to increase an existing
campaign's microbatch or resume it under a different accumulation contract.

## Equivalent preprocessing

Span tables retain the same prefix-sum/subtraction operations, executed per
span length rather than per cell. Energy features transfer static metadata in
bulk. Teacher normalization groups equal token counts to preserve scalar-divide
rounding on CUDA. Metric finiteness checks use one host synchronization on the
healthy path and still identify the offending key on failure.

Full-meta releases consumed gradient tuples before subsequent forwards and
optimizer-expression temporaries before hypergradient replay. Existing Adam
moments no longer allocate unused zero-tensor defaults. No update order, LR,
clipping rule, partition rule or trainable parameter set changes.

Measured on B200/Torch 2.11.0+cu130 in Modal app
`ap-wcspuKaPW9RBKpWHJAgunK`, five repetitions after two warmups:

| Operation | Atoms | Original ms | Optimized ms |
|---|---:|---:|---:|
| Energy features | 400 | 41.48 | 0.508 |
| Energy features | 1000 | 103.80 | 1.000 |
| Span tables (L2) | 400 | 37.88 | 0.259 |
| Span tables (L2) | 1000 | 93.66 | 0.255 |

These isolated kernels matched bitwise; feature-to-energy gradients also
matched. These factors must not be reported as end-to-end speedups.

## Short batch trials

1. Try one or two updates per candidate; stop on OOM and release its process.
2. Measure ten updates for feasible candidates; exclude the first two from
   steady-state statistics. Expand to thirty only if throughput is unstable.
3. Compare step wall time, tokens/second, peak allocated/reserved VRAM and
   actual rollout lengths. A fixed 1024-token probe does not qualify 4096 tokens.
4. Re-run pause/resume checks for the chosen microbatch before a new campaign.

`run_single_gpu.py` accepts `MP_MICRO_TRAIN_BATCH_SIZE` (divisor of 64) and
`MP_META_MICRO_BATCH_SIZE` (divisor of 16) for diagnostic limits 1-30 only.
The immutable campaign launcher strips inherited MP flags and retains its
original recipe. Diagnostic options are recorded in the launch manifest.

`experiments/modal/full_meta_capacity_modal.py` reuses the lhtu05 Gemma asset
volume and pinned runtime. It bounds one container to 1500 seconds and stops
on non-OOM errors. `--steps 10 --batch 64 --micros 4` measures synthetic
student/meta computation; it deliberately excludes teacher, on-policy rollout,
MP partition DP and company SFT weights. It is NOT a matched campaign test.
Results use a unique `--run-id` under `remote_artifacts/`.
