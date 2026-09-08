# MP-OPD Phi/Gemma 50-update diagnostic

Baseline: SimCT r3, source f72c8d7, verified complete at 100 updates.
This run starts again from Gemma-2-2b-it, not the baseline output checkpoint.
Teacher, immutable runtime, prompt snapshot, seed42, batch64, microbatch1,
LR1e-6, 2-epoch scheduler312/warmup16, sampling and sequence caps are inherited.
Differences: mp_opd atomic with maximum span1, 50-update limit, saves every10,
and a 3000-second subprocess limit / 3300-second Modal function limit.
The full 50 updates are a target, not guaranteed before the paid time cap.
Profile lhtu05: estimate8.50USD, reserve1USD, authorized total budget30USD.
No parity threshold is relaxed. Atomization/EOS behavior remains unchanged;
atomic differs from SimCT both in scalar loss and potentially boundary/exclusion
semantics, so this is not a loss-only ablation or full learned MP-OPD.
CPU preflight uses cached pinned assets and tests/mp_opd plus baseline trajectory
regressions. It must pass before GPU submission. Outputs are isolated in
mp-opd-phi-gemma-nosft-50-r1. No automatic retries after failure/preemption.
