# Phi to Gemma: 100-update OPD without task-specific SFT

User explicitly skips the SFT stage. Student google/gemma-2-2b-it revision299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8; teacher microsoft/Phi-4-mini-instruct revisioncfbefacb99257ffa30c83adab238a50856ac3083. This is a bounded no-warm-start experiment, not a paper reproduction.

Uses the qualified author-code10000 prompts unchanged; responses from the ongoing offline-generation jobs are unnecessary for OPD. Prompts are exported privately to codemaivanngu/simct-author-code-10k-prompts and pinned by Hub revision and parquet SHA256 before remote download. Model weights are downloaded directly on Modal CPU.

OPD follows Table4: AdamW, LR1e-6, warmup.05, cosine to0, weight decay0, effective batch64, microbatch1, BF16, max sequence4096, rollout max4096, temperature.6/top_p.95, reverse KL. Full two-epoch dataloader determines312 scheduler updates (10000//64*2), warmup16; diagnostic_max_updates100 stops early without shortening the schedule or input corpus. One B200 replaces8H20; original prompts/templates retained, exact token trajectory enabled. Adam betas/clipping/seed and inherited GH filtering follow repository defaults; these details are not fully specified by Table4.

Scoring option span_score_mode=mean_logprob normalizes original vocabulary logits independently at each token position before shared-token gathering and span averaging (Eq7). Default raw_logit remains unchanged for other experiments. New regression checks reference arithmetic, invariance to per-position logit offsets, and finite nonzero gradients; existing SpanCTKD and trajectory tests run remotely.

Runner: SIMCT_STAGE=export on phamvanvuhoan; SIMCT_STAGE=prepare then train on lhtu05, with --stage matching environment. Training runtime cap8400seconds plus cleanup. Existing invocation refuses duplicate training. GPU preemption leaves checkpoints for inspected recovery; it does not blindly restart training. W&B run simct-phi-gemma-nosft-100-r1; final completion requires run-summary showing100 optimizer updates.
