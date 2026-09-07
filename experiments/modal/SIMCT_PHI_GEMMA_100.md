# Phi to Gemma: 100-update OPD without task-specific SFT

User explicitly skips the SFT stage. Student google/gemma-2-2b-it revision299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8; teacher microsoft/Phi-4-mini-instruct revisioncfbefacb99257ffa30c83adab238a50856ac3083. This is a bounded no-warm-start experiment, not a paper reproduction.

Uses the qualified author-code10000 prompts unchanged; responses from the ongoing offline-generation jobs are unnecessary for OPD. Prompts are exported privately to codemaivanngu/simct-author-code-10k-prompts and pinned by Hub revision and parquet SHA256 before remote download. Model weights are downloaded directly on Modal CPU.

OPD follows Table4: AdamW, LR1e-6, warmup.05, cosine to0, weight decay0, effective batch64, microbatch1, BF16, max sequence4096, rollout max4096, temperature.6/top_p.95, reverse KL. Full two-epoch dataloader determines312 scheduler updates (10000//64*2), warmup16; diagnostic_max_updates100 stops early without shortening the schedule or input corpus. One B200 replaces8H20; original prompts/templates retained, exact token trajectory enabled. Adam betas/clipping/seed and inherited GH filtering follow repository defaults; these details are not fully specified by Table4.

Scoring option span_score_mode=mean_logprob normalizes original vocabulary logits independently at each token position before shared-token gathering and span averaging (Eq7). Default raw_logit remains unchanged for other experiments. New regression checks reference arithmetic, invariance to per-position logit offsets, and finite nonzero gradients; existing SpanCTKD and trajectory tests run remotely.

Runner: SIMCT_STAGE=export on phamvanvuhoan; SIMCT_STAGE=prepare then train on lhtu05, with --stage matching environment. Training runtime cap8400seconds plus cleanup. Existing invocation refuses duplicate training. GPU preemption leaves checkpoints for inspected recovery; it does not blindly restart training. W&B run simct-phi-gemma-nosft-100-r1; final completion requires run-summary showing100 optimizer updates.

Additional preflight findings: rendered chat templates own BOS, so exact prompt encoding uses add_special_tokens=False when templates are enabled. Source max_len was metadata-only for OPD; this run explicitly bounds each response to4096 minus its prompt length and one terminal sentinel. No prompt content is truncated or filtered. Audited Gemma prompt maximum1635 tokens; all10000 fit. Teacher retokenization retains its16384-token operational context.

## Deployment evidence

18 CPU tests passed in44.31seconds after the final scoring/BOS/sequence-cap changes. GPU execution source9a9efe5. App https://modal.com/apps/lhtu05/main/ap-bFZCziRy2ZTBqBnC9uo9HX ; FunctionCall fc-01M1YDSNWHAWGBGSF6QQ57R09X ; B200 container ta-01M1YDSP8JJA6PFYVHMN28KPQR. Submitted2026-09-07T17:14UTC (2026-09-08 00:14 local). Volume simct-phi-gemma-nosft-100-r1 stores invocation.json, train.log, result.json, checkpoint/, checkpoints/. Local submit log /home/tung/simct-data-evidence/opd-train-r1.log.

Prompt Hub revision681e17e797cc0ceef58c039510ec1a0f827200a8, parquetSHA256 cf9a13f4e0e7a4a9578325994414ad92148e8c95f8608a0d609218641f7921ae. lhtu05 guard before launch: reported1.23716677USD, estimate21USD, reserve1USD, projected23.23716677USD against28.5USD local hard limit (30USD user budget), OK. Running is not completion; require the final summary and checkpoint before reporting success.

R1 stopped before trainer initialization/update1: SGLang experimental piecewise CUDA graph capture hit cudaErrorIllegalAddress at startup. No training occurred. R2 only disables piecewise CUDA graphs on the rollout engine as recommended in the captured SGLang error; all scientific settings remain unchanged. R1 logs/result retained in its original volume, r2 uses a separate run/volume.
