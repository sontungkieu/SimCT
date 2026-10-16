# H2 pilot handoff to company compute

This is a prepared recipe, not a submitted company job. No node, GPU UUID,
queue receipt or company resource assignment exists for this experiment yet.
Use only the node/GPU explicitly assigned by Tung and the company queue/lease
workflow. No shared server should be stopped or replaced.

## Frozen pilot

Run `h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1` has100 prompts,
200 native Llama-3.2-3B-Instruct responses and1000 selected positions.
Keep `word-rollouts.json`, `word-prompts.json`, `word-fixtures.json` and
`word-candidates.json` unchanged for a hardware/backend replay. SHA256s are
in the delivery manifest. Preserve the original run as read-only evidence;
use a separate result directory and run ID for company measurements.

Pinned model revisions:

| Role | Repository | Revision |
|---|---|---|
| S | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| B | NousResearch/Meta-Llama-3.1-8B-Instruct | d10aef7999a2b5ba950ab3974312feeedbfe0b77 |
| Q | Qwen/Qwen2-7B-Instruct | f2826a00ceef68f0f2b946d945ecc0477ce4450c |

All B weight shards match official Meta public LFS hashes. Its
`tokenizer_config.json` differs: retain the actual mirror tokenizer/template
used in this pilot. Preserve `assets.json` and on-disk weight checksums.

The source bundle includes flat runtime files matching the Modal overlay,
the current scoring source, and the earlier source used for qualification
and generation. Receipts record source/config hashes; the bundle's source
manifest distinguishes those revisions. Git base is a41dc578 on
vdt/ops/b200-portable; the full task patch is included if signing remains
unavailable. Do not assume that base commit alone contains this experiment.

## Runtime and paths

Reuse the immutable company-compatible image:

`docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f`

It provides Torch2.11/CUDA13.0, Transformers5.6 and Python3.12.12 at
`/opt/venvs/simct-b200/bin/python`. The flat source overlay belongs at
`/opt/h2`. Model/data cache is mounted at `/assets`; result root at `/results`.
Bind the assigned workspace directories to those paths in the company runner;
the worker currently uses these container paths explicitly. Do not install
packages on Tung's laptop or rebuild this image for the pilot.

Download weights on the company node from the exact revisions above into
the `path` specified by each model in `assets.json`. HF credentials belong
in the approved secret mount/environment and must never enter source/results.
Validate the downloaded shard hashes before retaining `verified_weights`.
For an offline replay, qualify/scoring/report need the three model snapshots
and frozen pilot artifacts. Repeating audit/fixtures additionally needs the
historical Gemma/Qwen tokenizers recorded in `historical-assets.json`.

## Recommended order on one assigned B200

1. Read-only node/GPU UUID/runtime check and company lease/queue preflight.
2. New replay directory: copy frozen prompts, responses, fixtures, candidates
   and verified assets; keep the source config byte-for-byte for the replay.
   Do not copy finished score files into the new directory, because scoring
   would regard them as completed checkpoints. Delete nothing from the original.
3. Run `word-qualify` S, B, Q serially on the assigned GPU. It must pass on
   B200; Modal qualification does not transfer automatically. Keep FP32/eager
   and TF32 disabled until a separately qualified faster backend exists.
4. Score S, B, Q serially using the frozen candidates, then run CPU
   `word-report`. Record stage wall time, candidate counts, peak VRAM,
   cache/uncached checks and new receipts. Compare numeric results to Modal.
5. For an8k-position extension, use a separate source/config/result snapshot,
   increase `e3_positions` to8000, rerun fixtures/proposals/merge/scoring and
   requalify against that config hash. Retain the200 responses to isolate
   position sampling; a new generation seed or more prompts is another study.
   Copy the audit aggregate/index and its trace directory as well. The current
   index stores original absolute `/results/<pilot-run>/tokenizer-audit/...`
   paths: mount the original received pilot at that path read-only, or explicitly
   rebase only the index paths in a derived copy and record both manifests.
   Keep token IDs, offsets, text and audit statistics unchanged. Validate every
   indexed path before fixtures; copying just the index is insufficient.
6. For broader evidence, increase prompt count/seeds and add matched teacher
   quality/capacity controls. Keep pilot and confirmatory estimands distinct.

The worker invocation inside an already allocated container is
`/opt/venvs/simct-b200/bin/python -u /opt/h2/h2_word_pilot.py <stage> <arm>`.
Set `H2_RUN_ID` to the new directory name, `PYTHONPATH=/opt/h2`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`TOKENIZERS_PARALLELISM=false`, and `OMP_NUM_THREADS=4`.
This invocation is a runner specification, not a node launch command; an
executable company submission block requires the assigned node/GPU receipt.

## Resource interpretation

One B200 is sufficient for sequential FP32 eager qualification/scoring of
these models; measure actual peak VRAM rather than reserve all available GPUs.
Two or three assigned GPUs can split S/B/Q after each arm qualifies. A fourth
GPU is unnecessary for this three-arm pilot unless running independent seeds.
SGLang generation throughput of approximately7k tokens/s on Gemma2-2B with
concurrency256 does not predict this repeated candidate/KV-cache scoring cost.
Benchmark100 positions on the company runtime before extrapolating to8k.

The pilot measures hybrid exact ASCII atomic probabilities and canonical word
boundary proxies, with explicit OTHER, next-byte and conditional diagnostics.
It is exploratory H2 evidence; it does not certify the original draft's Gate1,
full text marginal invariance, or a causal tokenizer effect.
