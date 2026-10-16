# Plan siêu chi tiết: Structural vs. Behavioral Mismatch (sb-main-v2)

Ngày freeze: 2026-10-16/17. Nguồn contract: `structural_behavioral_plan_contract.json`,
`RUN_MATRIX.md`, `run_matrix.json` (139 DAG nodes). Trạng thái cập nhật tới commit
triển khai thực tế (G0 PASS, bench xong, D-pool đang chạy).

## 0. Cấu hình đóng băng (không đổi giữa chừng)

| Role | Model | Revision | Dtype vận hành |
|---|---|---|---|
| S student | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b7a12671c53f0838cd831a0843b95 | bf16 |
| B teacher | meta-llama/Llama-3.1-8B-Instruct (official) | 0e9e39f249a16976918f6564b8830bc894c89659 | bf16 |
| Q teacher | Qwen/Qwen2-7B-Instruct (không phải 2.5) | f2826a00ceef68f0f2b946d945ecc0477ce4450c | **fp32** (bf16 top20 overlap chỉ 16-18/20) |

- Student train luôn khởi tạo lại từ S. Teacher frozen. Không SFT warmup.
- Sinh: seed42, T=1.0/top-p=1.0/top-k=0, cap1024. Main pool 2000 prompts
  (1000 gsm8k-train + 1000 math-train, disjoint 100 pilot), 4 responses/prompt.
- Training recipe: full-param AdamW, reverse-KL trên audited shared byte support,
  150 updates, 64 prompts/update x 4 rollouts = 256 seq/update, cosine horizon150,
  warmup 8, clip 1.0, loss chia tổng response tokens trước mask.
  Checkpoint 25/50/75/100/125/150; primary = step150.
- Eval: mỗi checkpoint student chấm GSM8K-test 1319 + MATH-500, n8 samples,
  primary = mean correctness step150; mốc giữa chỉ dev-eval, không chọn best-test.

## 1. Chuẩn bị + qualify (V) — G0

| ID | Việc | Lượng | Gate | Trạng thái |
|---|---|---|---|---|
| P-freeze | Freeze dataset rev/row IDs, exclusions, dev/cal manifests, template policy, source hashes | CPU | — | XONG (manifest + 4 parquet verified) |
| V-qual-S/B/Q | identity/roundtrip, mass/residual, cached/uncached, batched/single, precision report | 3 arms x 2 dtype | G0 | XONG (G0=PASS 16/10) |
| V-B-bridge | Official B chấm lại 200 pilot positions/candidates/prefixes, so mirror archived (native + controlled-rendering) | 1 | correctness+manifest | CHƯA (driver chưa viết) |
| V-bench-S/B/Q | chọn 1 operating config: top-k {10,20,50}, cap {4,8}; đo s/position, VRAM, numeric error | 150 positions/model | G0+bench | XONG (1 GPU/job; S 7.2G/B 16.6G/Q-fp32 31.4G) |
| P-operating-freeze | freeze estimator/precision, optimizer defaults, forecast, mask rule, verifier/eval policy, training fit | CPU | G0+bench | XONG 1 phần (dtype + fit; còn optimizer/mask rule) |

## 2. Quality / calibration / template (C)

| ID | Việc | Lượng | Ghi chú |
|---|---|---|---|
| C-quality-screen-S/B/Q | 8 samples x 256 prompts/model (128/domain) | 2048 resp/model | mở pilot training; subset của full |
| C-cal-S/B/Q | score 512 neutral reference solutions (256/domain, disjoint pilot/main/eval) | 1 global token-temperature/model | raw vs calibrated tách entropy matching |
| C-template-S/B/Q | score 200 frozen student responses x 2 template policies | native vs controlled-rendering | không ép Q dùng Llama special tokens |
| C-quality-full-S/B/Q | 8 samples x (2000 train + 1319 gsm8k-test + 500 MATH) | 30552/model (tái dùng 4 samples S của pool, không sinh lại) | workload lớn, cuốn chiếu GPU rảnh |

## 3. Nhánh H2 inference (D) — insight chính

| ID | Việc | Lượng | Trạng thái |
|---|---|---|---|
| D-pool-00..09 | S sinh 200 prompts/shard x 4 | 8000 responses | ĐANG CHẠY (nemo 4 cards; shard 09 smoke 800/800) |
| D-census | audit S<->B, S<->Q, B<->Q toàn pool + native-vs-canonical | CPU | driver audit xong; preview shard-09: SB strict 0.9994, SQ 0.940 |
| P-positions | chốt 5000 joint-strict positions tập trung | CPU | chờ đủ 10 shards |
| D-candidates-00..09 | proposal S/B/Q + union bằng event IDs chung | 10 nhóm | CHƯA (driver chưa viết) |
| D-score-{S,B,Q}-{00..09} | score cùng event manifest (E1/E2/E3 + next-byte) | 30 jobs | CHƯA |
| D-H2 | merge + prompt-weighted means + bootstrap → mechanism table trước/sau controls | CPU | chờ score đủ |
| D-paired-P1 | B/Q targets trên cùng fixed strict positions (loss/gradient, equal-budget) | 1 nhóm | sau positions |

## 4. Chọn LR + training lấy insight (L/T) — 16 runs, seed42 duy nhất

| Đợt | Run | Teacher/supervision | Contrast | Gate mở |
|---|---|---|---|---|
| LR (v3) | L-B-5e7 / 1e6 / 2e6 | B-all, 30 updates, 256 dev problems | development accuracy, hòa thì lấy LR nhỏ | correctness+manifest |
| P-LR-freeze | chốt 1 LR cho mọi OPD arm (CPU) | — | — | xong 3 screens |
| Đầu (v3, song song H2) | T-B-all | B, mọi content positions | baseline không-mismatch | G0+LR freeze |
| | T-B-strict | B, joint-strict | vs B-all: giá mất coverage | + strict-mask audit |
| | T-Q-strict | Q, cùng joint-strict | vs B-strict: teacher effect | + Q support audit |
| | T-B-shift15 | B-strict + 15% sai target position | alignment-error cost | correctness gate |
| H1 dose (nemo/v3 khi pool xong) | T-B-shift05 / T-B-shift30 | như trên, rates 5/30% | đường corruption r={0,5,15,30} | xong shift15 |
| H3 mask (gated) | T-Q-top25 vs T-Q-random25 | Q-strict, bỏ 25% X cao nhất vs random twin | targeted filtering | G1 + mask qualification |
| | T-B-top25 vs T-B-random25 | B-strict, bỏ 25% B-disagreement cao | generic-disagreement control | mở cùng cặp Q |
| Comparators | B-Q-top16 | Q strict-only top16 | vs T-Q-strict full support | sau training chính |
| | B-Q-BPM | Q byte-prefix OPD + whitespace mask | vs T-Q-strict | + probability mapping audit |
| | B-GRPO | verified reward, matched 256 rollouts/update x150 | vs initial S và OPD | + reward/KL freeze riêng |

Mỗi main run: tối đa 38.400 trajectories (~39.3M tokens cap) + eval. 1 GPU/job cho tới khi đo xong fit.

## 5. Counterfactual H3 (F) — inference, gated G1

| ID | Lượng | Điều kiện mở |
|---|---|---|
| F-screen | 200 disagreement + 100 agreement positions, 2 branches x 8 continuations = 4800 | G1 hoặc diagnostic-only |
| F-local-screen | tối đa 200 prefixes x 2 teachers x 4 = 1600 | sau F-screen |
| F-full-00..09 | 2000 disagreement + 500 agreement = 40.000 S continuations | screen correctness/feasibility (không chọn bằng dấu effect) |
| F-local-{B,Q}-00..09 | mỗi teacher tối đa 2000 prefixes x 4 = 8000 (tổng 16000) | sau screen |
| F-analysis | prompt-held-out prediction + bootstrap (CPU) | sau full |

H2 null sau controls → dừng full behavioral, tiếp tục H1. H3 null → 1 P3 confirmation screen rồi dừng.

## 6. Phân bổ 12 cards (đã chốt UUID)

- v3 (84ea/b9db/3256/eaed): training dài — L-* rồi nối tiếp T-B-all/T-B-strict/T-Q-strict/T-B-shift15; gpu3 làm V-B-bridge trước.
- nemo (ac08/d54a/0edc/755a): inference H2 — pool 00-01/02-03/04-05-06/07-08(+09 smoke xong); tiếp candidates/score/F.
- v1 (9679/30b4/505a/5233): job ngắn — V-qual/bench (xong), C-cal/template, quality-full cuốn chiếu, eval.
- Không cross-node NCCL cho inference; không mượn GPU của 10 jobs GOLD/ULD cũ.

## 7. Tracker trạng thái (cập nhật 17/10 rạng sáng)

XONG: staging 36G verified • datasets 4/4 frozen • pool/dev/cal manifests • G0=PASS •
bench + operating dtype (S/B bf16, Q fp32) • D-pool smoke 800/800 (Modal) •
audit driver (census preview SB strict 0.9994) • fixtures/propose/score chain
xanh Modal (scores 311/311) • **V-B-bridge: official≡mirror weights (wmaxd=0.0,
200 positions), template khác nhau (0/200 match, official thêm system prompt),
migration effect mean ~1.6 nats — score pilot cũ không reuse**.
ĐANG: D-pool 9 shards (nemo) + audit từng shard.
CHƯA: P-positions (chờ đủ fixtures) • proposal/scoring production (drivers xanh,
chờ positions.json) • cal/template/quality • training wiring (A5) •
LR screens • 16 training runs • F-branch • comparators.
Lượt publish: b9a8df90 → 50dae79 → eb435d7 → c12c4ea → 0bd3767 → 6293fe8 →
fabc807 → e3695bc → b2eb4f1 → 07feec0 → 6ffe0cf → 48b8a2d → 94909aa →
44de543 (hf_rev f4222e03). Checkout node: SimCT-sb-b9a8df90 (shared).
Modal spend lhtu05: ~$16.3/$30 (debug sb ~$1.6+).
