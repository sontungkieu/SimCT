# Các run chính: Structural vs. Behavioral Mismatch

## Material Passport

- Ngày 2026-10-16; contract `sb-main-v2`; mode: experiment plan, inline.
- Nguồn: bản thảo HTML, user overrides và main plan ở [đây](../../STRUCTURAL_BEHAVIORAL_MAIN_PLAN.md).
- Mục tiêu trước mắt: một seed để tìm cơ chế; không nhân tất cả arms thành năm seed.
- Trạng thái: kế hoạch cụ thể, chưa submit GPU. [Manifest](run_matrix.json) là danh sách công việc khai báo, chưa phải launcher.
- Kiểm chứng mới: [receipt truy cập Meta](meta_access_receipt.json). Chưa tải hoặc load weights Meta trên node công ty trong lượt này.
- Các con số recipe dưới đây là đề xuất vận hành của study; không phải tất cả đều được HTML quy định.

## 1. Bộ model chốt cho các run mới

| Role | Model | Revision |
|---|---|---|
| S | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| B | **`meta-llama/Llama-3.1-8B-Instruct`** | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Q | **`Qwen/Qwen2-7B-Instruct`**, không phải Qwen2.5 | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` |

Mọi student training khởi tạo lại từ cùng S, teacher frozen. Không thêm SFT warmup. LR warmup trong optimizer là việc khác.

Pilot 100×2 cũ vẫn dùng B=NousResearch revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`. Bốn weight hashes trong metadata Meta khớp bốn shard đã được pilot kiểm checksum; `tokenizer_config.json` khác. Vì vậy không đổi tên model trong kết quả cũ và không tái sử dụng score B cũ như score official. Run `V-B-bridge` bên dưới đo phần chuyển này. File cấu hình pilot và handoff replay cũ giữ nguyên.

## 2. Thứ tự chạy thực tế

1. `V-*`: qualify runtime/model/precision và đo cost. CPU đóng băng data manifests, 10 shard và source hashes.
2. Đồng thời bắt đầu `C-*` quality/calibration/template và `D-pool-*`. `L-*` chỉ mở sau correctness + development verifier checks.
3. Chạy `D-census`, `D-candidates-*`, `D-score-*-*`, rồi `D-H2`. Đây là nhánh lấy insight H2, không phải mười training seeds.
4. Nhánh training đầu: `T-B-all`, `T-B-strict`, `T-Q-strict`, `T-B-shift15`. Nếu cần đường dose-response H1, thêm `T-B-shift05/30`, dùng lại r=0/15.
5. Nếu H2 còn đứng sau controls, chạy `F-screen`, rồi mở full counterfactual và local competence khi screen đạt correctness/feasibility. H2 null không chặn P1/P2.
6. Sau H2 + mask qualification, mở bốn `*-top25/random25`. G2 không có predictive increment thì chỉ một screen P3 này, không mở predictor/seed sweep.
7. Các comparator `B-*` hoàn thiện bảng baseline; không thay câu hỏi cơ chế bằng chạy thêm GOLD/ULD Countdown.

## 3. Các run chuẩn bị và inference

| ID / số job | Chạy gì, bao nhiêu | So với / đầu ra quyết định |
|---|---|---|
| `V-qual-S/B/Q` / 3 | Frozen pilot fixtures, mỗi model riêng; cached/uncached, batched/single, roundtrip và probability checks trên B200 | `qualification.json`; không mặc định BF16 pass vì runtime hỗ trợ |
| `V-B-bridge` / 1 | Official B trên cùng 200 pilot positions/candidates/prefixes; cùng precision, so score mirror archived; chấm riêng native-template và rendered-prompt-controlled conditions | tokenizer/template hashes, max/mean log-prob delta, X delta, rendered-prefix audit; không regenerate rollout |
| `V-bench-S/B/Q` / 3 | Cùng 100–200 development positions, proposed target 150; candidate top-k {10,20,50}, extension cap {4,8} | seconds/position, candidate/path count, peak VRAM, numeric error; chọn một operating config trước main |
| `C-quality-screen-S/B/Q` / 3 | Mỗi model 8 samples trên cùng 256 train prompts, 128/domain, subset của main pool | 2.048 responses/model; kiểm verifier, truncation, throughput và quality gap |
| `C-cal-S/B/Q` / 3 | Score cùng 512 neutral reference solutions, 256/domain; disjoint pilot/main/eval | một global token-temperature/model; raw vs calibrated tách biệt entropy matching |
| `C-template-S/B/Q` / 3 | Score cùng 200 frozen student responses dưới hai template policies đã đóng băng | native vs controlled-rendering sensitivity; không ép Q dùng Llama special tokens |
| `C-quality-full-S/B/Q` / 3 | 8 samples/problem/model trên 2.000 train + GSM8K test 1.319 + MATH-500 | 30.552/model, 91.656 tổng trước reuse; n8 quality/initial evaluation; không generate lại task keys đã có |
| `D-pool-00..09` / 10 | S sinh 200 prompts/shard ×4 responses; seed42; T1/top-p1/top-k0/cap1024 | 8.000 frozen responses; reuse đúng samples S từ quality, hoặc ngược lại |
| `D-census` / CPU | Audit S↔B, S↔Q, B↔Q trên toàn pool, kèm native-vs-canonical | tổng thể + từng sample/token; boundary F1, strict coverage, byte/vocab overlap, numeric runs, eligibility |
| `D-candidates-00..09` / 10 nhóm | Chọn tập 5.000 joint-strict positions tập trung; proposal S/B/Q và union bằng event IDs chung | cùng positions/candidates cho cả ba model; số positions mỗi shard theo manifest, không ép đúng 500 |
| `D-score-{S,B,Q}-{00..09}` / 30 | Score cùng event manifest; E1/E2/E3 đã audit, next-byte sensitivity riêng | per-position probabilities/mass/residual + raw/global-calibrated X; missing/failed inventory |
| `D-H2` / CPU | Merge theo expected IDs, prompt-weighted means và prompt bootstrap | mechanism table trước/sau calibration, template, support và quality controls; disposition G1 |
| `D-paired-P1` / 1 nhóm | B/Q targets trên cùng fixed strict positions: loss/gradient + equal-budget short update diagnostics | identical-context comparison, ghi rõ frozen-data/off-policy; khác estimand on-policy P1 |

`C-quality-full` là workload lớn, có thể chạy cuốn chiếu trên GPU rảnh. `C-quality-screen` chỉ giúp mở pilot training; trước kết luận H2/H3 cuối phải có quality evidence phù hợp và local competence. Không gọi screen 256 prompts là đã xong quality full.

Dataset repo/revision, row IDs, answer references, train/development exclusions, template policy và scorer version phải có trong freeze receipt trước launch. Chưa có các manifest này thì các family trên chưa launch-ready. LR development eval dùng 256 train problems riêng, disjoint main/pilot/calibration/test; không chọn LR bằng GSM8K test/MATH-500.

Quality samples và frozen pool dùng chung logical sample IDs cho bốn samples S đầu. Nếu full quality chạy trước, lấy đúng bốn samples đó; nếu pool chạy trước, quality chỉ bổ sung bốn. Screen là subset, không cộng thêm 6.144 responses nếu các task đã tồn tại.

## 4. Training recipe chung và ba run chọn LR

Proposed main recipe: seed42; full-parameter AdamW; reverse KL trên audited shared decoded-byte content support; 150 optimizer updates; 64 prompts/update ×4 on-policy rollouts = 256 sequences/update; cap1024; T1/top-p1/top-k0; cosine horizon150, warmup8 updates, clip1.0. Loss chia **tổng response tokens trước mask**. Freeze AdamW betas/epsilon/weight-decay và actual backend trong operating receipt; không kế thừa default khác nhau giữa methods.

| Run | LR | Budget | Chọn theo |
|---|---:|---|---|
| `L-B-5e7` | 5e-7 | B-all, 30 updates | cùng 256 development problems và evaluation seeds |
| `L-B-1e6` | 1e-6 | như trên | development accuracy, tie-break LR nhỏ hơn |
| `L-B-2e6` | 2e-6 | như trên | cùng scheduler horizon150, không co cosine xuống30 |

Chọn một LR, freeze cho mọi OPD arm. Main runs bắt đầu từ initial S, không tiếp nối một checkpoint LR winner rồi chỉ cho một arm thêm30 steps. Lưu checkpoints25/50/75/100/125/150; primary endpoint **step150**. Một main run tạo 38.400 response trajectories theo kế hoạch, tối đa39.321.600 generated tokens, không tính eval/teacher scoring. Đây là cap, chưa phải thời gian đo được.

## 5. Các run training lấy insight: một seed42

| Run | Teacher / supervision | Contrast chính | Khi mở |
|---|---|---|---|
| `T-B-all` | B, mọi content positions hợp lệ | baseline không mismatch tokenizer | G0 + LR freeze |
| `T-B-strict` | B, joint-strict S/B/Q | với B-all: giá của mất coverage | như trên + strict-mask audit |
| `T-Q-strict` | Q, cùng joint-strict rule | với B-strict: teacher effect sau eligibility control | như trên + Q probability/support audit |
| `T-B-shift15` | B-strict, sai target position tại15% eligible | với B-strict: alignment-error cost khi giữ teacher | cùng correctness gate, không đợi H2 positive |
| `T-Q-top25` | Q-strict, bỏ25% vị trí X cao nhất | với Q-random25: targeted filtering | G1 + online mask qualification |
| `T-Q-random25` | Q-strict, random twin | matched count + pre-mask loss mass | mở cùng Q-top25 |
| `T-B-top25` | B-strict, bỏ25% raw B-disagreement cao | generic-disagreement control | mở cùng Q mask pair |
| `T-B-random25` | B-strict, random twin của B-top25 | generic filtering / ít supervision | mở cùng B-top25 |

Đây là **8 main training arms**, không phải tám lần lặp lại. Bốn arm đầu chạy được về mặt thiết kế mà không cần H2 positive. Bốn mask arms chỉ mở cùng cặp đối chứng; không chạy riêng top25 rồi diễn giải improvement so với unmasked là H3 causal.

**Bổ sung để thật sự có H1 dose-response:** `T-B-shift05` và `T-B-shift30`, cùng recipe, dùng B-strict làm r0 và shift15 làm r15. Hoàn tất r={0,5,15,30} cần tổng ba corruption runs, chỉ thêm hai so với lean screen. Dùng nested deterministic selections trên eligible positions, t±1 direction fixed by task seed, cùng response và nội dung hợp lệ; báo boundary exclusions và realized injection fraction. Không chọn rates sau khi nhìn curve.

On-policy students tự sinh khác nhau sau updates. “Cùng rule strict” không có nghĩa actual positions giống hệt; `D-paired-P1` mới là identical-context diagnostic. B/Q quality khác nhau vẫn giới hạn diễn giải behavior; ba model này chưa đủ thiết kế factorial H4.

Mask online dùng globally calibrated X_E2 từ **current** S/B/Q tại current prefix; thêm B forward ở Q arms. Trước launch cần E2/E3 rank-concordance receipt trên held-out development positions và freeze acceptance rule. Nếu estimator không qualify, ghi blocked; không đổi sang E1 âm thầm. Random twin exact mask count/optimizer window, pre-mask loss-mass tolerance đề xuất1%; định nghĩa matching strata/algorithm trước chạy và ghi mismatches. Teacher target temperature vẫn1, không đổi cùng temperature dùng để rank mask.

## 6. Counterfactual H3: inference, không phải training seeds

| ID | Budget | Câu hỏi / điều kiện |
|---|---|---|
| `F-screen` | 200 disagreement +100 agreement positions; 2 branches ×8 S continuations =4.800 | forced event/byte-prefix hợp lệ? verifier/noise/length có đủ phân biệt? |
| `F-local-screen` | tối đa200 prefixes ×2 teachers ×4 =1.600 continuations | teacher có thực sự giải tiếp được từ prefix đó? |
| `F-full-00..09` | tổng2.000 disagreement +500 agreement; 40.000 S continuations | counterfactual advantage theo X/category/consensus |
| `F-local-{B,Q}-00..09` | mỗi teacher tối đa2.000 prefixes ×4 =8.000; tổng16.000 | local competence, quality-adjusted predictive analysis |
| `F-analysis` / CPU | prompt-held-out prediction và bootstrap | X có thêm giá trị ngoài B disagreement, entropy, category, local competence? |

Full chỉ mở sau screen correctness/feasibility, không chọn bằng dấu effect screen. Nếu screen tasks đúng full manifest/operating config đã freeze, reuse task keys và chỉ chạy phần thiếu; nếu đổi protocol, giữ screen development-only. Agreement-null dùng independent continuation seeds; treatment/control dùng paired logical seeds. Cùng event unit ở hai branches, không so whole word với một token. Nhãn helpful/neutral/harmful dùng delta preregister từ null-calibration split.

H2 sau controls null/equivalent: dừng mở full behavioral branch, tiếp tục H1 và báo null. H2 còn provisional do measurement: chỉ D4 diagnostic screen, chưa gọi main H3 test. H3 predictive null sau full: một P3 seed42 confirmation screen có đối chứng, chưa mở thêm predictor/seeds. Một seed training không chứng minh benefit ổn định qua training noise.

## 7. Comparator và phần HTML chưa phủ

Ba run bổ sung cho **adapted three-model** baseline table:

| ID | Cấu hình | Đối chứng |
|---|---|---|
| `B-Q-top16` | Q strict-only, shared support top16; retained/dropped mass và normalization ghi rõ | T-Q-strict full support |
| `B-Q-BPM` | Q byte-prefix OPD + whitespace mask; probability mapping audited | T-Q-strict; cùng init/rollouts/steps/LR |
| `B-GRPO` | verified reward, matched256 rollouts/update,150 updates | initial S và OPD; group4, reward/scaler/KL config phải qualify và freeze riêng |

Không gọi atomic MP-OPD là strict full-distribution OPD. Trainer/sampler/checkpoint của SimCT có sẵn; study-specific strict support, corruption/mask và GRPO/BPM wiring phải được kiểm chứng trước khi đặt queue job. Không bịa CLI flags.

Inventory nếu chạy hết recipe này: **3 LR screens +8 main arms +2 dose completion +3 comparator =16 training runs**, trong đó bốn H3 mask arms có gate. GPU-hours chưa biết trước benchmark; một run không mặc định fit một GPU. Không triển khai toàn bộ16 run cùng lúc.

Bản HTML chi tiết còn có shift/mis-route/drop đủ rates và method-on-identity ULD/GOLD/SimCT/BPM; placebo/capacity controls, D5/D6, transplant/P4, teacher/student mở rộng và generalization eval. Chúng nằm trong backlog có nhãn riêng; 16 runs trên **chưa** hoàn tất mọi cell của HTML. Các GOLD/ULD Countdown đã chạy không thay các method-on-identity cells này vì model/data khác. Không tự bổ sung bốn identity-method training runs trước khi các core contrasts cho insight.

## 8. Eval chung và cách dùng sáu GPU

Initial S và teacher B/Q: lấy n8 evaluation đã có từ quality full, không gọi sinh lại. Mỗi checkpoint student được chấm GSM8K test1319 + MATH-500, cùng frozen verifier/sampling/stop policies. Primary là mean correctness n8 tại step150; interval paired theo item. Mốc25/50/75/100/125 dùng development eval, không chọn best checkpoint bằng test. Báo numerator/denominator, missing/invalid, cap-hit, length, repetition, style diagnostics và scoring provenance. Gap closed chỉ báo khi denominator có nghĩa. T1/top-p1/top-k0/cap1024 là proposed primary; nếu cap quá nhiều, version policy áp đồng bộ trước compare, không tăng riêng một arm sau khi thấy accuracy.

| Wave | W1–W2 | W3–W4 | W5–W6 |
|---|---|---|---|
| A | S qualification/benchmark/pool/quality | B qualification/bridge/calibration/quality | Q qualification/benchmark/calibration/quality |
| B | S score shard queue | B score shard queue | Q score shard queue |
| C | F student forks trên toàn sáu slots khi cần | nhường slots theo queue task | nhường slots theo queue task |
| D | B local competence | B/Q local competence | Q local competence |

Training chạy xen vào slots đã xong inference, sau đo one-job fit: ưu tiên B-all, B-strict, Q-strict, shift15; tiếp05/30 khi cần H1 curve. Online Q mask arms cần S/B/Q scoring capacity; đo VRAM trước quyết định một hay hai GPU/job. Không chạy cross-node NCCL cho inference jobs độc lập và không mượn GPU của mười GOLD/ULD jobs cũ.

W1–W6 chỉ là slots logic. Trước block launch cần user-assigned node prompt labels + full GPU UUIDs + lease; benchmark ghi actual GPU-hours, wall-time DAG,20% reserve và forecast trước mở full scoring/forks/training.

## 9. Runner có thể làm gì ngay, còn thiếu gì

- Có source H2 pilot để replay/qualify/propose/score/report; **fixed config hiện tại vẫn trỏ mirror**, không dùng nguyên để gọi đó là official-B/main study.
- Cần versioned config loader + shard/task registry; main dataset manifests; global calibration/template/quality stages; E1/E2 mass audit; D4 fork/local-competence adapter; study-specific training support/corruption/mask adapters.
- `V-B-bridge`, calibration, D4 và main training table chưa có complete executable entrypoint. Bảng này là implementation contract để nối vào SimCT hiện có, không có nghĩa đã submit hay hoàn tất pipeline.
- Output root đề xuất `/workspace/storage-shared/nlp/tungks/tmp/structural-behavioral/sb-main-v2/<run-id>/<attempt-id>/`; source/config immutable, every expected logical task has receipt/hash/status. Retry missing tasks trong attempt mới; không sửa file dưới running process.
- Deliverable đầu tiên: bảng D1 match + H2 trước/sau controls, kèm một-seed contrasts B-all/B-strict/Q-strict/shift15. Tiếp theo mới là H3 advantage và mask-vs-random. Không cần chờ thêm năm seeds để báo insight ban đầu.
