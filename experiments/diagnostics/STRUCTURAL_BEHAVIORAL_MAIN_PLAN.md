# Kế hoạch chính: Structural vs. Behavioral Mismatch in Cross-Tokenizer OPD

Ngày: 2026-10-16, Asia/Saigon. Phiên bản: `sb-main-v2`.

**Danh sách run cụ thể và thứ tự chạy:** [RUN_MATRIX.md](artifacts/structural-behavioral-main-runs/RUN_MATRIX.md), kèm [manifest](artifacts/structural-behavioral-main-runs/run_matrix.json). Main study mới chuyển reference B sang official Meta theo cập nhật người dùng; pilot cũ giữ nguyên mirror và provenance.

## 1. Contract và phạm vi

Mục tiêu là kiểm chứng giả thuyết trong bản thảo của anh Tung: sau khi kiểm soát cấu trúc tokenization, phần bất đồng còn lại giữa teacher và student là gì, có giúp student giải bài hay không, và can thiệp vào phần đó có cải thiện training không. Đây là kế hoạch theo giả thuyết; mỗi workload phải trỏ tới một câu hỏi và một đối chứng.

Theo chỉ đạo mới nhất, giữ nguyên các kết quả GOLD/ULD đã chạy làm dữ kiện bổ sung. Không thu hồi, xóa, dừng hay ghi đè chúng. Chưa có manifest/config/metrics từ 10 GPU công ty trong lượt lập kế hoạch này: không suy ra nội dung thực chạy chỉ từ tài liệu từng gửi. GOLD/ULD Countdown chỉ được đưa vào cùng bảng so sánh khi model, dữ liệu, initialization, budget và evaluator thực sự khớp; trường hợp khác ghi ở phụ lục theo đúng provenance.

Kế hoạch này hoàn thiện thiết kế và phân việc, chưa phải thông báo đã launch. Sáu GPU trên bốn node là allocation do người dùng báo; node label, UUID, thời hạn sử dụng và runtime thực tế còn cần điền trước khi sinh block launch. Ưu tiên insight; chưa mở thêm seed training chỉ để tăng độ chính xác thống kê.

### Material Passport

- Origin: bản thảo HTML của người dùng + các đổi model được người dùng chốt + bằng chứng pilot hiện có.
- Mode: experiment-agent / plan, thực hiện inline.
- Verification: đã đọc source, config và summary pilot tại local; chưa xác minh trạng thái job công ty.
- Implementation: tái dùng SimCT; các adapter chưa có được đánh dấu ở mục 12.
- Remote mutation trong lượt này: không có; không phát sinh chi phí GPU.

## 2. Nguồn chuẩn và các thay đổi đã được chốt

Bản thảo: [artifact](https://claude.ai/artifact/8ZaYo6r8ggu937EF8dXM3m#1210cb5e-8513), các mục `Lean workflow`, `Run order and protocol`, `Evidence table` và `Baselines`.

Nguồn lưu tại `D:/dev/codex/research_vdt/structural-behavioral-opd/artifacts/`. HTML nội dung là `C:/Users/Tung/Downloads/Structural vs. Behavioral Mismatch in Cross-Tokenizer OPD_files/saved_resource(1).html`, SHA256 `24e8521d2cee71794f4f299250e79f9809f24f406c2c3855080f957535ee87ea`. `draft-original-extracted.txt` có SHA256 sau chuẩn hóa CRLF→LF `1fbeb8cabde59ce51f0f65fd84517122da8b780a154a9944f1119cdc4461c435`, khớp manifest cũ. Hash bytes CRLF hiện tại là `49d2aa366bfbce6cbcf8fa946c0b1b5d6d2de0a544dff0a38071d51cd7c1fa0d`.

**User override có hiệu lực:** dùng bộ Llama/Qwen2 đã chốt, không tự quay lại bộ Gemma/Qwen2.5 ban đầu.

| Role hiện tại | Model | Revision cho main study mới | Vai trò trong study |
|---|---|---|---|
| S | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | student khởi tạo, chưa thêm SFT warmup riêng |
| B | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | reference cùng tokenizer sau identity audit |
| Q | `Qwen/Qwen2-7B-Instruct` | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | cross-family teacher |

B của pilot cũ là mirror `NousResearch/Meta-Llama-3.1-8B-Instruct` revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`. Người dùng hiện đã lấy được official Meta. Đã kiểm authenticated config GET=200 và weight HEAD=302; bốn official LFS weight hashes khớp pilot verified checksums, còn `tokenizer_config.json` khác. [Receipt](artifacts/structural-behavioral-main-runs/meta_access_receipt.json) lưu bằng chứng truy cập, không phải xác minh full download/load trên node. Main study dùng official revision mới và chạy `V-B-bridge`; không đổi metadata/score trong pilot. Q là **Qwen2**, không phải Qwen2.5.

HTML gốc có hai cross-family teachers và reference G9. Setup hiện tại có một cross-family teacher Q và reference B. Ánh xạ phép so chính: `X = JS(Q,S) − JS(B,S)` tại cùng prefix/vị trí. Không tạo giả một teacher thứ tư hoặc giữ tên G9 cho B. Kết luận chỉ áp dụng cho bộ model này; chưa đủ xác định hiệu ứng causal cấp family nói chung.

Lean workflow của HTML là lịch mặc định: năm phép kiểm chứng. Capacity references, teacher 27B, D5 learnability, D6 drift, P3b predictor, P4 transplant, digit study và các ablation rộng là nhánh có điều kiện. Các gate trong phần detailed có yêu cầu placebo/capacity mạnh hơn lean version: báo rõ mức chứng cứ, không gọi lean-adapted gate là original detailed Gate 1.

## 3. Bằng chứng đã có và phần còn thiếu

Pilot `h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1`: 100 prompt GSM8K/MATH train, hai response/prompt, seed 42; 1.000 vị trí score đủ S/B/Q, 998 dùng được. Hai vị trí LaTeX chỉ còn một candidate hợp lệ được loại trước metric.

| Phép đo pilot, nats | Mean | CI95 bootstrap theo prompt |
|---|---:|---|
| X raw | 0.024866232461953505 | [0.018842436179683036, 0.03146766887074585] |
| X shape | 0.025318023992987538 | [0.017493241357622608, 0.034018151060619325] |
| X next-byte raw | 0.008564693248044654 | [0.0059120261700110945, 0.011161695799887631] |

Audit toàn bộ 200 response: strict student-token coverage S→Q = `46832/49307`, teacher-token coverage = `46832/52629`, strict byte coverage = `155944/161894`; identical whole numeric-run segmentation = `3557/5942`. Phải báo đúng denominator, không gom thành một con số “90%”.

Nguồn: `remote_artifacts/<pilot-run>/word-summary.json`, `tokenizer-audit-rollout.json`, `assets.json`; brief ở thư mục archive nêu trên. Pilot là development evidence, không phải bộ xác nhận độc lập. Không chạy lại chỉ để có kết quả mới đẹp hơn.

Những thiếu sót ảnh hưởng diễn giải: chưa có global token-temperature calibration; native templates khác nhau; chưa đo teacher quality/local competence; word events dùng canonical suffix/boundary proxy, chưa marginalize mọi token path; không có capacity/placebo control. BF16 gate student từng fail; pilot dùng eager FP32, TF32 off, chuẩn hóa xác suất FP64. Một runtime chạy được BF16 không tự chứng minh numeric parity.

## 4. Câu hỏi, phép kiểm chứng và giới hạn kết luận

| ID | Câu hỏi / giả thuyết | Phép đo quyết định | Kết quả khiến đổi hướng |
|---|---|---|---|
| E1 / D1 | Cấu trúc sai ở đâu, nhiều đến mức nào? | census byte-exact toàn response, theo category và numeric runs | mismatch rộng: ưu tiên H1; mismatch hẹp: tập trung các strata đó |
| E2 / D2–D3 / H2 | X còn tồn tại sau support/template/calibration/quality controls? | X raw/shape trên strict joint positions; E1/E2/E3 và next-byte sensitivity | X mất sau calibration/template/support correction: báo cơ chế tương ứng, không gọi behavioral harm |
| E3 / D4 / H3 predictive | Làm theo teacher tại high-X có tăng khả năng giải đúng không? | paired counterfactual advantage + teacher local competence; incremental prediction | high-X vẫn hữu ích như consensus: bằng chứng chống cách giải thích H3 |
| E4 / P1–P2 / H1 & H3 | Có cost khi áp cùng eligibility rule và có cost do alignment error không? | strict OPD B/Q; B với injected error; cùng init/budget/evaluator | không có training gap hoặc corruption không gây cost: không nhận hypothesis chỉ từ JS |
| E5 / P3 / H3 causal | Bỏ high-X có tốt hơn bỏ supervision tương đương không? | Q top-X mask vs random twin; B high-disagreement vs random twin | cả Q và B cùng hưởng lợi: generic noise; top-X = random: chưa có ích causal |

H4: correlation với structural density chỉ là mô tả. Muốn tách causal tokenizer × behavior cần P4 và manipulation/quality checks; P1/P3 hoặc correlation nhỏ riêng lẻ không chứng minh H4.

## 5. Pool, event space và estimand phải đóng băng

**Pool mới cho main study, đề xuất theo HTML:** 2.000 train prompts, 1.000 GSM8K + 1.000 MATH, khác 100 pilot prompts; bốn student responses/prompt, T=1, top-p=1, top-k=0, completion cap 1.024. Pin dataset revision, source row IDs, reference answers và exclusion list trước generation. Eval riêng: GSM8K test và MATH-500; không dùng test/eval để chọn X threshold, calibration hay mask.

Giữ cả initial frozen pool và native prefix/output IDs, rendered prompts, finish reason, actual length và stop set. Số response kỳ vọng 8.000; cap tổng là 8.192.000 generated tokens. Existing 100×2 pool vẫn là pilot độc lập. Mở từ 1.000 lên 8.000 positions trên cùng 200 response là sensitivity của pilot, không thay full pool hoặc thêm prompt diversity.

S0.3 quality theo detailed HTML: 8 samples/problem/model trên 2.000 train prompts và toàn bộ GSM8K test/MATH-500. Với ba model hiện tại, workload là (2.000+1.319+500)×8×3 = 91.656 responses, trừ các response thực sự cùng key/seed đã có. Bốn frozen student samples đầu được dùng chung, bốn quality samples bổ sung nằm ở artifact riêng; frozen pool vẫn chỉ có bốn. Trước full quality, screen 256 train prompts cân bằng hai domain để đo length/throughput/verifier và uncertainty; screen không thay full quality inventory. Calibration dùng 512 neutral reference solutions (256/domain), disjoint pilot/main/eval; không chọn nhiệt độ bằng task eval accuracy.

**Sampling position:** chọn một tập 5.000 strict joint positions cho E3 theo prompt/category strata; áp cùng tập cho S/B/Q. Loại whitespace-only/special/stop khỏi primary content analysis, báo riêng các strata này; first-five-response-token là stratum riêng. Lưu inclusion probabilities và weights. Primary estimand: equal-prompt mean của các weighted position means; bootstrap theo prompt, không theo token độc lập. Prompt ít/không có vị trí eligible phải được đếm và báo coverage.

**Event spaces:**

- E1: decoded-byte shared-token support, cùng normalization và mass dropped riêng cho mỗi model; chỉ ID trùng không đủ.
- E2: byte-prefix target với residual và kiểm chứng các điều kiện mapping. Phải triển khai/kiểm toán mass assignment cho trường hợp teacher token là prefix của student token; nếu cần nhiều teacher tokens để hoàn thành student token, phép single-token routing không tự cho exact student-event marginal. Ghi rõ estimator và phần residual; chưa pass thì không gắn nhãn exact.
- E3: next-word/event candidate union, scoring cùng event definition; bảo toàn prefix và audit boundary/token-crossing. Code hiện có là hybrid exact ASCII-atom + canonical-word proxy. Mọi kết luận phải giữ nhãn này tới khi có estimator marginal đúng hoặc bound đủ chặt.
- Next-byte distribution và candidate-conditional JS là sensitivity controls riêng; không thay E2/E3 bằng chúng hoặc lấy word-JS trừ byte-JS thành causal tokenizer effect.

Candidate development chỉ dùng pilot: so top-k 10/20/50, extension cap 4/8 và OTHER/retained mass trên cùng 100–200 positions. Chọn operating point trước score main pool. Nếu candidate probabilities overlap hoặc tổng mass >1 quá tolerance, dừng/fix grammar thay vì renormalize để che lỗi. Nếu không bound được approximation, H2 chỉ được phát biểu theo operational proxy đã ghi.

**Calibration:** neutral calibration split độc lập, cùng raw texts cho S/B/Q; chọn một global token-logit temperature/model, lưu objective/grid và hashes. Đo raw, global-calibrated và per-position event-entropy-matched riêng. Không gọi event-cell power-temperature là token-logit calibration. Template sensitivity dùng 200 responses, policy ghi rõ system/default date/stop; không suy ra zero template effect từ tokenizer identity.

## 6. Mười shard dữ liệu và mười work packages

Main frozen pool chia **10 shard prompt**, không chia thành 10 training seeds. Sort 2.000 prompt bằng stable hash của canonical dataset ID + row ID + study seed rồi round-robin vào shard 00–09: mỗi shard 200 prompts, 800 responses. Toàn bộ response/fork của một prompt ở cùng shard. S/B/Q dùng cùng prompt/response/position/candidate IDs. Chọn E3 positions tập trung trước khi rải scoring; không lấy independently 500 positions ở mỗi teacher.

Sharding không được đổi RNG theo số GPU/batch offset một cách âm thầm. Precompute seed theo logical task ID; pin batching/backend và ghi actual execution. Không bảo đảm bitwise reproducibility qua hardware/backend chỉ nhờ seed; parity phải đo. Merge cần expected-ID manifest, phát hiện duplicate/missing, retry đúng missing tasks trong attempt mới; không regenerate cả pool.

| WP | Đầu ra | Dependency / insight | Mức sẵn sàng |
|---|---|---|---|
| 01 | evidence index của pilot và 10 run cũ | tái dùng đúng dữ kiện, không trộn estimands | pilot có; 10 run công ty chờ inventory |
| 02 | node/backend qualification, probability/identity gates | mọi số đo có đáng tin không? | có single-GPU helper; company gate chưa xác minh |
| 03 | quality, calibration, template/stop manifest | loại teacher-quality/sharpness/template confounds | thiếu full controls |
| 04 | 2.000-prompt manifest + 8.000 frozen responses | common paired corpus | adapter cũ cần extension/sharding |
| 05 | D1 full census + mapping-method audit | cấu trúc còn gây vấn đề ở đâu? | census có, method-error adapters còn thiếu |
| 06 | E1/E2/E3 + calibrated D3, Gate G1 | bất đồng tồn tại ở space nào? | E3 proxy có; controls/E1/E2 cần hoàn thiện |
| 07 | D4 paired student forks + agreement null | làm theo teacher có giúp không? | adapter chưa có |
| 08 | local competence + H3 conditional analysis, Gate G2 | bất đồng hành vi hay teacher kém ở prefix? | adapter/analysis chưa có |
| 09 | P1/P2 short controlled training + required baseline screen | total training effect / structural corruption cost | SimCT core có; study adapters cần nối |
| 10 | P3 matched masking, eval và synthesis | targeted removal có ích hơn random removal không? | chờ G1 và study adapters |

Work packages là DAG khoa học; data shards là đơn vị queue. Không ép cả 10 WP chạy đồng thời hoặc dùng GPU cho CPU-only census/statistics.

## 7. Lịch dùng sáu GPU trên bốn node

Gọi sáu slot logic W1–W6; operator map sang node label/UUID đã giao. Mỗi worker chỉ thấy UUID của nó. Inference tasks độc lập, không cần dựng distributed training/NCCL giữa bốn node. Trao đổi manifest/results qua study directory trên shared storage được duyệt; cache/JIT ở node theo runtime contract. Không khởi tạo server dùng chung với các job đã chạy.

| Wave | W1–W2 | W3–W4 | W5–W6 | Điều kiện chuyển |
|---|---|---|---|---|
| A | student pool generation + student quality | B quality/calibration/template | Q quality/calibration/template + measurement benchmark | WP02 pass; manifest chung frozen |
| B | S candidate/score shards | B candidate/score shards | Q candidate/score shards | pool/candidates complete; per-model gates pass |
| C | CPU merge/census/D3; GPUs chỉ chạy missing/control tasks | như W1–W2 | như W1–W2 | G1 disposition có file; không giữ GPU cho bootstrap |
| D | cả sáu slot chia 10 prompt shards cho student counterfactual forks | cùng student weights | cùng student weights | G1 pass để mở full D4; diagnostic pilot có nhãn riêng |
| E | B local competence trên ba slot | B/Q phân lại theo remaining work | Q local competence trên ba slot | forks/prefix inventory frozen |
| F | queue các arms P1/P2, rồi P3 có điều kiện | cùng topology qualified | eval worker khi có checkpoint | một seed screening; không mở seed vô điều kiện |

Wave A chia cụ thể từng slot theo benchmark, không giả định W1/W2 ở cùng node. Giữ exact model/backend identity giữa các shard. Nếu production precision nhanh hơn FP32 chưa pass parity, dùng FP32; startup/compile/weights load tính riêng với steady-state throughput.

## 8. Gates và counterfactual H3

**G0 correctness:** response byte roundtrip; S/S divergence zero theo tolerance qualified; B/B qua scoring path tương tự; S/B tokenizer hashes và native/canonical audit; all probability vectors có normalization/residual đúng; Unicode/LaTeX/digit/whitespace fixtures; cached/uncached và batched/independent agreement. Tolerance freeze theo numeric qualification, không nới vì run fail. Pilot đã pass ở L40S FP32 không thay company B200 qualification.

**G1 adapted lean:** sau G0, kiểm tra X của Q trên primary word-initial content stratum, báo toàn strata với Holm correction nếu dùng để ra quyết định. Statistical screen: prompt-bootstrap CI95 lower bound >0 sau controls. Practical threshold đề xuất cho main mới là 0.005 nats (chọn từ scale pilot, không phải ngưỡng của HTML): lower bound phải vượt ngưỡng này và bound numerical/support/template error đã lượng hóa. Nếu approximation không có bound hoặc quality chưa match, trạng thái `provisional`, chỉ mở D4 diagnostic screen; không gọi original detailed H2 gate pass. Teacher global accuracy chênh >5 percentage points thì bỏ giả định global matching (ngưỡng đề xuất từ HTML), dùng local competence và sensitivity. Không tự thêm placebo/capacity models để hợp thức hóa detailed Gate 1.

G1 null sau controls: báo null/segmentation/sharpness tương ứng; ưu tiên P2 structural test. CI chứa0 tự nó là inconclusive, chưa chứng minh equivalence; equivalence chỉ khi CI nằm trong [-0.005,0.005] nats theo bound đề xuất. G1 positive: mở E3/D4 reliability. Không chọn category có hiệu ứng sau khi xem main results rồi gọi nó là primary.

**D4 staged screen:** 200 disagreement positions + 100 agreement-null positions trên các prompt có student success-rate estimate trong [0.2,0.8]. Kiểm feasibility/label noise, không test efficacy bằng số screen rồi gộp vô điều kiện. Full lean target: 2.000 disagreement + 500 agreement-null positions, stratify X quantile/category/prompt, freeze selection trước forks. Nếu không đủ eligible prompts/positions, báo thiếu; không mở sampling rule dựa trên outcome.

Tại mỗi position, dùng cùng exact prefix bytes/native student IDs; branch A buộc realized student event, branch B buộc preferred Q event; cùng event space, không so một whole word với một token. Canonical suffix encoding phải giữ prefix và decoded forced bytes; invalid continuations bị flag/exclude đối xứng. Student frozen sinh 8 continuations/branch với task-keyed paired seeds, T=1; verifier cố định. `A = success_Q-forced − success_S-forced`.

Agreement null: hai nhánh cùng forced event nhưng **independent continuation seeds** để đo sampling noise. Hai nhánh cùng prefix + cùng seed sẽ có thể trùng hoàn toàn và cho noise floor giả bằng 0. Paired seeds ở treatment/control không tự bảo đảm giảm variance; ghi actual sampling implementation. Đặt delta bằng quantile null đã preregister, trên null-calibration subset, không chọn delta từ effect cần chứng minh.

Từ cùng prefix, B và Q mỗi teacher sinh 4 lần bằng template riêng để đo local competence. Bốn samples cho estimate rời rạc/nhiễu: báo binomial uncertainty, sensitivity với sample count; không coi observed 3/4 là bằng chứng chắc q≥0.75. Lưu prefix familiarity/bits-per-byte, entropy, token category/frequency và generic disagreement làm covariates.

Ba-model triangulation: consensus correction (Q=B khác S), Q-specific disagreement (B=S khác Q), B-specific disagreement (Q=S khác B), three-way disagreement, full agreement. Không tái dựng nhãn four-model `Gemma-family split` của HTML. Q-specific disagreement vẫn có thể là B/S cùng sai; verifier/local competence mới kiểm tra được.

Phân tích: A distributions theo bucket, matched high-X/low-X trên B disagreement/entropy/category/frequency; nested models với prompt/response clustering, prompt-held-out prediction splits. Primary incremental endpoint theo HTML: ΔAUROC harmful ≥0.03, CI95 lower bound >0; báo effect size, rare-harm counts và continuous A thay vì chỉ nhãn. Không có predictive increment thì G2 null/uncertain; một P3 confirmation screen được phép trong kế hoạch, không scale predictor.

Full D4 student continuations = (2.000+500)×2×8 = 40.000; local competence tối đa 2.000×2×4 = 16.000 teacher continuations, deduplicate identical prefix jobs. Nếu one-event fork quá yếu, mở optional 8-token teacher-forced block trên 1.000 positions như HTML; ghi đây là intervention/estimand khác.

## 9. Controlled training: một seed để lấy cơ chế trước

Init dùng đúng S revision ở mục 2, teacher frozen. Không kế thừa SFT checkpoint/campaign default của SimCT. Reuse trainer, rollout, evaluator và checkpoint machinery; thêm study-specific support/masks/metrics. Learning-rate warmup là scheduler trong run, khác SFT warmup. Learning rate và scheduler chưa được main contract chốt: pilot sweep trên B với development prompts, sau đó freeze cùng effective batch, horizon và LR cho mọi arm. Không copy recipe Countdown của GOLD vào đây.

Recipe development đề xuất: full-parameter OPD, AdamW, primary reverse KL `KL(student || teacher_target)` trong shared decoded-byte E1 support; báo discarded mass và normalization của cả hai model. LR candidates {5e-7,1e-6,2e-6}; cùng 30-update screen và fixed development split, chọn bằng task score với tie-break LR nhỏ hơn. Main horizon150, scheduler cosine, LR warmup5% theo main horizon (8 optimizer updates), gradient clip1.0, teacher target temperature1.0. Các giá trị này là proposed study recipe, không phải recipe được HTML định sẵn hay đã qualify. Giữ cùng optimizer/batch/precision giữa compared arms; mask ranking được calibration riêng không tự đổi teacher targets. Primary tokenizer/content support phải audit trước dùng KL.

Planning budget theo HTML: 150 optimizer steps, 64 prompts/update ×4 rollouts = 256 sequences/update, cap 1.024. Đây là proposed study config, chưa qualify trên node. Nếu resource buộc giảm, version contract và chạy mọi compared arm cùng config. Benchmark đầu một run, lưu checkpoint ở steps 25/50/75/100/125/150; đánh giá primary ở fixed step 150, không chọn best checkpoint riêng mỗi arm.

**Lưu ý P1 cần nói đúng:** independently evolving on-policy students không sinh cùng actual tokens/prefixes ở mọi step dù dùng cùng prompt stream/seed. Chúng chỉ dùng chung eligibility rule `strict for S/B/Q`. Vì vậy OPD P1 đo total effect của teacher dưới rule này, chưa tự cô lập pure behavior. Muốn so đúng identical realized positions, thêm paired frozen-pool loss/gradient/short-update diagnostic và gọi đó là frozen-data/off-policy diagnostic; không đổi nhãn thành on-policy. Các báo cáo phải ghi rõ hai estimands.

| Arm | Cấu hình | Đối chứng / câu trả lời |
|---|---|---|
| T-B-all | B token-level OPD, all valid response positions | zero-tokenizer-mismatch reference |
| T-B-strict | B trên joint-strict mask | coverage cost so với B-all |
| T-Q-strict | Q trên cùng rule joint-strict | P1 teacher effect; chất lượng B/Q được đo |
| T-B-shift15 | B-strict, inject wrong-position targets tại 15% eligible positions | P2 lean corruption cost, baseline B-strict là r=0 |
| T-Q-top25 | Q-strict, mask top25% X | behavioral filtering |
| T-Q-random25 | Q-strict, random twin matched count + pre-mask loss mass | ít supervision có giải thích benefit không? |
| T-B-top25 | B-strict, mask top25% raw B divergence | generic-disagreement control |
| T-B-random25 | B-strict, matched random twin | generic-noise baseline |

Screen tám arms này với seed42. P3 arms cuối chỉ mở sau G1; G2 null thì một screen confirmation trước quyết định. Normalize loss theo **total pre-mask student response tokens**; không đổi denominator sang retained tokens để vô tình tăng strength. Random mask matching dùng deterministic stratified sampling: exact count trên optimizer window, sai lệch pre-mask loss mass tối đa1% relative (đề xuất), báo cả strata diagnostics. Nếu không match loss mass được trong tolerance, báo mismatch và sửa algorithm trên development data trước scientific run.

X online cần S/B/Q scores trên cùng current rollout, thêm B forward ở Q arms; không lấy lookup X từ initial pool rồi áp bừa lên unseen on-policy prefixes. Nếu dùng predictor/mask transfer cố định thì là arm riêng P3b. Bản reference/template/precision/denominator phải giống main diagnostics.

Để chi phí khả thi, P3 primary mask xếp hạng theo **global-calibrated X_E2 online**, trong khi H2 reporting dùng E3 estimator đã audit. Trước P3 phải đo E2/E3 rank concordance và top-quartile overlap trên fixed validation positions; nếu ranking không bền, không gọi P3 là intervention vào high-X_E3. E1 mask là fallback arm riêng với estimand riêng, không đổi ngầm. Báo forward/scoring cost của mask, vì thêm E3 scoring tại mọi token có thể lớn hơn nhiều so với một teacher forward.

Shift15 chỉ chứng minh effect của một corruption rate, không gọi dose-response. Để kết luận H1 dose-response, mở B rates {0,5,15,30}% với cùng injection rule; drop-only và mis-route là optional controls, reused r=0/15 tránh rerun. Report actual eligible/injected fraction và boundary exclusions.

HTML Baselines table còn yêu cầu initial S, GRPO matched rollouts, strict full-shared-vocab và top16, BPM byte-prefix + whitespace mask. Initial S và strict-full đã có trong main arms; top16/BPM/GRPO là comparator cells cần implementation qualification. Giữ trong final baseline checklist, ưu tiên sau core insight screen. SimCT/DPCA/GOLD/ULD runs cũ chỉ điền được cells nếu actual contract khớp. Không nói đã hoàn tất toàn bộ HTML khi baseline required chưa chạy.

H3 causal primary: Q-top25 hơn Q-random25 ít nhất 1 absolute accuracy point theo ngưỡng đề xuất trong HTML, paired-item CI và confirmation seeds khi cần; B control không cho cùng pattern generic filtering. Một seed chỉ screening, không đủ khẳng định qua training noise. Vì chỉ có một cross-family Q, không tuyên bố đã thỏa yêu cầu gốc “cả L và Q”. Mở thêm seed chỉ cho contrast quyết định sau screen theo fixed rule, không mở mọi cell tự động.

## 10. Metrics, evidence và quyết định

Mọi inference record có study_version, run/attempt ID, prompt/response/position ID, native IDs + byte offset, teacher revision, source/config hash, event/support ID, probabilities/residual, exclusions và method label. Mọi training receipt có optimizer steps thực, sampling/seed, masks count/loss mass, teacher freeze check, peak VRAM, throughput, checkpoint hash và eval provenance.

Metrics bắt buộc: structural census micro/macro/per-sample/category; X raw/calibrated/shape và support sensitivity; D4 A/uncertainty/local competence; GSM8K/MATH-500 pass numerator/denominator; gap closed chỉ khi denominator teacher−initial student xác định và có CI; style drift với frozen feature definitions, length/repetition/format; bits-per-byte trên teacher-written solutions như secondary diagnostics. Teacher closeness hoặc KL giảm không tự là task improvement.

Mỗi hypothesis có kết luận riêng: supported-at-estimand / null / contrary / inconclusive / not-tested. Không gộp H2 positive thành H3 harmful. Không biến missing/failed outcomes thành score0; generation failed là missing, verifier valid incorrect mới là0. Nếu harmful labels quá hiếm, báo power/uncertainty, không đổi sang chọn một threshold thuận lợi.

New loss chỉ đặt sau H2 + predictive/causal H3 có bằng chứng và can thiệp usable; H4 causal/transplant, IFEval/MMLU secondary, second student là conditional extensions. Với ưu tiên hiện tại, kết quả đủ tốt để gửi anh trước là một **mechanism table**: effect trước/sau calibration/template/support/quality controls, high-X vs consensus counterfactual A, và một contrast masked-vs-random nếu tới P3.

## 11. Estimate tài nguyên và điều kiện mở rộng

Không dùng throughput Gemma2 ~7k generation tokens/s để ước lượng repeated candidate scoring của Llama/Qwen. Không dùng estimate USD10 Modal cũ cho workload company mới. Không launch Modal mới trong kế hoạch này.

Company benchmark bắt buộc: 100–200 positions/model, actual candidate count, score-path lengths, cache clones, prefill time, effective positions/s, peak VRAM; generation throughput riêng từng model với completion-length distribution và truncation. `GPU-hours = tổng thời gian chiếm từng GPU`, `wall time` theo DAG/critical path; ghi cả startup và margin20%, không chia tổng GPU-hours cho6 rồi gọi đó là ETA chắc chắn.

HTML cho rough estimate steps1–3 khoảng20 GPU-hours, steps4–5 khoảng150–200 GPU-hours. Đó là estimate của draft trên setup khác, chưa phải đo cho model/pipeline hiện tại. Candidate path scoring FP32 có thể khác lớn. Runner phải dự báo lại từ benchmark và số task thực trước Wave B/full D4/training. Không tự đổi context length, số rollouts hoặc batch giữa arms để khớp estimate.

Durable shared logs/results và receipts theo company workflow; không để evidence duy nhất trong `/tmp` node-local. Theo chính sách node hiện tại, không tự move folders SimCT hoặc gold_all đang được process sử dụng. Runtime source/config bất biến theo run; collector có deadline và explicit failed/missing inventory.

## 12. Việc implementation còn thiếu và bước khởi động

Hiện có: `h2_word_pilot.py`, `h2_word_events.py`, `h2_word_metrics.py`, `h2_tokenizer_audit.py`, report/export helpers; `H2_COMPANY_HANDOFF.md` có runtime digest và replay instructions. H2 source đang có staged edits từ milestone cũ; HEAD một mình không đại diện overlay. Dùng existing per-file source manifest/task patch, không reset hay commit cuốn chúng vào thay đổi kế hoạch.

Cần làm trước full study: shard-aware immutable config + task registry/merge, stable per-task RNG, quality/template/global calibration, E1/E2 estimators và probability-contract audit, D4 fork adapter/local competence + verifier, study-specific masks/teacher scoring tích hợp SimCT, required comparator qualifications. Không tạo lại trainer hoặc sampler framework mới. Không đưa lệnh CLI tưởng tượng vào runner khi adapter chưa tồn tại.

**Bước đầu của runner:** đọc [run matrix](artifacts/structural-behavioral-main-runs/RUN_MATRIX.md)/JSON contract; inventory kết quả 10 run cũ và pilot; map sáu UUID được giao; verify B200/backend bằng frozen pilot và official-B bridge trước; chạy benchmark measurement cùng quality/template/calibration tasks. Sau receipt G0 + benchmark, freeze operating config và thực hiện WP04–06 theo 10 data shards. Mỗi transition ghi gate disposition; các phần chưa triển khai giữ `planned`, không ghi `running` hay `pass`.

## References nội bộ

- [Machine-readable plan](structural_behavioral_plan_contract.json).
- [Pilot probability/measurement contract](H2_WORD_PILOT.md).
- [Company pilot replay handoff](H2_COMPANY_HANDOFF.md).
- `D:/dev/codex/research_vdt/structural-behavioral-opd/artifacts/H2_BRIEF_FOR_BROTHER_20261015.md`.
- `remote_artifacts/h2-llama32-qwen2-s42-p100-n2-e1000-20261015-r1/word-summary.json` và `assets.json`.
