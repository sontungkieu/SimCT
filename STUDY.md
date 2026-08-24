# Learning lab: Dynamic Span Learning cho cross-tokenizer distillation

Lab này tách sáu ý tưởng cốt lõi thành các bài tập CPU nhỏ, xác định được bằng
test và không phụ thuộc `torch`, model weights, tokenizer library, Ray hay GPU.
Mục tiêu là hiểu đúng interface toán học trước khi ghép nó vào pipeline KD thật.

> Trạng thái khởi đầu có chủ ý: smoke tests phải pass; các test bài tập phải
> fail bằng `NotImplementedError` cho tới khi bạn tự hoàn thiện TODO trong
> `vdt_span/`.

## 1. Môi trường tối thiểu

Yêu cầu duy nhất là Python 3.10 trở lên. Không cài package `kdflow` và không dùng
`requirements.txt` của repo cho lab này vì file đó kéo theo stack training lớn.

Chạy trực tiếp bằng Python có sẵn:

```bash
cd /home/tung/vdt-dynamic-span-learning
python3 scripts/grade_learning.py --smoke
python3 scripts/grade_learning.py
```

Hoặc tạo môi trường rỗng, tái lập bằng `uv`:

```bash
cd /home/tung/vdt-dynamic-span-learning
uv venv --python 3.10 .venv-learning
uv pip install --python .venv-learning/bin/python -r requirements-learning.txt
.venv-learning/bin/python scripts/grade_learning.py --smoke
.venv-learning/bin/python scripts/grade_learning.py
```

Lệnh grader trả exit code `0` chỉ khi phần được chấm pass hoàn toàn, `1` khi còn
bài tập fail, và `2` khi smoke/environment fail. Dùng `--verbose` để xem
traceback hoặc `--exercise alignment` (tương tự cho `candidates`, `viterbi`,
`continuation`, `coarsening`, `policy`) để tập trung vào một bài.

## 2. Bức tranh chung và ranh giới khái niệm

Hai tokenizer không chia cùng chuỗi byte thành cùng token IDs. Vì vậy ta không
được so sánh ID trực tiếp. Lab dùng chuỗi byte đã decode làm event interface:

```text
teacher pieces ─┐
                ├─ shared byte-boundary lattice ─ atomic aligned spans
student pieces ─┘                              │
                                               ├─ candidate spans
context + training state ─ adaptive budget ───┤
                                               └─ semi-Markov Viterbi path

teacher continuation probabilities ─ coarsen onto shared events ─ supervision
```

Ba câu hỏi phải được giữ riêng:

- `c_align`: hai phía đang nói về cùng byte interval nào?
- `c_supervision`: xác suất/score nào được chiếu lên shared event?
- `c_credit` hoặc policy: score span được phân bổ/chọn như thế nào?

Lab này kiểm tra interface và invariant. Nó không chứng minh rằng một span
policy cụ thể cải thiện training, không chạy benchmark, và không coi score
heuristic là causal token credit.

## 3. Exercise 1 — exact byte-boundary alignment (18 điểm)

File: `vdt_span/alignment.py`.

Với token pieces dạng bytes, đặt cumulative boundaries:

\[
B_T = \{0, |t_1|, |t_1t_2|, \ldots\},\qquad
B_S = \{0, |s_1|, |s_1s_2|, \ldots\}.
\]

Các shared boundaries là `B_T ∩ B_S`. Mỗi cặp shared boundaries liên tiếp tạo
đúng một `AlignedSpan`; không được tách nhỏ hơn tại boundary chỉ xuất hiện ở một
tokenizer. Đây là alignment theo byte, nên một token piece được phép chứa nửa
đầu của code point UTF-8 và không cần decode độc lập được.

Contract:

- Reject token piece rỗng và reject khi concatenated teacher bytes khác student
  bytes bằng `ValueError`.
- Trả tuple các `AlignedSpan` phủ toàn bộ chuỗi, không gap/overlap.
- Các range token và byte dùng half-open convention `[start, end)`.

Gợi ý: viết ra cumulative boundary kèm token index trước; đừng dùng `str`, số
character, `.strip()`, hay thay thế marker `Ġ`/`▁`.

## 4. Exercise 2 — candidate span enumeration (16 điểm)

File: `vdt_span/candidates.py`.

Input là các atomic spans từ Exercise 1. Một candidate hợp lệ là union của một
dãy atom liên tiếp `[i, j)`. Nó phải đồng thời thỏa:

\[
\Delta_T \le L_T,\qquad \Delta_S \le L_S,\qquad
\Delta_{byte} \le L_B.
\]

Contract:

- Xác nhận atom input liên tục tuyệt đối ở teacher range, student range và byte
  range; input hỏng phải gây `ValueError`.
- Ba limit phải là số nguyên dương.
- Enumerate mọi union hợp lệ đúng một lần, kể cả singleton.
- Sort lexicographically theo `(atom_start, atom_end)`; limit là inclusive.

Gợi ý: vì width chỉ tăng khi kéo `j` sang phải, có thể dừng inner scan khi một
resource limit đã bị vượt.

## 5. Exercise 3 — semi-Markov Viterbi (20 điểm)

File: `vdt_span/viterbi.py`.

Mỗi `ScoredSpan(i, j, score)` là một edge từ boundary `i` tới `j`. Tìm path phủ
đúng `[0, N)` và tối đa hóa tổng score:

\[
V(j)=\max_{(i,j)\in\mathcal C}\{V(i)+s(i,j)\},\qquad V(0)=0.
\]

Contract:

- Reject `num_atoms < 0`, range candidate không hợp lệ, hoặc score không finite.
- Với `num_atoms == 0`, trả `ViterbiPath(0.0, ())`.
- Nếu không có complete path, raise `ValueError`.
- Tie-break xác định: score lớn hơn; nếu score bằng nhau thì ít span hơn; nếu
  vẫn bằng thì tuple path `(start, end)` nhỏ hơn theo lexicographic order.

Gợi ý: state DP cần giữ đủ cả score và path/tie-break key; chỉ giữ scalar score
sẽ làm output phụ thuộc thứ tự input.

## 6. Exercise 4 — autoregressive continuation scoring (14 điểm)

File: `vdt_span/continuation.py`.

Với prefix `x` và continuation `y_1...y_m`, joint continuation log-probability:

\[
\log p(y_{1:m}\mid x)=\sum_{k=1}^{m}
\log p(y_k\mid x,y_{<k}).
\]

Contract:

- Gọi callback đúng một lần mỗi token với prefix đã nối các continuation token
  trước đó; không dùng frozen prefix.
- Trả tổng, không lấy mean và không exponentiate.
- Continuation rỗng có score `0.0` và không gọi callback.
- Mỗi log-probability phải finite và `<= 0`; nếu không, raise `ValueError`.

Gợi ý: chuyển prefix thành tuple một lần để tránh mutation input rồi cập nhật
prefix cục bộ sau mỗi query.

## 7. Exercise 5 — mass-preserving coarsening (16 điểm)

File: `vdt_span/coarsening.py`.

Cho fine events `e` có mass `p(e)` và map bucket `g(e)`, coarse mass là:

\[
q(z)=\sum_{e:g(e)=z}p(e).
\]

Event không có trong `bucket_of` phải đi vào `residual_bucket`. Đây là điểm
quan trọng: drop tail rồi renormalize phần top-k không phải mass-preserving.

Contract:

- Input phải không rỗng, mọi mass finite/không âm, `atol >= 0`, và tổng mass
  cách `1.0` không quá `atol`; nếu sai raise `ValueError`.
- Aggregate collision và residual bằng phép cộng; không renormalize.
- Output không cần tạo residual bucket nếu không có residual mass.
- Tổng output phải giữ nguyên tổng input trong sai số số thực.

Gợi ý: `math.fsum` giúp kiểm tra tổng ổn định hơn `sum`.

## 8. Exercise 6 — adaptive context/training-dependent policy (16 điểm)

File: `vdt_span/policy.py`.

Để policy audit được trước khi thử learned router, lab dùng một curriculum có
hard bounds. Đặt:

\[
p=\min(\text{step}/\text{warmup\_steps},1),
\]

\[
r=p\,c_{boundary}(1-d_{teacher})(1-r_{context}),
\]

\[
L= L_{min}+\left\lfloor (L_{max}-L_{min})r \right\rfloor.
\]

Contract:

- Ba context feature nằm trong `[0, 1]`; `step >= 0`, `warmup_steps > 0`;
  `1 <= min_width <= max_width`. Vi phạm phải raise `ValueError`.
- Clip training progress ở `1`, nhưng không clip input context sai.
- Trả `int` trong hard bounds. Policy phải thay đổi theo cả context và training
  state; không cache một width tĩnh.

Đây chỉ là baseline minh bạch: confidence cao, disagreement/risk thấp và
training muộn cho phép span dài hơn. Sau này chỉ nên thay bằng learned policy
khi uniform/deterministic baselines có oracle gap đo được trên held-out utility.

## 9. Rubric và lộ trình làm bài

| Nhóm | Điểm | Điều được chấm |
|---|---:|---|
| Exact byte alignment | 18 | shared boundaries, UTF-8 byte pieces, validation |
| Candidate enumeration | 16 | completeness, exact ranges, limits/order, contiguity |
| Semi-Markov Viterbi | 20 | optimum path, two tie-breaks, empty/unreachable |
| Continuation scoring | 14 | sum, evolving prefix, empty case, validation |
| Mass-preserving coarsening | 16 | collision, residual tail, no renorm, validation |
| Adaptive span policy | 16 | exact formula, schedule, context response, validation |
| **Tổng** | **100** | |

Mỗi public test có số điểm cố định trong `scripts/grade_learning.py`; không có
random seed, network, clock hoặc GPU. Trình tự học đề xuất:

1. Chạy smoke và grader đầy đủ để thấy baseline đỏ có chủ ý.
2. Làm E1 rồi E2 để tạo candidate lattice đúng.
3. Làm E4 và E5 để hiểu score/event mass trước khi chọn path.
4. Làm E3, cuối cùng nối E6 làm resource budget cho E2.
5. Chạy `python3 -m unittest discover -s tests/learning -v` để xem toàn bộ public
   assertions, rồi chạy grader để nhận điểm chính xác.

Không sửa tests để làm điểm tăng. Nếu muốn thử thêm, tạo test riêng trước cho
emoji/byte fallback, duplicate candidate, floating-point tie, và residual mass
rất nhỏ.
