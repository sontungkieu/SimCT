"""Offline, stdlib-only delivery report for a completed small H2 pilot.

Usage: python h2_word_report_export.py --run-dir PATH
Fails closed on missing/unfinished rollout, scoring, or qualification evidence.
Generated text is escaped, never evaluated as Markdown/HTML or spreadsheet code.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote


ARMS = ("S", "B", "Q")
COVERAGE = ("strict_student_token_fraction", "strict_teacher_token_fraction", "strict_byte_fraction")


def read(path, optional=False):
    if optional and not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def percent(value):
    return "NA" if value is None else format(value * 100, ".8f") + "%"


def fraction(numerator, denominator):
    if numerator is None or denominator is None:
        return "NA"
    return f"{numerator}/{denominator} = {percent(numerator / denominator if denominator else None)}"


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(flatten(item, name))
        else:
            out[name] = item
    return out


def csv_write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                item = scalar(value)
                # Model/source text may begin with a spreadsheet formula marker.
                if isinstance(value, str) and item.lstrip().startswith(("=", "+", "-", "@")):
                    item = "'" + item
                safe[key] = item
            writer.writerow(safe)


class Document:
    def __init__(self):
        self.md, self.body = [], []

    def heading(self, title, level=2):
        self.md.append("#" * level + " " + title + "\n")
        self.body.append(f"<h{level}>{html.escape(title)}</h{level}>")

    def paragraph(self, text):
        self.md.append(text + "\n")
        self.body.append("<p>" + html.escape(text) + "</p>")

    def table(self, columns, rows):
        rows = [[scalar(cell) for cell in row] for row in rows]
        def md_cell(cell):
            cell = cell.replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")
            return re.sub(r"([|`*_\[\]{}()])", r"\\\1", cell)
        self.md.append("| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" +
                       "\n".join("| " + " | ".join(md_cell(cell) for cell in row) + " |" for row in rows) + "\n")
        self.body.append('<div class="scroll"><table><thead><tr>' + "".join("<th>" + html.escape(c) + "</th>" for c in columns) +
                         "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(c) + "</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>")

    def json_detail(self, label, value, markdown=True):
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        if markdown:
            # Four-backtick fencing safely contains ordinary triple backticks.
            longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
            fence = "`" * max(4, longest + 1)
            self.md.append(f"{label}\n\n{fence}json\n{text}\n{fence}\n")
        self.body.append("<details><summary>" + html.escape(label) + "</summary><pre>" + html.escape(text) + "</pre></details>")

    def link(self, label, href):
        self.md.append(f"[{label}]({href})\n")
        self.body.append('<p><a href="' + html.escape(href, quote=True) + '">' + html.escape(label) + "</a></p>")

    def code(self, text):
        self.md.append("```bash\n" + text + "\n```\n")
        self.body.append("<pre>" + html.escape(text) + "</pre>")

    def save(self, delivery):
        (delivery / "REPORT.md").write_text("\n".join(self.md), encoding="utf-8")
        css = "body{font:16px/1.6 system-ui,sans-serif;max-width:1450px;margin:36px auto;padding:0 22px;color:#17212b}h1,h2{line-height:1.25}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccd4dc;padding:7px;text-align:left;vertical-align:top}th{background:#eef3f7}td{overflow-wrap:anywhere}.scroll{overflow:auto;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px;font:12px/1.5 monospace}details{border:1px solid #d9e1e8;padding:12px;margin:10px 0}summary{cursor:pointer;font-weight:600}a{color:#1252a0}input{padding:10px;width:min(600px,90%)}@media print{body{font-size:11px;margin:0;max-width:none}details{break-inside:avoid}table{font-size:8px}input{display:none}}"
        page = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H2 pilot — báo cáo và bàn giao</title><style>' + css + '</style><body><label>Tìm nội dung trong bảng: <input id="search" type="search" placeholder="prompt, response, category…"></label>' + "\n".join(self.body) + '<script>document.getElementById("search").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q)});});</script></body></html>'
        (delivery / "REPORT.html").write_text(page, encoding="utf-8")


def validated(run):
    summary = read(run / "word-summary.json")
    rollouts = read(run / "word-rollouts.json")
    assets = read(run / "assets.json")
    audits = {label: read(run / f"tokenizer-audit-{label}.json") for label in ("reference", "rollout")}
    config = summary["config"]
    if (config.get("prompts"), config.get("samples_per_prompt"), config.get("e3_positions")) != (100, 2, 1000):
        raise ValueError("delivery requires the authorized 100 prompts × 2 rollouts × 1000 scored positions")
    prompt_counts = Counter(str(r["prompt_id"]) for r in rollouts)
    if len(rollouts) != 200 or len({r["id"] for r in rollouts}) != 200 or len(prompt_counts) != 100 or set(prompt_counts.values()) != {2}:
        raise ValueError("rollout evidence is incomplete or duplicated")
    if summary.get("requested_positions") != 1000 or summary.get("valid_positions", 0) + len(summary.get("failures", [])) != 1000:
        raise ValueError("1000-position report accounting is incomplete")
    if audits["reference"].get("rows") != 100 or audits["rollout"].get("rows") != 200:
        raise ValueError("reference/rollout audit row counts do not cover the authorized corpus")
    expected = {r["id"] for r in rollouts}
    if {r["rollout_id"] for r in audits["rollout"]["index"]} != expected:
        raise ValueError("audit and rollout corpus IDs disagree")
    scores, qualifications = {}, {}
    for arm in ARMS:
        scores[arm] = read(run / f"word-scores-{arm}.json")
        keys = [r["key"] for r in scores[arm]["rows"]]
        if len(keys) != 1000 or len(set(keys)) != 1000:
            raise ValueError(f"{arm}: scoring must contain 1000 unique positions")
        qualifications[arm] = read(run / f"word-qualification-{arm}.json")
        q = qualifications[arm]
        if not q.get("pass") or q.get("model_revision") != assets["models"][arm]["revision"]:
            raise ValueError(f"{arm}: latest qualification is failed or stale")
        if config.get("backbone_dtype", "float32") not in q.get("backend", ""):
            raise ValueError(f"{arm}: qualification dtype differs from completed config")
    if len({tuple(sorted(r["key"] for r in scores[a]["rows"])) for a in ARMS}) != 1:
        raise ValueError("model scoring key sets differ")
    metrics = read(run / "word-metrics.json")
    valid_keys = [row["key"] for row in metrics["rows"]]
    failed_keys = [row["key"] for row in summary.get("failures", [])]
    if len(valid_keys) != summary["valid_positions"] or len(set(valid_keys)) != len(valid_keys) or len(set(failed_keys)) != len(failed_keys):
        raise ValueError("position-level metric/failure counts are incomplete or duplicated")
    if set(valid_keys) & set(failed_keys) or set(valid_keys) | set(failed_keys) != set(r["key"] for r in scores["S"]["rows"]):
        raise ValueError("position-level metric/failure keys do not account for scored positions")
    return summary, rollouts, assets, audits, scores, qualifications


def local_trace(run, index):
    path = str(index.get("path", ""))
    marker = "/tokenizer-audit/"
    if marker not in path:
        return None
    relative = "tokenizer-audit/" + path.split(marker, 1)[1]
    candidate = (run / relative).resolve()
    if not candidate.is_relative_to(run.resolve()):
        return None
    return candidate if candidate.is_file() else None


def audit_table_row(corpus, pair, aggregate):
    micro = aggregate["global_micro"]
    counts = micro["counts"]
    coverage = micro.get("coverage", {})
    boundary = micro.get("boundaries", {})
    return [corpus, pair, aggregate["rollout_count"], aggregate["prompt_count"],
            fraction(counts.get("strict_pairs"), counts.get("student_tokens")),
            fraction(counts.get("strict_pairs"), counts.get("teacher_tokens")),
            fraction(counts.get("strict_bytes"), counts.get("bytes")),
            fraction(counts.get("student_tokens_available_teacher_vocab"), counts.get("student_tokens")),
            fraction(counts.get("strict_eligible_student_tokens"), counts.get("eligible_student_tokens")),
            percent(aggregate.get("global_macro_rollout_coverage", {}).get(COVERAGE[0])),
            percent(aggregate.get("global_macro_prompt_coverage", {}).get(COVERAGE[0])),
            scalar(boundary.get("jaccard")),
            scalar(counts.get("teacher_tokens", 0) / counts["student_tokens"] if counts.get("student_tokens") else None)]


def generate(run):
    summary, rollouts, assets, audits, scores, qualifications = validated(run)
    delivery = run / "delivery"
    delivery.mkdir(exist_ok=True)
    doc = Document()
    cfg = summary["config"]
    doc.heading("Báo cáo H2 pilot: Llama 3.2 3B — Llama 3.1 8B / Qwen2 7B", 1)
    doc.paragraph("Lượt inference và xử lý đã đủ 100 prompts, 200 responses và 1.000 positions được score bởi cả ba model. Đây là kết quả thăm dò với hybrid probability contract: single-ASCII atoms dùng exact next-byte marginal, các word events còn lại dùng prefix-preserving canonical suffix/boundary proxy. Không phải phép tính xác suất đầy đủ của mọi word tokenization và không xác nhận original Gate 1. Positions bị loại và shape bất khả thi được giữ trong accounting dưới đây.")
    doc.table(["Đại lượng", "Bằng chứng"], [["Run", summary["run_id"]], ["Prompts / responses / scored positions", "100 / 200 / 1000"], ["Positions có raw hợp lệ", summary["valid_positions"]], ["Position failures", len(summary.get("failures", []))], ["Shape valid / missing", f"{summary['X_shape'].get('valid_position_count')} / {summary['X_shape'].get('missing_position_count')}"], ["Prompt clusters có raw / shape", f"{summary['X_raw'].get('valid_prompt_count')} / {summary['X_shape'].get('valid_prompt_count')}"], ["Global calibration", summary.get("global_calibration", "not run")]])
    doc.table(["Kết quả chính (nats)", "Mean", "Prompt-bootstrap CI95"],
              [[key, summary[key].get("mean"), summary[key].get("ci95")] for key in ("X_raw", "X_shape")])
    positive = all(summary[key].get("ci95") and summary[key]["ci95"][0] > 0 for key in ("X_raw", "X_shape"))
    doc.paragraph(("Cả raw và shape có CI trên 0: pilot có tín hiệu ủng hộ H2 theo estimand hiện tại. " if positive else
                   "CI của pilot chưa đồng thời trên 0 cho raw và shape; chưa đủ tín hiệu theo tiêu chí này. ") +
                  "Kết luận áp dụng cho corpus, support và probability contract đã đo. Chưa chứng minh causal tokenizer effect; native template, năng lực teacher và canonical word proxy vẫn là các giới hạn. X = JS(Q,S) − JS(B,S).")
    doc.heading("Protocol và cách đọc các đơn vị mẫu")
    doc.paragraph("100 prompts là số cụm độc lập dùng trong bootstrap; mỗi prompt có 2 rollout. 1.000 positions là các điểm đo bên trong các responses, không phải 1.000 bài toán độc lập. Chọn vị trí trong tập strict alignment chung S/B/Q, bỏ vùng đầu và whitespace theo fixture; sampling_weight = N_stratum/n_stratum. Trong prompt lấy mean theo sampling weights, sau đó mean các prompt với trọng số bằng nhau. CI bootstrap chỉ phản ánh biến thiên trên corpus và proxy hiện tại; không tách năng lực teacher, họ model hoặc native chat template.")
    doc.table(["Thông số", "Giá trị log/config"], [[key, cfg.get(key)] for key in ("seed", "temperature", "top_p", "top_k", "max_new_tokens", "generation_batch_size", "backbone_dtype", "gpu", "dataset_mix", "proposal_top_k", "proposal_max_tokens", "bootstrap_replicates", "template_policy", "shape_contract")] +
              [["Initial config probability_contract", cfg.get("probability_contract")],
               ["Effective measured probability_contract", summary.get("effective_probability_contract")]])
    doc.table(["Rollout diagnostic", "Count"], [["Content tokens sinh ra", sum(len(r.get("content_ids", [])) for r in rollouts)],
              ["Responses chạm max_new_tokens cap", sum(r.get("finish_reason") == "max_new_tokens" for r in rollouts)],
              ["Finish-reason histogram", dict(Counter(r.get("finish_reason") for r in rollouts))]])
    doc.heading("Model, revision và mirror provenance")
    doc.table(["Arm", "Vai trò", "Model repo", "Revision", "Tokenizer hash", "Verified weight shards"], [[arm, assets["models"][arm].get("role"), assets["models"][arm]["repo"], assets["models"][arm]["revision"], assets["models"][arm].get("tokenizer_sha256"), len(assets["models"][arm].get("verified_weights", {}))] for arm in ARMS])
    mirror = assets["models"]["B"].get("official_comparison", {})
    doc.paragraph("B dùng mirror NousResearch được người dùng cho phép. So sánh official dựa trên metadata public LFS và Git blob của các file nhỏ; shard mirror tải xuống còn được hash tại chỗ. Không suy ra mọi cấu hình tokenizer giống nhau chỉ từ việc weights giống nhau.")
    doc.table(["Official comparison", "Kết quả"], [["Official repo / revision", f"{mirror.get('repo')} / {mirror.get('revision')}"], ["Weight shards identical by name/hash", mirror.get("identical_weight_shards_by_name_and_hash")]] + [[name, record.get("identical")] for name, record in mirror.get("small_file_comparisons", {}).items()])
    for arm in ARMS:
        doc.json_detail(f"Provenance và hashes đầy đủ: {arm}", assets["models"][arm])
    historical_assets = read(run / "historical-assets.json", optional=True)
    if historical_assets:
        doc.heading("Provenance của historical tokenizer controls", 3)
        doc.table(["Control role", "Original arm", "Repo", "Revision", "Tokenizer hash"],
                  [[role, original_arm, historical_assets.get("models", {}).get(original_arm, {}).get("repo"),
                    historical_assets.get("models", {}).get(original_arm, {}).get("revision"),
                    historical_assets.get("models", {}).get(original_arm, {}).get("tokenizer_sha256")]
                   for role, original_arm in cfg.get("historical_tokenizer_controls", {}).items()])
        doc.json_detail("Historical assets manifest", historical_assets, markdown=False)
    else:
        doc.paragraph("historical-assets.json chưa đóng gói: historical audit vẫn có occurrence counts, nhưng người nhận chưa có manifest repo/revision/hash của toàn bộ old tokenizer controls. Cần manifest assets của config.source_run để tái lập old/new comparison tại công ty.")
    doc.heading("Qualification: backend nào thực sự được dùng")
    doc.paragraph("Bảng sau dùng qualification mới nhất, phải pass và khớp revision/dtype của lượt hoàn tất. Các attempt cũ được giữ riêng: failure BF16 là failure qualification, không phải kết quả H2; không trộn số liệu BF16 với FP32. Cached-vs-uncached word JS kiểm tra implementation độc lập hơn phép JS(p,p)=0 mang tính đại số.")
    doc.paragraph("Backbone inference dùng FP32 eager; TF32 bị tắt trong loader. Scorer thực hiện log-softmax normalization và probability accumulation bằng float64 để kiểm tra mass conservation; điều này không biến backbone thành FP64. Bảng dữ liệu từng arm ghi normalization_dtype và kiểm tra tổng next-byte mass ở dưới.")
    qrows = []
    for arm in ARMS:
        q = qualifications[arm]
        qrows.append([arm, q.get("backend"), q["pass"], q.get("generation_mean_abs_logprob"), q.get("generation_max_abs_logprob"), q.get("cached_max_abs_logprob"), q.get("canonical_head_max_abs"), q.get("same_model_cached_vs_uncached_word_js"), q.get("stop_token_ids")])
    doc.table(["Arm", "Backend", "Pass", "Generation mean Δlogp", "Generation max Δlogp", "Cached max Δlogp", "Head max Δlogit", "Word JS", "Stop IDs"], qrows)
    attempts = []
    for path in sorted(run.glob("word-qualification-*-a*.json")):
        item = read(path)
        attempts.append({"file": path.name, **item})
    doc.json_detail("Qualification attempt history", attempts)
    receipts = [read(p) for p in sorted(run.glob("*-receipt.json"))]
    if receipts:
        doc.table(["Stage", "Arm", "Exit", "Runtime digest", "Config hash", "Word config hash", "Source hashes"],
                  [[r.get("stage"), r.get("arm"), r.get("exit_code"), r.get("runtime_image"), r.get("config_sha256"), r.get("word_config_sha256"), len(r.get("source_sha256", {}))] for r in receipts])
        doc.json_detail("Execution receipts và source hashes đầy đủ", receipts, markdown=False)
    else:
        doc.paragraph("Execution receipt files chưa được đóng gói trong run-dir; không suy ra runtime digest từ tên backend. Tại công ty phải lấy đúng immutable image digest/source hashes từ stage receipts trước khi tái lập.")
    doc.heading("Tokenizer match: ba khái niệm cần phân biệt")
    doc.paragraph("Static vocabulary overlap đo unique decoded non-special byte strings trong vocabularies; không đo tần suất sử dụng. Shared-vocabulary occurrence coverage đo token xuất hiện có byte string nằm trong vocabulary phía kia. Strict 1:1 occurrence match yêu cầu hai token có đúng cùng byte start/end trên cùng text. Cùng numeric token ID không chứng minh lexical compatibility; cùng source bytes với IDs khác nhau vẫn có thể strict match.")
    doc.paragraph("H2-eligible denominator là student tokens sau năm tokens đầu, không phải whitespace và có UTF-8 prefix hoàn chỉnh; strict_eligible_student_tokens/eligible_student_tokens đo pair alignment trong tập đó. Không thay denominator toàn response bằng denominator eligible mà giữ cùng tên tỷ lệ. Common S/B/Q eligible positions còn hẹp hơn từng pair riêng.")
    doc.paragraph("Không có nguồn, corpus hay denominator cho con số ‘90%’ được nhắc đến. Báo cáo này chỉ đo corpus hiện tại, không xác nhận hoặc bác bỏ một nhận định chưa rõ nghĩa. Cần nói rõ student-token, teacher-token hay byte denominator; có loại whitespace/special hay không. Special tokens không nằm trong byte spans. Boundary metrics bỏ source start/end; nếu hai interior-boundary sets cùng rỗng thì convention Jaccard=1, không được diễn giải thành có nhiều evidence về token match. Tỷ lệ bảng hiển thị 8 chữ số thập phân phần trăm; CSV giữ full float precision.")
    doc.table(["Audit label", "S token path", "Teacher token path", "Cách diễn giải"],
              [["rollout/new-B", "Sampled native content IDs", "Cùng native IDs, chỉ sau vocabulary hash equality assertion", "Same-tokenizer on-policy control; strict=100% không phải evidence độc lập về canonical segmentation"],
               ["rollout/new-Q", "Sampled native content IDs", "Canonical retokenization của cùng raw text", "On-policy alignment; có thể gồm stochastic token-path effects"],
               ["rollout/canonical-B hoặc canonical-Q", "Canonical retokenization", "Canonical retokenization", "Control tách sampled path khỏi segmentation"],
               ["historical-B hoặc historical-Q", "Historical canonical retokenization", "Historical canonical retokenization", "Dùng canonical_control intersection để so old/new trên cùng texts"],
               ["reference/*", "Canonical reference solution encoding", "Canonical encoding", "Reference corpus độc lập với generated rollout corpus"]])
    doc.paragraph("Word scoring giữ nguyên native prefix IDs đã chọn, encode canonical candidate suffix riêng, rồi bắt buộc full-context decode khớp prefix+candidate. Không canonicalize lại lịch sử sampled tokens. Đây là prefix-preserving canonical suffix proxy; các alternative suffix tokenizations vẫn không được cộng đầy đủ. Bằng chứng qualification phải theo đúng prefix convention và backend của lượt này.")
    doc.paragraph("Riêng candidate là một ASCII digit hoặc punctuation atom, mass lấy từ exact first-byte marginal của tất cả native next-token IDs: token ‘123’ đóng góp vào event ‘1’, token ‘...’ đóng góp vào event ‘.’. Non-ASCII atoms, letter và LaTeX events vẫn dùng canonical suffix/boundary proxy. Vì vậy không được mô tả mọi word-cell mass là canonical path hoặc mọi word-cell mass là exact marginal; mass_methods trong CSV phân biệt từng cell.")
    vocab = read(run / "vocabulary-overlap.json")
    doc.json_detail("Static vocabulary overlap", vocab.get("pairs", vocab))
    overall_rows, alignment_csv, prompt_csv = [], [], []
    for corpus, audit in audits.items():
        for pair, stats in audit.get("pairs", {}).items():
            overall_rows.append(audit_table_row(corpus, pair, stats))
            for row in stats["per_rollout"]:
                alignment_csv.append({"corpus": corpus, "pair": pair, "prompt_id": row["prompt_id"], "rollout_id": row["rollout_id"], **flatten({k: v for k, v in row.items() if k not in ("prompt_id", "rollout_id")})})
            for row in stats["per_prompt"]:
                prompt_csv.append({"corpus": corpus, "pair": pair, **flatten(row)})
    doc.table(["Corpus", "Pair", "Rows", "Prompts", "Strict student tokens", "Strict teacher tokens", "Strict bytes", "Student shared-vocab usage", "Strict H2-eligible S", "Macro response strict", "Macro prompt strict", "Boundary Jaccard", "Teacher/student token ratio"], overall_rows)
    doc.heading("Old/new matched-corpus comparison và quantiles")
    for corpus, audit in audits.items():
        paired = audit.get("paired_comparison")
        if not paired:
            doc.paragraph(f"{corpus}: artifact chưa có paired-intersection summary; independent summaries ở trên không đủ để kết luận old/new trên cùng subset khi có failures.")
        else:
            paired_rows = []
            for arm, item in paired.items():
                selected = item.get("canonical_control", item)
                label = "canonical-control" if "canonical_control" in item else "matched-text"
                for family in ("new", "historical"):
                    paired_rows.append(audit_table_row(corpus, label + "/" + family + "-" + arm, selected[family]))
                doc.paragraph(f"{corpus}/{arm}/{label}: intersection {selected.get('common_count')} responses; omitted {selected.get('omitted_count', selected.get('omitted_from_paired_count'))}. Cùng text và cùng successful response IDs; không gọi reference text là student rollout.")
                if corpus == "rollout" and "canonical_control" not in item:
                    doc.paragraph("Artifact chưa có canonical-control intersection: không quy chênh lệch native/new-vs-historical hoàn toàn cho tokenizer vì sampled token path khác canonical path.")
            doc.table(["Corpus", "Paired pair", "Rows", "Prompts", "Strict S", "Strict teacher", "Strict bytes", "Shared-vocab S", "Strict eligible S", "Macro response", "Macro prompt", "Boundary Jaccard", "Expansion"], paired_rows)
        details = {pair: {k: stats.get(k) for k in ("rollout_strict_coverage_quantiles", "prompt_strict_coverage_quantiles")}
                   for pair, stats in audit.get("pairs", {}).items()}
        quantile_rows = []
        category_rows, interval_rows, numeric_rows = [], [], []
        for pair, stats in audit.get("pairs", {}).items():
            for unit in ("rollout", "prompt"):
                for metric, values in stats.get(unit + "_strict_coverage_quantiles", {}).items():
                    quantile_rows.append([pair, unit, metric, *[values.get("p" + str(p)) for p in (0, 10, 25, 50, 75, 90, 100)]])
            micro = stats["global_micro"]
            for category, values in micro.get("categories", {}).items():
                category_rows.append([pair, category, fraction(values.get("strict_student_tokens"), values.get("student_tokens")), fraction(values.get("strict_bytes"), values.get("bytes"))])
            for kind, values in micro.get("interval_classes", {}).items():
                interval_rows.append([pair, kind, values.get("intervals"), values.get("student_tokens"), values.get("teacher_tokens"), values.get("bytes"), values.get("student_token_fraction"), values.get("byte_fraction")])
            numeric = micro.get("numeric_runs", {})
            numeric_rows.append([pair, numeric.get("count"), numeric.get("exact_run_segmentation_match_count"), numeric.get("exact_run_segmentation_match_fraction"), numeric.get("character_length_histogram"), micro.get("source"), micro.get("excluding_whitespace")])
        doc.table(["Pair", "Unit", "Coverage metric", "p0", "p10", "p25", "p50", "p75", "p90", "p100"], quantile_rows)
        doc.table(["Pair", "Category", "Strict student tokens", "Strict bytes"], category_rows)
        doc.table(["Pair", "Interval class", "Intervals", "S tokens", "Teacher tokens", "Bytes", "S-token fraction", "Byte fraction"], interval_rows)
        doc.table(["Pair", "Numeric runs", "Matched runs", "Match fraction", "Digit-length histogram", "Unicode/newlines", "Excluding whitespace"], numeric_rows)
        doc.json_detail(f"Quantiles p0/p10/p25/p50/p75/p90/p100: {corpus}", details, markdown=False)
        doc.json_detail(f"Category, numeric runs, Unicode, whitespace và coarse intervals: {corpus}",
                        {pair: stats["global_micro"] for pair, stats in audit.get("pairs", {}).items()}, markdown=False)
        doc.json_detail(f"Audit encoding failures: {corpus}", audit.get("failures", []))
    doc.heading("H2 raw và shape: kết quả và missing accounting")
    doc.paragraph(f"Bảng metric bên dưới tính trên {summary['valid_positions']} positions qua bước kiểm tra candidates. Missing=0 trong bảng này là conditional trên tập valid đó; {len(summary.get('failures', []))} failures trước metric vẫn được tính riêng trong tổng 1.000 positions đã score, không được biến thành giá trị 0 hoặc coi là không tồn tại.")
    doc.paragraph("X_raw = JS(Q,S) − JS(B,S), đơn vị nats. X_shape dùng teacher được entropy-match tới student bằng power-temperature trên shared hybrid word cells cộng OTHER. Đây không phải temperature tác động lên token logits; global temperature calibration chưa chạy. X>0 nghĩa là cross teacher xa student hơn baseline theo estimand ở corpus này; không tự chứng minh nguyên nhân là tokenizer hoặc teacher khác họ. OTHER giữ residual probability ngoài shared candidate cells; OTHER quá lớn làm phép đo phụ thuộc mạnh vào support chọn được.")
    doc.paragraph("X_byte_raw/X_byte_shape là coarse diagnostics trên 258 next-byte cells: raw bytes 0..255, EOS=256, nontext/padded=257. Chúng marginalize toàn bộ native next-token probability vào first-byte cells, nhưng không phải whole-word marginal. Không diễn giải word JS − byte JS thành causal decomposition hoặc phần trăm tokenizer contribution. X_candidate_conditional renormalize candidate mass và loại OTHER: chỉ là sensitivity check, không thay estimand primary.")
    doc.paragraph("Effective probability contract được stage report ghi: " + str(summary.get("effective_probability_contract", "not packaged")))
    metric_keys = [key for key in ("X_raw", "X_shape", "X_byte_raw", "X_byte_shape", "X_candidate_conditional") if isinstance(summary.get(key), dict)]
    doc.table(["Metric", "Vai trò", "Equal-prompt mean", "CI95", "Valid positions", "Missing positions", "Valid prompts"],
              [[key, "primary hybrid word+OTHER" if key in ("X_raw", "X_shape") else ("conditional sensitivity; excludes OTHER" if key == "X_candidate_conditional" else "coarse next-byte diagnostic"),
                summary[key].get("mean"), summary[key].get("ci95"), summary[key].get("valid_position_count"), summary[key].get("missing_position_count"), summary[key].get("valid_prompt_count")] for key in metric_keys])
    category_rows = []
    for key in metric_keys:
        for category, item in summary[key].get("categories", {}).items():
            category_rows.append([key, category, item.get("mean"), item.get("ci95"), item.get("valid_position_count"), item.get("missing_position_count"), item.get("valid_prompt_count")])
    doc.table(["Metric", "Category", "Mean", "CI95", "Valid positions", "Missing", "Prompt clusters"], category_rows)
    doc.table(["Support diagnostic", "Count"], [["Dropped candidates across valid rows", summary.get("dropped_candidates")]] + [[f"OTHER dominant: {a}", n] for a, n in summary.get("other_dominant_counts", {}).items()])
    doc.json_detail("Triangulation: cross disagrees khi baseline agrees", summary.get("cross_disagrees_when_baseline_agrees", {}))
    doc.paragraph("Top1 agreement dùng overlap của candidate tie sets và báo OTHER dominance riêng; count này không phải accuracy hoặc teacher correctness. Native realized-token logprob, top64 retained mass và entropy được lưu để triangulate. Native-token entropy giữa vocabularies khác nhau không trực tiếp tương đương; không dùng entropy đó làm thay thế shared-cell entropy matching.")
    score_flags, score_csv, method_counts, telemetry_rows = Counter(), [], Counter(), []
    candidate_file = read(run / "word-candidates.json", optional=True)
    candidate_map = {r["key"]: r.get("candidates", []) for r in (candidate_file or {}).get("positions", [])}
    for arm, data in scores.items():
        for item in data["rows"]:
            finite = [x for x in item["masses"] if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
            flags = {"missing_mass_count": sum(x is None for x in item["masses"]), "nonfinite_count": sum(x is not None and not math.isfinite(x) for x in item["masses"]),
                     "negative_mass_count": sum(x < 0 for x in finite), "sum_valid_canonical_mass": math.fsum(finite),
                     "mass_sum_above_one_tolerance": math.fsum(finite) > 1 + 1e-6}
            for key, value in flags.items():
                if key != "sum_valid_canonical_mass":
                    score_flags[f"{arm}.{key}"] += int(value)
            for reason in item.get("path_errors", []):
                if reason:
                    score_flags[f"{arm}.path_error.{reason}"] += 1
            for method in item.get("mass_methods", []):
                method_counts[f"{arm}.{method or 'missing_method'}"] += 1
            byte_values = item.get("byte_probs")
            if byte_values is not None:
                byte_sum = math.fsum(byte_values)
                flags.update(byte_cell_count=len(byte_values), byte_mass_sum=byte_sum, byte_mass_sum_abs_error=abs(byte_sum - 1),
                             byte_mass_sum_failed=abs(byte_sum - 1) > 1e-6, byte_negative_cells=sum(v < 0 for v in byte_values),
                             byte_wrong_cell_count=len(byte_values) != 258, byte_nonfinite_cells=sum(not math.isfinite(v) for v in byte_values),
                             byte_eos_mass=byte_values[256] if len(byte_values) > 256 else None,
                             byte_nontext_padded_mass=byte_values[257] if len(byte_values) > 257 else None)
                score_flags[f"{arm}.byte_mass_sum_failed"] += int(flags["byte_mass_sum_failed"])
                score_flags[f"{arm}.byte_negative_cells"] += flags["byte_negative_cells"]
                score_flags[f"{arm}.byte_wrong_cell_count"] += int(flags["byte_wrong_cell_count"])
                score_flags[f"{arm}.byte_nonfinite_cells"] += flags["byte_nonfinite_cells"]
            telemetry = item.get("token_telemetry", {})
            telemetry_rows.append({"arm": arm, "key": item["key"], **telemetry,
                                   "sampled_prefix_cached_uncached_word_js": item.get("sampled_prefix_cached_uncached_word_js"),
                                   "normalization_dtype": data.get("normalization_dtype"), "backbone_dtype": data.get("dtype")})
            score_csv.append({"arm": arm, "key": item["key"], **flags, "candidates": candidate_map.get(item["key"]),
                              "masses": item["masses"], "path_errors": item.get("path_errors"), "native_candidate_paths": item.get("native_candidate_paths"),
                              "mass_methods": item.get("mass_methods"), "byte_probs": byte_values,
                              "sampled_prefix_cached_uncached_word_js": item.get("sampled_prefix_cached_uncached_word_js"),
                              **flatten({"token_telemetry": telemetry}), "wall_seconds": item.get("wall_seconds")})
    doc.table(["Scoring flag", "Count"], sorted(score_flags.items()))
    doc.table(["Mass method / arm", "Candidate cells"], sorted(method_counts.items()))
    telemetry_summary = []
    for arm in ARMS:
        records = [r for r in telemetry_rows if r["arm"] == arm]
        check = [r["sampled_prefix_cached_uncached_word_js"] for r in records if r["sampled_prefix_cached_uncached_word_js"] is not None]
        fields = []
        for key in ("logprob_realized_token", "top64_retained_mass", "native_token_entropy_nats"):
            values = [r[key] for r in records if r.get(key) is not None]
            fields.extend([min(values) if values else None, math.fsum(values) / len(values) if values else None, max(values) if values else None])
        byte_errors = [r["byte_mass_sum_abs_error"] for r in score_csv if r["arm"] == arm and r.get("byte_mass_sum_abs_error") is not None]
        telemetry_summary.append([arm, scores[arm].get("dtype"), scores[arm].get("normalization_dtype"), len(check), max(check) if check else None,
                                  max(byte_errors) if byte_errors else None, *fields])
    doc.table(["Arm", "Backbone", "Normalization", "Sampled-prefix checks", "Max cached/uncached word JS", "Max byte-mass sum error",
               "Realized LP min", "LP mean", "LP max", "Top64 mass min", "Mass mean", "Mass max", "Native entropy min", "Entropy mean", "Entropy max"], telemetry_summary)
    metrics_file = read(run / "word-metrics.json")
    metrics = metrics_file["rows"]
    shape_reasons = Counter()
    for row in metrics:
        for teacher, match in row.get("shape_calibration", {}).items():
            if not match.get("feasible"):
                shape_reasons[teacher + ":" + str(match.get("reason"))] += 1
    doc.table(["Shape infeasible reason", "Count in packaged metric rows"], sorted(shape_reasons.items()))
    doc.json_detail("Position failures", summary.get("failures", []))
    examples = summary.get("highest_X_shape_examples", [])[:30]
    doc.heading("30 vị trí X_shape cao nhất: để kiểm tra định tính")
    doc.paragraph("Các ví dụ được chọn theo chính metric đã đo, có selection bias. Chúng giúp tìm failure modes; không thay cho đánh giá tổng thể. Context và candidate strings được escape khi hiển thị.")
    doc.table(["Key", "Category", "Reference continuation", "X_raw", "X_shape", "Candidates kept", "Dropped", "Context tail"], [[r["key"], r.get("category"), r.get("reference_word"), r.get("X_raw"), r.get("X_shape"), len(r.get("candidates", [])), r.get("dropped_candidates"), r.get("prefix_text", "")[-160:]] for r in examples])
    for row in examples:
        doc.json_detail("Chi tiết example " + row["key"], {k: row.get(k) for k in ("prompt_id", "response_id", "byte_start", "category", "prefix_text", "reference_word", "candidates", "masses", "probabilities", "raw", "shape", "shape_calibration", "top_cells", "X_raw", "X_shape", "X_byte_raw", "X_byte_shape", "X_candidate_conditional", "cross_disagrees_when_baseline_agrees", "native_token_telemetry")}, markdown=False)
    doc.heading("Bảng đủ 200 responses và bàn giao trace")
    doc.paragraph("Không chọn riêng responses đẹp. Bảng liệt kê tất cả IDs. NA ở auxiliary shared-vocabulary/eligible fields của canonical-control rows nghĩa là field đó chưa tính cho control; không đồng nghĩa pair audit thất bại. Encoding failures và successful row counts được báo riêng. CSV chứa mọi denominator và metric không rút gọn; HTML có details riêng từng prompt/response. Trace JSON chứa per-student-token byte offsets, teacher indices, strict/context alignment, categories và minimal coarse intervals.")
    by_pair = {pair: {str(r["rollout_id"]): r for r in stats["per_rollout"]} for pair, stats in audits["rollout"].get("pairs", {}).items()}
    table_rows, rollout_csv = [], []
    grouped = defaultdict(list)
    index_map = {str(r["rollout_id"]): r for r in audits["rollout"]["index"]}
    for row in rollouts:
        rid = row["id"]
        stats = {a: by_pair.get("new-" + a, {}).get(rid, {}) for a in ("B", "Q")}
        cells = []
        for a in ("B", "Q"):
            c = stats[a].get("counts", {})
            cells.extend([fraction(c.get("strict_pairs"), c.get("student_tokens")), fraction(c.get("student_tokens_available_teacher_vocab"), c.get("student_tokens")),
                          fraction(c.get("strict_bytes"), c.get("bytes")), fraction(c.get("strict_eligible_student_tokens"), c.get("eligible_student_tokens"))])
        table_rows.append([rid, row.get("source"), len(row.get("content_ids", [])), row.get("finish_reason"), *cells])
        rollout_csv.append({"prompt_id": row["prompt_id"], "response_id": rid, "source": row.get("source"), "finish_reason": row.get("finish_reason"),
                            "sampled_content_tokens": len(row.get("content_ids", [])), "sampling_seed": row.get("sampling_seed"), "batch_offset": row.get("batch_offset"),
                            "prompt": row.get("prompt"), "text": row.get("text"), **flatten({a: stats[a] for a in ("B", "Q")})})
        grouped[row["prompt_id"]].append(row)
    doc.table(["Response", "Dataset", "S tokens", "Finish", "B strict S", "B shared-vocab usage", "B strict bytes", "B strict eligible", "Q strict S", "Q shared-vocab usage", "Q strict bytes", "Q strict eligible"], table_rows)
    per_prompt_maps = {arm: {str(r["prompt_id"]): r for r in audits["rollout"].get("pairs", {}).get("new-" + arm, {}).get("per_prompt", [])} for arm in ("B", "Q")}
    prompt_rows = []
    for prompt, members in sorted(grouped.items()):
        cells = []
        for arm in ("B", "Q"):
            c = per_prompt_maps[arm].get(str(prompt), {}).get("micro", {}).get("counts", {})
            cells.extend([fraction(c.get("strict_pairs"), c.get("student_tokens")), fraction(c.get("student_tokens_available_teacher_vocab"), c.get("student_tokens")), fraction(c.get("strict_bytes"), c.get("bytes")), fraction(c.get("strict_eligible_student_tokens"), c.get("eligible_student_tokens"))])
        prompt_rows.append([prompt, len(members), *cells])
    doc.heading("Bảng đủ 100 prompts: gộp counts của hai responses")
    doc.table(["Prompt", "Responses", "B strict S", "B shared-vocab usage", "B strict bytes", "B strict eligible", "Q strict S", "Q shared-vocab usage", "Q strict bytes", "Q strict eligible"], prompt_rows)
    trace_count = 0
    canonical_count = 0
    for prompt, members in sorted(grouped.items()):
        details = ['<details><summary>' + html.escape(prompt) + '</summary><p>' + html.escape(str(members[0].get("prompt", ""))) + "</p>"]
        for row in members:
            rid = row["id"]
            trace = local_trace(run, index_map.get(rid, {}))
            payload = read(trace) if trace else None
            diagnostic = {"generation": {k: row.get(k) for k in ("response_index", "finish_reason", "sampling_seed", "batch_offset", "backend")},
                          "pairs": {pair: by_pair.get(pair, {}).get(rid) for pair in by_pair}}
            if payload:
                trace_count += 1
                canonical_count += int(bool(payload.get("student_path_is_canonical")))
                diagnostic["category_numeric_interval_details"] = {pair: {k: audit.get(k) for k in ("counts", "excluding_whitespace", "categories", "numeric_runs", "interval_classes", "source")}
                                                                  for pair, audit in payload.get("pairs", {}).items()}
                diagnostic["student_path_is_canonical"] = payload.get("student_path_is_canonical")
                diagnostic["baseline_content_policy"] = payload.get("baseline_content_policy")
                if payload.get("native_vs_canonical_student"):
                    native_control = payload["native_vs_canonical_student"]
                    diagnostic["native_vs_canonical_student"] = {k: native_control.get(k) for k in ("counts", "coverage", "boundaries", "interval_classes")}
                    alignment_csv.append({"corpus": "rollout", "pair": "native-vs-canonical-S", "prompt_id": prompt, "rollout_id": rid,
                                          "student_path_is_canonical": payload.get("student_path_is_canonical"),
                                          **flatten(diagnostic["native_vs_canonical_student"])})
                for pair, audit in payload.get("pairs", {}).items():
                    alignment_csv.append({"corpus": "rollout", "pair": pair, "prompt_id": prompt, "rollout_id": rid, "record_kind": "trace_enrichment",
                                          **flatten({k: audit.get(k, {}) for k in ("counts", "coverage", "excluding_whitespace", "boundaries", "categories", "numeric_runs", "interval_classes", "operations", "source", "lexical_occurrence_coverage")})})
            href = quote("../" + trace.relative_to(run).as_posix(), safe="/") if trace else None
            details.append('<details><summary>' + html.escape(rid) + '</summary><pre>' + html.escape(row.get("text", "")) + '</pre><pre>' + html.escape(json.dumps(diagnostic, ensure_ascii=False, indent=2)) + '</pre>' +
                           ('<a href="' + html.escape(href, quote=True) + '">Full per-token trace JSON</a>' if href else '<p>Trace file chưa được đóng gói; xem aggregate JSON.</p>') + '</details>')
        details.append("</details>")
        doc.body.append("".join(details))
    doc.paragraph(f"Full response trace files đã đóng gói: {trace_count}/200. Thiếu trace files không được suy diễn là audit match=0; aggregate audit vẫn chỉ phản ánh các encodings thành công được đếm rõ.")
    doc.paragraph(f"Native student path trùng canonical retokenization: {canonical_count}/{trace_count} responses có trace. Đây là lý do giữ cả native và canonical controls. Full interval operations được giữ trong trace JSON/CSV thay vì lặp lại trong HTML.")
    csv_write(delivery / "responses.csv", rollout_csv)
    csv_write(delivery / "alignment-per-response.csv", alignment_csv)
    csv_write(delivery / "alignment-per-prompt.csv", prompt_csv)
    csv_write(delivery / "scoring-path-flags.csv", score_csv)
    csv_write(delivery / "native-token-telemetry.csv", telemetry_rows)
    csv_write(delivery / "word-position-metrics.csv", [flatten(row) for row in metrics])
    doc.heading("Chi phí và source checkpoint")
    billing = read(run / "billing-observed.json", optional=True)
    if billing is None:
        doc.paragraph("Chưa đóng gói billing-observed.json. Không gán estimate thành actual charge; Modal billing có thể trễ. Absolute cap được người dùng cho phép là $10 cho lượt này, không phải bằng chứng đã chi đúng $10 hoặc quyền tự mở rộng compute.")
    else:
        doc.paragraph("Bên dưới là billing observation đóng gói; cần giữ timestamp và phân biệt charged/observed usage với estimate/reserve. Billing có thể cập nhật trễ sau khi app kết thúc.")
        doc.json_detail("Billing observation", billing)
    patch = run / "source-staged.patch"
    if patch.exists():
        doc.paragraph("Source-staged.patch giữ scoped source changes. Theo trạng thái bàn giao, local signed commit bị chặn bởi GPG timeout; patch là snapshot phục hồi source, không phải xác nhận đã commit/push. Không bypass signing và không coi patch là remote Git backup.")
        doc.link("Scoped source patch", "../source-staged.patch")
    doc.heading("Bàn giao để chạy lớn tại công ty")
    doc.paragraph("Không có lệnh submit công ty tự động trong báo cáo. Trước khi scaleup, chọn đúng GPU/node đã được giao; giữ nguyên model repos/revisions, verified weight hashes, tokenizer files, native templates, stop IDs, runtime digest và probability contract. Reproduce một fixture bằng HF eager FP32 trước. Nếu đổi sang SGLang hoặc batch scoring để tăng throughput, qualify generation/cache/logprob/word probabilities trên cùng fixtures rồi mới so sánh khoa học; throughput Gemma trước đây không phải benchmark cho bộ Llama/Qwen2 này.")
    doc.paragraph("Tăng số prompts giúp mở rộng independent clusters. Tăng positions trong 100 prompts chỉ làm đo kỹ hơn corpus đó. Bootstrap theo prompt; báo raw và shape missing denominators. Nếu làm calibration hoặc thay event/candidate support, đó là protocol revision cần run ID mới và provenance mới. Không trộn corpus on-policy từ model khác vào paired old/new tokenizer comparison; dùng cùng raw texts và common successful response IDs.")
    doc.code("# Chỉ đọc/export local bằng Python có sẵn; không tải dependency/model.\npython experiments/diagnostics/h2_word_report_export.py --run-dir /path/to/extracted/run\n\n# Tại công ty, sau khi đã được cấp đúng node/GPU và copy đúng assets/runtime:\n# export H2_RUN_ID=<new-company-run-id>\n# export PYTHONPATH=/path/to/staged/diagnostics\n# /path/to/existing/runtime/bin/python /path/to/staged/diagnostics/h2_word_pilot.py word-qualify S\n# Lặp qualification cho B/Q; inspect pass/revision/config hashes trước stage tiếp theo.\n# /results/<run-id>/assets.json và word-prompts.json phải trỏ đúng assets đã kiểm chứng.")
    doc.paragraph("Điểm cần đọc cùng kết quả: finite candidate union; canonical path không gom mọi tokenization; boundary mask thiếu incomplete UTF-8 fragments được nêu là proxy limitation; native templates khác nhau; chưa có teacher quality/capacity/placebo controls; shape entropy match có thể bất khả thi do ties/zero support; global calibration chưa chạy. Pilot này cung cấp tín hiệu và failure diagnostics để quyết định thiết kế lượt lớn, chưa chứng minh causal tokenizer effect.")
    doc.heading("Tệp dữ liệu đầy đủ")
    for name in ("responses.csv", "alignment-per-response.csv", "alignment-per-prompt.csv", "scoring-path-flags.csv", "native-token-telemetry.csv", "word-position-metrics.csv"):
        doc.link(name, name)
    for name in ("word-summary.json", "word-metrics.json", "tokenizer-audit-reference.json", "tokenizer-audit-rollout.json", "vocabulary-overlap.json", "assets.json", "historical-assets.json"):
        if (run / name).exists():
            doc.link(name, "../" + name)
    for name in ("H2_COMPANY_HANDOFF.md", "task-source-snapshot.zip", "apps-terminal-status.json"):
        if (run / name).exists(): doc.link(name, "../" + name)
    doc.save(delivery)
    manifest = {"run_id": summary["run_id"], "completeness": {"prompts": 100, "responses": 200, "positions_per_model": 1000, "valid_positions": summary["valid_positions"], "trace_files_packaged": trace_count},
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in sorted(delivery.iterdir()) if path.is_file() and path.name != "delivery-manifest.json"}}
    (delivery / "delivery-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    delivery = generate(args.run_dir.resolve())
    print(str(delivery / "REPORT.html"))


if __name__ == "__main__":
    main()
