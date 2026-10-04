"""一页式 HTML 训练报告渲染器的测试：固定输入（mini metrics dict）下产出合法 HTML。

不要求像素级；断言：章节标题齐全、无未替换占位符、base64 图片内嵌、
缺产物时如实降级（渲染“—”而非虚构数字）、真实 artifacts 可渲染出文档化指标。
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from creditrisk.html_report import (
    extract_readme_limitations,
    render_html_report,
    rerender_from_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 标准 1×1 透明 PNG（仅用于验证 base64 内嵌机制，不参与任何指标）
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

SECTION_TITLES = ["① 数据概况", "② 特征工程摘要", "③ 模型对比", "④ 概率校准",
                  "⑤ 阈值-业务扫描", "⑥ 公平性审计", "⑦ 已知限制"]

README_MINI = """# demo

## 已知限制

1. **限制一**：规模很小（1000 条），指标不外推。
2. **合规视角**：未做去偏处理，详见
   [reports/fairness.md](reports/fairness.md)。

## License

MIT
"""


@pytest.fixture()
def mini_metrics():
    """结构完整但数值极小的 metrics dict（与 metrics.json 同 schema）。"""
    return {
        "data_source": "openml:credit-g:v1",
        "train_size": 8,
        "test_size": 2,
        "bad_rate_train": 0.5,
        "bad_rate_test": 0.5,
        "experiments": [
            {"model": "logistic_regression", "strategy": "weight",
             "cv_auc_mean": 0.75, "cv_auc_std": 0.05, "cv_ks_mean": 0.40,
             "cv_ks_std": 0.06, "test_auc": 0.80, "test_ks": 0.50, "test_gini": 0.60},
            {"model": "lightgbm", "strategy": "weight",
             "cv_auc_mean": 0.70, "cv_auc_std": 0.04, "cv_ks_mean": 0.35,
             "cv_ks_std": 0.05, "test_auc": 0.72, "test_ks": 0.40, "test_gini": 0.44},
        ],
        "selected": {"model": "logistic_regression", "strategy": "weight"},
        "final_test_metrics": {"auc": 0.80, "ks": 0.50, "gini": 0.60, "n": 2, "bad_rate": 0.5},
        "calibration": {
            "n_bins": 10,
            "test_n": 2,
            "deployed": {"model": "lr+weight (before)", "brier": 0.20, "ece": 0.10,
                         "mean_predicted_pd": 0.60, "observed_bad_rate": 0.5, "reliability": []},
            "repair": {
                "method": "sigmoid",
                "protocol": "oof held-out",
                "selection": {"selection_fit_n": 4, "selection_valid_n": 4, "n_bins": 10,
                              "ece_valid_raw": 0.12, "brier_valid_raw": 0.20,
                              "ece_valid_sigmoid": 0.07, "brier_valid_sigmoid": 0.18,
                              "ece_valid_isotonic": 0.08, "brier_valid_isotonic": 0.19,
                              "method": "sigmoid"},
                "test": {"n_bins": 10, "test_n": 2,
                         "test_ece_before": 0.10, "test_ece_after": 0.05,
                         "test_brier_before": 0.20, "test_brier_after": 0.17,
                         "test_mean_pd_before": 0.60, "test_mean_pd_after": 0.51,
                         "test_auc_before": 0.80, "test_auc_after": 0.80,
                         "observed_bad_rate": 0.5},
                "sigmoid_coefficients": {"a": 0.9, "b": -0.5},
                "model": "lr+weight (after)", "brier": 0.17, "ece": 0.05,
                "mean_predicted_pd": 0.51, "observed_bad_rate": 0.5, "reliability": [],
            },
            "reference": {"model": "lr+none (reference)", "brier": 0.18, "ece": 0.04,
                          "mean_predicted_pd": 0.50, "observed_bad_rate": 0.5, "reliability": []},
        },
        "threshold_tradeoff": {
            "decision_rule": "approve if PD < threshold",
            "test_n": 2,
            "grid_quantiles": [0.5],
            "table": [{"threshold": 0.5, "n_approved": 1, "approval_rate": 0.5,
                       "bad_rate_approved": 0.0, "n_rejected": 1, "bad_rate_rejected": 1.0}],
            "targets": [
                {"target_approval_rate": 0.5, "threshold": 0.5, "n_approved": 1,
                 "approval_rate": 0.5, "bad_rate_approved": 0.0,
                 "n_rejected": 1, "bad_rate_rejected": 1.0},
                {"target_approval_rate": 0.9, "threshold": 0.9, "n_approved": 2,
                 "approval_rate": 1.0, "bad_rate_approved": 0.5,
                 "n_rejected": 0, "bad_rate_rejected": None},
            ],
        },
        "fairness": {
            "group_column": "personal_status",
            "decision_rule": "approve if PD < threshold",
            "threshold": 0.5, "n_total": 2,
            "overall_bad_rate": 0.5, "overall_selection_rate": 0.5,
            "groups": {
                "gA": {"n": 1, "bad_rate": 0.0, "selection_rate": 1.0,
                       "tpr_good": 1.0, "auc": 1.0},
                "gB": {"n": 1, "bad_rate": 1.0, "selection_rate": 0.0,
                       "tpr_good": float("nan"), "auc": float("nan")},
            },
            "demographic_parity_gap": 1.0,
            "equal_opportunity_gap": 0.0,
        },
        "global_importance_method": "shap_mean_abs(lightgbm+weight)",
        "deployed_model_local_explanation": "coef_x_woe_deviation",
        "local_example": {"pd": 0.5, "y_true": 0,
                          "top_features": [{"feature": "f1", "contribution": 0.1}]},
        "shap_local_example_lightgbm": None,
    }


def _write_pngs(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "calibration_curve.png").write_bytes(PNG_1PX)
    (reports_dir / "threshold_tradeoff.png").write_bytes(PNG_1PX)


def test_render_mini_metrics_produces_valid_html_with_sections_and_images(tmp_path, mini_metrics):
    reports_dir = tmp_path / "reports"
    _write_pngs(reports_dir)
    (tmp_path / "README.md").write_text(README_MINI, encoding="utf-8")
    out = tmp_path / "report.html"
    result = render_html_report(mini_metrics, reports_dir, out,
                                readme_path=tmp_path / "README.md")
    assert result == out and out.is_file()
    html = out.read_text(encoding="utf-8")

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    for title in SECTION_TITLES:
        assert title in html, f"缺少章节：{title}"

    # 关键实测数字按格式化规则出现（AUC 4 位小数、校准前后 ECE）
    assert "0.8000" in html
    assert "0.1000" in html and "0.0500" in html
    # sigmoid 系数与择优标记
    assert "sigmoid" in html and "✔ 选中" in html
    # 已知限制来自传入 README，链接剥掉只留文字
    assert "限制一" in html and "合规视角" in html
    assert "](reports/fairness.md)" not in html

    # 两张图均 base64 内嵌
    assert html.count('src="data:image/png;base64,') == 2

    # 无未替换占位符、无脚本、无外网资源
    assert "{{" not in html and "}}" not in html
    assert "%(" not in html and "TODO" not in html.upper()
    assert "<script" not in html.lower()
    assert "http://" not in html and "https://" not in html


def test_render_with_missing_inputs_degrades_honestly(tmp_path, mini_metrics):
    """无图、无 meta、无 README、若干键缺失 → 章节齐全、渲染“—”、不虚构数字。"""
    skeleton = {k: mini_metrics[k] for k in
                ("data_source", "train_size", "test_size", "bad_rate_train",
                 "bad_rate_test", "final_test_metrics")}
    out = tmp_path / "report.html"
    render_html_report(skeleton, tmp_path / "no_such_reports", out,
                       meta=None, readme_path=None)
    html = out.read_text(encoding="utf-8")

    for title in SECTION_TITLES:
        assert title in html
    assert html.count("data:image/png") == 0          # 缺图如实省略，不放占位图
    assert "—" in html                                # 缺失数值渲染为“—”
    assert "nan" not in html.lower() and "None" not in html
    assert "已知限制" in html                          # 降级提示仍指回 README
    assert "synthetic_fallback" not in html           # 不虚构数据来源
    assert "稳定性表略" in html                       # 缺 PSI 产物如实省略整块


def test_render_psi_table_from_reports_dir(tmp_path, mini_metrics):
    """reports/ 里有 PSI CSV 时，②节渲染 top-8 稳定性表与判定文案。"""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    pd.DataFrame([
        {"feature": "age", "psi": 0.0123, "level": "stable"},
        {"feature": "credit_amount", "psi": 0.31, "level": "significant"},
    ]).to_csv(reports_dir / "psi_train_vs_test.csv", index=False)
    out = tmp_path / "report.html"
    render_html_report(mini_metrics, reports_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "特征稳定性（自实现 PSI，train vs test）" in html
    assert "特征（PSI 最高前 8）" in html
    assert "age" in html and "credit_amount" in html
    assert "稳定（&lt;0.1）" in html and "显著漂移（&gt;0.25）" in html
    assert "0.3100" in html and "0.0123" in html


def test_render_scorecard_section_when_metrics_present(tmp_path, mini_metrics):
    """metrics 带 scorecard 时⑤节渲染标度表；缺 key 的降级路径由其余用例隐式覆盖。"""
    mini_metrics["scorecard"] = {
        "pdo": 20.0, "base_odds": 50.0, "base_score": 600.0,
        "n": 2, "min": 512.3, "max": 640.7, "mean": 576.5, "median": 576.5,
    }
    out = tmp_path / "report.html"
    render_html_report(mini_metrics, tmp_path / "no_such_reports", out)
    html = out.read_text(encoding="utf-8")

    assert "信用分标度（Scorecard，PD 的单调刻度变换）" in html
    assert "640.7" in html and "512.3" in html
    assert "50 : 1 → 600 分" in html
    assert "不改变排序、不改变阈值决策" in html


def test_extract_readme_limitations_parses_numbered_items_and_merges_wrapped_lines():
    items = extract_readme_limitations(README_MINI)
    assert len(items) == 2
    assert "限制一" in items[0]
    assert "合规视角" in items[1]
    assert "reports/fairness.md" in items[1]          # 链接文字保留（剥离由渲染层做）
    # 章节之后的内容不混入
    assert "MIT" not in items[-1]
    assert extract_readme_limitations("# 没有该章节\n正文") == []


def test_rerender_from_real_artifacts(tmp_path):
    """真实 artifacts/metrics.json（同构 schema）可渲染出 README 文档化的指标。"""
    out = tmp_path / "report.html"
    rerender_from_artifacts(PROJECT_ROOT, out_path=out)
    html = out.read_text(encoding="utf-8")

    for title in SECTION_TITLES:
        assert title in html
    # README「真实评估指标」钉住的数字必须原样出现（0.8013/0.5119/ECE 0.1420→0.0681）
    assert "0.8013" in html and "0.5119" in html
    assert "0.1420" in html and "0.0681" in html
    # 公平性差距与阈值关键读数
    assert "0.336" in html and "0.321" in html
    assert "16.4%" in html and "相对下降 34%" in html
    # 已知限制与 README 同步（首条标题）
    assert "数据规模与年代" in html
    # ②节渲染真实 PSI 稳定性表（reports/psi_train_vs_test.csv 已提交）
    assert "特征稳定性（自实现 PSI，train vs test）" in html
    assert "特征（PSI 最高前 8）" in html
    # 自包含：两张真实图表内嵌，无外链
    assert html.count('src="data:image/png;base64,') == 2
    assert "https://" not in html and "<script" not in html.lower()
