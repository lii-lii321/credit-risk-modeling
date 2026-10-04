"""一页式 HTML 训练报告渲染器（单文件自包含、纯静态、零 JS、零外网资源）。

把训练全链路聚合为单页 reports/report.html，章节顺序固定：
①数据概况 ②特征工程摘要（WOE/IV + PSI 稳定性表） ③模型对比（LR vs LightGBM）
④校准：发现→修复→复测 ⑤阈值-业务扫描 ⑥公平性审计 ⑦已知限制（从 README 同步）。

数据纪律（军规）：
- 页面上每个数字只能来自调用方传入的 metrics（即 artifacts/metrics.json 的内容）、
  artifacts/model_meta.json、reports/ 下既有 CSV/PNG 产物与代码常量（seed、分箱数等），
  绝不新造数；缺失键渲染为“—”，不虚构 0。
- 已知限制清单直接解析 README「已知限制」章节（单一事实来源），解析不到则如实提示。
- 自包含约束：CSS 内联、PNG 转 base64 内嵌，不引用任何 CDN/字体/脚本外链。

目标场景：面试官 3 分钟内在单页看懂数据 → 特征 → 模型 → 校准 → 业务 → 公平性全链路。
"""
from __future__ import annotations

import base64
import html as _html
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
from .fairness import fairness_conclusion

_MODEL_LABELS = {
    "logistic_regression": "LogisticRegression",
    "lightgbm": "LightGBM",
}
_SOURCE_LABELS = {
    "openml:credit-g:v1": "OpenML credit-g v1（真实公开数据）",
    "synthetic_fallback": "合成降级数据（OpenML 拉取失败，以下指标不代表真实数据表现）",
}
_STRENGTH_LABELS = {
    "useless": "无信息（IV<0.02）",
    "weak": "弱（0.02–0.1）",
    "medium": "中（0.1–0.3）",
    "strong": "强（0.3–0.5）",
    "suspicious": "可疑（>0.5，需人工复核）",
}
_PSI_LEVELS = {
    "stable": "稳定（<0.1）",
    "moderate": "中度漂移（0.1–0.25）",
    "significant": "显著漂移（>0.25）",
}

# ---------------------------------------------------------------- primitives


def _short_model(label) -> str:
    """metrics 里的长模型标签（如 "logistic_regression+weight (deployed, before calibration)"）
    截为展示用短标签 "LogisticRegression + weight"。只裁剪口径后缀，不触碰任何数字。"""
    base = str(label).split(" (")[0]
    model, _, strategy = base.partition("+")
    pretty = _MODEL_LABELS.get(model, model)
    return f"{pretty} + {strategy}" if strategy else pretty


def _esc(value) -> str:
    return _html.escape(str(value), quote=True)


def _num(value, spec: str | int = ".4f") -> str:
    """格式化数字；缺失/NaN/None 一律渲染为“—”（不虚构 0）。

    spec 可传格式串（".4f"）或小数位数（3 → ".3f"）。
    """
    if isinstance(spec, int):
        spec = f".{spec}f"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return format(v, spec)


def _pct(value, digits: int = 1) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return format(v * 100.0, f".{digits}f") + "%"


def _signed(value, spec: str = ".3f") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return ("+" if v >= 0 else "−") + format(abs(v), spec)


def _sub(a, b) -> float | None:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x - y


def _img_src(path: Path) -> str | None:
    """PNG → base64 data URI；文件不存在返回 None（如实省略，不放占位图）。"""
    path = Path(path)
    if not path.is_file():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _inline_md(text: str) -> str:
    """受限 inline markdown：剥掉链接只留文字，**粗体**与 `代码` 转 HTML，其余转义。"""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], highlight: int = -1) -> str:
    """数据表；cells 由调用方预先转义（允许内联 <strong>/<code>）。highlight<0 不高亮。"""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cells = "".join(f"<td>{c}</td>" for c in row)
        cls = ' class="hl"' if i == highlight else ""
        body.append(f"<tr{cls}>{cells}</tr>")
    return (
        f'<table class="data"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


# ------------------------------------------------------------ README 限制解析


def extract_readme_limitations(readme_text: str) -> list[str]:
    """解析 README「已知限制」章节的编号条目（跨行合并为一条）；找不到返回空表。"""
    lines = readme_text.splitlines()
    start = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#") and "已知限制" in s:
            start = i + 1
            break
    if start < 0:
        return []
    items: list[str] = []
    current: str | None = None
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("#"):  # 下一个章节标题 → 结束
            break
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            if current:
                items.append(current)
            current = m.group(2)
        elif current is not None and s:
            current += " " + s
    if current:
        items.append(current)
    return items


# ------------------------------------------------------------------ sections


def _section_data(metrics: dict) -> str:
    source_raw = metrics.get("data_source", "")
    source_label = _SOURCE_LABELS.get(str(source_raw), _esc(source_raw) or "—")
    train_n = int(metrics.get("train_size") or 0)
    test_n = int(metrics.get("test_size") or 0)
    total = train_n + test_n
    bad_train = metrics.get("bad_rate_train")
    bad_test = metrics.get("bad_rate_test")
    ratio = "—"
    if isinstance(bad_train, (int, float)) and 0 < float(bad_train) < 1:
        ratio = f"{(1 - float(bad_train)) / float(bad_train):.1f} : 1"

    rows = [
        ("数据来源", _esc(source_label)),
        ("样本量", f"{total} 行 × 20 特征" if total else "—"),
        ("切分", f"分层 train/test = {train_n}/{test_n}"
                 f"（seed={RANDOM_STATE}，test 占 {TEST_SIZE:.0%}）"),
        ("类别比（正常:违约）", f"{_pct(_sub(1.0, bad_train))} : {_pct(bad_train)}（约 {ratio}）"),
        ("坏样本率", f"train {_pct(bad_train)} / test {_pct(bad_test)}"),
    ]
    if source_raw == "openml:credit-g:v1":
        rows.append(("缺失值", "0（EDA 实测，见 reports/eda_summary.md）"))
    kv = "".join(
        f'<tr><th scope="row">{k}</th><td>{v}</td></tr>' for k, v in rows
    )
    warn = ""
    if source_raw == "synthetic_fallback":
        warn = ('<p class="warn">警告：OpenML 拉取失败，本次训练使用 schema 一致的合成降级数据，'
                "页面上所有指标不代表真实数据表现。</p>")
    return f'<h2 id="s1">① 数据概况</h2>{warn}<table class="kv">{kv}</table>'


def _section_features(reports_dir: Path) -> str:
    parts = [
        '<h2 id="s2">② 特征工程摘要（自实现 WOE/IV）</h2>',
        '<p>自实现 WOE 分位数分箱编码：数值特征等频 10 箱、类别特征按取值分箱；'
        "加法平滑 α=0.5（防零频次 ln(0)）；缺失/未见值映射中性 0，不引入人为方向性。"
        "WOE&gt;0 表示该箱违约占比高于总体。</p>",
    ]
    iv_path = reports_dir / "iv_table.csv"
    if iv_path.is_file():
        iv = pd.read_csv(iv_path)
        rows = [
            (_esc(r["feature"]), _num(r["iv"]),
             _esc(_STRENGTH_LABELS.get(str(r["strength"]), str(r["strength"]))))
            for _, r in iv.head(8).iterrows()
        ]
        parts.append(_table(["特征（IV 前 8）", "IV", "经验评级"], rows))
        if (iv["strength"].astype(str) == "suspicious").any():
            top = str(iv.iloc[0]["feature"])
            parts.append(
                f'<p class="note"><strong>如实记录</strong>：{top} IV&gt;0.5 被标记为 suspicious'
                "（潜在信息泄漏式强变量，评分卡实务中需人工复核，本项目保留并标注，不静默删除）。</p>"
            )
        parts.append('<p class="note">完整 20 特征 IV 表见 reports/iv_table.csv。</p>')
    else:
        parts.append('<p class="note">reports/iv_table.csv 缺失，IV 表略。</p>')

    psi_path = reports_dir / "psi_train_vs_test.csv"
    if psi_path.is_file():
        psi = pd.read_csv(psi_path)
        mx = float(psi["psi"].max())
        verdict = "全部 &lt;0.1，同分布随机切分符合预期" if mx < 0.1 else "存在漂移特征，上线前需复核"
        rows = [
            (_esc(r["feature"]), _num(r["psi"]),
             _esc(_PSI_LEVELS.get(str(r["level"]), str(r["level"]))))
            for _, r in psi.head(8).iterrows()
        ]
        parts.append(
            f'<p><strong>特征稳定性（自实现 PSI，train vs test）</strong>：'
            f"最大 PSI {_num(mx)} → {verdict}。</p>"
        )
        parts.append(_table(["特征（PSI 最高前 8）", "PSI", "经验评级"], rows))
        parts.append(
            '<p class="note">PSI = Σ (actual%−expected%)·ln(actual%/expected%)：'
            "数值特征按 expected 分位数等频 10 箱、类别特征按取值并集对齐，零比例以 ε=1e-4 截断；"
            "含缺失值独立成箱。完整 20 特征表与人为漂移对照实验（age+15 / credit_amount×1.5 / "
            "duration+12，应检出显著漂移）见 reports/psi_train_vs_test.csv 与 "
            "reports/stability_report.md。</p>"
        )
    else:
        parts.append('<p class="note">reports/psi_train_vs_test.csv 缺失，稳定性表略。</p>')
    return "\n".join(parts)


def _section_models(metrics: dict, reports_dir: Path) -> str:
    parts = ['<h2 id="s3">③ 模型对比（LR vs LightGBM × none/weight/SMOTE）</h2>']
    experiments = metrics.get("experiments") or []
    selected = metrics.get("selected") or {}
    rows, highlight = [], -1
    for i, e in enumerate(experiments):
        model = str(e.get("model", ""))
        strategy = str(e.get("strategy", ""))
        if selected and model == selected.get("model") and strategy == selected.get("strategy"):
            highlight = i
        model_txt = _esc(_MODEL_LABELS.get(model, model))
        if i == highlight:
            model_txt += "（部署）"
        rows.append((
            model_txt,
            _esc(strategy),
            f'{_num(e.get("cv_auc_mean"))} ± {_num(e.get("cv_auc_std"))}',
            f'{_num(e.get("cv_ks_mean"))} ± {_num(e.get("cv_ks_std"))}',
            f"<strong>{_num(e.get('test_auc'))}</strong>" if i == highlight else _num(e.get("test_auc")),
            _num(e.get("test_ks")),
        ))
    if rows:
        parts.append(_table(
            ["model", "strategy", "CV AUC", "CV KS", "test AUC", "test KS"],
            rows, highlight,
        ))
        parts.append('<p class="note">5 折分层 CV（均值 ± 标准差）+ 独立测试集；'
                     "完整矩阵见 reports/model_comparison.csv。</p>")

    final = metrics.get("final_test_metrics") or {}
    sel_model = _MODEL_LABELS.get(str(selected.get("model")), selected.get("model", "—"))
    parts.append(
        f'<p><strong>最优组合：{_esc(sel_model)} + {_esc(selected.get("strategy", "—"))}</strong>'
        f"（部署模型）。测试集（n={_num(final.get('n'), '.0f')}）："
        f"AUC=<strong>{_num(final.get('auc'))}</strong>，"
        f"KS=<strong>{_num(final.get('ks'))}</strong>，"
        f"Gini={_num(final.get('gini'))}。</p>"
    )

    # 不平衡策略读数（同一模型下 weight vs smote 的 CV AUC 差距，数据驱动）
    by_model: dict[str, dict[str, float]] = {}
    for e in experiments:
        by_model.setdefault(str(e.get("model")), {})[str(e.get("strategy"))] = float(
            e.get("cv_auc_mean") or float("nan")
        )
    for model, strat in by_model.items():
        w, s = strat.get("weight"), strat.get("smote")
        gap = _sub(w, s)
        if gap is not None:
            verdict = "几乎无差异——credit-g 不平衡温和（约 30% 坏样本），样本加权以更低复杂度等效" \
                if abs(gap) < 0.01 else "存在可见差异，取 CV AUC 更高者"
            parts.append(
                f'<p class="note">{_esc(_MODEL_LABELS.get(model, model))}：weight 与 smote 的 '
                f"CV AUC 差距 {abs(gap):.4f}——{verdict}。</p>"
            )

    # 可解释性与稳定性补充（全部读自既有产物）
    shap_csv = reports_dir / "shap_global_importance.csv"
    if shap_csv.is_file():
        imp = pd.read_csv(shap_csv)
        top3 = "、".join(str(f) for f in imp["feature"].head(3))
        method = metrics.get("global_importance_method", "shap_mean_abs")
        parts.append(
            f'<p class="note">全局重要性（{_esc(method)}）Top 3：{top3}'
            "（完整表见 reports/shap_global_importance.csv）。</p>"
        )
    psi_csv = reports_dir / "psi_train_vs_test.csv"
    if psi_csv.is_file():
        psi = pd.read_csv(psi_csv)
        mx = float(psi["psi"].max())
        verdict = "全部 &lt;0.1，同分布切分符合预期" if mx < 0.1 else "存在特征漂移，需复核"
        parts.append(
            f'<p class="note">稳定性（自实现 PSI，train vs test）：最大 PSI {mx:.4f} → {verdict}'
            "（表见第②节，完整报告见 reports/stability_report.md）。</p>"
        )
    local = metrics.get("local_example") or {}
    if local.get("top_features"):
        tops = "、".join(
            f'{t.get("feature")}({_signed(t.get("contribution"))})'
            for t in local["top_features"][:3]
        )
        parts.append(
            f'<p class="note">单样本解释示例（测试集第 1 条，方法 '
            f"{_esc(metrics.get('deployed_model_local_explanation', '—'))}）："
            f"PD={_num(local.get('pd'), 3)}，真实 y={local.get('y_true', '—')}，"
            f"top 归因：{tops}。</p>"
        )
    return "\n".join(parts)


def _section_calibration(metrics: dict, reports_dir: Path) -> str:
    parts = ['<h2 id="s4">④ 概率校准：发现 → 修复 → 复测</h2>']
    cal = metrics.get("calibration") or {}
    dep = cal.get("deployed") or {}
    ref = cal.get("reference") or {}
    repair = cal.get("repair") or {}
    sel = repair.get("selection") or {}
    test = repair.get("test") or {}

    bias = _sub(dep.get("mean_predicted_pd"), dep.get("observed_bad_rate"))
    parts.append(
        '<div class="step"><span class="tag">发现</span>'
        "<p>排序指标（AUC/KS）不约束 PD 绝对值；balanced 类加权改善排序但把 PD 系统性抬高——"
        f"部署模型（{_esc(_short_model(dep.get('model', '—')))}，校准前）平均预测 PD "
        f"{_num(dep.get('mean_predicted_pd'), 3)} vs 实际违约率 "
        f"{_num(dep.get('observed_bad_rate'), 3)}（偏移 {_signed(bias)}），"
        f"校准前 ECE=<strong>{_num(dep.get('ece'))}</strong>、Brier={_num(dep.get('brier'))}；"
        f"未加权参照（{_esc(_short_model(ref.get('model', '—')))}）ECE={_num(ref.get('ece'))}。"
        "此发现按项目纪律保留原始数字，不作粉饰。</p></div>"
    )

    sel_rows = [
        ("raw（未校准）", _num(sel.get("ece_valid_raw")), _num(sel.get("brier_valid_raw")), ""),
        ("sigmoid（Platt）", _num(sel.get("ece_valid_sigmoid")),
         _num(sel.get("brier_valid_sigmoid")),
         "✔ 选中" if repair.get("method") == "sigmoid" else ""),
        ("isotonic", _num(sel.get("ece_valid_isotonic")), _num(sel.get("brier_valid_isotonic")),
         "✔ 选中" if repair.get("method") == "isotonic" else ""),
    ]
    coef = repair.get("sigmoid_coefficients") or {}
    coef_txt = (
        f"（PD_cal = σ(a·logit(PD_raw)+b)，a={_num(coef.get('a'))}，b={_num(coef.get('b'))}，严格单调）"
        if repair.get("method") == "sigmoid" and coef else ""
    )
    parts.append(
        '<div class="step"><span class="tag">修复</span>'
        "<p>协议（防泄漏）：训练集 5 折分层 CV 产出 OOF held-out 预测 → 拟合 sigmoid 与 isotonic，"
        "验证段按 ECE 择优，择优者用全部 OOF 重拟合为部署校准器；测试集只做单调变换，"
        "绝不参与校准器拟合。</p>"
        + _table(["候选（验证段）", "ECE ↓", "Brier ↓", "择优"], sel_rows)
        + f'<p class="note">选中方法：{_esc(repair.get("method", "—"))}{coef_txt}</p></div>'
    )

    fixed = (
        isinstance(test.get("test_ece_after"), (int, float))
        and isinstance(test.get("test_ece_before"), (int, float))
        and float(test["test_ece_after"]) < float(test["test_ece_before"])
    )
    verdict = (
        "校准后平均 PD 与实际违约率偏差收窄，可谨慎用于需要绝对 PD 的场景"
        "（测试集样本有限，逐桶读数仍有抽样噪声）"
        if fixed else
        "本次修复未能在测试集上降低 ECE，如实记录：校准后 PD 仍不能当作真实违约概率使用"
    )
    after_bias = _sub(test.get("test_mean_pd_after"), test.get("observed_bad_rate"))
    parts.append(
        '<div class="step"><span class="tag">复测</span>'
        f"<p>测试集（n={_num(test.get('test_n'), '.0f')}）校准前 → 校准后：</p>"
        + _table(
            ["指标", "校准前", "校准后"],
            [
                ("ECE ↓", _num(test.get("test_ece_before")), f"<strong>{_num(test.get('test_ece_after'))}</strong>"),
                ("Brier ↓", _num(test.get("test_brier_before")), _num(test.get("test_brier_after"))),
                ("平均预测 PD", _num(test.get("test_mean_pd_before"), 3), _num(test.get("test_mean_pd_after"), 3)),
                ("PD 偏差（vs 实际违约率）", "—", _signed(after_bias)),
                ("AUC（排序保持）", _num(test.get("test_auc_before")), _num(test.get("test_auc_after"))),
            ],
        )
        + f"<p class=\"note\">实际违约率 {_num(test.get('observed_bad_rate'), 3)}。{verdict}。</p></div>"
    )

    img = _img_src(reports_dir / "calibration_curve.png")
    if img:
        parts.append(
            '<img class="figure" alt="reliability diagram（校准前/后/参照）" src="' + img + '">'
        )
        parts.append('<p class="note">可靠性曲线：校准前/校准后/未加权参照三组并排，'
                     "对角线为完美校准；逐桶数据见 reports/calibration_table.csv。</p>")
    else:
        parts.append('<p class="note">reports/calibration_curve.png 缺失，图略。</p>')
    return "\n".join(parts)


def _section_thresholds(metrics: dict, reports_dir: Path) -> str:
    parts = ['<h2 id="s5">⑤ 阈值-业务扫描（批准率 / 坏账率）</h2>']
    trade = metrics.get("threshold_tradeoff") or {}
    targets = trade.get("targets") or []
    test_n = trade.get("test_n")
    parts.append(
        f"<p>决策规则：{_esc(trade.get('decision_rule', 'approve if PD < threshold'))}"
        f"（阈值取测试集 PD 分位数，排序型阈值，不受 PD 校准偏移影响；n={_num(test_n, '.0f')}）。</p>"
    )
    if targets:
        rows = [
            (_pct(t.get("target_approval_rate")), _num(t.get("threshold"), 3),
             _pct(t.get("approval_rate")), _pct(t.get("bad_rate_approved")),
             _pct(t.get("bad_rate_rejected")))
            for t in targets
        ]
        parts.append(_table(
            ["目标批准率", "阈值", "实际批准率", "批内坏账率", "拒件坏账率"], rows))
        tightest, loosest = targets[0], targets[-1]
        drop = _sub(loosest.get("bad_rate_approved"), tightest.get("bad_rate_approved"))
        base = loosest.get("bad_rate_approved")
        if drop is not None and isinstance(base, (int, float)) and float(base) > 0:
            relative = float(drop) / float(base)
            parts.append(
                f'<p><strong>关键读数</strong>：批准率从 {_pct(loosest.get("approval_rate"), 0)} '
                f'收紧到 {_pct(tightest.get("approval_rate"), 0)}，'
                f'批内坏账率由 {_pct(loosest.get("bad_rate_approved"))} 降至 '
                f'{_pct(tightest.get("bad_rate_approved"))}'
                f"（相对下降 {relative:.0%}），被拒人群坏账率 "
                f'{_pct(tightest.get("bad_rate_rejected"))}——模型排序确实把高风险申请人'
                "集中到了拒绝侧。</p>"
            )
        parts.append(
            '<p class="note">描述性扫描，未引入利润/损失矩阵，不构成阈值最优化建议；'
            "测试集样本有限，最细分档坏账率对单样本波动敏感。</p>"
        )

    # 信用分标度：metrics 有该字段才渲染（缺失时如实省略整块，不虚构）
    sc = metrics.get("scorecard") or {}
    if sc:
        parts.append("<p><strong>信用分标度（Scorecard，PD 的单调刻度变换）</strong></p>")
        sc_rows = [
            ("PDO（odds 翻倍所需分数）", _num(sc.get("pdo"), ".0f")),
            ("基准 odds（正常:违约）→ 基准分",
             f"{_num(sc.get('base_odds'), '.0f')} : 1 → {_num(sc.get('base_score'), '.0f')} 分"),
            ("测试集分数范围",
             f"{_num(sc.get('min'), '.1f')} ~ {_num(sc.get('max'), '.1f')}"
             f"（n={_num(sc.get('n'), '.0f')}）"),
            ("中位 / 均值", f"{_num(sc.get('median'), '.1f')} / {_num(sc.get('mean'), '.1f')}"),
        ]
        parts.append(_table(["参数 / 统计", "取值"], sc_rows))
        parts.append(
            '<p class="note">score = Offset + (PDO/ln2)·ln((1−PD)/PD)，基准 odds 落在基准分，'
            "PD 越低分数越高。与校准后 PD 一一对应的单调变换：不改变排序、不改变阈值决策，"
            "仅把概率量纲翻译成业务方习惯的分数量纲；参数已写入 artifacts/model_meta.json。</p>"
        )

    img = _img_src(reports_dir / "threshold_tradeoff.png")
    if img:
        parts.append('<img class="figure" alt="threshold trade-off curves" src="' + img + '">')
        parts.append('<p class="note">完整十分位扫描见 reports/threshold_tradeoff.csv、'
                     "按目标反查见 reports/threshold_targets.csv。</p>")
    else:
        parts.append('<p class="note">reports/threshold_tradeoff.png 缺失，图略。</p>')
    return "\n".join(parts)


def _section_fairness(metrics: dict) -> str:
    parts = ['<h2 id="s6">⑥ 公平性审计（personal_status 分组，只测量不修正）</h2>']
    fair = metrics.get("fairness") or {}
    groups = fair.get("groups") or {}
    if not groups:
        return "\n".join(parts + ['<p class="note">metrics 中无公平性审计数据，本节略。</p>'])
    parts.append(
        f"<p>分组特征 <code>{_esc(fair.get('group_column', 'personal_status'))}</code>"
        "（<strong>含性别编码</strong>）；决策规则与部署一致，批准阈值 t="
        f"{_num(fair.get('threshold'), 3)}（校准后 PD 低风险档下界）；"
        f"测试集 n={_num(fair.get('n_total'), '.0f')}，整体批准率 "
        f"{_pct(fair.get('overall_selection_rate'))}，整体实际违约率 "
        f"{_pct(fair.get('overall_bad_rate'))}。</p>"
    )
    rows = [
        (_esc(name), _num(v.get("n"), ".0f"), _pct(v.get("bad_rate")),
         _pct(v.get("selection_rate")), _pct(v.get("tpr_good")), _num(v.get("auc"), 3))
        for name, v in groups.items()
    ]
    parts.append(_table(
        ["分组", "n", "实际违约率", "选择率（批准率）", "TPR（批准|实际正常）", "组内 AUC"], rows))
    parts.append(
        f'<p><strong>如实记录的差距</strong>：demographic parity 差距 '
        f'<strong>{_num(fair.get("demographic_parity_gap"), 3)}</strong>，'
        f'等机会差距 <strong>{_num(fair.get("equal_opportunity_gap"), 3)}</strong>。</p>'
    )
    try:
        conclusion = fairness_conclusion(fair)
    except Exception:  # noqa: BLE001 — 渲染不应因结论文本失败而中断
        conclusion = ""
    if conclusion:
        for line in conclusion.splitlines():
            if line.strip():
                parts.append(f'<p class="note">{_esc(line)}</p>')

    # 阈值敏感性扫描（三档对比）：metrics 有该字段才渲染，缺失时如实省略整块
    sens = metrics.get("fairness_sensitivity") or {}
    tiers = [t for t in (sens.get("tiers") or []) if isinstance(t.get("report"), dict)]
    if tiers:
        group_names = list(tiers[0]["report"].get("groups") or {})
        headers = (["档位", "阈值 t", "目标批准率", "整体批准率"]
                   + [_esc(g) for g in group_names] + ["DP 差", "等机会差"])
        sens_rows = []
        for tier in tiers:
            rep = tier["report"]
            target = tier.get("target_approval_rate")
            cells = [
                _esc("部署档（现行）" if target is None else f"批准率 {float(target):.0%} 目标"),
                _num(tier.get("threshold"), 4),
                "—" if target is None else _pct(target),
                _pct(rep.get("overall_selection_rate")),
            ]
            cells += [
                _pct((rep.get("groups") or {}).get(g, {}).get("selection_rate"))
                for g in group_names
            ]
            cells += [
                _num(rep.get("demographic_parity_gap"), 3),
                _num(rep.get("equal_opportunity_gap"), 3),
            ]
            sens_rows.append(tuple(cells))
        parts.append(_table(headers, sens_rows))
        targets_txt = " / ".join(
            f"{float(t):.0%}" for t in (sens.get("approval_targets") or [])
        )
        parts.append(
            '<p class="note">阈值敏感性扫描：部署档之外，另按整体批准率 '
            f'{_esc(targets_txt)} 目标反查阈值（阈值 = 校准后 PD 的目标分位数，'
            "排序型），档位按阈值从紧到松排列；各组列为该组选择率。"
            "完整表见 reports/fairness_sensitivity.md。</p>"
        )

    parts.append('<p class="note">本项目只做描述性审计，未做再平衡/去偏/受保护属性移除，'
                 "不构成合规结论；完整表见 reports/fairness.md 与 reports/fairness.csv。</p>")
    return "\n".join(parts)


def _section_limitations(readme_path: Path | None) -> str:
    parts = ['<h2 id="s7">⑦ 已知限制（与 README 同步）</h2>']
    items: list[str] = []
    if readme_path and Path(readme_path).is_file():
        items = extract_readme_limitations(Path(readme_path).read_text(encoding="utf-8"))
    if items:
        lis = "".join(f"<li>{_inline_md(t)}</li>" for t in items)
        parts.append(f'<ol class="limits">{lis}</ol>')
    else:
        parts.append('<p class="note">README「已知限制」章节暂不可读，'
                     "请直接查看 README.md 的「已知限制」——本报告不在此转述，避免与原文失同步。</p>")
    return "\n".join(parts)


# ------------------------------------------------------------------- CSS/HTML

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; color: #334155;
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
  font-size: 14px; line-height: 1.7; }
.wrap { max-width: 960px; margin: 0 auto; padding: 44px 28px 64px; }
header h1 { color: #1a365d; font-size: 26px; margin: 0 0 6px; letter-spacing: 0.5px; }
.meta { color: #64748b; font-size: 13px; margin: 0 0 14px; }
.chips { margin: 10px 0 4px; }
.chips span { display: inline-block; border: 1px solid #e2e8f0; background: #f8fafc;
  border-radius: 4px; padding: 2px 12px; margin: 3px 8px 3px 0; font-size: 13px; }
.chips b { color: #1a365d; }
nav.toc { border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;
  padding: 8px 0; margin: 18px 0 6px; }
nav.toc a { color: #2563eb; text-decoration: none; margin-right: 16px; font-size: 13px; }
h2 { color: #1a365d; font-size: 18px; border-left: 4px solid #1a365d;
  padding-left: 10px; margin: 42px 0 12px; }
h2:first-of-type { margin-top: 28px; }
p { margin: 8px 0; }
table.data, table.kv { border-collapse: collapse; width: 100%; font-size: 13px;
  margin: 10px 0 14px; break-inside: avoid; }
table.data th, table.kv th { text-align: left; color: #1a365d; background: #f8fafc;
  border-bottom: 2px solid #1a365d; padding: 6px 10px; font-weight: 600; }
table.data td, table.kv td { padding: 5px 10px; border-bottom: 1px solid #e2e8f0; }
table.kv th[scope="row"] { width: 170px; color: #64748b; font-weight: 400;
  border-bottom: 1px solid #e2e8f0; background: transparent; }
tr.hl td { background: #eff6ff; font-weight: 600; color: #1a365d; }
.step { border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px 18px 6px;
  margin: 12px 0; break-inside: avoid; }
.step .tag { display: inline-block; background: #1a365d; color: #ffffff;
  font-size: 12px; letter-spacing: 2px; border-radius: 3px; padding: 1px 10px; }
.note { font-size: 12.5px; color: #64748b; }
.warn { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
  border-radius: 6px; padding: 10px 14px; font-size: 13.5px; }
img.figure { max-width: 100%; height: auto; border: 1px solid #e2e8f0;
  border-radius: 4px; margin: 10px 0 4px; break-inside: avoid; }
ol.limits { padding-left: 22px; margin: 8px 0; }
ol.limits li { margin-bottom: 9px; }
code { background: #f1f5f9; border-radius: 3px; padding: 0 5px;
  font-family: Consolas, "Courier New", monospace; font-size: 12.5px; }
footer { margin-top: 52px; border-top: 1px solid #e2e8f0; padding-top: 16px;
  color: #64748b; font-size: 12.5px; }
@media print { .wrap { padding: 0; max-width: none; } h2 { margin-top: 26px; } }
"""

_TOC = (
    '<nav class="toc">'
    '<a href="#s1">① 数据</a><a href="#s2">② 特征工程</a><a href="#s3">③ 模型对比</a>'
    '<a href="#s4">④ 校准</a><a href="#s5">⑤ 阈值-业务</a><a href="#s6">⑥ 公平性</a>'
    '<a href="#s7">⑦ 已知限制</a></nav>'
)


def render_html_report(
    metrics: dict,
    reports_dir: Path,
    out_path: Path,
    meta: dict | None = None,
    readme_path: Path | None = None,
) -> Path:
    """把训练指标与既有产物聚合渲染为单页自包含 HTML，写入 out_path 并返回。

    参数：
    - metrics：artifacts/metrics.json 的内容（run_training 内存中的 metrics_payload
      或从磁盘读回的同构 dict）；
    - reports_dir：既有图表/CSV 产物目录（图转 base64、IV/SHAP/PSI 表读此处）；
    - meta：artifacts/model_meta.json 的内容（可选，用于页眉版本与环境信息）；
    - readme_path：README 路径（已知限制从此解析；缺省 None 表示解析不到则如实提示）。
    """
    reports_dir = Path(reports_dir)
    out_path = Path(out_path)
    meta = meta or {}

    libs = meta.get("library_versions") or {}
    lib_txt = "、".join(
        f"{k} {v}" for k, v in libs.items() if v
    ) or "—"
    trained_at = meta.get("trained_at", "—")

    final = metrics.get("final_test_metrics") or {}
    cal = (metrics.get("calibration") or {}).get("repair", {}).get("test", {}) or {}
    selected = metrics.get("selected") or {}
    sel_model = _MODEL_LABELS.get(str(selected.get("model")), selected.get("model", "—"))
    source_label = _SOURCE_LABELS.get(
        str(metrics.get("data_source", "")), str(metrics.get("data_source", "—"))
    )
    chips = (
        f'<span>test AUC <b>{_num(final.get("auc"))}</b></span>'
        f'<span>test KS <b>{_num(final.get("ks"))}</b></span>'
        f'<span>ECE {_num(cal.get("test_ece_before"))} → <b>{_num(cal.get("test_ece_after"))}</b></span>'
        f'<span>部署 <b>{_esc(sel_model)} + {_esc(selected.get("strategy", "—"))}</b> + '
        f'{_esc((metrics.get("calibration") or {}).get("repair", {}).get("method", "—"))}</span>'
    )

    body = "\n".join([
        _section_data(metrics),
        _section_features(reports_dir),
        _section_models(metrics, reports_dir),
        _section_calibration(metrics, reports_dir),
        _section_thresholds(metrics, reports_dir),
        _section_fairness(metrics),
        _section_limitations(readme_path),
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>credit-risk-modeling · 一页式训练报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>credit-risk-modeling · 一页式训练报告</h1>
<p class="meta">数据：{_esc(source_label)}　|　模型版本 {_esc(meta.get("model_version", "—"))}
　|　训练时间 {_esc(trained_at)}　|　环境：{_esc(lib_txt)}</p>
<div class="chips">{chips}</div>
{_TOC}
</header>
<main>
{body}
</main>
<footer>
<p>本页由训练管线（scripts/run_training.py）在训练末尾自动生成；也可用
<code>python scripts/render_report.py</code> 从已提交的 artifacts/ 与 reports/ 产物免训练重渲染。
页面所有数字均来自 artifacts/metrics.json、artifacts/model_meta.json 与 reports/ 下既有产物，
缺失项渲染为“—”，不虚构数据。单文件自包含（内联 CSS + base64 图片），无任何外网依赖。</p>
<p>渲染时间：{_esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
</footer>
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def rerender_from_artifacts(project_root: Path | None = None,
                            out_path: Path | None = None) -> Path:
    """从既有 artifacts/ + reports/ 产物重渲染报告，不加载模型、不重复训练。"""
    root = Path(project_root) if project_root else PROJECT_ROOT
    metrics = json.loads(
        (root / "artifacts" / "metrics.json").read_text(encoding="utf-8")
    )
    meta_path = root / "artifacts" / "model_meta.json"
    meta = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    readme = root / "README.md"
    target = Path(out_path) if out_path else root / "reports" / "report.html"
    return render_html_report(
        metrics, root / "reports", target,
        meta=meta, readme_path=readme if readme.is_file() else None,
    )
