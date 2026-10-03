# -*- coding: utf-8 -*-
"""公平性审计（描述性测量，不修正）：按 personal_status 分组的部署阈值公平性指标。

背景（README「已知限制」第 2 条曾长期记录"公平性审计未实现"）：gender 经
personal_status 编码于特征中（"female div/dep/mar" / "male div/sep" /
"male mar/wid" / "male single"），模型可见受保护属性。本模块把该发现转化为
基础审计能力：按 personal_status 分组实测部署决策规则下的公平性指标——
只测量，不做再平衡 / 去偏 / 受保护属性移除（这些仍未实现，见 README）。

审计口径（与部署完全一致，不重新建模）：
1. 复用训练管线的测试集切分与预测概率（seed=42，n=200），本模块不拟合任何模型；
2. 决策规则与阈值分析/风险分档一致：PD < threshold 批准，PD ≥ threshold 拒绝；
   部署阈值取校准后 PD 的低风险档下界（run_training 传入 t_low，与 API/线上同尺度）；
3. 指标定义（credit-g 中 y=1 表示违约，"有利结果"为按期还款）：
   - 选择率（selection rate）= P(批准)；各组 max−min 即 demographic parity 差距；
   - 等机会 TPR = P(批准 | 实际正常)（Hardt et al. 2016 的 equal opportunity，
     正类取有利结果），各组 max−min 即等机会差距；组内无正常样本时无定义，记 NaN；
   - 分组 AUC：组内 PD 对违约标签的排序能力；组内单一类别时无定义，记 NaN（不虚构）。

限制：测试集仅 200 条，分组读数受抽样噪声影响；审计为描述性测量，
不构成合规结论（未做任何公平性约束或修正）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import _validate
from .evaluate import roc_auc

DECISION_RULE = "approve if PD < threshold"


def _max_min_gap(values) -> float:
    """有限值集合的 max−min；全部无定义（NaN）时返回 NaN（不虚构 0 差距）。"""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.max() - arr.min())


def fairness_audit(y_true, y_proba, groups, threshold: float, group_column: str = "") -> Dict:
    """按分组计算部署阈值下的公平性指标，返回可直接写入 metrics.json 的报告字典。

    参数：
    - y_true / y_proba：与训练管线一致的测试集标签与预测概率（校准后 PD）；
    - groups：与样本一一对应的分组标识（如 X_test["personal_status"]）；
    - threshold：部署阈值（批准 = y_proba < threshold），需在 [0, 1]；
    - group_column：分组特征名（仅用于报告标注）。

    返回字段：threshold / decision_rule / n_total / overall_bad_rate /
    overall_selection_rate / groups（逐组 n、bad_rate、selection_rate、
    tpr_good、auc）/ demographic_parity_gap / equal_opportunity_gap。
    """
    y, p = _validate(y_true, y_proba)
    g = pd.Series(groups).reset_index(drop=True)
    if len(g) != len(y):
        raise ValueError("groups 与 y_true 长度不一致")
    if g.isna().any() or g.astype(str).str.strip().eq("").any():
        raise ValueError("groups 存在缺失或空白值")
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold 需在 [0, 1] 内")

    approved = p < threshold
    report: Dict = {
        "group_column": str(group_column),
        "decision_rule": DECISION_RULE,
        "threshold": threshold,
        "n_total": int(len(y)),
        "overall_bad_rate": float(y.mean()),
        "overall_selection_rate": float(approved.mean()),
        "groups": {},
    }
    for name in pd.unique(g):  # 保留数据中首次出现的顺序
        mask = (g == name).to_numpy()
        yg, ag, pg = y[mask], approved[mask], p[mask]
        n_good = int((yg == 0).sum())
        n_bad = int((yg == 1).sum())
        both_classes = n_good > 0 and n_bad > 0
        report["groups"][str(name)] = {
            "n": int(mask.sum()),
            "bad_rate": float(yg.mean()),
            "selection_rate": float(ag.mean()),
            "tpr_good": float(ag[yg == 0].mean()) if n_good else float("nan"),
            "auc": float(roc_auc(yg, pg)) if both_classes else float("nan"),
        }

    report["demographic_parity_gap"] = _max_min_gap(
        v["selection_rate"] for v in report["groups"].values()
    )
    report["equal_opportunity_gap"] = _max_min_gap(
        v["tpr_good"] for v in report["groups"].values()
    )
    return report


def fairness_table(report: Dict) -> pd.DataFrame:
    """分组指标表（每组一行），用于 fairness.csv 与 markdown 渲染。"""
    rows = []
    n_total = report["n_total"]
    for name, v in report["groups"].items():
        rows.append({
            "group": name,
            "n": v["n"],
            "n_share": v["n"] / n_total,
            "bad_rate": round(v["bad_rate"], 4),
            "selection_rate": round(v["selection_rate"], 4),
            "tpr_good": round(v["tpr_good"], 4),
            "auc": round(v["auc"], 4),
        })
    return pd.DataFrame(rows, columns=[
        "group", "n", "n_share", "bad_rate", "selection_rate", "tpr_good", "auc",
    ])


def _extreme_group(report: Dict, metric: str) -> Tuple[Optional[str], Optional[str], float, float]:
    """按指标取（最高组, 最低组, 最高值, 最低值）；全组无定义时返回 (None, None, NaN, NaN)。"""
    pairs = [
        (name, float(v[metric])) for name, v in report["groups"].items()
        if np.isfinite(v[metric])
    ]
    if not pairs:
        return None, None, float("nan"), float("nan")
    hi = max(pairs, key=lambda kv: kv[1])
    lo = min(pairs, key=lambda kv: kv[1])
    return hi[0], lo[0], hi[1], lo[1]


def fairness_conclusion(report: Dict) -> str:
    """三行数据驱动结论：选择率 → 等机会/分组 AUC → 审计边界（如实、不粉饰）。"""
    sel_hi, sel_lo, sel_hi_v, sel_lo_v = _extreme_group(report, "selection_rate")
    tpr_hi, tpr_lo, tpr_hi_v, tpr_lo_v = _extreme_group(report, "tpr_good")
    auc_vals = [v["auc"] for v in report["groups"].values() if np.isfinite(v["auc"])]
    min_n = min(v["n"] for v in report["groups"].values())
    auc_txt = (
        f"{min(auc_vals):.3f}–{max(auc_vals):.3f}" if auc_vals else "无定义"
    )
    tpr_txt = (
        f"最高 {tpr_hi}（{tpr_hi_v:.1%}）、最低 {tpr_lo}（{tpr_lo_v:.1%}），"
        f"等机会差距 {report['equal_opportunity_gap']:.3f}"
        if tpr_hi is not None else "等机会差距无定义（存在无正常样本的组）"
    )
    return "\n".join([
        f"选择率（部署阈值 t={report['threshold']:.3f}，整体批准率 {report['overall_selection_rate']:.1%}）："
        f"最高 {sel_hi}（{sel_hi_v:.1%}），最低 {sel_lo}（{sel_lo_v:.1%}），"
        f"demographic parity 差距 {report['demographic_parity_gap']:.3f}。",
        f"等机会（TPR = 批准 | 实际正常）：{tpr_txt}；分组 AUC 区间 {auc_txt}。",
        f"以上为 n={report['n_total']}、最小组 {min_n} 条的描述性审计（seed=42 切分，"
        "复用部署管线预测，未做再平衡/去偏/特征移除）；小样本读数噪声大，"
        "不构成合规结论。",
    ])


def fairness_markdown(report: Dict) -> str:
    """渲染 reports/fairness.md：口径说明 + 分组指标表 + 三行以内文字结论。"""
    tbl = fairness_table(report)
    eo_gap = report["equal_opportunity_gap"]
    eo_txt = f"{eo_gap:.4f}" if np.isfinite(eo_gap) else "无定义"
    lines = [
        "# 公平性审计（personal_status 分组）",
        "",
        f"- 独立测试集 n={report['n_total']}（与训练管线同一切分与预测，seed=42，不重新建模）；"
        f"分组特征：`{report['group_column']}`（**含性别编码**）",
        f"- 决策规则与部署一致：{report['decision_rule']}，阈值 t={report['threshold']:.4f}"
        "（校准后 PD 低风险档下界，与 API/线上输出同尺度）",
        f"- 整体选择率 {report['overall_selection_rate']:.1%}，整体实际违约率 {report['overall_bad_rate']:.1%}；"
        "TPR = P(批准 | 实际正常)（等机会的正类取有利结果，Hardt et al. 2016）",
        "",
        "## 分组指标",
        "",
        tbl.to_markdown(index=False),
        "",
        f"- **Demographic parity 差距（选择率 max−min）：{report['demographic_parity_gap']:.4f}**",
        f"- **等机会差距（各组 TPR max−min）：{eo_txt}**",
        "",
        "## 结论",
        "",
        fairness_conclusion(report),
        "",
    ]
    return "\n".join(lines)
