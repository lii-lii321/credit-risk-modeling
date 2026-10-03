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

本模块另提供 fairness_sensitivity：部署档之外再按整体批准率目标反查阈值
（复用 thresholds.thresholds_for_approval_rates，不重写反查逻辑），逐档复跑
同一分组审计，观察 DP/等机会差距对阈值松紧的敏感性——仍只测量，不修正。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .calibration import _validate
from .evaluate import roc_auc
from .thresholds import thresholds_for_approval_rates

DECISION_RULE = "approve if PD < threshold"


def _max_min_gap(values) -> float:
    """有限值集合的 max−min；全部无定义（NaN）时返回 NaN（不虚构 0 差距）。"""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.max() - arr.min())


def fairness_audit(y_true, y_proba, groups, threshold: float, group_column: str = "") -> dict:
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
    report: dict = {
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


def fairness_table(report: dict) -> pd.DataFrame:
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


def _extreme_group(report: dict, metric: str) -> tuple[str | None, str | None, float, float]:
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


def fairness_conclusion(report: dict) -> str:
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


def fairness_markdown(report: dict) -> str:
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


def _sensitivity_tier_label(tier: dict) -> str:
    """档位展示名：部署档 / 70% 目标档（数据驱动，不硬编码目标值）。"""
    target = tier.get("target_approval_rate")
    if target is None:
        return "部署档"
    try:
        return f"{float(target):.0%} 目标档"
    except (TypeError, ValueError):
        return str(tier.get("label", "未知档"))


def fairness_sensitivity(
    y_true,
    y_proba,
    groups,
    deployed_threshold: float,
    approval_targets: Sequence[float] = (0.7, 0.9),
    group_column: str = "",
) -> dict:
    """三档阈值下的分组公平性敏感性扫描（只测量，不修正）。

    档位：
    1. 部署档：deployed_threshold（与 API/线上同尺度的现行决策阈值）；
    2+. 目标档：按整体批准率目标反查阈值——复用 thresholds.thresholds_for_approval_rates
       （阈值 = y_proba 的目标分位数，排序型阈值）；PD 并列时实际批准率可能偏离目标，
       各档报告中的 overall_selection_rate 如实给出实际值。

    每档复用 fairness_audit（各组选择率 / DP 差 / 等机会差 / 分组 AUC），
    档位按阈值升序（最紧 → 最松，稳定排序）排列；返回 dict 可直接写入 metrics.json。
    """
    deployed_threshold = float(deployed_threshold)
    if not (0.0 <= deployed_threshold <= 1.0):
        raise ValueError("deployed_threshold 需在 [0, 1] 内")
    targets = np.sort(np.asarray(approval_targets, dtype=float))
    if targets.size == 0:
        raise ValueError("approval_targets 不能为空")
    if not ((targets > 0) & (targets < 1)).all():
        raise ValueError("approval_targets 需在 (0, 1) 开区间内")

    back = thresholds_for_approval_rates(y_true, y_proba, targets)
    tiers: list[dict] = [{
        "label": "deployed",
        "target_approval_rate": None,
        "threshold": deployed_threshold,
        "report": fairness_audit(
            y_true, y_proba, groups, threshold=deployed_threshold,
            group_column=group_column,
        ),
    }]
    for row in back.to_dict("records"):
        threshold = float(row["threshold"])
        tiers.append({
            "label": f"target_{int(round(float(row['target_approval_rate']) * 100))}",
            "target_approval_rate": float(row["target_approval_rate"]),
            "threshold": threshold,
            "report": fairness_audit(
                y_true, y_proba, groups, threshold=threshold,
                group_column=group_column,
            ),
        })
    tiers.sort(key=lambda t: t["threshold"])  # 稳定：同阈值时部署档保持在前
    return {
        "group_column": str(group_column),
        "decision_rule": DECISION_RULE,
        "n_total": int(tiers[0]["report"]["n_total"]),
        "deployed_threshold": deployed_threshold,
        "approval_targets": [float(t) for t in targets],
        "tiers": tiers,
    }


def fairness_sensitivity_table(scan: dict) -> pd.DataFrame:
    """三档对比表（每档一行）：阈值、目标/实际整体批准率、各组选择率、DP 差、等机会差。

    target_approval_rate 部署档为 NaN（无目标口径，展示层渲染“—”，不虚构 0）。
    """
    tiers = scan["tiers"]
    group_order = list(tiers[0]["report"]["groups"].keys())
    rows = []
    for tier in tiers:
        rep = tier["report"]
        row = {
            "tier": tier["label"],
            "target_approval_rate": tier["target_approval_rate"],
            "threshold": round(float(tier["threshold"]), 4),
            "overall_selection_rate": round(rep["overall_selection_rate"], 4),
            "demographic_parity_gap": round(rep["demographic_parity_gap"], 4),
            "equal_opportunity_gap": round(rep["equal_opportunity_gap"], 4),
        }
        for g in group_order:
            row[f"sel:{g}"] = round(rep["groups"][g]["selection_rate"], 4)
        rows.append(row)
    return pd.DataFrame(rows, columns=[
        "tier", "target_approval_rate", "threshold", "overall_selection_rate",
        "demographic_parity_gap", "equal_opportunity_gap",
    ] + [f"sel:{g}" for g in group_order])


def fairness_sensitivity_conclusion(scan: dict) -> str:
    """三行以内数据驱动结论：阈值松紧 → DP/等机会差距变化 → 始终偏低组 → 审计边界。"""
    tiers = sorted(scan["tiers"], key=lambda t: t["threshold"])
    dps = [float(t["report"]["demographic_parity_gap"]) for t in tiers]
    eos = [float(t["report"]["equal_opportunity_gap"]) for t in tiers]
    overall = [float(t["report"]["overall_selection_rate"]) for t in tiers]
    labels = [_sensitivity_tier_label(t) for t in tiers]

    def _direction(values: list[float]) -> str:
        if all(v == values[0] for v in values):
            return "不随阈值松紧变化"
        if all(b <= a for a, b in zip(values, values[1:], strict=False)):
            return "随阈值放松收窄"
        if all(b >= a for a, b in zip(values, values[1:], strict=False)):
            return "随阈值放松扩大"
        return "随阈值松紧非单调变化"

    lowest: list[str | None] = []
    for t in tiers:
        pairs = [
            (g, float(v["selection_rate"]))
            for g, v in t["report"]["groups"].items()
            if np.isfinite(v["selection_rate"])
        ]
        lowest.append(min(pairs, key=lambda kv: kv[1])[0] if pairs else None)
    if lowest and len(set(lowest)) == 1 and lowest[0] is not None:
        lowest_vals = [
            float(t["report"]["groups"][lowest[0]]["selection_rate"]) for t in tiers
        ]
        lowest_txt = (
            f"选择率最低组三档均为 {lowest[0]}"
            f"（依次 {'、'.join(f'{v:.1%}' for v in lowest_vals)}），"
            "在所有档位下获批机会都最小"
        )
    else:
        lowest_txt = "选择率最低组随档位变化（" + "、".join(
            f"{lab}:{g}" for lab, g in zip(labels, lowest, strict=False)
        ) + "）"

    tier_txt = "、".join(
        f"{lab} t={t['threshold']:.4f}" for lab, t in zip(labels, tiers, strict=False)
    )
    return "\n".join([
        f"阈值从紧到松（{tier_txt}，整体批准率 {'→'.join(f'{v:.1%}' for v in overall)}）："
        f"demographic parity 差距 {'→'.join(f'{v:.3f}' for v in dps)}，{_direction(dps)}；"
        f"等机会差距 {'→'.join(f'{v:.3f}' for v in eos)}，{_direction(eos)}。",
        f"{lowest_txt}。",
        f"以上为 n={scan['n_total']} 的描述性敏感性扫描（seed=42 切分，复用部署决策规则，"
        "阈值=PD 目标分位数反查，未做再平衡/去偏）；小样本组读数噪声大，不构成合规结论。",
    ])


def fairness_sensitivity_markdown(scan: dict) -> str:
    """渲染 reports/fairness_sensitivity.md：口径说明 + 三档对比表 + 三行以内结论。"""
    tbl = fairness_sensitivity_table(scan)
    disp = tbl.astype(object).where(pd.notna(tbl), "—")  # 部署档无目标口径 → “—”
    targets_txt = " / ".join(f"{t:.0%}" for t in scan["approval_targets"])
    lines = [
        "# 公平性敏感性扫描（三档阈值）",
        "",
        f"- 独立测试集 n={scan['n_total']}（与训练管线同一切分与预测，seed=42，不重新建模）；"
        f"分组特征：`{scan['group_column']}`（**含性别编码**）",
        f"- 决策规则与部署一致：{scan['decision_rule']}；档位：部署档 t={scan['deployed_threshold']:.4f}"
        f"（校准后 PD 低风险档下界）+ 按整体批准率 {targets_txt} 反查"
        "（阈值 = 校准后 PD 的目标分位数，排序型阈值，复用 thresholds_for_approval_rates）",
        "- `sel:*` 列为该组选择率（批准率）；DP 差 = 各组选择率 max−min；"
        "等机会差 = 各组 TPR（批准|实际正常）max−min；档位按阈值从紧到松排列",
        "",
        "## 三档对比",
        "",
        disp.to_markdown(index=False),
        "",
        "## 结论",
        "",
        fairness_sensitivity_conclusion(scan),
        "",
    ]
    return "\n".join(lines)
