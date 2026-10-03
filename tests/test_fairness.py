"""公平性审计测试：手算可核对的小样例（选择率 / parity 差 / TPR 差 / 分组 AUC）、
无定义指标的诚实降级（NaN 不虚构）、输入校验、三档阈值敏感性扫描（手算分位数反查），
以及 credit-g 真实数据端到端。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.data import CATEGORY_VALUES, load_credit_data
from creditrisk.fairness import (
    fairness_audit,
    fairness_conclusion,
    fairness_markdown,
    fairness_sensitivity,
    fairness_sensitivity_conclusion,
    fairness_sensitivity_markdown,
    fairness_sensitivity_table,
    fairness_table,
)
from creditrisk.models import fit_deployable_pipeline

THRESHOLD = 0.5

# 手算样例（threshold=0.5，批准 = p < 0.5）：
# A: p=[0.1,0.2,0.3,0.9], y=[0,0,0,1] → 批准 3/4=0.75；违约率 1/4=0.25；
#    TPR=批准|正常=3/3=1.0；AUC：违约 0.9 高于全部正常 → 1.0
# B: p=[0.4,0.6,0.7,0.8], y=[0,1,0,1] → 批准 1/4=0.25；违约率 2/4=0.5；
#    TPR=1/2=0.5（正常 0.4 获批、正常 0.7 被拒）；AUC=3/4=0.75
#    （4 个正反对：仅 (0.7 正常 vs 0.6 违约) 一对逆序）
@pytest.fixture(scope="module")
def hand():
    groups = ["A"] * 4 + ["B"] * 4
    proba = [0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.7, 0.8]
    y = [0, 0, 0, 1, 0, 1, 0, 1]
    return fairness_audit(y, proba, groups, threshold=THRESHOLD)


def test_hand_computed_group_metrics(hand):
    assert hand["n_total"] == 8
    assert hand["groups"]["A"] == {
        "n": 4, "bad_rate": 0.25, "selection_rate": 0.75, "tpr_good": 1.0, "auc": 1.0,
    }
    assert hand["groups"]["B"] == {
        "n": 4, "bad_rate": 0.5, "selection_rate": 0.25, "tpr_good": 0.5, "auc": 0.75,
    }
    assert hand["overall_selection_rate"] == pytest.approx(0.5)
    assert hand["overall_bad_rate"] == pytest.approx(0.375)


def test_hand_computed_demographic_parity_gap(hand):
    assert hand["demographic_parity_gap"] == pytest.approx(0.75 - 0.25)


def test_hand_computed_equal_opportunity_gap(hand):
    assert hand["equal_opportunity_gap"] == pytest.approx(1.0 - 0.5)


def test_hand_computed_group_auc(hand):
    assert hand["groups"]["A"]["auc"] == pytest.approx(1.0)
    assert hand["groups"]["B"]["auc"] == pytest.approx(0.75)


def test_group_auc_and_tpr_nan_for_single_class():
    """组内只有违约样本：AUC 与 TPR 无定义，必须记 NaN（不虚构 0）。"""
    report = fairness_audit(
        [1, 1, 1], [0.9, 0.95, 0.2], ["C", "C", "C"], threshold=THRESHOLD,
    )
    group = report["groups"]["C"]
    assert np.isnan(group["auc"])
    assert np.isnan(group["tpr_good"])
    assert group["bad_rate"] == pytest.approx(1.0)
    assert group["selection_rate"] == pytest.approx(1 / 3)
    assert np.isnan(report["equal_opportunity_gap"])
    assert report["demographic_parity_gap"] == pytest.approx(0.0)


def test_gaps_ignore_undefined_groups_and_use_defined_ones():
    """存在无定义组时，差距只在有定义的组之间计算。"""
    report = fairness_audit(
        [1, 1, 0, 0], [0.1, 0.9, 0.2, 0.6], ["C", "C", "A", "B"], threshold=THRESHOLD,
    )
    # C 全为违约：tpr_good 无定义；A 正常获批（0.2<0.5）TPR=1.0；B 正常被拒（0.6≥0.5）TPR=0.0
    assert np.isnan(report["groups"]["C"]["tpr_good"])
    assert report["groups"]["A"]["tpr_good"] == pytest.approx(1.0)
    assert report["groups"]["B"]["tpr_good"] == pytest.approx(0.0)
    assert report["equal_opportunity_gap"] == pytest.approx(1.0)


@pytest.mark.parametrize("threshold,expected_sel", [(0.0, 0.0), (1.0, 1.0)])
def test_threshold_extremes(threshold, expected_sel):
    report = fairness_audit(
        [0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9], ["A", "A", "B", "B"], threshold=threshold,
    )
    assert all(g["selection_rate"] == pytest.approx(expected_sel) for g in report["groups"].values())
    assert report["demographic_parity_gap"] == pytest.approx(0.0)


def test_input_validation():
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 0.2, 0.3], ["A", "B"], threshold=0.5)  # groups 长度不一致
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 0.2], ["A", None], threshold=0.5)  # groups 含缺失
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 0.2], ["A", "  "], threshold=0.5)  # groups 含空白
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 0.2], ["A", "B"], threshold=1.5)  # 阈值越界
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 0.2], ["A", "B"], threshold=-0.1)  # 阈值越界
    with pytest.raises(ValueError):
        fairness_audit([0, 2], [0.1, 0.2], ["A", "B"], threshold=0.5)  # 标签非 0/1
    with pytest.raises(ValueError):
        fairness_audit([0, 1], [0.1, 1.2], ["A", "B"], threshold=0.5)  # 概率越界
    with pytest.raises(ValueError):
        fairness_audit([], [], [], threshold=0.5)  # 空输入


def test_fairness_table_matches_report(hand):
    tbl = fairness_table(hand)
    assert list(tbl["group"]) == ["A", "B"]
    assert tbl.loc[tbl["group"] == "A", "n"].item() == 4
    assert tbl.loc[tbl["group"] == "A", "n_share"].item() == pytest.approx(0.5)
    assert tbl.loc[tbl["group"] == "B", "selection_rate"].item() == pytest.approx(0.25)
    assert tbl.loc[tbl["group"] == "B", "tpr_good"].item() == pytest.approx(0.5)
    assert tbl.loc[tbl["group"] == "B", "auc"].item() == pytest.approx(0.75)


def test_markdown_contains_real_numbers(hand):
    md = fairness_markdown(hand)
    assert md.startswith("# 公平性审计")
    assert "personal_status" in md or "group_column" in md or "分组特征" in md
    assert f"{hand['demographic_parity_gap']:.4f}" in md
    assert f"{hand['equal_opportunity_gap']:.4f}" in md
    assert str(THRESHOLD) in fairness_table(hand).to_string()  # 表中数字可复现


def test_conclusion_three_lines_with_real_numbers(hand):
    conclusion = fairness_conclusion(hand)
    lines = [ln for ln in conclusion.splitlines() if ln.strip()]
    assert len(lines) <= 3
    assert "75.0%" in conclusion and "25.0%" in conclusion  # A/B 选择率如实出现
    assert "demographic parity 差距 0.500" in conclusion
    assert "等机会差距 0.500" in conclusion
    assert "未做再平衡/去偏" in conclusion


# ------------------------------------- credit-g 真实数据端到端（复用部署管线口径）
@pytest.fixture(scope="module")
def real_audit():
    data = load_credit_data()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, stratify=data.y, random_state=42
    )
    pipeline = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "weight")
    proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = float(np.quantile(proba, 0.6))  # 与部署低风险档同口径
    return fairness_audit(
        y_test, proba, X_test["personal_status"], threshold=threshold,
        group_column="personal_status",
    )


def test_end_to_end_credit_g_covers_all_groups(real_audit):
    assert real_audit["n_total"] == 200
    assert sum(g["n"] for g in real_audit["groups"].values()) == 200
    assert set(real_audit["groups"]) == set(CATEGORY_VALUES["personal_status"])
    for v in real_audit["groups"].values():
        assert 0.0 <= v["selection_rate"] <= 1.0
        assert 0.0 <= v["bad_rate"] <= 1.0
        assert np.isnan(v["auc"]) or 0.0 <= v["auc"] <= 1.0


def test_end_to_end_gaps_consistent_with_groups(real_audit):
    sel = [g["selection_rate"] for g in real_audit["groups"].values()]
    tpr = [g["tpr_good"] for g in real_audit["groups"].values() if np.isfinite(g["tpr_good"])]
    assert real_audit["demographic_parity_gap"] == pytest.approx(max(sel) - min(sel))
    assert real_audit["equal_opportunity_gap"] == pytest.approx(max(tpr) - min(tpr))


def test_end_to_end_report_renders(real_audit):
    tbl = fairness_table(real_audit)
    md = fairness_markdown(real_audit)
    assert len(tbl) == len(real_audit["groups"])
    assert all(name in md for name in real_audit["groups"])
    assert f"n={real_audit['n_total']}" in md


# ------------------------------------- credit-g 真实数据端到端（三档敏感性扫描 smoke）
@pytest.fixture(scope="module")
def real_scan():
    data = load_credit_data()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, stratify=data.y, random_state=42
    )
    pipeline = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "weight")
    proba = pipeline.predict_proba(X_test)[:, 1]
    return fairness_sensitivity(
        y_test, proba, X_test["personal_status"],
        deployed_threshold=float(np.quantile(proba, 0.6)),
        approval_targets=(0.7, 0.9),
        group_column="personal_status",
    )


def test_end_to_end_sensitivity_credit_g(real_scan):
    assert real_scan["n_total"] == 200
    tiers = real_scan["tiers"]
    assert [t["label"] for t in tiers] == ["deployed", "target_70", "target_90"]
    thresholds = [t["threshold"] for t in tiers]
    assert thresholds == sorted(thresholds)
    for tier in tiers:
        rep = tier["report"]
        assert set(rep["groups"]) == set(CATEGORY_VALUES["personal_status"])
        sel = [v["selection_rate"] for v in rep["groups"].values()]
        assert rep["demographic_parity_gap"] == pytest.approx(max(sel) - min(sel))
        for v in rep["groups"].values():
            assert 0.0 <= v["selection_rate"] <= 1.0
    # 阈值从紧到松：整体批准率不降（决策规则 p < t 的直接推论）
    overall = [t["report"]["overall_selection_rate"] for t in tiers]
    assert overall == sorted(overall)
    md = fairness_sensitivity_markdown(real_scan)
    assert all(name in md for name in tiers[0]["report"]["groups"])


# ------------------------------------- 三档阈值敏感性扫描（手算可核对的小样例）
# 手算样例：p 升序 [0.1..0.6]，批准 = p < t；A: p=[0.1,0.2,0.3] y=[0,0,0]；B: p=[0.4,0.5,0.6] y=[1,0,1]
# np.quantile 线性插值：q(0.5)=0.35（(n-1)q=2.5 → 0.30/0.40 中点）；q(0.9)=0.55（4.5 → 0.50/0.60 中点）
# t=0.35：A 全批（sel=1.0）、B 全拒（sel=0.0）→ DP=1.0；B 的正常样本(0.5)被拒 → A TPR=1.0、B TPR=0 → EO=1.0
# t=0.55：A 全批（1.0）、B 批 2/3（0.4、0.5 获批）→ DP=1/3；B 的正常样本获批 → EO=0.0；整体批准 5/6
@pytest.fixture(scope="module")
def hand_scan():
    groups = ["A"] * 3 + ["B"] * 3
    proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    y = [0, 0, 0, 1, 0, 1]
    return fairness_sensitivity(
        y, proba, groups, deployed_threshold=0.35, approval_targets=(0.5, 0.9),
        group_column="status",
    )


def test_sensitivity_hand_computed_tiers(hand_scan):
    assert hand_scan["deployed_threshold"] == pytest.approx(0.35)
    assert hand_scan["approval_targets"] == pytest.approx([0.5, 0.9])
    assert [t["label"] for t in hand_scan["tiers"]] == ["deployed", "target_50", "target_90"]
    assert [t["threshold"] for t in hand_scan["tiers"]] == pytest.approx([0.35, 0.35, 0.55])

    deployed = hand_scan["tiers"][0]["report"]
    assert deployed["groups"]["A"]["selection_rate"] == pytest.approx(1.0)
    assert deployed["groups"]["B"]["selection_rate"] == pytest.approx(0.0)
    assert deployed["demographic_parity_gap"] == pytest.approx(1.0)
    assert deployed["equal_opportunity_gap"] == pytest.approx(1.0)

    loose = hand_scan["tiers"][2]["report"]
    assert loose["groups"]["B"]["selection_rate"] == pytest.approx(2 / 3)
    assert loose["groups"]["B"]["tpr_good"] == pytest.approx(1.0)
    assert loose["demographic_parity_gap"] == pytest.approx(1 - 2 / 3)
    assert loose["equal_opportunity_gap"] == pytest.approx(0.0)
    assert loose["overall_selection_rate"] == pytest.approx(5 / 6)


def test_sensitivity_monotone_in_threshold(hand_scan):
    """阈值升序排列（同阈值部署档在前）；各组选择率随阈值不降，整体批准率不降。"""
    tiers = hand_scan["tiers"]
    thresholds = [t["threshold"] for t in tiers]
    assert thresholds == sorted(thresholds)
    labels = [t["label"] for t in tiers]
    assert labels.index("deployed") < labels.index("target_50")  # 稳定排序
    group_names = list(tiers[0]["report"]["groups"])
    for g in group_names:
        sels = [t["report"]["groups"][g]["selection_rate"] for t in tiers]
        assert sels == sorted(sels)
    overall = [t["report"]["overall_selection_rate"] for t in tiers]
    assert overall == sorted(overall)


def test_sensitivity_matches_fairness_audit_at_same_threshold():
    """同阈值下敏感性档位与独立调用 fairness_audit 的结果完全一致（复用而非重写）。"""
    y = [0, 0, 0, 1, 0, 1]
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    g = ["A"] * 3 + ["B"] * 3
    standalone = fairness_audit(y, p, g, threshold=0.55, group_column="status")
    scan = fairness_sensitivity(
        y, p, g, deployed_threshold=0.2, approval_targets=(0.9,), group_column="status",
    )

    def norm(obj):
        """NaN 归一为哨兵值后再比较（NaN != NaN 会让 dict 直接比较误报）。"""
        if isinstance(obj, dict):
            return {k: norm(v) for k, v in obj.items()}
        if isinstance(obj, float) and np.isnan(obj):
            return "<nan>"
        return obj

    assert norm(scan["tiers"][1]["report"]) == norm(standalone)


def test_sensitivity_input_validation():
    y, p, g = [0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], ["A", "A", "B", "B"]
    with pytest.raises(ValueError):
        fairness_sensitivity(y, p, g, deployed_threshold=1.5)  # 部署阈值越界
    with pytest.raises(ValueError):
        fairness_sensitivity(y, p, g, deployed_threshold=0.5, approval_targets=[])  # 目标为空
    with pytest.raises(ValueError):
        fairness_sensitivity(y, p, g, deployed_threshold=0.5, approval_targets=(0.0,))  # 越界
    with pytest.raises(ValueError):
        fairness_sensitivity(y, p, g, deployed_threshold=0.5, approval_targets=(1.0,))  # 越界
    with pytest.raises(ValueError):
        fairness_sensitivity(y, p, g[:3], deployed_threshold=0.5)  # groups 长度不一致
    with pytest.raises(ValueError):
        fairness_sensitivity(y, [0.1, 0.2], g, deployed_threshold=0.5)  # y/p 长度不一致


def test_sensitivity_table_columns_and_placeholder(hand_scan):
    tbl = fairness_sensitivity_table(hand_scan)
    assert len(tbl) == 3
    assert list(tbl["tier"]) == ["deployed", "target_50", "target_90"]
    assert list(tbl.columns[:6]) == [
        "tier", "target_approval_rate", "threshold", "overall_selection_rate",
        "demographic_parity_gap", "equal_opportunity_gap",
    ]
    assert list(tbl.columns[6:]) == ["sel:A", "sel:B"]
    assert pd.isna(tbl.loc[0, "target_approval_rate"])          # 部署档无目标口径
    assert tbl.loc[1, "target_approval_rate"] == pytest.approx(0.5)
    assert tbl.loc[2, "demographic_parity_gap"] == pytest.approx(1 - 2 / 3, abs=1e-4)


def test_sensitivity_markdown_and_conclusion(hand_scan):
    md = fairness_sensitivity_markdown(hand_scan)
    assert md.startswith("# 公平性敏感性扫描")
    assert "0.3333" in md          # 最松档 DP 差（4 位小数，手算 1−2/3）
    assert "—" in md               # 部署档目标批准率如实渲染“—”，不虚构 0
    for label in ("deployed", "target_50", "target_90"):
        assert label in md

    conclusion = fairness_sensitivity_conclusion(hand_scan)
    lines = [ln for ln in conclusion.splitlines() if ln.strip()]
    assert len(lines) <= 3
    assert "随阈值放松收窄" in conclusion          # DP 1.000→1.000→0.333
    assert "demographic parity 差距 1.000→1.000→0.333" in conclusion
    assert "等机会差距 1.000→1.000→0.000" in conclusion
    assert "选择率最低组三档均为 B" in conclusion
    assert "未做再平衡/去偏" in conclusion
