"""阈值-业务指标单元测试：已知构造、单调性与计数守恒、目标反查与输入校验。"""
import numpy as np
import pandas as pd
import pytest

from creditrisk.thresholds import (
    plot_tradeoff_curves,
    thresholds_for_approval_rates,
    tradeoff_table,
)


def test_tradeoff_table_perfect_separation():
    """坏样本 PD 全 0.9、好样本全 0.1：阈值 0.5 → 批内坏账率 0，拒件坏账率 1。"""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    table = tradeoff_table(y, p, [0.5])
    assert len(table) == 1
    row = table.iloc[0]
    assert row["threshold"] == pytest.approx(0.5)
    assert row["n_approved"] == 4 and row["n_rejected"] == 4
    assert row["approval_rate"] == pytest.approx(0.5)
    assert row["bad_rate_approved"] == pytest.approx(0.0)
    assert row["bad_rate_rejected"] == pytest.approx(1.0)


def test_tradeoff_table_decision_rule_is_strictly_below():
    """PD 恰等于阈值时拒绝（PD < t 才批准）。"""
    y = np.array([1, 0])
    p = np.array([0.5, 0.4])
    row = tradeoff_table(y, p, [0.5]).iloc[0]
    assert row["n_approved"] == 1 and row["n_rejected"] == 1
    assert row["bad_rate_approved"] == pytest.approx(0.0)
    assert row["bad_rate_rejected"] == pytest.approx(1.0)


def test_counts_sum_and_approval_rate_monotonic():
    """n_approved + n_rejected = n；阈值升高批准率单调不减。"""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 300)
    p = rng.random(300)
    grid = np.linspace(0.05, 0.95, 19)
    table = tradeoff_table(y, p, grid)
    assert (table["n_approved"] + table["n_rejected"]).eq(300).all()
    assert table["approval_rate"].is_monotonic_increasing
    assert table["threshold"].is_monotonic_increasing
    assert ((table["bad_rate_approved"] >= 0) & (table["bad_rate_approved"] <= 1)).all()
    assert ((table["bad_rate_rejected"] >= 0) & (table["bad_rate_rejected"] <= 1)).all()


def test_bad_rate_weighted_identity():
    """整体坏账率 = 批内坏账率×批准占比 + 拒件坏账率×拒绝占比（守恒关系）。"""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 400)
    p = rng.random(400)
    overall = y.mean()
    for row in tradeoff_table(y, p, [0.2, 0.5, 0.8]).itertuples():
        reconstructed = (
            row.bad_rate_approved * row.approval_rate
            + row.bad_rate_rejected * (1 - row.approval_rate)
        )
        assert reconstructed == pytest.approx(overall, abs=1e-9)


def test_empty_group_bad_rate_is_nan():
    """批准组或拒绝组为空时坏账率记 NaN，不虚构 0。"""
    y = np.array([1, 0, 1, 0])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    table = tradeoff_table(y, p, [0.0, 1.0])
    no_approval = table.iloc[0]
    assert no_approval["n_approved"] == 0
    assert np.isnan(no_approval["bad_rate_approved"])
    assert no_approval["bad_rate_rejected"] == pytest.approx(0.5)
    all_approved = table.iloc[1]
    assert all_approved["n_rejected"] == 0
    assert np.isnan(all_approved["bad_rate_rejected"])


def test_thresholds_for_approval_rates_achieves_target():
    """连续无并列分数下，分位数阈值精确达成目标批准率。"""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, 500)
    p = rng.random(500)
    table = thresholds_for_approval_rates(y, p, (0.7, 0.8, 0.9))
    assert list(table["target_approval_rate"]) == [0.7, 0.8, 0.9]
    assert np.allclose(table["approval_rate"], [0.7, 0.8, 0.9])
    # 阈值升序、与 target 行对齐
    assert table["threshold"].is_monotonic_increasing


def test_thresholds_for_approval_rates_rejects_bad_targets():
    y = np.array([0, 1])
    p = np.array([0.3, 0.6])
    with pytest.raises(ValueError):
        thresholds_for_approval_rates(y, p, (0.0, 0.5))
    with pytest.raises(ValueError):
        thresholds_for_approval_rates(y, p, (1.0,))
    with pytest.raises(ValueError):
        thresholds_for_approval_rates(y, p, ())


def test_input_validation_shared_with_calibration():
    with pytest.raises(ValueError):
        tradeoff_table([], [], [0.5])  # 空输入
    with pytest.raises(ValueError):
        tradeoff_table([0, 1], [0.5], [0.5])  # 长度不一致
    with pytest.raises(ValueError):
        tradeoff_table([0, 2], [0.5, 0.5], [0.5])  # 标签越界
    with pytest.raises(ValueError):
        tradeoff_table([0, 1], [1.5, 0.5], [0.5])  # 概率越界
    with pytest.raises(ValueError):
        tradeoff_table([0, 1], [0.5, 0.5], [])  # 阈值列表为空


def test_pandas_inputs_equivalent_to_arrays():
    y = pd.Series([1, 1, 0, 0])
    p = pd.Series([0.9, 0.8, 0.2, 0.1])
    from_pandas = tradeoff_table(y, p, [0.5])
    from_array = tradeoff_table(y.to_numpy(), p.to_numpy(), [0.5])
    pd.testing.assert_frame_equal(from_pandas, from_array)


def test_plot_tradeoff_curves_writes_file(tmp_path):
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.3, 300)
    p = rng.random(300)
    out = plot_tradeoff_curves(y, p, np.linspace(0.1, 0.9, 9), tmp_path / "threshold_tradeoff.png")
    assert out.exists() and out.stat().st_size > 0
