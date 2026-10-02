# -*- coding: utf-8 -*-
"""概率校准单元测试：Brier 已知构造值、可靠性分箱、ECE 行为与输入校验。"""
import numpy as np
import pandas as pd
import pytest

from creditrisk.calibration import (
    brier_score,
    expected_calibration_error,
    plot_reliability_diagram,
    reliability_data,
)


def test_brier_known_values():
    """全对预测 Brier=0；常数 0.5 预测 Brier=0.25（无信息上限之一）。"""
    assert brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    # 全错预测 Brier=1
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_matches_manual_construction():
    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.3, 0.4])
    manual = ((0.8 - 1) ** 2 + (0.3 - 0) ** 2 + (0.4 - 1) ** 2) / 3
    assert brier_score(y, p) == pytest.approx(manual)


def test_brier_input_validation():
    with pytest.raises(ValueError):
        brier_score([], [])  # 空输入
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5])  # 长度不一致
    with pytest.raises(ValueError):
        brier_score([0, 2], [0.5, 0.5])  # 标签越界
    with pytest.raises(ValueError):
        brier_score([0, 1], [-0.1, 0.5])  # 概率越界


def test_reliability_data_known_construction():
    """8 个样本 4 桶（每桶 2 个），逐桶均值可手工验证。"""
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    table = reliability_data(y, p, n_bins=4)
    assert list(table["n"]) == [2, 2, 2, 2]
    assert table["n"].sum() == 8
    assert table["mean_predicted_pd"].iloc[0] == pytest.approx(0.15)
    assert table["observed_bad_rate"].iloc[0] == pytest.approx(0.5)
    # 按预测 PD 升序：桶均值单调不减
    assert table["mean_predicted_pd"].is_monotonic_increasing


def test_reliability_data_columns_and_rate_bounds():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = rng.random(500)
    table = reliability_data(y, p, n_bins=10)
    assert list(table.columns) == ["bin", "n", "mean_predicted_pd", "observed_bad_rate"]
    assert table["n"].sum() == 500
    assert ((table["observed_bad_rate"] >= 0) & (table["observed_bad_rate"] <= 1)).all()
    assert ((table["mean_predicted_pd"] >= 0) & (table["mean_predicted_pd"] <= 1)).all()


def test_reliability_data_collapses_tied_probabilities():
    """全部概率并列 → 单桶，不崩溃且统计正确。"""
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.full(8, 0.5)
    table = reliability_data(y, p, n_bins=4)
    assert len(table) == 1
    assert table["n"].iloc[0] == 8
    assert table["mean_predicted_pd"].iloc[0] == pytest.approx(0.5)
    assert table["observed_bad_rate"].iloc[0] == pytest.approx(0.5)


def test_ece_near_zero_for_calibrated_scores():
    """按真实概率生成的标签应近似校准：ECE 在抽样噪声量级内。"""
    rng = np.random.default_rng(7)
    n = 20000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_ece_detects_constant_shift():
    """系统性高估 0.2 的预测，ECE 应接近该偏移量。"""
    rng = np.random.default_rng(11)
    n = 5000
    p_true = rng.uniform(0.05, 0.75, n)
    y = rng.binomial(1, p_true)
    p_pred = np.clip(p_true + 0.2, 0, 1)
    ece = expected_calibration_error(y, p_pred, n_bins=10)
    assert 0.15 < ece < 0.25
    # 对照：同一组标签下，校准版本的 ECE 显著更小
    assert expected_calibration_error(y, p_true, n_bins=10) < ece


def test_ece_bounds_and_bins_validation():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, 200)
    p = rng.random(200)
    value = expected_calibration_error(y, p, n_bins=5)
    assert 0.0 <= value <= 1.0
    with pytest.raises(ValueError):
        expected_calibration_error(y, p, n_bins=0)
    with pytest.raises(ValueError):
        reliability_data(y, p, n_bins=-1)


def test_plot_reliability_diagram_writes_file(tmp_path):
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.3, 300)
    p_a = np.clip(y * 0.6 + rng.random(300) * 0.4, 0.01, 0.99)
    p_b = rng.random(300)
    out = plot_reliability_diagram(
        [("model_a", y, p_a), ("model_b", y, p_b)],
        tmp_path / "calibration_curve.png",
        n_bins=5,
    )
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError):
        plot_reliability_diagram([], tmp_path / "empty.png")


def test_reliability_data_accepts_pandas_inputs():
    """与仓库其余模块一致：pandas Series 输入应等价于 ndarray。"""
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    p = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    from_pandas = reliability_data(y, p, n_bins=4)
    from_array = reliability_data(y.to_numpy(), p.to_numpy(), n_bins=4)
    pd.testing.assert_frame_equal(from_pandas, from_array)
