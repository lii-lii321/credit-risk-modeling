"""PSI 单元测试：恒等分布、漂移分布、类别列与零频次鲁棒性。"""
import numpy as np
import pandas as pd
import pytest

from creditrisk.psi import max_psi, psi, psi_categorical, psi_numeric, psi_table


def test_identical_distribution_psi_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi_numeric(pd.Series(x), pd.Series(x)) < 0.01


def test_shifted_distribution_psi_large():
    rng = np.random.default_rng(1)
    expected = pd.Series(rng.normal(0, 1, 5000))
    actual = pd.Series(rng.normal(3, 1, 5000))  # 均值右移 3 个标准差
    value = psi_numeric(expected, actual)
    assert value > 0.25


def test_psi_monotone_in_shift():
    rng = np.random.default_rng(2)
    expected = pd.Series(rng.normal(0, 1, 8000))
    small = psi_numeric(expected, pd.Series(rng.normal(0.5, 1, 8000)))
    large = psi_numeric(expected, pd.Series(rng.normal(2.0, 1, 8000)))
    assert large > small > 0.0


def test_categorical_psi_detects_mixing_shift():
    rng = np.random.default_rng(3)
    expected = pd.Series(rng.choice(["A", "B"], 5000, p=[0.7, 0.3]))
    same = psi_categorical(expected, pd.Series(rng.choice(["A", "B"], 5000, p=[0.7, 0.3])))
    shifted = psi_categorical(expected, pd.Series(rng.choice(["A", "B"], 5000, p=[0.2, 0.8])))
    assert same < 0.02
    assert shifted > 0.25


def test_zero_proportion_handled_by_epsilon():
    """某箱在 actual 完全缺失时不应产生 inf/NaN。"""
    expected = pd.Series(["A", "A", "A", "B"])
    actual = pd.Series(["A", "A", "A", "A"])
    value = psi_categorical(expected, actual)
    assert np.isfinite(value)
    assert value > 0.1


def test_psi_dispatches_by_dtype():
    rng = np.random.default_rng(4)
    e_num, a_num = pd.Series(rng.normal(size=1000)), pd.Series(rng.normal(2, 1, 1000))
    e_cat, a_cat = pd.Series(["x"] * 500 + ["y"] * 500), pd.Series(["x"] * 900 + ["y"] * 100)
    assert psi(e_num, a_num) > 0.25
    assert psi(e_cat, a_cat) > 0.1


def test_psi_table_levels_and_ordering():
    rng = np.random.default_rng(5)
    expected = pd.DataFrame({
        "drift": rng.normal(0, 1, 3000),
        "stable": rng.normal(0, 1, 3000),
        "cat_stable": rng.choice(["p", "q"], 3000),
    })
    actual = pd.DataFrame({
        "drift": rng.normal(4, 1, 3000),
        "stable": rng.normal(0, 1, 3000),
        "cat_stable": rng.choice(["p", "q"], 3000),
    })
    table = psi_table(expected, actual)
    assert list(table["feature"]) == ["drift", "stable", "cat_stable"]  # 按 PSI 降序
    assert table.iloc[0]["level"] == "significant"
    assert table.iloc[1]["level"] == "stable"
    assert max_psi(table) == pytest.approx(table["psi"].max())


def test_missing_values_do_not_crash():
    rng = np.random.default_rng(6)
    expected = pd.Series(rng.normal(size=1000))
    actual = pd.Series(rng.normal(size=800))
    actual.iloc[:50] = np.nan
    assert np.isfinite(psi_numeric(expected, actual))
    e_cat = pd.Series(["a", None, "b"] * 100)
    a_cat = pd.Series(["a", "b", None] * 90)
    assert np.isfinite(psi_categorical(e_cat, a_cat))


def test_constant_column_psi_finite():
    expected = pd.Series([1.0] * 500)
    assert np.isfinite(psi_numeric(expected, pd.Series([1.0] * 300)))
