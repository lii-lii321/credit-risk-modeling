"""WoEEncoder 单元测试：编码方向、IV 单调性、未见类别与缺失值鲁棒性。"""
import numpy as np
import pandas as pd
import pytest

from creditrisk.woe import WoEEncoder, iv_strength


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n = 400
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    num = rng.normal(0, 1, size=n)
    # 违约率随 cat=A、num 增大而升高，保证有真实信号
    logit = 0.9 * (cat == "A") + 1.2 * num - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"cat": cat, "num": num}), pd.Series(y)


def test_fit_produces_woe_columns_and_shapes(tiny_df):
    X, y = tiny_df
    enc = WoEEncoder().fit(X, y)
    out = enc.transform(X)
    assert list(out.columns) == ["cat_woe", "num_woe"]
    assert out.shape == (len(X), 2)
    assert np.isfinite(out.to_numpy()).all()


def test_woe_sign_matches_default_risk(tiny_df):
    """违约率最高的箱 WOE 应为正、最低的箱应为负（事件=bad 的约定）。"""
    X, y = tiny_df
    enc = WoEEncoder().fit(X, y)
    woe = enc.woe_maps_["cat"]
    bad_rate = X["cat"].groupby(y).apply(lambda s: (s == "A").mean())
    rate_A_given_bad = bad_rate.loc[1]
    rate_A_given_good = bad_rate.loc[0]
    if rate_A_given_bad > rate_A_given_good:
        assert woe["A"] > 0
    else:
        assert woe["A"] < 0


def test_woe_monotone_with_risk_on_num(tiny_df):
    """数值列 WOE 应大致随分箱违约率单调上升（允许平局）。"""
    X, y = tiny_df
    enc = WoEEncoder(n_bins=5).fit(X, y)
    ids = enc._numeric_bin_ids(X["num"], enc.bin_edges_["num"])
    order = sorted({i for i in ids if i != "__MISSING__"}, key=lambda b: int(b.split("_")[1]))
    woe = [enc.woe_maps_["num"][b] for b in order]
    rates = []
    for b in order:
        mask = ids == b
        rates.append(y[mask].mean())
    assert np.corrcoef(woe, rates)[0, 1] > 0.9


def test_iv_large_for_separating_feature():
    """完美区分两类的特征 IV 应显著大于噪声特征。"""
    y = pd.Series([0] * 200 + [1] * 200)
    X = pd.DataFrame({"sep": ["G"] * 200 + ["B"] * 200,
                      "noise": np.random.default_rng(1).choice(["x", "y"], 400)})
    enc = WoEEncoder().fit(X, y)
    assert enc.iv_["sep"] > 1.0
    assert enc.iv_["noise"] < 0.05


def test_unseen_category_maps_to_zero(tiny_df):
    X, y = tiny_df
    enc = WoEEncoder().fit(X, y)
    unseen = pd.DataFrame({"cat": ["Z"], "num": [0.0]})
    assert enc.transform(unseen)["cat_woe"].iloc[0] == 0.0


def test_missing_values_do_not_crash(tiny_df):
    X, y = tiny_df
    X.loc[0, "num"] = np.nan
    X.loc[1, "cat"] = np.nan
    enc = WoEEncoder().fit(X, y)
    out = enc.transform(X)
    assert np.isfinite(out.to_numpy()).all()
    # 训练时见过 missing → 有独立箱；transform 输入新 missing 也安全
    probe = pd.DataFrame({"cat": [np.nan], "num": [np.nan]})
    assert np.isfinite(enc.transform(probe).to_numpy()).all()


def test_extreme_value_clipped_into_edge_bins(tiny_df):
    X, y = tiny_df
    enc = WoEEncoder().fit(X, y)
    probe = pd.DataFrame({"cat": ["A"], "num": [1e9]})
    edges = enc.bin_edges_["num"]
    assert enc.transform(probe)["num_woe"].iloc[0] == pytest.approx(
        enc.woe_maps_["num"][f"bin_{len(edges) - 2}"]
    )


def test_iv_table_sorted_and_labeled(tiny_df):
    X, y = tiny_df
    enc = WoEEncoder().fit(X, y)
    table = enc.iv_table()
    assert list(table["feature"]) == sorted(table["feature"], key=lambda f: -table.set_index("feature").loc[f, "iv"])
    assert set(table["strength"]).issubset({"useless", "weak", "medium", "strong", "suspicious"})


def test_invalid_smoothing_raises(tiny_df):
    X, y = tiny_df
    with pytest.raises(ValueError):
        WoEEncoder(smoothing=0).fit(X, y)


def test_single_class_raises():
    X = pd.DataFrame({"cat": ["A"] * 10, "num": np.arange(10, dtype=float)})
    y = pd.Series([1] * 10)
    with pytest.raises(ValueError):
        WoEEncoder().fit(X, y)


@pytest.mark.parametrize(
    "iv,expected",
    [(0.01, "useless"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong"), (0.9, "suspicious")],
)
def test_iv_strength_mapping(iv, expected):
    assert iv_strength(iv) == expected


# ---------------------------------------------------------- 缺失成箱正确性
def test_missing_bin_numeric_woe_matches_formula():
    """回归：缺失样本的 WOE 必须按真实 bad/good 计数计算，而非静默为 0。

    构造：10 个缺失全为 bad、10 个非缺失全为 good，n_bins=5 → 分位数边界
    去重后 6 个边界 → 5 个数据箱 + 1 个缺失箱 = 6 labels；
    平滑公式 WOE_missing = ln((n_bad+α)/(n_good+α))（分母相同抵消）
                        = ln(10.5/0.5) = ln(21) ≈ 3.0445。
    """
    X = pd.DataFrame({"f": [np.nan] * 10 + list(range(10, 20))})
    y = pd.Series([1] * 10 + [0] * 10)
    enc = WoEEncoder(n_bins=5).fit(X, y)
    alpha, n_labels = 0.5, 6
    expected = float(np.log((10 + alpha) / (0 + alpha)))  # 分母 (10+α·6) 两侧相同
    assert enc.woe_maps_["f"]["__MISSING__"] == pytest.approx(expected)
    assert enc.woe_maps_["f"]["__MISSING__"] == pytest.approx(np.log(21), rel=1e-9)
    assert n_labels == len(enc.woe_maps_["f"])


def test_missing_bin_categorical_all_good_is_strongly_negative():
    """类别列缺失单成箱：缺失全为 good 时 WOE 应显著为负，而非 0。"""
    X = pd.DataFrame({"c": ["A"] * 10 + [np.nan] * 10})
    y = pd.Series([0] * 10 + [1] * 10)
    enc = WoEEncoder().fit(X, y)
    assert enc.woe_maps_["c"]["__MISSING__"] > 2.0  # 缺失侧全是 bad
    assert enc.woe_maps_["c"]["A"] < -2.0


def test_transform_maps_nan_to_missing_bin_woe(tiny_df):
    """transform 输入缺失时必须取缺失箱的（非零）WOE，而非 0。"""
    X, y = tiny_df
    X = X.copy()
    X.loc[X.index[:50], "num"] = np.nan
    X.loc[X.index[:50], "cat"] = np.nan
    enc = WoEEncoder().fit(X, y)
    probe = pd.DataFrame({"cat": [np.nan], "num": [np.nan]})
    out = enc.transform(probe)
    assert out["num_woe"].iloc[0] == pytest.approx(enc.woe_maps_["num"]["__MISSING__"])
    assert out["cat_woe"].iloc[0] == pytest.approx(enc.woe_maps_["cat"]["__MISSING__"])
    # 50 个缺失样本不应与全零退化一致：至少一个方向的 WOE 非零
    assert abs(enc.woe_maps_["num"]["__MISSING__"]) > 0 or abs(enc.woe_maps_["cat"]["__MISSING__"]) > 0


def test_all_nan_numeric_column_degrades_to_missing_bin_only():
    """全缺失数值列：无分位边界（edges 为空 → bin_ids 为 np.nan）也不崩溃，
    且全部样本落入缺失箱。"""
    X = pd.DataFrame({"f": [np.nan] * 20, "g": np.arange(20, dtype=float)})
    y = pd.Series([0, 1] * 10)
    enc = WoEEncoder(n_bins=5).fit(X, y)
    out = enc.transform(X)
    assert np.isfinite(out.to_numpy()).all()
    assert len(enc.woe_maps_["f"]) == 1  # 只有缺失箱
    assert out["f_woe"].nunique() == 1   # 全部同箱
