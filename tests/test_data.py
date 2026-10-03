"""数据层测试：缓存加载、schema 校验、合成降级路径。"""
import numpy as np
import pandas as pd
import pytest

from creditrisk.config import CAT_FEATURES, FEATURES, NUM_FEATURES
from creditrisk.data import assert_schema, generate_synthetic_credit, load_credit_data


@pytest.fixture(scope="module")
def real_data():
    return load_credit_data()


def test_load_credit_data_from_cache(real_data):
    """仓库内已提交 data/raw/credit-g.csv 缓存（源自 OpenML credit-g v1）。"""
    assert not real_data.is_synthetic
    assert real_data.X.shape == (1000, 20)
    assert list(real_data.X.columns) == FEATURES
    assert 0.28 < real_data.y.mean() < 0.32  # credit-g 坏样本率 ≈ 30%
    assert set(np.unique(real_data.y)) == {0, 1}


def test_dtypes_normalized(real_data):
    for col in CAT_FEATURES:
        assert real_data.X[col].map(type).eq(str).all()
    for col in NUM_FEATURES:
        assert pd.api.types.is_float_dtype(real_data.X[col])


def test_real_data_passes_schema(real_data):
    assert_schema(real_data)


def test_synthetic_fallback_schema_identical():
    syn = generate_synthetic_credit(n=500, seed=1)
    assert syn.is_synthetic
    assert syn.X.shape == (500, 20)
    assert list(syn.X.columns) == FEATURES
    assert_schema(syn)  # 列集合与类别取值与真实 schema 完全一致
    assert 0.1 < syn.y.mean() < 0.6


def test_synthetic_deterministic():
    a = generate_synthetic_credit(n=200, seed=7)
    b = generate_synthetic_credit(n=200, seed=7)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)


def test_fetch_failure_falls_back_to_synthetic(monkeypatch, tmp_path):
    """在线拉取失败时自动降级为合成数据并打 is_synthetic 标记。"""
    import creditrisk.data as data_mod

    def boom(*args, **kwargs):
        raise ConnectionError("simulated offline")

    monkeypatch.setattr(data_mod, "fetch_openml", boom)
    data = data_mod.load_credit_data(cache_path=tmp_path / "nope.csv")
    assert data.is_synthetic
    assert data.X.shape[1] == 20
