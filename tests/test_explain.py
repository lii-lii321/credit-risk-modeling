# -*- coding: utf-8 -*-
"""可解释性测试：SHAP 路径、线性降级路径、permutation 全局降级路径。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import creditrisk.explain as explain_mod
from creditrisk.data import generate_synthetic_credit
from creditrisk.explain import SHAP_AVAILABLE, explain_instance, global_importance
from creditrisk.features import build_pipeline


@pytest.fixture(scope="module")
def fitted_lr_pipeline():
    data = generate_synthetic_credit(n=500, seed=3)
    pipe = build_pipeline(LogisticRegression(max_iter=2000))
    pipe.fit(data.X, data.y)
    return pipe, data.X


@pytest.fixture(scope="module")
def fitted_lgbm_pipeline():
    from lightgbm import LGBMClassifier

    data = generate_synthetic_credit(n=500, seed=3)
    pipe = build_pipeline(LGBMClassifier(n_estimators=60, learning_rate=0.1,
                                         random_state=0, verbose=-1))
    pipe.fit(data.X, data.y)
    return pipe, data.X


def test_global_importance_shap_for_lgbm(fitted_lgbm_pipeline):
    pipe, X = fitted_lgbm_pipeline
    table = global_importance(pipe, X)
    assert len(table) == 20
    assert table["importance"].is_monotonic_decreasing
    assert (table["importance"] >= 0).all()
    assert table["method"].iloc[0] == "shap_mean_abs"


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap 未安装")
def test_shap_available_in_this_env():
    assert SHAP_AVAILABLE


def test_global_importance_permutation_fallback(fitted_lr_pipeline):
    """线性模型走 permutation 降级路径（需要 y）。"""
    pipe, X = fitted_lr_pipeline
    data = generate_synthetic_credit(n=500, seed=3)
    table = global_importance(pipe, X, y=data.y, use_shap=False, n_repeats=2)
    assert len(table) == 20
    assert table["method"].iloc[0] == "permutation_auc_drop"


def test_explain_instance_linear_coef_path(fitted_lr_pipeline):
    """线性模型单样本解释：contribution = coef * (x_woe - mean)。"""
    pipe, X = fitted_lr_pipeline
    out = explain_instance(pipe, X.iloc[[0]], top_k=3)
    assert len(out) == 3
    assert set(out[0]) == {"feature", "contribution"}
    contributions = [abs(item["contribution"]) for item in out]
    assert contributions == sorted(contributions, reverse=True)  # 按|贡献|降序


def test_explain_instance_shap_path(fitted_lgbm_pipeline):
    pipe, X = fitted_lgbm_pipeline
    out = explain_instance(pipe, X.iloc[[5]], top_k=3)
    assert len(out) == 3
    assert all(np.isfinite(item["contribution"]) for item in out)


def test_explain_instance_falls_back_without_shap(fitted_lgbm_pipeline, monkeypatch):
    """模拟 shap 不可用：树模型解释应降级为 coef 路径或报出明确错误，
    这里 LightGBM 没有 coef_，因此 monkeypatch 后应抛 ValueError。"""
    pipe, X = fitted_lgbm_pipeline
    monkeypatch.setattr(explain_mod, "SHAP_AVAILABLE", False)
    with pytest.raises(ValueError):
        explain_instance(pipe, X.iloc[[0]], top_k=3)


def test_top_k_respects_k(fitted_lgbm_pipeline):
    pipe, X = fitted_lgbm_pipeline
    out = explain_instance(pipe, X.iloc[[1]], top_k=5)
    assert len(out) == 5
