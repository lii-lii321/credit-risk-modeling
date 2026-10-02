# -*- coding: utf-8 -*-
"""特征管线与模型实验测试（小数据量、快速）。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from creditrisk.features import build_pipeline
from creditrisk.models import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    cross_validate_model,
    fit_deployable_pipeline,
    make_estimator,
    make_pipeline,
    select_best,
)
from creditrisk.data import generate_synthetic_credit


@pytest.fixture(scope="module")
def small_split():
    data = generate_synthetic_credit(n=600, seed=11)
    X, y = data.X, data.y
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(0.75 * len(X))
    return X.iloc[idx[:cut]].reset_index(drop=True), y.iloc[idx[:cut]].reset_index(drop=True), \
        X.iloc[idx[cut:]].reset_index(drop=True), y.iloc[idx[cut:]].reset_index(drop=True)


def test_build_pipeline_end_to_end(small_split):
    X_tr, y_tr, X_te, _ = small_split
    pipe = build_pipeline(LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_make_pipeline_variants():
    for model in MODEL_NAMES:
        for strategy in IMBALANCE_STRATEGIES:
            pipe = make_pipeline(model, strategy)
            names = [name for name, _ in pipe.steps]
            assert names[0] == "woe" and names[-1] == "model"
            if strategy == "smote":
                assert "smote" in names


def test_make_estimator_unknown_name():
    with pytest.raises(ValueError):
        make_estimator("xgboost", "none")
    with pytest.raises(ValueError):
        make_estimator("lightgbm", "undersample")


def test_cross_validate_model_returns_stats(small_split):
    X_tr, y_tr, _, _ = small_split
    stats = cross_validate_model(X_tr, y_tr, "logistic_regression", "weight", n_splits=3)
    assert 0.0 <= stats["cv_auc_mean"] <= 1.0
    assert 0.0 <= stats["cv_ks_mean"] <= 1.0
    assert stats["cv_auc_std"] >= 0.0


def test_weighting_beats_none_on_imbalanced_cv(small_split):
    """合成数据 25-30% 坏样本，加权策略的 CV AUC 不应显著劣于不处理。"""
    X_tr, y_tr, _, _ = small_split
    none_stats = cross_validate_model(X_tr, y_tr, "logistic_regression", "none", n_splits=3)
    weight_stats = cross_validate_model(X_tr, y_tr, "logistic_regression", "weight", n_splits=3)
    assert weight_stats["cv_auc_mean"] >= none_stats["cv_auc_mean"] - 0.02


def test_smote_pipeline_only_resamples_training(small_split):
    """SMOTE 管线 predict 时不应重采样：predict_proba 行数 == 输入行数。"""
    X_tr, y_tr, X_te, _ = small_split
    pipe = make_pipeline("logistic_regression", "smote")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)
    assert proba.shape[0] == len(X_te)


def test_fit_deployable_pipeline_is_plain_sklearn(small_split):
    X_tr, y_tr, X_te, _ = small_split
    pipe = fit_deployable_pipeline(X_tr, y_tr, "lightgbm", "smote")
    import sklearn.pipeline

    assert isinstance(pipe, sklearn.pipeline.Pipeline)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)


def test_select_best_prefers_lr_on_tie():
    results = pd.DataFrame([
        {"model": "logistic_regression", "strategy": "weight", "cv_auc_mean": 0.750},
        {"model": "lightgbm", "strategy": "weight", "cv_auc_mean": 0.753},
    ])
    model, strategy = select_best(results)
    assert (model, strategy) == ("logistic_regression", "weight")


def test_select_best_picks_clear_winner():
    results = pd.DataFrame([
        {"model": "logistic_regression", "strategy": "weight", "cv_auc_mean": 0.700},
        {"model": "lightgbm", "strategy": "none", "cv_auc_mean": 0.800},
    ])
    model, strategy = select_best(results)
    assert (model, strategy) == ("lightgbm", "none")
