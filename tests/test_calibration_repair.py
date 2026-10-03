# -*- coding: utf-8 -*-
"""校准修正层测试：sigmoid/isotonic 行为、择优协议、bundle 序列化往返、
以及固定 seed 的真实数据全流程断言「校准后 ECE < 校准前 ECE」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.calibration import expected_calibration_error
from creditrisk.calibration_repair import (
    ProbabilityCalibrator,
    apply_calibrated_proba,
    build_bundle,
    calibrate_pipeline,
    load_bundle,
    logit,
    out_of_fold_proba,
    save_bundle,
    select_and_fit_calibrator,
)
from creditrisk.data import load_credit_data
from creditrisk.features import build_pipeline
from creditrisk.models import fit_deployable_pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def shift_data():
    """系统性高估 0.2 的固定构造：p_pred = p_true + 0.2，标签按 p_true 抽样。"""
    rng = np.random.default_rng(11)
    n = 8000
    p_true = rng.uniform(0.05, 0.75, n)
    y = rng.binomial(1, p_true)
    p_pred = np.clip(p_true + 0.2, 0, 1)
    return p_pred[:4000], y[:4000], p_pred[4000:], y[4000:], p_true[4000:]


# ------------------------------------------------- sigmoid / isotonic 行为
def test_sigmoid_reduces_ece_and_brier_on_constant_shift(shift_data):
    p_fit, y_fit, p_val, y_val, _ = shift_data
    raw_ece = expected_calibration_error(y_val, p_val, n_bins=10)
    raw_brier = float(np.mean((p_val - y_val) ** 2))
    calibrator = ProbabilityCalibrator("sigmoid").fit(p_fit, y_fit)
    calibrated = calibrator.apply(p_val)
    assert expected_calibration_error(y_val, calibrated, n_bins=10) < raw_ece
    assert float(np.mean((calibrated - y_val) ** 2)) < raw_brier
    # Platt 参数：a>0（保持方向），截距为负（修正整体高估）
    a, b = calibrator.coefficients_["a"], calibrator.coefficients_["b"]
    assert a > 0 and b < 0


def test_isotonic_reduces_ece_on_constant_shift(shift_data):
    p_fit, y_fit, p_val, y_val, _ = shift_data
    raw_ece = expected_calibration_error(y_val, p_val, n_bins=10)
    calibrator = ProbabilityCalibrator("isotonic").fit(p_fit, y_fit)
    calibrated = calibrator.apply(p_val)
    assert expected_calibration_error(y_val, calibrated, n_bins=10) < raw_ece


def test_sigmoid_near_identity_on_calibrated_input():
    """已校准的输入上拟合 sigmoid：应接近恒等映射（ECE 不显著恶化）。"""
    rng = np.random.default_rng(7)
    n = 20000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)
    calibrator = ProbabilityCalibrator("sigmoid").fit(p, y)
    calibrated = calibrator.apply(p)
    assert expected_calibration_error(y, calibrated, n_bins=10) < 0.03
    a, b = calibrator.coefficients_["a"], calibrator.coefficients_["b"]
    assert abs(a - 1.0) < 0.15 and abs(b) < 0.2


def test_apply_bounds_and_monotonicity(shift_data):
    p_fit, y_fit, _, _, _ = shift_data
    grid = np.linspace(0.0, 1.0, 101)
    for method in ("sigmoid", "isotonic"):
        calibrated = ProbabilityCalibrator(method).fit(p_fit, y_fit).apply(grid)
        assert ((calibrated >= 0) & (calibrated <= 1)).all()
        assert np.all(np.diff(calibrated) >= -1e-12), f"{method} 应保持单调不减"


def test_calibrator_input_validation():
    with pytest.raises(ValueError):
        ProbabilityCalibrator("bogus")
    with pytest.raises(ValueError):
        ProbabilityCalibrator("sigmoid").fit([0.1, 0.2], [1])  # 长度不一致
    with pytest.raises(ValueError):
        ProbabilityCalibrator("isotonic").fit([1.5, 0.2], [0, 1])  # 概率越界
    with pytest.raises(ValueError):
        ProbabilityCalibrator("sigmoid").fit([0.1, 0.2], [1, 1])  # 单一类别
    with pytest.raises(RuntimeError):
        ProbabilityCalibrator("sigmoid").apply([0.5])  # 未 fit 先 apply


def test_logit_clips_extremes():
    z = logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(z).all()
    assert z[1] == pytest.approx(0.0)
    assert z[0] < 0 < z[2]


# ------------------------------------------------------------- 择优协议
def test_select_and_fit_calibrator_picks_one_of_two_and_improves(shift_data):
    p_fit, y_fit, p_val, y_val, _ = shift_data
    calibrator, report = select_and_fit_calibrator(p_fit, y_fit, p_val, y_val)
    assert calibrator.method in ("sigmoid", "isotonic")
    assert report["method"] == calibrator.method
    # 验证段三方 ECE 都在报告中，且择优者优于 raw
    candidates = {report["ece_valid_sigmoid"], report["ece_valid_isotonic"]}
    assert report["ece_valid_raw"] > min(candidates)
    assert report[f"ece_valid_{calibrator.method}"] == min(candidates)


def test_out_of_fold_proba_covers_all_rows_once():
    data = load_credit_data()
    X, y = data.X, data.y
    oof = out_of_fold_proba(X, y, lambda: make_pipeline("logistic_regression", "weight"), n_splits=3)
    assert oof.shape == (len(y),)
    assert np.isfinite(oof).all()
    assert ((oof >= 0) & (oof <= 1)).all()


# ---------------------------------------------------------- bundle 往返
def _tiny_bundle(tmp_path):
    data = load_credit_data()
    X, y = data.X, data.y
    pipeline = build_pipeline(LogisticRegression(max_iter=2000), n_bins=5).fit(X.iloc[:400], y.iloc[:400])
    raw = pipeline.predict_proba(X.iloc[400:600])[:, 1]
    calibrator = ProbabilityCalibrator("isotonic").fit(raw, y.iloc[400:600])
    meta = {
        "features": list(X.columns),
        "model_version": "test",
        "risk_band_thresholds": {"low_below": 0.3, "medium_below": 0.7},
    }
    path = save_bundle(tmp_path / "bundle.joblib", build_bundle(pipeline, calibrator, meta))
    return path, pipeline, calibrator, X


def test_bundle_round_trip_preserves_predictions(tmp_path):
    path, pipeline, calibrator, X = _tiny_bundle(tmp_path)
    loaded = load_bundle(path)
    assert loaded["meta"]["risk_band_thresholds"] == {"low_below": 0.3, "medium_below": 0.7}
    row = X.iloc[[0]]
    raw_before = float(pipeline.predict_proba(row)[0, 1])
    raw_after = float(loaded["pipeline"].predict_proba(row)[0, 1])
    assert raw_after == pytest.approx(raw_before)
    assert float(loaded["calibrator"].apply([raw_after])[0]) == pytest.approx(
        float(calibrator.apply([raw_before])[0])
    )


def test_load_bundle_missing_file_has_actionable_message(tmp_path):
    with pytest.raises(RuntimeError) as exc_info:
        load_bundle(tmp_path / "nope.joblib")
    assert "run_training" in str(exc_info.value)


def test_build_bundle_requires_features_meta():
    from sklearn.linear_model import LogisticRegression

    pipeline = build_pipeline(LogisticRegression())
    with pytest.raises(ValueError):
        build_bundle(pipeline, None, {"model": "x"})


def test_apply_calibrated_proba_fallback_without_calibrator():
    out = apply_calibrated_proba(None, np.array([0.25, 0.75]))
    assert np.allclose(out, [0.25, 0.75])


# ------------------------------------- 真实数据全流程：校准后 ECE 必须下降
@pytest.fixture(scope="module")
def real_split():
    data = load_credit_data()
    from sklearn.model_selection import train_test_split

    return train_test_split(data.X, data.y, test_size=0.2, stratify=data.y, random_state=42)


def test_full_pipeline_calibration_improves_ece(real_split):
    """固定 seed 全流程：LR+weight 管线 → OOF 校准器择优 → 测试集 ECE 必须下降。"""
    X_train, X_test, y_train, y_test = real_split
    pipeline = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "weight")
    calibrator, selection, test_report = calibrate_pipeline(
        pipeline, X_train, y_train, X_test, y_test,
        make_pipeline_fn=lambda: make_pipeline("logistic_regression", "weight"),
        n_splits=5, random_state=42,
    )
    assert calibrator.method in ("sigmoid", "isotonic")
    # 核心断言：校准后 ECE 严格小于校准前
    assert test_report["test_ece_after"] < test_report["test_ece_before"]
    # 单调校准不允许显著损失排序能力
    assert test_report["test_auc_after"] >= test_report["test_auc_before"] - 0.001
    # 择优依据真实记录
    assert test_report["test_brier_after"] < test_report["test_brier_before"]
    assert selection["ece_valid_raw"] > min(
        selection["ece_valid_sigmoid"], selection["ece_valid_isotonic"]
    )


def test_calibrated_bundle_end_to_end_round_trip(real_split, tmp_path):
    """calibrate_pipeline 产出的部署校准器经 bundle 序列化后预测一致。"""
    X_train, X_test, y_train, y_test = real_split
    pipeline = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "weight")
    calibrator, _, test_report = calibrate_pipeline(
        pipeline, X_train, y_train, X_test, y_test,
        make_pipeline_fn=lambda: make_pipeline("logistic_regression", "weight"),
        n_splits=5, random_state=42,
    )
    path = save_bundle(tmp_path / "deploy.joblib", build_bundle(
        pipeline, calibrator,
        {"features": list(X_train.columns), "model_version": "1.0.0",
         "risk_band_thresholds": {"low_below": 0.1, "medium_below": 0.5}},
    ))
    loaded = load_bundle(path)
    proba = loaded["pipeline"].predict_proba(X_test)[:, 1]
    expected = calibrator.apply(proba)
    got = loaded["calibrator"].apply(proba)
    assert np.allclose(got, expected)
    assert np.mean(got) == pytest.approx(test_report["test_mean_pd_after"], abs=1e-9)
