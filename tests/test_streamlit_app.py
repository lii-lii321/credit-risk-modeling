# -*- coding: utf-8 -*-
"""Streamlit Demo 冒烟测试：模块可安全 import、纯函数打分行为正确。

依赖仓库内真实产物 artifacts/deploy_bundle.joblib（随仓库提交，
由 scripts/run_training.py 产出，非 mock）。
"""
from __future__ import annotations

import numpy as np
import pytest

import streamlit_app as demo
from creditrisk.config import CATEGORY_VALUES, NUM_RANGES


@pytest.fixture(scope="module")
def bundle():
    return demo.load_deploy_bundle()


def test_module_import_is_side_effect_free():
    """import 不触发 UI / 不加载模型：纯函数可直接调用。"""
    assert callable(demo.score_applicant)
    assert callable(demo.load_deploy_bundle)
    assert demo._BUNDLE_CACHE == {} or all(isinstance(v, dict) for v in demo._BUNDLE_CACHE.values())


def test_default_applicant_matches_schema():
    applicant = demo.default_applicant()
    assert set(applicant) == set(CATEGORY_VALUES) | set(NUM_RANGES)
    for name, value in applicant.items():
        if name in CATEGORY_VALUES:
            assert value in CATEGORY_VALUES[name]
        else:
            lo, hi = NUM_RANGES[name]
            assert lo <= value <= hi


def test_score_applicant_returns_valid_probability_and_decision(bundle):
    result = demo.score_applicant(bundle, demo.default_applicant(), threshold=0.5)
    assert 0.0 <= result["pd"] <= 1.0
    assert 0.0 <= result["pd_raw"] <= 1.0
    assert result["decision"] == demo.approval_decision(result["pd"], 0.5)
    assert result["risk_band"] in ("low", "medium", "high")
    # 优质默认画像在 0.5 阈值下应可批准
    assert result["decision"] == "approve"
    assert len(result["top_features"]) == 3


def test_top_features_structure_and_order(bundle):
    result = demo.score_applicant(bundle, demo.default_applicant(), threshold=0.5)
    for item in result["top_features"]:
        assert set(item) == {"feature", "contribution"}
        assert item["feature"] in CATEGORY_VALUES or item["feature"] in NUM_RANGES
    contributions = [abs(item["contribution"]) for item in result["top_features"]]
    assert contributions == sorted(contributions, reverse=True)


def test_risky_applicant_scores_higher_than_good(bundle):
    good = demo.score_applicant(bundle, demo.default_applicant(), threshold=0.5)
    risky_profile = {**demo.default_applicant(),
                     "checking_status": "<0", "duration": 48.0, "credit_amount": 12000.0,
                     "credit_history": "delayed previously", "savings_status": "<100",
                     "employment": "unemployed", "installment_commitment": 4.0,
                     "property_magnitude": "no known property", "age": 22.0,
                     "housing": "rent", "existing_credits": 4.0, "num_dependents": 2.0,
                     "other_payment_plans": "bank", "foreign_worker": "yes"}
    risky = demo.score_applicant(bundle, risky_profile, threshold=0.5)
    assert risky["pd"] > good["pd"]
    # 单调校准不改判风险分档的相对高低
    assert demo.risk_band(risky["pd"], bundle["meta"]["risk_band_thresholds"]) in ("medium", "high")


def test_approval_decision_threshold_semantics():
    assert demo.approval_decision(0.12, 0.5) == "approve"
    assert demo.approval_decision(0.50, 0.5) == "reject"  # 等于阈值 → 拒绝
    assert demo.approval_decision(0.87, 0.5) == "reject"


def test_risk_band_boundaries():
    thresholds = {"low_below": 0.3, "medium_below": 0.6}
    assert demo.risk_band(0.299, thresholds) == "low"
    assert demo.risk_band(0.3, thresholds) == "medium"
    assert demo.risk_band(0.6, thresholds) == "high"


def test_load_bundle_missing_dir_has_actionable_message(tmp_path):
    demo._BUNDLE_CACHE.pop(str(tmp_path.resolve()), None)
    with pytest.raises(RuntimeError) as exc_info:
        demo.load_deploy_bundle(tmp_path)
    assert "run_training" in str(exc_info.value)


def test_streamlit_ui_renders_and_threshold_flips_decision():
    """AppTest 真实渲染薄壳 UI：无异常、PD 指标存在、阈值滑杆联动审批结论。"""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=180)
    at.run()
    assert not at.exception
    assert at.metric[0].label == "违约概率 PD（校准后）"
    assert len(at.selectbox) == 13 and len(at.slider) == 8

    thr = [s for s in at.slider if "阈值" in s.label][0]
    thr.set_value(0.01).run()
    assert not at.exception
    assert at.error  # 极低阈值下默认画像转为拒绝
    assert "拒绝" in at.error[0].value
