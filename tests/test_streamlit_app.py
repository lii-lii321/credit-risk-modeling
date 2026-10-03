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


def test_streamlit_ui_renders_and_profile_flips_decision():
    """AppTest 真实渲染薄壳 UI：无异常、PD 指标存在、画像改动联动审批结论。

    注意：不要用 set_value 设滑杆越界值（低于 min_value 会被新版 Streamlit
    静默裁剪回默认值，导致断言不稳定），决策联动改由高风险画像驱动。
    """
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    def decision_text(at):
        if at.error:
            return at.error[0].value
        if at.success:
            return at.success[0].value
        return ""

    at = AppTest.from_file(str(app_path), default_timeout=180)
    at.run()
    assert not at.exception
    assert at.metric[0].label == "违约概率 PD（校准后）"
    assert len(at.selectbox) == 13 and len(at.slider) == 8
    # 默认优质画像：应显示批准
    assert "批准" in decision_text(at)

    # 通过界面控件切到完整高风险画像（与 tests/test_api.py 的 RISKY 同参）：
    # 校准后 PD 升至约 90%，默认阈值 0.5 下应转为拒绝
    selectboxes = {s.label: s for s in at.selectbox}
    sliders = {s.label: s for s in at.slider}
    selectboxes["checking_status"].set_value("<0")
    selectboxes["credit_history"].set_value("delayed previously")
    selectboxes["savings_status"].set_value("<100")
    selectboxes["employment"].set_value("unemployed")
    selectboxes["property_magnitude"].set_value("no known property")
    selectboxes["housing"].set_value("rent")
    selectboxes["other_payment_plans"].set_value("bank")
    selectboxes["foreign_worker"].set_value("yes")
    sliders["贷款期限（月）"].set_value(48.0).run()
    sliders["贷款金额（DM）"].set_value(12000.0).run()
    assert not at.exception
    assert "拒绝" in decision_text(at)
