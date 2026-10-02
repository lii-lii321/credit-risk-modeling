# -*- coding: utf-8 -*-
"""API 契约测试：/health 与 /score 的正常流、校验失败流与解释口径。

依赖 artifacts/pipeline.joblib + model_meta.json（随仓库提交，
由 scripts/run_training.py 真实训练产出，非 mock）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


GOOD_APPLICANT = {
    "checking_status": ">=200",
    "duration": 12.0,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 1500.0,
    "savings_status": ">=1000",
    "employment": ">=7",
    "installment_commitment": 1.0,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 4.0,
    "property_magnitude": "real estate",
    "age": 45.0,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1.0,
    "job": "skilled",
    "num_dependents": 1.0,
    "own_telephone": "yes",
    "foreign_worker": "no",
}

RISKY_APPLICANT = {
    "checking_status": "<0",
    "duration": 48.0,
    "credit_history": "delayed previously",
    "purpose": "new car",
    "credit_amount": 12000.0,
    "savings_status": "<100",
    "employment": "unemployed",
    "installment_commitment": 4.0,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 1.0,
    "property_magnitude": "no known property",
    "age": 22.0,
    "other_payment_plans": "bank",
    "housing": "rent",
    "existing_credits": 4.0,
    "job": "unskilled resident",
    "num_dependents": 2.0,
    "own_telephone": "none",
    "foreign_worker": "yes",
}


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] in ("logistic_regression", "lightgbm")
    assert body["model_version"]
    assert body["trained_at"]


def test_score_valid_request(client):
    resp = client.post("/score", json=GOOD_APPLICANT)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["risk_band"] in ("low", "medium", "high")
    assert body["model_version"]
    assert body["explanation_method"] in ("shap", "coef_x_woe_deviation")
    assert len(body["top_features"]) == 3
    for item in body["top_features"]:
        assert set(item) == {"feature", "contribution"}
        assert item["feature"] in GOOD_APPLICANT


def test_score_top_features_sorted_by_abs_contribution(client):
    body = client.post("/score", json=GOOD_APPLICANT).json()
    contributions = [abs(item["contribution"]) for item in body["top_features"]]
    assert contributions == sorted(contributions, reverse=True)


def test_score_risky_applicant_scores_higher_than_good(client):
    """确定性模型下，高风险画像的违约概率应显著高于优质画像。"""
    good = client.post("/score", json=GOOD_APPLICANT).json()["probability_of_default"]
    risky = client.post("/score", json=RISKY_APPLICANT).json()["probability_of_default"]
    assert risky > good + 0.2


def test_score_risky_applicant_top_feature_is_negative_signal(client):
    """高风险画像的 top 解释应包含已知负向信号特征（checking_status/savings_status 等）。"""
    body = client.post("/score", json=RISKY_APPLICANT).json()
    top = {item["feature"] for item in body["top_features"]}
    assert top & {"checking_status", "savings_status", "credit_history", "duration"}


def test_score_rejects_unknown_category(client):
    bad = {**GOOD_APPLICANT, "checking_status": "一百万存款"}
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422
    assert "checking_status" in resp.text


def test_score_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD_APPLICANT.items() if k != "age"}
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422


def test_score_rejects_negative_numeric(client):
    bad = {**GOOD_APPLICANT, "age": -1.0}
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422


def test_score_accepts_boundary_values(client):
    """数值越界不报错：WOE 分箱把值裁剪到边界箱（服务契约：仅类别做硬校验）。"""
    extreme = {**GOOD_APPLICANT, "age": 200.0, "credit_amount": 9_999_999.0}
    resp = client.post("/score", json=extreme)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["probability_of_default"] <= 1.0
