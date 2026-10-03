# -*- coding: utf-8 -*-
"""API 错误规范化契约测试：所有非 2xx 统一返回 {"error", "detail", "hint"} 三键结构。

覆盖分支：422 校验失败（类别非法/缺字段）、404 未知路径、405 方法不允许、
500 内部异常兜底（不泄漏堆栈）、503 模型产物缺失（monkeypatch 指向不存在路径，
服务以降级模式启动）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as api_main
from app.main import app
from test_api import GOOD_APPLICANT


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _assert_error_shape(body: dict, expected_error: str) -> None:
    assert set(body) == {"error", "detail", "hint"}
    assert body["error"] == expected_error
    assert body["detail"]
    assert body["hint"]


def test_validation_error_unknown_category(client):
    bad = {**GOOD_APPLICANT, "checking_status": "一百万存款"}
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422
    body = resp.json()
    _assert_error_shape(body, "validation_error")
    assert "checking_status" in body["detail"]


def test_validation_error_missing_field(client):
    bad = {k: v for k, v in GOOD_APPLICANT.items() if k != "age"}
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422
    body = resp.json()
    _assert_error_shape(body, "validation_error")
    assert "age" in body["detail"]


def test_unknown_path_returns_not_found(client):
    resp = client.get("/no/such/path")
    assert resp.status_code == 404
    _assert_error_shape(resp.json(), "not_found")


def test_method_not_allowed(client):
    resp = client.get("/score")
    assert resp.status_code == 405
    _assert_error_shape(resp.json(), "method_not_allowed")


def test_internal_error_masks_exception_detail(client, monkeypatch):
    """评分过程抛异常 → 500 internal_error，异常细节不得出现在响应中。"""

    def _boom(*args, **kwargs):
        raise RuntimeError("boom secret stack detail")

    monkeypatch.setattr(api_main, "explain_instance", _boom)
    resp = client.post("/score", json=GOOD_APPLICANT)
    assert resp.status_code == 500
    body = resp.json()
    _assert_error_shape(body, "internal_error")
    assert "boom" not in resp.text
    assert "Traceback" not in resp.text


def test_model_unavailable_when_artifacts_missing(monkeypatch):
    """产物缺失场景：monkeypatch 指向不存在路径，服务降级启动并返回 503。"""
    empty_dir = Path(__file__).parent / "__nonexistent_artifacts__"
    monkeypatch.setattr(api_main, "META_PATH", empty_dir / "model_meta.json")
    monkeypatch.setattr(api_main, "BUNDLE_PATH", empty_dir / "deploy_bundle.joblib")
    monkeypatch.setattr(api_main, "PIPELINE_PATH", empty_dir / "pipeline.joblib")

    try:
        with TestClient(app) as degraded:
            health = degraded.get("/health")
            assert health.status_code == 503
            _assert_error_shape(health.json(), "model_unavailable")

            score = degraded.post("/score", json=GOOD_APPLICANT)
            assert score.status_code == 503
            _assert_error_shape(score.json(), "model_unavailable")
            assert "run_training" in score.json()["hint"]
    finally:
        # 恢复全局 _state，避免污染后续测试（monkeypatch 自动还原路径常量）
        api_main._load_artifacts()
