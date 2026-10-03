# -*- coding: utf-8 -*-
"""FastAPI 评分服务：POST /score（违约概率 + 风险分档 + top 特征解释）、GET /health。

启动时优先加载 artifacts/deploy_bundle.joblib（模型 + 校准器 + schema 元数据，
由 scripts/run_training.py 产出），PD 输出为校准后概率；bundle 缺失时回退到
pipeline.joblib（校准前 PD，响应中如实标记 calibrated=False）。两者皆缺失时服务
以降级模式启动，/score 与 /health 返回 503 model_unavailable（而非崩溃退出）。
单样本解释复用 creditrisk.explain.explain_instance：部署模型为逻辑回归时走
coef × WOE 偏移路径，为 LightGBM 时走 Tree SHAP 路径（与训练报告口径一致）。

错误响应规范化：所有非 2xx 统一返回
``{"error": "<机器可读码>", "detail": "<人话说明>", "hint": "<怎么修>"}``。
错误码清单见 README「评分服务契约」。
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# 让 app 包在没有 pip install 的情况下也能找到 src/creditrisk
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.schemas import HealthResponse, ScoreRequest, ScoreResponse, TopFeature  # noqa: E402
from creditrisk.calibration_repair import apply_calibrated_proba  # noqa: E402
from creditrisk.config import ARTIFACTS_DIR, FEATURES  # noqa: E402
from creditrisk.explain import explain_instance  # noqa: E402

logger = logging.getLogger("uvicorn.error")

PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
BUNDLE_PATH = ARTIFACTS_DIR / "deploy_bundle.joblib"
META_PATH = ARTIFACTS_DIR / "model_meta.json"

_state: Dict[str, Any] = {}

# 统一错误目录：error 码 → 默认 detail / hint（README 错误码表的单一来源）
_ERROR_CATALOG: Dict[str, Dict[str, str]] = {
    "validation_error": {
        "detail": "请求体校验失败",
        "hint": "detail 中含具体字段与原因；对照 README「评分服务契约」的 20 个特征名"
                "与类别白名单修正请求体（字段名区分大小写）。",
    },
    "model_unavailable": {
        "detail": "模型产物缺失或加载失败",
        "hint": "先运行 python scripts/run_training.py 生成 artifacts/ 产物，再重启服务。",
    },
    "not_found": {
        "detail": "请求的路径不存在",
        "hint": "可用端点：POST /score、GET /health、GET /docs。",
    },
    "method_not_allowed": {
        "detail": "该路径不支持此 HTTP 方法",
        "hint": "例如 /score 仅接受 POST、/health 仅接受 GET。",
    },
    "internal_error": {
        "detail": "服务器内部错误",
        "hint": "完整堆栈已记录在服务端日志；请重试，若持续失败请附带请求时间提 issue。",
    },
    "http_error": {
        "detail": "请求未被接受",
        "hint": "请检查请求后重试。",
    },
}


def _error_response(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    """所有非 2xx 的统一出口：{"error", "detail", "hint"} 三键结构。"""
    catalog = _ERROR_CATALOG.get(error, _ERROR_CATALOG["http_error"])
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail or catalog["detail"],
            "hint": catalog["hint"],
        },
    )


def _load_error_detail() -> str:
    err = _state.get("load_error")
    if err:
        return f"模型产物加载失败：{err}"
    return "模型未加载"


def _load_artifacts() -> None:
    try:
        if not META_PATH.exists():
            raise RuntimeError(
                f"缺少模型产物（{META_PATH}）。请先运行 python scripts/run_training.py 生成。"
            )
        _state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))

        if BUNDLE_PATH.exists():
            # 首选：含校准器的部署 bundle（模型 + 校准器 + schema 元数据单一来源）
            bundle = joblib.load(BUNDLE_PATH)
            _state["pipeline"] = bundle["pipeline"]
            _state["calibrator"] = bundle.get("calibrator")
            _state["bundle_features"] = bundle["meta"].get("features", FEATURES)
        elif PIPELINE_PATH.exists():
            # 回退：仅校准前管线（诚实标记未校准）
            _state["pipeline"] = joblib.load(PIPELINE_PATH)
            _state["calibrator"] = None
            _state["bundle_features"] = FEATURES
        else:
            raise RuntimeError(
                f"缺少模型产物（{PIPELINE_PATH} / {BUNDLE_PATH}）。"
                "请先运行 python scripts/run_training.py 生成。"
            )
    except Exception as exc:  # noqa: BLE001 —— 产物缺失/损坏时降级启动而非崩溃退出
        _state.clear()
        _state["load_error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("模型产物加载失败，服务以降级模式启动（/score 与 /health 返回 503）：%s",
                       _state["load_error"])


def _risk_band(pd_value: float, thresholds: Dict[str, float]) -> str:
    if pd_value < thresholds["low_below"]:
        return "low"
    if pd_value < thresholds["medium_below"]:
        return "medium"
    return "high"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="credit-risk-modeling",
    description="信贷违约概率评分 API（credit-g 端到端建模）",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422：Pydantic 校验失败 → 统一 validation_error 结构，detail 保留字段级原因。"""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        msg = err.get("msg", "校验失败")
        parts.append(f"{loc}: {msg}" if loc else msg)
    detail = "；".join(parts) if parts else None
    return _error_response(422, "validation_error", f"请求体校验失败 → {detail}")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """404/405/503 及其他显式 HTTPException → 统一错误结构。"""
    code_by_status = {
        404: "not_found",
        405: "method_not_allowed",
        500: "internal_error",
        503: "model_unavailable",
    }
    error = code_by_status.get(exc.status_code, "http_error")
    # 503 的 detail 携带具体加载失败原因；其余用目录默认文案，避免回显内部信息
    detail = str(exc.detail) if (error == "model_unavailable" and exc.detail) else None
    return _error_response(exc.status_code, error, detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底 500：完整堆栈只进服务端日志，响应不泄漏任何异常细节。"""
    logger.exception("未处理异常（%s %s）", request.method, request.url.path)
    return _error_response(500, "internal_error")


@app.get("/health")
def health() -> HealthResponse:
    meta = _state.get("meta")
    if meta is None:
        raise HTTPException(status_code=503, detail=_load_error_detail())
    calibrator = _state.get("calibrator")
    calibration_meta = meta.get("calibration") or {}
    return HealthResponse(
        status="ok",
        model_version=meta["model_version"],
        model=meta["model"],
        trained_at=meta["trained_at"],
        calibrated=calibrator is not None,
        calibration_method=(calibration_meta.get("method") or "none") if calibrator is not None else "none",
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    pipeline = _state.get("pipeline")
    meta = _state.get("meta")
    calibrator = _state.get("calibrator")
    features = _state.get("bundle_features", FEATURES)
    if pipeline is None or meta is None:
        raise HTTPException(status_code=503, detail=_load_error_detail())

    try:
        frame = pd.DataFrame(req.to_feature_frame(), columns=FEATURES)
        pd_raw = float(pipeline.predict_proba(frame)[0, 1])
        proba = float(apply_calibrated_proba(calibrator, np.array([pd_raw]))[0])
        top = explain_instance(pipeline, frame, top_k=3)
    except Exception:  # noqa: BLE001 —— 评分异常统一转 500，堆栈仅进服务端日志
        logger.exception("评分失败")
        raise HTTPException(status_code=500, detail="评分过程发生内部错误") from None

    calibration_meta = meta.get("calibration") or {}
    calibrated = calibrator is not None
    return ScoreResponse(
        probability_of_default=proba,
        probability_of_default_raw=pd_raw if calibrated else None,
        calibrated=calibrated,
        calibration_method=(calibration_meta.get("method") or "none") if calibrated else "none",
        risk_band=_risk_band(proba, meta["risk_band_thresholds"]),
        top_features=[TopFeature(**item) for item in top],
        model_version=meta["model_version"],
        model=meta["model"],
        explanation_method="shap" if meta["model"] == "lightgbm" else "coef_x_woe_deviation",
    )
