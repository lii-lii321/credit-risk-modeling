# -*- coding: utf-8 -*-
"""FastAPI 评分服务：POST /score（违约概率 + 风险分档 + top 特征解释）、GET /health。

启动时优先加载 artifacts/deploy_bundle.joblib（模型 + 校准器 + schema 元数据，
由 scripts/run_training.py 产出），PD 输出为校准后概率；bundle 缺失时回退到
pipeline.joblib（校准前 PD，响应中如实标记 calibrated=False）。
单样本解释复用 creditrisk.explain.explain_instance：部署模型为逻辑回归时走
coef × WOE 偏移路径，为 LightGBM 时走 Tree SHAP 路径（与训练报告口径一致）。
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# 让 app 包在没有 pip install 的情况下也能找到 src/creditrisk
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.schemas import ScoreRequest, ScoreResponse, TopFeature  # noqa: E402
from creditrisk.calibration_repair import apply_calibrated_proba  # noqa: E402
from creditrisk.config import ARTIFACTS_DIR, FEATURES  # noqa: E402
from creditrisk.explain import explain_instance  # noqa: E402

PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
BUNDLE_PATH = ARTIFACTS_DIR / "deploy_bundle.joblib"
META_PATH = ARTIFACTS_DIR / "model_meta.json"

_state: Dict[str, Any] = {}


def _load_artifacts() -> None:
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


@app.get("/health")
def health() -> Dict[str, str]:
    meta = _state.get("meta")
    if meta is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {
        "status": "ok",
        "model_version": meta["model_version"],
        "model": meta["model"],
        "trained_at": meta["trained_at"],
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    pipeline = _state.get("pipeline")
    meta = _state.get("meta")
    calibrator = _state.get("calibrator")
    features = _state.get("bundle_features", FEATURES)
    if pipeline is None or meta is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        frame = pd.DataFrame(req.to_feature_frame(), columns=FEATURES)
        pd_raw = float(pipeline.predict_proba(frame)[0, 1])
        proba = float(apply_calibrated_proba(calibrator, np.array([pd_raw]))[0])
        top = explain_instance(pipeline, frame, top_k=3)
    except Exception as exc:  # noqa: BLE001 —— 评分异常统一转 500，避免泄漏堆栈
        raise HTTPException(status_code=500, detail=f"评分失败: {exc}") from exc

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
