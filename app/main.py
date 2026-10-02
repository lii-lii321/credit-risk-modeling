# -*- coding: utf-8 -*-
"""FastAPI 评分服务：POST /score（违约概率 + 风险分档 + top 特征解释）、GET /health。

启动时加载 artifacts/pipeline.joblib 与 model_meta.json（由 scripts/run_training.py 产出）；
单样本解释复用 creditrisk.explain.explain_instance：部署模型为逻辑回归时走
coef × WOE 偏移路径，为 LightGBM 时走 Tree SHAP 路径（与训练报告口径一致）。
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# 让 app 包在没有 pip install 的情况下也能找到 src/creditrisk
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.schemas import ScoreRequest, ScoreResponse, TopFeature  # noqa: E402
from creditrisk.config import ARTIFACTS_DIR, FEATURES  # noqa: E402
from creditrisk.explain import explain_instance  # noqa: E402

PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
META_PATH = ARTIFACTS_DIR / "model_meta.json"

_state: Dict[str, Any] = {}


def _load_artifacts() -> None:
    if not PIPELINE_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            f"缺少模型产物（{PIPELINE_PATH} / {META_PATH}）。"
            "请先运行 python scripts/run_training.py 生成。"
        )
    # joblib 反序列化仅加载本仓库训练脚本产出的第一方产物（可信来源）；
    # 生产环境应从带签名/校验的模型注册中心拉取，而非直接加载外部提供的文件。
    _state["pipeline"] = joblib.load(PIPELINE_PATH)
    _state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))


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
    if pipeline is None or meta is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        frame = pd.DataFrame(req.to_feature_frame(), columns=FEATURES)
        proba = float(pipeline.predict_proba(frame)[0, 1])
        top = explain_instance(pipeline, frame, top_k=3)
    except Exception as exc:  # noqa: BLE001 —— 评分异常统一转 500，避免泄漏堆栈
        raise HTTPException(status_code=500, detail=f"评分失败: {exc}") from exc

    return ScoreResponse(
        probability_of_default=proba,
        risk_band=_risk_band(proba, meta["risk_band_thresholds"]),
        top_features=[TopFeature(**item) for item in top],
        model_version=meta["model_version"],
        model=meta["model"],
        explanation_method="shap" if meta["model"] == "lightgbm" else "coef_x_woe_deviation",
    )
