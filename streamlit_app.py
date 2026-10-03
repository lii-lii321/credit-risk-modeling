# -*- coding: utf-8 -*-
"""Streamlit 交互 Demo：单笔信贷评分（PD + top 特征贡献 + 审批阈值）。

本地运行：streamlit run streamlit_app.py
模型产物为仓库内 artifacts/deploy_bundle.joblib（含校准器与特征 schema 元数据，
由 scripts/run_training.py 产出并随仓库提交），Streamlit Community Cloud 可直接加载。

代码组织：全部评分逻辑为不依赖 st.* 的纯函数（可单测，见 tests/test_streamlit_app.py），
Streamlit 只做输入采集与结果呈现的薄壳；UI 代码在 `__main__` 守卫内，
模块可被安全 import（pytest 冒烟）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from creditrisk.calibration_repair import apply_calibrated_proba, load_bundle  # noqa: E402
from creditrisk.config import ARTIFACTS_DIR, CATEGORY_VALUES, NUM_RANGES  # noqa: E402

BUNDLE_PATH = ARTIFACTS_DIR / "deploy_bundle.joblib"

# 数值特征滑杆规格：(label, min, max, default, step, help)
NUMERIC_SPECS: Dict[str, Dict] = {
    "duration": {"label": "贷款期限（月）", "step": 1, "default": 12},
    "credit_amount": {"label": "贷款金额（DM）", "step": 100, "default": 1500},
    "installment_commitment": {"label": "还款收入占比档（1-4）", "step": 1, "default": 2},
    "residence_since": {"label": "现住址居住年限（年）", "step": 1, "default": 4},
    "age": {"label": "年龄", "step": 1, "default": 35},
    "existing_credits": {"label": "本行已有信贷笔数", "step": 1, "default": 1},
    "num_dependents": {"label": "被抚养人数", "step": 1, "default": 1},
}

# 类别特征默认值（与 README/API 示例的优质画像一致，首屏呈现可批准的低风险样例）
CATEGORICAL_DEFAULTS: Dict[str, str] = {
    "checking_status": ">=200",
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "savings_status": ">=1000",
    "employment": ">=7",
    "personal_status": "male single",
    "other_parties": "none",
    "property_magnitude": "real estate",
    "other_payment_plans": "none",
    "housing": "own",
    "job": "skilled",
    "own_telephone": "yes",
    "foreign_worker": "no",
}

_BUNDLE_CACHE: Dict[str, dict] = {}


def load_deploy_bundle(artifacts_dir: Optional[Path] = None) -> dict:
    """加载部署 bundle（进程内缓存）；缺失时抛出带修复指引的 RuntimeError。

    纯函数：不依赖 st.*，供 UI 与测试共用。
    """
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    key = str(artifacts_dir.resolve())
    if key not in _BUNDLE_CACHE:
        bundle = load_bundle(artifacts_dir / "deploy_bundle.joblib")
        meta_path = artifacts_dir / "model_meta.json"
        if meta_path.exists():
            bundle["meta"].setdefault("trained_at",
                                      json.loads(meta_path.read_text(encoding="utf-8")).get("trained_at"))
        _BUNDLE_CACHE[key] = bundle
    return _BUNDLE_CACHE[key]


def default_applicant() -> Dict[str, object]:
    """一组合法的默认申请画像：类别取默认值，数值取各特征默认/下界。"""
    features: Dict[str, object] = {name: default for name, default in CATEGORICAL_DEFAULTS.items()}
    for name, spec in NUMERIC_SPECS.items():
        lo = NUM_RANGES[name][0]
        value = spec["default"]
        features[name] = float(max(lo, value))
    return features


def approval_decision(pd_value: float, threshold: float) -> str:
    """审批决策纯函数：PD < 阈值 → 批准，否则拒绝。"""
    return "approve" if pd_value < threshold else "reject"


def risk_band(pd_value: float, thresholds: Dict[str, float]) -> str:
    if pd_value < thresholds["low_below"]:
        return "low"
    if pd_value < thresholds["medium_below"]:
        return "medium"
    return "high"


def score_applicant(bundle: dict, features: Dict[str, object], threshold: float,
                    top_k: int = 3) -> Dict[str, object]:
    """单笔评分纯函数：bundle + 画像 + 阈值 → PD/解释/风险分档/审批决策。

    - pd：校准后违约概率（部署口径，可直接解读为概率）；
    - pd_raw：校准前概率（参照）；
    - top_features：coef × WOE 偏移贡献（对数几率尺度，正=推高违约），
      复用 creditrisk.explain.explain_instance，与训练报告/ API 同口径。
    """
    from creditrisk.explain import explain_instance

    pipeline = bundle["pipeline"]
    meta = bundle["meta"]
    frame = pd.DataFrame([features])[meta["features"]]
    pd_raw = float(pipeline.predict_proba(frame)[0, 1])
    pd_calibrated = float(apply_calibrated_proba(bundle.get("calibrator"), pd_raw)[0])
    return {
        "pd": pd_calibrated,
        "pd_raw": pd_raw,
        "top_features": explain_instance(pipeline, frame, top_k=top_k),
        "risk_band": risk_band(pd_calibrated, meta["risk_band_thresholds"]),
        "decision": approval_decision(pd_calibrated, threshold),
        "threshold": float(threshold),
    }


# ------------------------------------------------------------------- UI
if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="信贷违约概率评分 Demo", page_icon="🏦", layout="wide")

    try:
        bundle = load_deploy_bundle()
    except RuntimeError as exc:
        st.error(f"**模型产物加载失败**：{exc}")
        st.caption("产物会随仓库提交；若在全新环境运行，请在仓库目录执行 "
                   "`python scripts/run_training.py` 重新生成 artifacts/deploy_bundle.joblib。")
        st.stop()

    meta = bundle["meta"]
    calibration_meta = meta.get("calibration") or {}

    st.title("信贷违约概率（PD）评分 Demo")
    st.caption(
        f"模型：{meta.get('model')} + {meta.get('strategy')}（v{meta.get('model_version')}） · "
        f"概率校准：{calibration_meta.get('method', 'none')}（测试集 ECE "
        f"{calibration_meta.get('test_ece_before', float('nan')):.3f} → "
        f"{calibration_meta.get('test_ece_after', float('nan')):.3f}） · "
        f"数据：OpenML credit-g · 仅限方法演示，不构成真实信贷决策依据"
    )

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("申请人画像")
        inputs: Dict[str, object] = {}
        for name in meta.get("categorical_features", list(CATEGORICAL_DEFAULTS)):
            inputs[name] = st.selectbox(name, CATEGORY_VALUES[name],
                                        index=CATEGORY_VALUES[name].index(CATEGORICAL_DEFAULTS[name]))
        for name, spec in NUMERIC_SPECS.items():
            lo, hi = NUM_RANGES[name]
            inputs[name] = float(st.slider(spec["label"], min_value=float(lo), max_value=float(hi),
                                           value=float(spec["default"]), step=float(spec["step"])))

        st.divider()
        threshold = st.slider("审批阈值（PD < 阈值 → 批准）", min_value=0.05, max_value=0.95,
                              value=0.50, step=0.01, format="%.2f")

    with right:
        result = score_applicant(bundle, inputs, threshold)

        col_pd, col_raw = st.columns(2)
        col_pd.metric("违约概率 PD（校准后）", f"{result['pd']:.1%}")
        col_raw.metric("校准前 PD（参照）", f"{result['pd_raw']:.1%}")

        band_text = {"low": "低风险 🟢", "medium": "中风险 🟡", "high": "高风险 🔴"}[result["risk_band"]]
        st.caption(f"风险分档（校准后 PD 的 60%/85% 分位阈值）：**{band_text}**")

        if result["decision"] == "approve":
            st.success(f"✅ 批准 —— PD {result['pd']:.1%} < 阈值 {threshold:.0%}")
        else:
            st.error(f"⛔ 拒绝 —— PD {result['pd']:.1%} ≥ 阈值 {threshold:.0%}")

        st.subheader("Top 特征贡献（coef × WOE 偏移，对数几率尺度）")
        top = result["top_features"]
        chart = pd.DataFrame(
            {"contribution": [item["contribution"] for item in top]},
            index=[item["feature"] for item in top],
        )
        st.bar_chart(chart)
        st.caption("正值 = 推高违约风险，负值 = 降低违约风险；贡献为逻辑回归系数 × "
                   "WOE 偏离基准的量，与训练报告、API 的解释口径一致。")

        with st.expander("完整评分结果（JSON）"):
            st.json(result, expanded=True)
