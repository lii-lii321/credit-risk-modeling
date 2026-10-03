# -*- coding: utf-8 -*-
"""请求/响应契约：20 个 credit-g 特征的强校验 Pydantic v2 模型。

类别取值契约来自 creditrisk.config.CATEGORY_VALUES（OpenML credit-g 固定结构），
非法类别 → 422；数值特征不做硬范围拒绝（WOE 分箱会把越界值裁剪到边界箱，
并在 risk_band 之外原样返回解释）。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from creditrisk.config import CATEGORY_VALUES, NUM_FEATURES


class ScoreRequest(BaseModel):
    checking_status: str = Field(..., description="活期账户状态，如 '<0'")
    duration: float = Field(..., ge=0, description="贷款期限（月）")
    credit_history: str = Field(..., description="历史信用记录")
    purpose: str = Field(..., description="贷款用途")
    credit_amount: float = Field(..., ge=0, description="贷款金额")
    savings_status: str = Field(..., description="储蓄账户状态")
    employment: str = Field(..., description="现职工作年限")
    installment_commitment: float = Field(..., ge=0, description="可支配收入还款占比档（1-4）")
    personal_status: str = Field(..., description="婚姻与性别状态")
    other_parties: str = Field(..., description="其他担保人/共同申请人")
    residence_since: float = Field(..., ge=0, description="现住址居住年限（年）")
    property_magnitude: str = Field(..., description="财产状况")
    age: float = Field(..., ge=0, description="年龄")
    other_payment_plans: str = Field(..., description="其他分期计划")
    housing: str = Field(..., description="住房状况")
    existing_credits: float = Field(..., ge=0, description="本行已有信贷笔数")
    job: str = Field(..., description="职业资历")
    num_dependents: float = Field(..., ge=0, description="被抚养人数")
    own_telephone: str = Field(..., description="是否自有电话")
    foreign_worker: str = Field(..., description="是否外籍工人")

    @field_validator(*CATEGORY_VALUES.keys())
    @classmethod
    def category_must_be_known(cls, v: str, info) -> str:
        allowed = CATEGORY_VALUES.get(info.field_name)
        if allowed is not None and v not in allowed:
            raise ValueError(f"{info.field_name} 取值 '{v}' 不在合法类别中: {allowed}")
        return v

    def to_feature_frame(self):
        from creditrisk.config import FEATURES

        data = self.model_dump()
        return {name: [data[name]] for name in FEATURES}


class TopFeature(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    probability_of_default_raw: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="校准前（原始模型）PD；未部署校准器时为 null"
    )
    calibrated: bool = Field(..., description="PD 是否经过概率校准（isotonic/sigmoid）")
    calibration_method: str = Field(..., description="校准方法：isotonic / sigmoid / none")
    risk_band: str
    top_features: list[TopFeature]
    model_version: str
    model: str
    explanation_method: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model: str
    trained_at: str
    calibrated: bool = Field(..., description="是否加载了概率校准器（deploy_bundle.joblib）")
    calibration_method: str = Field(..., description="校准方法：isotonic / sigmoid / none")


def numeric_features() -> list[str]:
    return list(NUM_FEATURES)
