"""全局配置：路径、随机种子、目标定义与 credit-g 特征 schema。

schema 常量来自 OpenML credit-g (version 1) 的实际导出
（scripts/inspect_schema.py），属于公开 Statlog German Credit 数据的固定结构，
不含任何敏感信息。
"""
from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 目标定义：credit-g 的 class 字段，bad=违约（正类），good=正常
TARGET_COL = "default"  # 1=违约(bad), 0=正常(good)
POSITIVE_LABEL = 1

CAT_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

NUM_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

FEATURES = CAT_FEATURES + NUM_FEATURES

# 每个类别特征的合法取值（与 OpenML credit-g v1 一致，API 侧用于请求校验）
CATEGORY_VALUES: dict[str, list[str]] = {
    "checking_status": ["0<=X<200", "<0", ">=200", "no checking"],
    "credit_history": [
        "all paid",
        "critical/other existing credit",
        "delayed previously",
        "existing paid",
        "no credits/all paid",
    ],
    "purpose": [
        "business",
        "domestic appliance",
        "education",
        "furniture/equipment",
        "new car",
        "other",
        "radio/tv",
        "repairs",
        "retraining",
        "used car",
    ],
    "savings_status": ["100<=X<500", "500<=X<1000", "<100", ">=1000", "no known savings"],
    "employment": ["1<=X<4", "4<=X<7", "<1", ">=7", "unemployed"],
    "personal_status": ["female div/dep/mar", "male div/sep", "male mar/wid", "male single"],
    "other_parties": ["co applicant", "guarantor", "none"],
    "property_magnitude": ["car", "life insurance", "no known property", "real estate"],
    "other_payment_plans": ["bank", "none", "stores"],
    "housing": ["for free", "own", "rent"],
    "job": ["high qualif/self emp/mgmt", "skilled", "unemp/unskilled non res", "unskilled resident"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["no", "yes"],
}

# 数值特征的训练集观测范围（用于合成降级与 API 合理性提示，非硬校验）
NUM_RANGES: dict[str, tuple[float, float]] = {
    "duration": (4.0, 72.0),
    "credit_amount": (250.0, 18424.0),
    "installment_commitment": (1.0, 4.0),
    "residence_since": (1.0, 4.0),
    "age": (19.0, 75.0),
    "existing_credits": (1.0, 4.0),
    "num_dependents": (1.0, 2.0),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit-g.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
