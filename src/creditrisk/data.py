"""数据加载层：OpenML credit-g → 本地缓存 → schema 一致的合成降级。

加载优先级（保证离线可复现、断网可用）：
1. data/raw/credit-g.csv 本地缓存（由首次 fetch_openml 写入，随仓库提交）；
2. sklearn fetch_openml 在线拉取（成功后回写缓存）；
3. 两者均失败时生成 schema 一致的合成数据，并通过 logger 警告
   （使用方应检查 returned 字段 `is_synthetic`）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from .config import (
    CAT_FEATURES,
    CATEGORY_VALUES,
    FEATURES,
    NUM_FEATURES,
    NUM_RANGES,
    RAW_DATA_PATH,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

OPENML_DATASET = "credit-g"
OPENML_VERSION = 1


@dataclass
class Dataset:
    """统一的数据容器：X（特征）、y（违约标签）与来源标记。"""

    X: pd.DataFrame
    y: pd.Series
    is_synthetic: bool = False

    @property
    def df(self) -> pd.DataFrame:
        return self.X.assign(**{TARGET_COL: self.y})


_LABEL_MAP = {"bad": 1, "good": 0, "1": 1, "0": 0}


def _coerce_target(y: pd.Series) -> pd.Series:
    """目标标签统一为 0/1：兼容 OpenML 原始标签（bad/good）与缓存整型（0/1）。"""
    s = y.astype(str).str.strip().str.lower()
    unknown = set(s.unique()) - set(_LABEL_MAP)
    if unknown:
        raise ValueError(f"未知目标取值: {sorted(unknown)}")
    return s.map(_LABEL_MAP).astype(int).rename(TARGET_COL)


def _normalize(X: pd.DataFrame, y: pd.Series) -> Dataset:
    """统一 dtypes：类别列转 str、数值列转 float，保证缓存读写与在线拉取一致。"""
    X = X.copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)
    for col in NUM_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return Dataset(X=X[FEATURES], y=_coerce_target(y))


def load_credit_data(cache_path: Path | None = RAW_DATA_PATH) -> Dataset:
    """加载 credit-g：缓存 → 在线拉取 → 合成降级。"""
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
        X, y = cached[FEATURES], cached[TARGET_COL]
        return _normalize(X, y)

    logger.info("本地缓存不存在，尝试从 OpenML 拉取 %s ...", OPENML_DATASET)
    try:
        raw_X, raw_y = fetch_openml(
            OPENML_DATASET,
            version=OPENML_VERSION,
            as_frame=True,
            return_X_y=True,
            parser="auto",
        )
    except Exception as exc:  # noqa: BLE001 —— 网络/解析失败均降级为合成数据
        logger.warning("OpenML 拉取失败（%s），使用 schema 一致的合成数据替代。", exc)
        return generate_synthetic_credit(is_synthetic=True)

    data = _normalize(raw_X, raw_y)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        data.df.to_csv(cache_path, index=False)
        logger.info("已写入本地缓存：%s", cache_path)
    return data


def generate_synthetic_credit(n: int = 1000, seed: int = 42, is_synthetic: bool = True) -> Dataset:
    """生成与 credit-g schema 一致的合成数据（列名/类别/数值范围相同）。

    信号构造：基于少量与违约强相关的特征做线性打分 + 逻辑函数，
    使坏样本占比约 30%，仅供管线联调与降级演示，指标不代表真实业务。
    """
    rng = np.random.default_rng(seed)

    def draw_cat(col: str) -> np.ndarray:
        values = CATEGORY_VALUES[col]
        return rng.choice(values, size=n)

    X = pd.DataFrame({col: draw_cat(col) for col in CAT_FEATURES})
    for col in NUM_FEATURES:
        lo, hi = NUM_RANGES[col]
        X[col] = np.round(rng.uniform(lo, hi, size=n), 2)

    risk = (
        1.2 * (X["checking_status"] == "<0").to_numpy()
        + 0.8 * (X["credit_history"] == "delayed previously").to_numpy()
        + 0.6 * (X["savings_status"] == "<100").to_numpy()
        + 0.5 * (X["employment"] == "unemployed").to_numpy()
        + 0.00008 * (X["credit_amount"].to_numpy() - NUM_RANGES["credit_amount"][0])
        + 0.02 * (X["duration"].to_numpy() - NUM_RANGES["duration"][0])
        + rng.normal(0, 0.8, size=n)
    )
    # 偏置 -3.4 经验校准：使合成坏样本率约 30%（credit-g 原始约 30%）
    prob = 1.0 / (1.0 + np.exp(-(risk - 3.4)))
    y = pd.Series(rng.binomial(1, prob, size=n), name=TARGET_COL)

    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)
    for col in NUM_FEATURES:
        X[col] = X[col].astype(float)
    return Dataset(X=X[FEATURES], y=y, is_synthetic=is_synthetic)


def assert_schema(data: Dataset) -> None:
    """校验数据符合 credit-g schema（列集合、类别取值），失败抛 ValueError。"""
    missing = set(FEATURES) - set(data.X.columns)
    if missing:
        raise ValueError(f"缺少特征列: {sorted(missing)}")
    for col, allowed in CATEGORY_VALUES.items():
        observed = set(data.X[col].dropna().unique())
        illegal = observed - set(allowed)
        if illegal:
            raise ValueError(f"{col} 出现未知类别: {sorted(illegal)}")
