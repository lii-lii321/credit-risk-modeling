"""自实现的 PSI（Population Stability Index）特征漂移检验。

定义：PSI = Σ ( actual%_i - expected%_i ) * ln( actual%_i / expected%_i )

- 数值特征：以 expected 的分位数等频分箱（bins 档），两分布共用同一分箱；
- 类别特征：按类别值对齐，取并集；
- 零比例用 epsilon 下限截断，避免 ln(0)；
- 经验判读：<0.1 稳定，0.1-0.25 中度漂移需关注，>0.25 显著漂移。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EPSILON = 1e-4


def _smooth(p: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, epsilon)


def psi_categorical(expected: pd.Series, actual: pd.Series, epsilon: float = DEFAULT_EPSILON) -> float:
    """类别特征 PSI：按类别值对齐计算。"""
    expected = expected.astype(object).fillna("__MISSING__")
    actual = actual.astype(object).fillna("__MISSING__")
    categories = sorted(set(expected.unique()) | set(actual.unique()))
    if len(categories) == 0:
        return 0.0
    p_exp = _smooth(expected.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy(), epsilon)
    p_act = _smooth(actual.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy(), epsilon)
    return float(np.sum((p_act - p_exp) * np.log(p_act / p_exp)))


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10,
                epsilon: float = DEFAULT_EPSILON) -> float:
    """数值特征 PSI：expected 分位数分箱，actual 投影到同一分箱。"""
    e = pd.to_numeric(expected, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(e.dropna(), quantiles))
    if edges.size < 2:
        # expected 常数列：退化为两箱（缺省 vs 该值）
        edges = np.array([e.min() - 0.5, e.max() + 0.5]) if len(e) else np.array([0.0, 1.0])

    def proportions(s: pd.Series) -> np.ndarray:
        clipped = np.clip(s.to_numpy(dtype=float), edges[0], edges[-1])
        idx = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, edges.size - 2)
        counts = np.bincount(idx, minlength=edges.size - 1)
        missing = int(pd.isna(s).sum())
        counts = np.append(counts, missing)  # 缺失值单列一箱
        return counts / counts.sum()

    p_exp = _smooth(proportions(e), epsilon)
    p_act = _smooth(proportions(a), epsilon)
    return float(np.sum((p_act - p_exp) * np.log(p_act / p_exp)))


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10,
        epsilon: float = DEFAULT_EPSILON) -> float:
    """按 dtype 自动选择数值/类别 PSI。"""
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        return psi_numeric(expected, actual, bins=bins, epsilon=epsilon)
    return psi_categorical(expected, actual, epsilon=epsilon)


def psi_table(expected_df: pd.DataFrame, actual_df: pd.DataFrame, bins: int = 10,
              epsilon: float = DEFAULT_EPSILON) -> pd.DataFrame:
    """DataFrame 级 PSI 汇总：每特征一行，附漂移等级。"""
    rows = []
    for col in expected_df.columns:
        value = psi(expected_df[col], actual_df[col], bins=bins, epsilon=epsilon)
        if value < 0.1:
            level = "stable"
        elif value < 0.25:
            level = "moderate"
        else:
            level = "significant"
        rows.append({"feature": col, "psi": value, "level": level})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def max_psi(table: pd.DataFrame) -> float:
    """方便 CI/告警使用的整表最大 PSI。"""
    return float(table["psi"].max()) if len(table) else 0.0
