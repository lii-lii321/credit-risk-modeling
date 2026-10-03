"""评价指标：AUC、Gini 与自实现的 KS 统计量。

KS（Kolmogorov-Smirnov）定义：max_s | F_bad(s) - F_good(s) |，
即按模型分数排序后，违约与正常两组累计分布的最大间距。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_EPSILON = 1e-6


def ks_statistic(y_true, y_score) -> float:
    """KS 统计量 ∈ [0, 1]：违约(1)与正常(0)分数累计分布的最大绝对差。"""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    if len(y) == 0:
        raise ValueError("输入为空")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y_true 只能包含 0/1")
    if not ((y == 1).any() and (y == 0).any()):
        raise ValueError("KS 需要两类样本同时存在")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted)
    cum_good = np.cumsum(1 - y_sorted)
    # KS 定义在阈值上：CDF 在平分处是阶跃，只在每个分数组最后一个位置评估，
    # 否则结果会依赖平分内部的排序，产生偏大的 KS。
    is_last = np.empty(len(s), dtype=bool)
    is_last[:-1] = s_sorted[:-1] != s_sorted[1:]
    is_last[-1] = True
    n_bad = int(y_sorted.sum())
    n_good = int(len(y_sorted) - n_bad)
    diff = np.abs(cum_bad[is_last] / n_bad - cum_good[is_last] / n_good)
    return float(diff.max())


def roc_auc(y_true, y_score) -> float:
    """AUC（封装 sklearn，保持模块内聚便于替换）。"""
    return float(roc_auc_score(y_true, y_score))


def gini_coefficient(y_true, y_score) -> float:
    """Gini = 2*AUC - 1，信贷风控常用衍生指标。"""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def evaluate_predictions(y_true, y_score) -> dict:
    """一次输出全部核心指标。"""
    return {
        "auc": roc_auc(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "gini": gini_coefficient(y_true, y_score),
        "n": int(len(y_true)),
        "bad_rate": float(np.mean(np.asarray(y_true).astype(int))),
    }
