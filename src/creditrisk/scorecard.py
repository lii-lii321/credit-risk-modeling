"""PD → 信用分（Scorecard）标度转换：行业惯例 PDO 参数化。

公式（分数越高越安全，与 PD 严格单调递减一一对应）：
    Factor = PDO / ln(2)
    Offset = base_score - Factor * ln(base_odds)
    score  = Offset + Factor * ln((1 - p) / p)

默认基准：base_odds=50（正常:违约）落在 base_score=600，PDO=20（分数每 +20 分，
好坏 odds 翻倍）。这只是刻度变换：不改变排序、不改变任何阈值决策，
作用是把校准后 PD 翻译成业务方更习惯的整数分数量纲。
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_PDO = 20.0
DEFAULT_BASE_ODDS = 50.0
DEFAULT_BASE_SCORE = 600.0
_P_EPSILON = 1e-6


def _validate_params(pdo: float, base_odds: float, base_score: float) -> None:
    if pdo <= 0:
        raise ValueError(f"PDO 必须 > 0，收到：{pdo}")
    if base_odds <= 0:
        raise ValueError(f"base_odds 必须 > 0，收到：{base_odds}")
    if base_score <= 0:
        raise ValueError(f"base_score 必须 > 0，收到：{base_score}")


def score_from_pd(
    p: float | np.ndarray,
    pdo: float = DEFAULT_PDO,
    base_odds: float = DEFAULT_BASE_ODDS,
    base_score: float = DEFAULT_BASE_SCORE,
) -> np.ndarray:
    """向量化转换；p 截断到 [_P_EPSILON, 1-_P_EPSILON]，0/1 端点不产生 inf。"""
    _validate_params(pdo, base_odds, base_score)
    values = np.asarray(p, dtype=float)
    clipped = np.clip(values, _P_EPSILON, 1.0 - _P_EPSILON)
    odds = (1.0 - clipped) / clipped
    factor = pdo / math.log(2.0)
    offset = base_score - factor * math.log(base_odds)
    return offset + factor * np.log(odds)


def scorecard_summary(
    pds: np.ndarray,
    pdo: float = DEFAULT_PDO,
    base_odds: float = DEFAULT_BASE_ODDS,
    base_score: float = DEFAULT_BASE_SCORE,
) -> dict:
    """一批 PD（通常为测试集校准后 PD）的分数汇总，供报告与 model_meta 使用。"""
    scores = score_from_pd(pds, pdo=pdo, base_odds=base_odds, base_score=base_score)
    return {
        "pdo": float(pdo),
        "base_odds": float(base_odds),
        "base_score": float(base_score),
        "n": int(scores.size),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
    }
