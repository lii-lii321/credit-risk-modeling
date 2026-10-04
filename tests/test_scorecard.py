"""Scorecard 标度转换测试：锚点值、PDO 语义、单调性与端点防护。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from creditrisk.scorecard import (
    DEFAULT_BASE_ODDS,
    DEFAULT_BASE_SCORE,
    DEFAULT_PDO,
    score_from_pd,
    scorecard_summary,
)


def test_anchor_point_base_odds_maps_to_base_score():
    """锚点：p = 1/(1+base_odds) 时 odds=base_odds，分数恰为 base_score。"""
    p_anchor = 1.0 / (1.0 + DEFAULT_BASE_ODDS)
    score = float(score_from_pd(p_anchor))
    assert score == pytest.approx(DEFAULT_BASE_SCORE, abs=1e-9)


def test_pdo_doubling_adds_exactly_pdo_points():
    """PDO 语义：odds 翻倍（p 减半），分数恰好 +PDO。"""
    base = float(score_from_pd(1.0 / 51.0))          # odds = 50
    doubled = float(score_from_pd(1.0 / 101.0))      # odds = 100
    assert doubled - base == pytest.approx(DEFAULT_PDO, abs=1e-9)


def test_score_strictly_decreasing_in_pd():
    """单调性：PD 越高分数越低（验收要求的单调性测试）。"""
    grid = np.array([0.01, 0.02, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99])
    scores = score_from_pd(grid)
    assert np.all(np.diff(scores) < 0)
    # 任意两端点也保持方向
    assert scores[0] > scores[-1]


def test_endpoints_clipped_no_infinity():
    """p=0/1 截断到 epsilon，分数有限不爆炸。"""
    scores = score_from_pd(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(scores))
    lo = float(score_from_pd(1e-6))
    hi = float(score_from_pd(1.0 - 1e-6))
    assert float(scores[0]) == lo and float(scores[1]) == hi


def test_scalar_and_array_consistent():
    scalar = float(score_from_pd(0.02))
    arr = score_from_pd(np.array([0.02]))
    assert math.isfinite(scalar)
    assert float(arr[0]) == pytest.approx(scalar, abs=1e-12)


def test_custom_params_change_scale_consistently():
    """换基准后锚点与 PDO 语义仍在：score(base_odds)=base_score，翻倍 +PDO。"""
    pdo, base_odds, base_score = 30.0, 20.0, 700.0
    anchor = float(score_from_pd(1.0 / 21.0, pdo=pdo, base_odds=base_odds, base_score=base_score))
    assert anchor == pytest.approx(base_score, abs=1e-9)
    dbl = float(score_from_pd(1.0 / 41.0, pdo=pdo, base_odds=base_odds, base_score=base_score))
    assert dbl - anchor == pytest.approx(pdo, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"pdo": 0.0},
        {"pdo": -1.0},
        {"base_odds": 0.0},
        {"base_odds": -5.0},
        {"base_score": 0.0},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        score_from_pd(0.5, **kwargs)


def test_summary_shape_and_consistency():
    rng = np.random.default_rng(42)
    pds = rng.uniform(0.01, 0.9, size=200)
    summary = scorecard_summary(pds)
    assert summary["n"] == 200
    assert summary["pdo"] == DEFAULT_PDO and summary["base_odds"] == DEFAULT_BASE_ODDS
    assert summary["base_score"] == DEFAULT_BASE_SCORE
    assert summary["min"] < summary["median"] < summary["max"]
    assert 0.0 < summary["mean"] < 1000.0
    # 与直接逐点转换一致
    direct = score_from_pd(pds)
    assert float(np.min(direct)) == pytest.approx(summary["min"], abs=1e-9)
    assert float(np.max(direct)) == pytest.approx(summary["max"], abs=1e-9)


def test_summary_constant_pds_bounds_equal():
    summary = scorecard_summary(np.full(5, 0.02))
    assert summary["min"] == summary["max"] == summary["median"] == summary["mean"]
