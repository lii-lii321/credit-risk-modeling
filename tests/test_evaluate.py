# -*- coding: utf-8 -*-
"""评价指标单元测试：KS 已知构造值、AUC 与 Gini 关系。"""
import numpy as np
import pytest

from creditrisk.evaluate import evaluate_predictions, gini_coefficient, ks_statistic, roc_auc


def test_ks_perfect_separation_is_one():
    """坏样本分数全部低于好样本时，两组分布完全分离 → KS=1。"""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert ks_statistic(y, score) == pytest.approx(1.0)


def test_ks_identical_distributions_is_zero():
    """两组分数同分布 → KS=0（所有样本同分时两组累计曲线重合）。"""
    y = np.array([1, 0, 1, 0])
    score = np.array([0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y, score) == pytest.approx(0.0, abs=1e-9)


def test_ks_independent_random_scores_stays_small():
    """分数与标签独立时 KS 应远离 1（固定种子下约 0.1 量级）。"""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, 2000)
    score = rng.random(2000)
    assert ks_statistic(y, score) < 0.2


def test_ks_known_construction():
    """手工构造：坏组 [0.1,0.5]，好组 [0.3,0.7]，最大累计差=0.5。"""
    y = np.array([1, 0, 1, 0])
    score = np.array([0.1, 0.3, 0.5, 0.7])
    assert ks_statistic(y, score) == pytest.approx(0.5)


def test_ks_bounded_and_robust_to_ties():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    score = rng.random(500)
    value = ks_statistic(y, score)
    assert 0.0 <= value <= 1.0
    # 平分（相同分数）不应崩溃
    score_tied = np.round(score, 1)
    assert 0.0 <= ks_statistic(y, score_tied) <= 1.0


def test_ks_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ks_statistic([], [])
    with pytest.raises(ValueError):
        ks_statistic(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))  # 只有一类


def test_auc_and_gini_relation():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(50), np.ones(50)])
    score = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.5, 1, 50)])
    auc = roc_auc(y, score)
    assert 0.5 < auc < 1.0
    assert gini_coefficient(y, score) == pytest.approx(2 * auc - 1)


def test_auc_random_scores_near_half():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.5, 4000)
    score = rng.random(4000)
    assert abs(roc_auc(y, score) - 0.5) < 0.05


def test_evaluate_predictions_bundle():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    out = evaluate_predictions(y, score)
    assert out["auc"] == 1.0
    assert out["ks"] == 1.0
    assert out["gini"] == 1.0
    assert out["n"] == 4
    assert out["bad_rate"] == 0.5
