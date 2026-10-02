# -*- coding: utf-8 -*-
"""EDA 回归测试：报告图片链接必须指向真实落盘文件；统计量由数据计算而非硬编码。"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from creditrisk.config import CAT_FEATURES, NUM_FEATURES, TARGET_COL
from creditrisk.eda import run_eda


def _tiny_credit_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for col in CAT_FEATURES:
        data[col] = rng.choice(["opt1", "opt2"], size=n)
    for col in NUM_FEATURES:
        data[col] = rng.uniform(0, 100, size=n).round(2)
    df = pd.DataFrame(data)
    df[TARGET_COL] = (rng.random(n) < 0.3).astype(int)
    return df


def test_eda_summary_links_point_to_real_files(tmp_path):
    """回归：eda_summary.md 中每个图片链接都必须对应 out_dir 里真实存在的文件。"""
    df = _tiny_credit_df()
    run_eda(df, tmp_path)
    summary = (tmp_path / "eda_summary.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", summary)
    assert len(links) == 5
    saved = {p.name for p in tmp_path.glob("*.png")}
    for link in links:
        assert (tmp_path / link).exists(), f"报告链接 {link} 在输出目录中不存在"
        assert link in saved


def test_eda_stats_computed_not_hardcoded(tmp_path):
    """回归：违约率/样本数按数据计算（不得出现硬编码的 300/1000）。"""
    df = _tiny_credit_df(n=60, seed=1)
    stats = run_eda(df, tmp_path)
    summary = (tmp_path / "eda_summary.md").read_text(encoding="utf-8")
    n_bad = int((df[TARGET_COL] == 1).sum())
    assert stats["n_rows"] == 60
    assert stats["bad_rate"] == df[TARGET_COL].mean()
    assert f"（{n_bad}/60）" in summary
    assert "300/1000" not in summary


def test_eda_max_abs_corr_matches_independent_computation(tmp_path):
    """回归：max_abs_corr 必须等于对角置零后相关矩阵的最大值（placeholder 死行已删）。"""
    df = _tiny_credit_df(n=80, seed=2)
    stats = run_eda(df, tmp_path)
    corr_matrix = df[NUM_FEATURES].corr().abs()
    corr_values = corr_matrix.to_numpy()
    np.fill_diagonal(corr_values, 0)  # 与实现一致：排除自相关
    assert stats["max_abs_corr"] == corr_values.max()
    assert stats["top_corr_pair"] == list(
        corr_matrix.stack().sort_values(ascending=False).index[0]
    )


def test_eda_missing_note_adapts_to_data(tmp_path):
    """数据有缺失时，摘要不得宣称「无缺失」。"""
    df = _tiny_credit_df(n=60, seed=3)
    df.loc[df.index[:5], NUM_FEATURES[0]] = np.nan
    run_eda(df, tmp_path)
    summary = (tmp_path / "eda_summary.md").read_text(encoding="utf-8")
    assert "无缺失" not in summary
    assert "5 个缺失值" in summary
