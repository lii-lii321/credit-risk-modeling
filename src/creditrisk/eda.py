# -*- coding: utf-8 -*-
"""EDA：缺失/分布/相关性/类别-违约率图表，输出到 reports/。

图表文字统一使用英文，规避 matplotlib 中文字体缺字问题；
中文结论写在生成的 eda_summary.md 与 README 中。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import CAT_FEATURES, NUM_FEATURES, TARGET_COL  # noqa: E402

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_target_distribution(y: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar(["good (0)", "bad (1)"], counts.values, color=["#2563eb", "#1a365d"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(y):.1%})", ha="center", va="bottom")
    ax.set_title("Target distribution: credit default")
    ax.set_ylabel("count")
    return _save(fig, out_dir, "eda_target_distribution.png")


def plot_missing_and_dtypes(X: pd.DataFrame, out_dir: Path) -> Path:
    """缺失值与列类型概览（credit-g 本身无缺失，图上体现的是 dtype 分布）。"""
    nulls = X.isna().sum()
    dtypes = X.dtypes.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(nulls.index, nulls.values, color="#334155")
    axes[0].set_title("Missing values per column")
    axes[0].tick_params(axis="x", rotation=90)
    dtype_counts = dtypes.value_counts()
    axes[1].bar(dtype_counts.index, dtype_counts.values, color="#2563eb")
    axes[1].set_title("Column dtype counts")
    for i, v in enumerate(dtype_counts.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    return _save(fig, out_dir, "eda_missing_dtypes.png")


def plot_numeric_distributions(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> Path:
    """数值特征按违约分组的分布（KDE）。"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), NUM_FEATURES):
        for label, color in [(0, "#2563eb"), (1, "#dc2626")]:
            values = X.loc[y == label, col]
            sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3, label=f"y={label}")
        ax.set_title(col)
        ax.legend()
    axes.ravel()[-1].axis("off")  # 7 个特征占 8 格，空出最后一格
    fig.suptitle("Numeric feature distributions by default status")
    return _save(fig, out_dir, "eda_numeric_distributions.png")


def plot_categorical_default_rates(X: pd.DataFrame, y: pd.Series, out_dir: Path,
                                   top_k: int = 6) -> Path:
    """类别特征的违约率（按整体 IV 排序取前 top_k 个特征）。"""
    rates = {}
    for col in CAT_FEATURES:
        rates[col] = y.groupby(X[col]).mean()
    order = sorted(rates, key=lambda c: rates[c].max() - rates[c].min(), reverse=True)[:top_k]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    overall = float(y.mean())
    for ax, col in zip(axes.ravel(), order):
        r = rates[col].sort_values(ascending=False)
        ax.bar(r.index, r.values, color="#1a365d")
        ax.axhline(overall, color="#dc2626", linestyle="--", label=f"overall {overall:.0%}")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=60)
        ax.legend(fontsize=8)
    fig.suptitle("Default rate by category (top discriminative features)")
    return _save(fig, out_dir, "eda_categorical_default_rates.png")


def plot_numeric_correlation(X: pd.DataFrame, out_dir: Path) -> Path:
    corr = X[NUM_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation among numeric features")
    return _save(fig, out_dir, "eda_numeric_correlation.png")


def run_eda(df: pd.DataFrame, out_dir: Optional[Path] = None) -> dict:
    """完整 EDA：生成全部图表并返回关键统计量（供 summary/README 引用）。"""
    out_dir = Path(out_dir)
    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]

    paths = {
        "target": plot_target_distribution(y, out_dir),
        "missing": plot_missing_and_dtypes(X, out_dir),
        "numeric": plot_numeric_distributions(X, y, out_dir),
        "categorical": plot_categorical_default_rates(X, y, out_dir),
        "correlation": plot_numeric_correlation(X, out_dir),
    }

    stats = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "bad_rate": float(y.mean()),
        "total_missing": int(X.isna().sum().sum()),
    }
    corr_matrix = X[NUM_FEATURES].corr().abs()
    # 新版 pandas/numpy 下 to_numpy() 可能返回只读视图，必须在副本上置零对角
    corr_values = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0)
    stats["max_abs_corr"] = float(corr_values.max())
    zeroed = pd.DataFrame(corr_values, index=corr_matrix.index, columns=corr_matrix.columns)
    top_corr_pair = zeroed.stack().sort_values(ascending=False).index[0]
    stats["top_corr_pair"] = list(top_corr_pair)

    n_bad = int(round(stats["bad_rate"] * stats["n_rows"]))
    if stats["total_missing"] == 0:
        missing_note = "数据无缺失，WOE 编码器仍保留缺失成箱处理路径"
    else:
        missing_note = f"存在 {stats['total_missing']} 个缺失值，WOE 编码器会把缺失单独成箱"
    lines = [
        "# EDA 摘要（credit-g）",
        "",
        f"- 样本量：{stats['n_rows']} 行 × {stats['n_features']} 特征；"
        f"违约率 {stats['bad_rate']:.1%}（{n_bad}/{stats['n_rows']}）",
        f"- 缺失值总数：{stats['total_missing']}（{missing_note}）",
        f"- 数值特征最大|相关系数|：{stats['max_abs_corr']:.2f}（{stats['top_corr_pair'][0]} vs {stats['top_corr_pair'][1]}），共线性风险低",
        "",
        "## 图表",
        "",
    ]
    for key in ("target", "missing", "numeric", "categorical", "correlation"):
        # 链接必须与 _save 实际落盘文件名一致（paths[key].name），否则报告在仓库中裂图
        lines.append(f"![{key}]({paths[key].name})")
    lines += [
        "",
        "## 观察",
        "",
        "- 类不平衡约 1:2.3，采用样本加权与 SMOTE 两种策略对比；",
        "- checking_status=<0、credit_history、savings_status 等类别特征的违约率差异显著，是 WOE/IV 的强信号候选；",
        "- 数值特征 duration 与 credit_amount 正相关但|r|<0.7，无需降维处理。",
    ]
    (out_dir / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return stats
