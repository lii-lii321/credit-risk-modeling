"""阈值-业务指标：给定批准阈值下的批准率与批内/拒件坏账率。

模型输出 PD 后，业务真正执行的是"在阈值 t 处批准多少、批准的人群坏多少"。
本模块把 PD 排序能力翻译成决策表（描述性分析，不含利润/损失矩阵与最优化）：

- 决策规则：PD < t 批准（分数越低越安全），PD ≥ t 拒绝；
- approval_rate = n(PD < t) / n；
- bad_rate_approved：被批准人群的实际违约率；
- bad_rate_rejected：被拒人群的实际违约率（代表被拦截的损失）。

批准组为空时坏账率无定义，记 NaN（不虚构 0）。
阈值若取自 PD 分位数，则属于排序型阈值——不受 PD 绝对值校准偏移的影响。
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 复用校准模块的输入校验（同包内私有函数）：非空、等长、标签 0/1、概率在 [0, 1]
from .calibration import _validate  # noqa: E402


def tradeoff_table(y_true, y_proba, thresholds) -> pd.DataFrame:
    """逐阈值决策表：批准率、批内坏账率、拒件坏账率，按阈值升序排列。"""
    y, p = _validate(y_true, y_proba)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        raise ValueError("thresholds 不能为空")
    n = len(y)
    rows = []
    for t in thresholds:
        approved = p < t
        n_approved = int(approved.sum())
        n_rejected = n - n_approved
        rows.append({
            "threshold": float(t),
            "n_approved": n_approved,
            "approval_rate": n_approved / n,
            "bad_rate_approved": float(y[approved].mean()) if n_approved else float("nan"),
            "n_rejected": n_rejected,
            "bad_rate_rejected": float(y[~approved].mean()) if n_rejected else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("threshold", ignore_index=True)


def thresholds_for_approval_rates(
    y_true, y_proba, approval_targets: Iterable[float] = (0.7, 0.8, 0.9)
) -> pd.DataFrame:
    """按目标批准率反查阈值：threshold = PD 的 target 分位数，附实际达成口径。

    PD 存在并列时，实际批准率可能偏离目标值，表中两列同时给出、不掩饰偏差。
    """
    y, p = _validate(y_true, y_proba)
    # 升序排列，保证与 tradeoff_table 的阈值排序保持行对齐
    targets = np.sort(np.asarray(approval_targets, dtype=float))
    if targets.size == 0:
        raise ValueError("approval_targets 不能为空")
    if not ((targets > 0) & (targets < 1)).all():
        raise ValueError("approval_targets 需在 (0, 1) 开区间内")
    cutoffs = np.quantile(p, targets)
    table = tradeoff_table(y, p, cutoffs)
    table.insert(0, "target_approval_rate", targets)
    return table


def plot_tradeoff_curves(y_true, y_proba, thresholds, out_path: Path) -> Path:
    """批准率（左轴）与批内/拒件坏账率（右轴）随阈值变化的曲线。

    图表文字统一英文（与 EDA/校准图一致，规避 matplotlib 中文字体缺字）。
    """
    table = tradeoff_table(y_true, y_proba, thresholds)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax_right = ax.twinx()
    line_approval = ax.plot(
        table["threshold"], table["approval_rate"],
        color="#2563eb", marker="o", label="approval rate (left)",
    )
    line_approved = ax_right.plot(
        table["threshold"], table["bad_rate_approved"],
        color="#1a365d", marker="s", label="bad rate, approved (right)",
    )
    line_rejected = ax_right.plot(
        table["threshold"], table["bad_rate_rejected"],
        color="#dc2626", marker="^", linestyle="--", label="bad rate, rejected (right)",
    )
    ax.set_xlabel("Approval threshold (approve if PD < threshold)")
    ax.set_ylabel("Approval rate")
    ax_right.set_ylabel("Default rate")
    ax.set_title("Threshold vs approval rate / default rate")
    lines = line_approval + line_approved + line_rejected
    # matplotlib 类型桩把 Artist.get_label() 标为 object（运行时恒为 str），故豁免 misc；
    # lint 环境无 matplotlib 时该行为 Any，双码豁免 unused-ignore 以兼容有桩/无桩两种环境
    ax.legend(
        lines,
        [line.get_label() for line in lines],  # type: ignore[misc, unused-ignore]
        loc="center right", fontsize=9,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
