"""概率校准：Brier 分数、等频分桶可靠性表与期望校准误差（ECE）。

AUC/KS 只衡量排序能力（坏样本分数是否整体更低），不约束分数的绝对值；
信贷实务中 PD 常被直接用于定价、拨备与阈值解读，因此还需回答：
"预测 PD=0.6 的群组，实际违约率是否接近 60%？"

定义（均为自实现，便于审计）：
- Brier = mean((p - y)^2)，取值 [0, 1]，越小越好；
- 可靠性表：按预测 PD 等频分桶，逐桶对比平均预测 PD 与实际违约率；
- ECE = Σ_b (n_b/n) · |mean_p_b − rate_b|，按桶样本量加权的绝对校准差。

注意：class_weight="balanced" 会系统性抬高预测 PD 的绝对值（排序能力不变），
本模块用于量化这一副作用，结论写入训练报告与 README。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _validate(y_true, y_proba) -> tuple[np.ndarray, np.ndarray]:
    """输入校验：同长非空、标签 0/1、概率在 [0, 1]。"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    if len(y) == 0 or len(p) == 0:
        raise ValueError("输入为空")
    if len(y) != len(p):
        raise ValueError("y_true 与 y_proba 长度不一致")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y_true 只能包含 0/1")
    if not ((p >= 0) & (p <= 1)).all() or not np.isfinite(p).all():
        raise ValueError("y_proba 必须为 [0, 1] 内的有限值")
    return y, p


def brier_score(y_true, y_proba) -> float:
    """Brier 分数 ∈ [0, 1]：预测概率与真实结局的均方误差，越小越好。"""
    y, p = _validate(y_true, y_proba)
    return float(np.mean((p - y) ** 2))


def reliability_data(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """等频分桶可靠性表：逐桶对比平均预测 PD 与实际违约率。

    按预测概率的分位数分箱；概率大量并列时唯一分位边界减少，
    实际桶数可能少于 n_bins（完全并列时合并为单桶）。
    返回列：bin（区间标签）、n（样本数）、mean_predicted_pd、observed_bad_rate，
    按预测 PD 升序排列。
    """
    y, p = _validate(y_true, y_proba)
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins 至少为 1")

    quantiles = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)  # 并列概率会使边界塌缩
    if len(edges) < 2:  # 所有概率相同 → 单桶
        bin_id: np.ndarray = np.zeros(len(p), dtype=int)
    else:
        # side="right"−1：样本落入最后一个 ≤ 自身的边界桶；最大值裁剪进末桶
        bin_id = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    frame = pd.DataFrame({"p": p, "y": y, "bin_id": bin_id})
    table = (
        frame.groupby("bin_id")
        .agg(n=("y", "size"), mean_predicted_pd=("p", "mean"), observed_bad_rate=("y", "mean"))
        .reset_index()
    )
    if len(edges) < 2:
        labels = [f"[{edges[0]:.3f}, {edges[0]:.3f}]"]
    else:
        labels = [f"[{edges[i]:.3f}, {edges[i + 1]:.3f})" for i in range(len(edges) - 1)]
    table["bin"] = [labels[i] for i in table["bin_id"]]
    return table[["bin", "n", "mean_predicted_pd", "observed_bad_rate"]]


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """ECE ∈ [0, 1]：按桶样本量加权的 |平均预测 PD − 实际违约率| 之和，越小越校准。"""
    table = reliability_data(y_true, y_proba, n_bins)
    weights = table["n"] / table["n"].sum()
    gap = (table["mean_predicted_pd"] - table["observed_bad_rate"]).abs()
    return float(np.sum(weights.to_numpy() * gap.to_numpy()))


def plot_reliability_diagram(
    series: Sequence[tuple[str, object, object]],
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """可靠性曲线（reliability diagram）：逐桶平均预测 PD vs 实际违约率。

    series 为若干 (label, y_true, y_proba) 元组；对角线代表完美校准。
    图表文字统一英文（与 EDA 图一致，规避中文字体缺字）。
    """
    if not series:
        raise ValueError("至少需要一组校准数据")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1, label="perfectly calibrated")
    palette = ["#1a365d", "#2563eb", "#334155"]
    for i, (label, y_true, y_proba) in enumerate(series):
        table = reliability_data(y_true, y_proba, n_bins=n_bins)
        ece = expected_calibration_error(y_true, y_proba, n_bins=n_bins)
        ax.plot(
            table["mean_predicted_pd"],
            table["observed_bad_rate"],
            marker="o",
            color=palette[i % len(palette)],
            label=f"{label} (ECE={ece:.3f})",
        )
    ax.set_xlabel("Mean predicted PD (quantile bins)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
