# -*- coding: utf-8 -*-
"""自实现的 WOE（Weight of Evidence）/ IV（Information Value）编码器。

约定（与评分卡主流教材一致）：
- 正类 = 违约（bad, y=1）；
- WOE_bin = ln( dist_bad_bin / dist_good_bin )，WOE > 0 表示该分箱违约占比高于总体；
- IV = Σ ( dist_bad_bin - dist_good_bin ) * WOE_bin；
- 零频次/小样本使用加法平滑 α=0.5：dist = (n + α) / (N + α * n_bins)，
  避免 ln(0) 与无穷大；
- 数值特征按分位数等频分箱（n_bins），类别特征按类别值分箱；
- transform 时：数值越界落到边界箱，类别未见值与缺失值映射为 0（中性，
  不引入人为方向性），并在 fit 阶段记录 missing 是否成箱。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_BIN = "__MISSING__"
IV_STRENGTH = [
    (0.02, "useless"),
    (0.1, "weak"),
    (0.3, "medium"),
    (0.5, "strong"),
    (float("inf"), "suspicious"),
]


def iv_strength(iv: float) -> str:
    """IV 经验评级：<0.02 无用 / 0.02-0.1 弱 / 0.1-0.3 中 / 0.3-0.5 强 / >0.5 可疑。"""
    for threshold, label in IV_STRENGTH:
        if iv < threshold:
            return label
    return "suspicious"


class WoEEncoder(BaseEstimator, TransformerMixin):
    """对 DataFrame 整体做 WOE 编码，输出 `<col>_woe` 列。

    Parameters
    ----------
    n_bins:
        数值特征的分位数分箱数（类别数不足时自动取唯一值数）。
    smoothing:
        比例平滑系数 α，防止零频次导致的 ln(0)。
    """

    def __init__(self, n_bins: int = 10, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.smoothing <= 0:
            raise ValueError("smoothing 必须为正数（加法平滑）")
        y = pd.Series(y).reset_index(drop=True).astype(int)
        X = X.reset_index(drop=True)
        n_bad = int((y == 1).sum())
        n_good = int((y == 0).sum())
        if n_bad == 0 or n_good == 0:
            raise ValueError("WOE 编码要求两类样本均非空")

        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.iv_: Dict[str, float] = {}
        self.feature_names_out_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.numeric_columns_: List[str] = []

        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                self.numeric_columns_.append(col)
                woe_map, edges, iv = self._fit_numeric(s, y, n_bad, n_good)
                self.bin_edges_[col] = edges
            else:
                self.categorical_columns_.append(col)
                woe_map, edges, iv = self._fit_categorical(s, y, n_bad, n_good)
            self.woe_maps_[col] = woe_map
            self.iv_[col] = iv
            self.feature_names_out_.append(f"{col}_woe")
        return self

    def _fit_numeric(self, s: pd.Series, y: pd.Series, n_bad: int, n_good: int):
        """数值列：分位数等频分箱 + WOE。缺失值单独成箱（若训练集存在缺失）。"""
        finite = s.dropna()
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(finite, quantiles)) if len(finite) else np.array([])
        has_missing = bool(s.isna().any())

        bin_ids = self._numeric_bin_ids(s, edges)
        labels = [f"bin_{i}" for i in range(len(edges) - 1)]
        if has_missing:
            labels.append(MISSING_BIN)

        woe_map, iv = self._compute_woe_iv(bin_ids, labels, y, n_bad, n_good)
        return woe_map, edges, iv

    def _fit_categorical(self, s: pd.Series, y: pd.Series, n_bad: int, n_good: int):
        """类别列：每个取值一箱，NaN 单独一箱。"""
        s = s.astype(object).where(s.notna(), MISSING_BIN)
        labels = sorted(s.astype(str).unique())
        bin_ids = s.astype(str).to_numpy()
        woe_map, iv = self._compute_woe_iv(bin_ids, labels, y, n_bad, n_good)
        return woe_map, None, iv

    def _compute_woe_iv(self, bin_ids: np.ndarray, labels: List[str], y: pd.Series,
                        n_bad: int, n_good: int):
        """给定每个样本的分箱归属，计算各箱 WOE 与总 IV（加法平滑）。"""
        alpha = self.smoothing
        n_bins = len(labels)
        woe_map: Dict[str, float] = {}
        iv = 0.0
        for label in labels:
            if label == MISSING_BIN:
                # bin_ids 各路径（数值/类别）产出的都是 object 数组：缺失位置为
                # "__MISSING__" 字符串或 np.nan（全缺失列退化分支），
                # 单用 pd.isna 会恒 False（字符串非 NaN），必须两者都查。
                arr = np.asarray(bin_ids)
                mask = (arr == MISSING_BIN) | pd.isna(arr)
            else:
                mask = np.asarray(bin_ids) == label
            n_bin_bad = int(((y == 1) & mask).sum())
            n_bin_good = int(((y == 0) & mask).sum())
            dist_bad = (n_bin_bad + alpha) / (n_bad + alpha * n_bins)
            dist_good = (n_bin_good + alpha) / (n_good + alpha * n_bins)
            woe = float(np.log(dist_bad / dist_good))
            woe_map[label] = woe
            iv += (dist_bad - dist_good) * woe
        return woe_map, float(iv)

    # -------------------------------------------------------------- transform
    def _numeric_bin_ids(self, s: pd.Series, edges: np.ndarray) -> np.ndarray:
        """数值 → 分箱索引（字符串标签）；NaN 返回 NaN 占位。"""
        if edges.size == 0:
            return np.full(len(s), np.nan, dtype=object)
        clipped = np.clip(s.to_numpy(dtype=float), edges[0], edges[-1])
        idx = np.searchsorted(edges, clipped, side="right") - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        ids = np.array([f"bin_{i}" for i in idx], dtype=object)
        ids[pd.isna(s.to_numpy())] = MISSING_BIN
        return ids

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "woe_maps_"):
            raise RuntimeError("WoEEncoder 尚未 fit")
        missing_cols = set(X.columns) - set(self.woe_maps_)
        if missing_cols:
            raise ValueError(f"transform 输入缺少训练时特征: {sorted(missing_cols)}")

        out = {}
        for col, woe_map in self.woe_maps_.items():
            s = X[col]
            if col in self.numeric_columns_:
                ids = self._numeric_bin_ids(s, self.bin_edges_[col])
                ids = np.where(pd.isna(ids) | (ids == MISSING_BIN), MISSING_BIN, ids)
                values = [woe_map.get(i, 0.0) for i in ids]
            else:
                keys = s.astype(object).where(s.notna(), MISSING_BIN).astype(str)
                values = [woe_map.get(k, 0.0) for k in keys]
            out[f"{col}_woe"] = np.asarray(values, dtype=float)
        return pd.DataFrame(out, index=X.index)

    # ---------------------------------------------------------------- utils
    def iv_table(self) -> pd.DataFrame:
        """各特征 IV 及其经验强度评级，按 IV 降序。"""
        rows = [
            {"feature": col, "iv": iv, "strength": iv_strength(iv)}
            for col, iv in sorted(self.iv_.items(), key=lambda kv: -kv[1])
        ]
        return pd.DataFrame(rows)
