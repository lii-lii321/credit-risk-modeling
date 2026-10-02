# -*- coding: utf-8 -*-
"""可解释性模块：SHAP 全局重要性 + 单样本归因；提供两级降级路径。

优先路径：shap.TreeExplainer（LightGBM）——精确的树模型 Shapley 值；
降级路径 A（模型为线性模型或 shap 不可用时的单样本解释）：
    contribution_i = coef_i * (x_woe_i - mean_woe_i)，即「偏离基准多少个 WOE 点 × 系数」；
降级路径 B（全局重要性）：sklearn permutation importance（AUC 下降量）。

两条路径都在 tests/test_explain.py 覆盖，README 如实声明当前实际使用的是哪条。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:  # shap 为可选依赖：导入失败时走降级路径并如实标记
    import shap

    SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001
    shap = None
    SHAP_AVAILABLE = False


def _tree_shap_values(model, X_woe: pd.DataFrame) -> np.ndarray:
    """统一各 shap 版本的返回结构 → (n_samples, n_features) 的正类贡献。"""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_woe)
    if isinstance(values, list):  # 旧版：按类别返回列表
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # 新版： (n, features, n_classes)
        values = values[:, :, -1]
    return values


def _strip_woe(names: List[str]) -> List[str]:
    return [n[: -len("_woe")] if n.endswith("_woe") else n for n in names]


def global_importance(pipeline, X: pd.DataFrame, y: Optional[pd.Series] = None,
                      use_shap: bool = True, n_repeats: int = 5) -> pd.DataFrame:
    """全局特征重要性 DataFrame（feature, importance, method），按重要性降序。

    树模型且 shap 可用 → mean(|SHAP|)；否则 permutation importance（需要 y）。
    """
    model = pipeline.steps[-1][1]
    is_tree = hasattr(model, "boosting_type") or hasattr(model, "get_booster")
    if use_shap and SHAP_AVAILABLE and is_tree:
        X_woe = _transform_prefix(pipeline, X)
        values = _tree_shap_values(model, X_woe)
        importance = np.abs(values).mean(axis=0)
        names = _strip_woe(list(X_woe.columns))
        method = "shap_mean_abs"
    else:
        if y is None:
            raise ValueError("permutation importance 需要 y")
        from sklearn.inspection import permutation_importance

        result = permutation_importance(pipeline, X, y, scoring="roc_auc",
                                        n_repeats=n_repeats, random_state=0)
        importance = result.importances_mean
        names = list(X.columns)
        method = "permutation_auc_drop"
    table = pd.DataFrame({"feature": names, "importance": importance, "method": method})
    return table.sort_values("importance", ascending=False, ignore_index=True)


def _transform_prefix(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """跑通 Pipeline 前缀（除最后一步外的所有 transformer）。"""
    X_t = X
    for _, transformer in pipeline.steps[:-1]:
        X_t = transformer.transform(X_t)
    return X_t


def explain_instance(pipeline, X_row: pd.DataFrame, top_k: int = 3,
                     background_means: Optional[pd.Series] = None) -> List[dict]:
    """单样本解释：返回按 |贡献| 降序的 top_k 个 {feature, contribution}。

    - 树模型 + shap 可用：该样本的 Tree SHAP 值（单位：对数几率贡献）；
    - 线性模型或 shap 不可用：coef_i * (x_woe_i - mean_i)，
      background_means 缺省用 0（WOE 特征均值近似为 0）。
    """
    X_woe = _transform_prefix(pipeline, X_row)
    model = pipeline.steps[-1][1]

    if SHAP_AVAILABLE and (hasattr(model, "boosting_type") or hasattr(model, "get_booster")):
        values = _tree_shap_values(model, X_woe)[0]
    else:
        coef = getattr(model, "coef_", None)
        if coef is None:
            raise ValueError("该模型既不支持 SHAP 也没有 coef_，无法做单样本解释")
        base = background_means if background_means is not None else pd.Series(0.0, index=X_woe.columns)
        values = coef[0] * (X_woe.iloc[0].to_numpy() - base.reindex(X_woe.columns).to_numpy())

    features = _strip_woe(list(X_woe.columns))
    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {"feature": features[i], "contribution": float(values[i])}
        for i in order
    ]
