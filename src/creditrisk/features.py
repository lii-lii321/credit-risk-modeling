"""特征工程管线：WOE 编码 + 模型的可部署 Pipeline 组装。"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from .woe import WoEEncoder


def build_pipeline(estimator, n_bins: int = 10) -> Pipeline:
    """统一评分卡式特征管线：原始 DataFrame → WOE 特征 → 模型。"""
    return Pipeline([
        ("woe", WoEEncoder(n_bins=n_bins)),
        ("model", estimator),
    ])
