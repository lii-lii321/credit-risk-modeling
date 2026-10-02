# -*- coding: utf-8 -*-
"""模型对比实验：逻辑回归基线 vs LightGBM；类不平衡用样本加权并与 SMOTE 对比。

实验矩阵（2 模型 × 3 不平衡策略）：
- none   ：不做任何处理；
- weight ：class_weight="balanced"（样本加权）；
- smote  ：imblearn SMOTE 过采样（只作用于训练折，验证/测试集保持原分布）。

统一在 WOE 特征上训练（评分卡传统做法），使两个模型可比、
PSI/SHAP 解释口径一致。评估：5 折分层 CV 的 AUC/KS + 独立测试集指标。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE
from .evaluate import evaluate_predictions
from .features import build_pipeline
from .woe import WoEEncoder

MODEL_NAMES = ["logistic_regression", "lightgbm"]
IMBALANCE_STRATEGIES = ["none", "weight", "smote"]

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def make_estimator(model_name: str, strategy: str):
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"未知不平衡策略: {strategy}")
    if model_name == "logistic_regression":
        params = dict(max_iter=5000, random_state=RANDOM_STATE)
        if strategy == "weight":
            params["class_weight"] = "balanced"
        return LogisticRegression(**params)
    if model_name == "lightgbm":
        params = dict(LGBM_PARAMS)
        if strategy == "weight":
            params["class_weight"] = "balanced"
        return LGBMClassifier(**params)
    raise ValueError(f"未知模型: {model_name}")


def make_pipeline(model_name: str, strategy: str, n_bins: int = 10):
    """构建 sklearn/imblearn Pipeline；SMOTE 策略使用 imblearn Pipeline。"""
    estimator = make_estimator(model_name, strategy)
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([
            ("woe", WoEEncoder(n_bins=n_bins)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ])
    if strategy not in ("none", "weight"):
        raise ValueError(f"未知不平衡策略: {strategy}")
    return build_pipeline(estimator, n_bins=n_bins)


def cross_validate_model(X: pd.DataFrame, y: pd.Series, model_name: str, strategy: str,
                         n_splits: int = 5) -> Dict[str, float]:
    """5 折分层 CV：每折独立 fit，汇总 AUC/KS 均值与标准差。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs: List[float] = []
    kss: List[float] = []
    for train_idx, valid_idx in skf.split(X, y):
        pipe = make_pipeline(model_name, strategy)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        metrics = evaluate_predictions(y.iloc[valid_idx], proba)
        aucs.append(metrics["auc"])
        kss.append(metrics["ks"])
    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_ks_mean": float(np.mean(kss)),
        "cv_ks_std": float(np.std(kss)),
    }


def run_all_experiments(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        n_splits: int = 5) -> pd.DataFrame:
    """2 模型 × 3 策略全量实验，返回逐行结果（CV + 测试集指标）。"""
    rows = []
    for model_name in MODEL_NAMES:
        for strategy in IMBALANCE_STRATEGIES:
            cv = cross_validate_model(X_train, y_train, model_name, strategy, n_splits=n_splits)
            pipe = make_pipeline(model_name, strategy)
            pipe.fit(X_train, y_train)
            test_proba = pipe.predict_proba(X_test)[:, 1]
            test_metrics = evaluate_predictions(y_test, test_proba)
            rows.append({
                "model": model_name,
                "strategy": strategy,
                **cv,
                "test_auc": test_metrics["auc"],
                "test_ks": test_metrics["ks"],
                "test_gini": test_metrics["gini"],
            })
    return pd.DataFrame(rows)


def select_best(results: pd.DataFrame, tie_tolerance: float = 0.005) -> Tuple[str, str]:
    """按 CV AUC 选最优；差距在 tie_tolerance 内时偏向逻辑回归（可解释性优先）。"""
    best_auc = results["cv_auc_mean"].max()
    candidates = results[results["cv_auc_mean"] >= best_auc - tie_tolerance]
    lr_rows = candidates[candidates["model"] == "logistic_regression"]
    chosen = lr_rows.iloc[0] if len(lr_rows) else candidates.sort_values("cv_auc_mean", ascending=False).iloc[0]
    return str(chosen["model"]), str(chosen["strategy"])


def fit_deployable_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                            model_name: str, strategy: str):
    """产出可部署的纯 sklearn Pipeline。

    SMOTE 属于训练期手段：先把训练集重采样，再装回普通 Pipeline，
    保证线上只有 transform + predict，无过采样副作用。
    """
    pipe = make_pipeline(model_name, strategy)
    if strategy == "smote":
        woe = pipe.named_steps["woe"]
        smote = pipe.named_steps["smote"]
        estimator = pipe.named_steps["model"]
        X_woe = woe.fit_transform(X_train, y_train)
        X_res, y_res = smote.fit_resample(X_woe, y_train)
        estimator.fit(X_res, y_res)
        from sklearn.pipeline import Pipeline

        return Pipeline([("woe", woe), ("model", estimator)])
    pipe.fit(X_train, y_train)
    return pipe
