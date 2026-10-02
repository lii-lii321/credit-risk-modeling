# -*- coding: utf-8 -*-
"""端到端训练入口：实验对比 → 最优模型 → 产物落盘 → 可解释性 → 稳定性报告。

运行：python scripts/run_training.py
产物：
- artifacts/pipeline.joblib、model_meta.json、metrics.json
- reports/model_comparison.csv、iv_table.csv、shap_*.csv/png、
  stability_report.md、training_report.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

try:  # shap 为可选依赖：不可用时训练入口照常运行，
    # 全局重要性自动降级为 permutation importance（与 explain.SHAP_AVAILABLE 同源判断）
    import shap
except Exception:  # noqa: BLE001
    shap = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditrisk.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CATEGORY_VALUES,
    FEATURES,
    REPORTS_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COL,
)
from creditrisk.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    plot_reliability_diagram,
    reliability_data,
)
from creditrisk.data import load_credit_data  # noqa: E402
from creditrisk.evaluate import evaluate_predictions  # noqa: E402
from creditrisk.explain import SHAP_AVAILABLE, explain_instance, global_importance  # noqa: E402
from creditrisk.models import (  # noqa: E402
    fit_deployable_pipeline,
    run_all_experiments,
    select_best,
)
from creditrisk.psi import psi_table  # noqa: E402
from creditrisk.woe import WoEEncoder  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def log(msg: str) -> None:
    print(f"[training] {msg}", flush=True)


def main() -> None:
    started = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    data = load_credit_data()
    if data.is_synthetic:
        log("警告：OpenML 拉取失败，当前为合成降级数据！以下指标不代表真实数据表现。")
    X, y = data.X, data.y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log(f"数据：train={len(X_train)} test={len(X_test)} 坏样本率 train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # ------------------------------------------------------- experiments
    results = run_all_experiments(X_train, y_train, X_test, y_test)
    results = results.sort_values("cv_auc_mean", ascending=False, ignore_index=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    log("模型对比（按 CV AUC 排序）：")
    log(results.to_string(index=False))

    best_model, best_strategy = select_best(results)
    log(f"最优组合：model={best_model} strategy={best_strategy}")

    # --------------------------------------------------- final artifact
    pipeline = fit_deployable_pipeline(X_train, y_train, best_model, best_strategy)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_predictions(y_test, test_proba)
    log(f"最终模型测试集指标：{final_metrics}")
    joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.joblib")

    # risk band 阈值：测试集 PD 的 60% / 85% 分位
    t_low, t_high = np.quantile(test_proba, [0.60, 0.85])

    # ------------------------------------------------------- calibration
    # 排序指标（AUC/KS）不约束 PD 的绝对值；class_weight="balanced" 会系统性抬高
    # 预测 PD。这里用未加权 LR 作参照量化该副作用：Brier / ECE / 可靠性曲线。
    ref_pipe = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "none")
    ref_proba = ref_pipe.predict_proba(X_test)[:, 1]
    cal_deployed_table = reliability_data(y_test, test_proba, n_bins=10)
    cal_ref_table = reliability_data(y_test, ref_proba, n_bins=10)
    deployed_label = f"{best_model}+{best_strategy} (deployed)"
    reference_label = "logistic_regression+none (reference)"
    calibration = {
        "n_bins": 10,
        "test_n": int(len(y_test)),
        "deployed": {
            "model": deployed_label,
            "brier": brier_score(y_test, test_proba),
            "ece": expected_calibration_error(y_test, test_proba, n_bins=10),
            "mean_predicted_pd": float(np.mean(test_proba)),
            "observed_bad_rate": float(np.mean(y_test)),
            "reliability": cal_deployed_table.to_dict("records"),
        },
        "reference": {
            "model": reference_label,
            "brier": brier_score(y_test, ref_proba),
            "ece": expected_calibration_error(y_test, ref_proba, n_bins=10),
            "mean_predicted_pd": float(np.mean(ref_proba)),
            "observed_bad_rate": float(np.mean(y_test)),
            "reliability": cal_ref_table.to_dict("records"),
        },
    }
    cal_csv = pd.concat([
        cal_deployed_table.assign(model=deployed_label),
        cal_ref_table.assign(model=reference_label),
    ])[["model", "bin", "n", "mean_predicted_pd", "observed_bad_rate"]]
    cal_csv.to_csv(REPORTS_DIR / "calibration_table.csv", index=False)
    plot_reliability_diagram(
        [(deployed_label, y_test, test_proba), (reference_label, y_test, ref_proba)],
        REPORTS_DIR / "calibration_curve.png",
        n_bins=10,
    )
    log(f"校准：部署模型 Brier={calibration['deployed']['brier']:.4f} "
        f"ECE={calibration['deployed']['ece']:.4f}，平均预测 PD "
        f"{calibration['deployed']['mean_predicted_pd']:.3f} vs 实际违约率 "
        f"{calibration['deployed']['observed_bad_rate']:.3f}")

    # ------------------------------------------------------------- IV
    woe = WoEEncoder(n_bins=10).fit(X_train, y_train)
    iv_table = woe.iv_table()
    iv_table.to_csv(REPORTS_DIR / "iv_table.csv", index=False)
    log(f"IV 前 5：{iv_table.head(5).to_dict('records')}")

    # ------------------------------------------------------- explain
    # SHAP 全局重要性固定在 LightGBM 上做树 SHAP（任务要求的可解释性产物）；
    # 若最终部署模型是逻辑回归，则其单样本解释走 coef 路径并如实记录方法。
    if SHAP_AVAILABLE:
        lgbm_pipe = fit_deployable_pipeline(X_train, y_train, "lightgbm", "weight")
        imp = global_importance(lgbm_pipe, X_train)
        shap_method = "shap_mean_abs(lightgbm+weight)"
        local_lgbm = explain_instance(lgbm_pipe, X_test.iloc[[0]], top_k=3)
        log(f"LightGBM SHAP 单样本解释（PD={lgbm_pipe.predict_proba(X_test.iloc[[0]])[0, 1]:.3f}）：{local_lgbm}")
    else:
        imp = global_importance(pipeline, X_train, y=y_train, use_shap=False)
        shap_method = "permutation_auc_drop"
        local_lgbm = None
        log("提示：shap 不可用，全局重要性已降级为 permutation importance（诚实降级）。")

    imp.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#1a365d")
    ax.set_title(f"Global feature importance ({shap_method})")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_global_importance.png", dpi=150)
    plt.close(fig)

    example_row = X_test.iloc[[0]]
    try:
        local = explain_instance(pipeline, example_row, top_k=3)
        log(f"部署模型单样本解释示例（真实 y={int(y_test.iloc[0])}, PD={test_proba[0]:.3f}）：{local}")
    except ValueError as exc:
        # 部署模型为树模型且 shap 不可用时，单样本解释明确不可得（不静默降级到错误方法）
        local = None
        log(f"提示：部署模型单样本解释不可用（{exc}）")

    # ------------------------------------------------------- artifacts
    metrics_payload = {
        "data_source": "openml:credit-g:v1" if not data.is_synthetic else "synthetic_fallback",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "bad_rate_train": float(y_train.mean()),
        "bad_rate_test": float(y_test.mean()),
        "experiments": results.to_dict("records"),
        "selected": {"model": best_model, "strategy": best_strategy},
        "final_test_metrics": final_metrics,
        "calibration": calibration,
        "global_importance_method": shap_method,
        "deployed_model_local_explanation": (
            ("shap" if best_model == "lightgbm" else "coef_x_woe_deviation")
            if local is not None else "unavailable"
        ),
        "local_example": {
            "pd": float(test_proba[0]),
            "y_true": int(y_test.iloc[0]),
            "top_features": local,
        },
        "shap_local_example_lightgbm": local_lgbm,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model_version": "1.0.0",
        "model": best_model,
        "strategy": best_strategy,
        "target": TARGET_COL,
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "category_values": CATEGORY_VALUES,
        "risk_band_thresholds": {"low_below": float(t_low), "medium_below": float(t_high)},
        "test_metrics": final_metrics,
        "global_importance_method": shap_method,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__ if (SHAP_AVAILABLE and shap is not None) else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    (ARTIFACTS_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------- stability
    psi_train_test = psi_table(X_train, X_test)
    shifted = X_test.copy()
    shifted["age"] = shifted["age"] + 15
    shifted["credit_amount"] = shifted["credit_amount"] * 1.5
    shifted["duration"] = shifted["duration"] + 12
    psi_shifted = psi_table(X_train, shifted)
    psi_train_test.to_csv(REPORTS_DIR / "psi_train_vs_test.csv", index=False)
    psi_shifted.to_csv(REPORTS_DIR / "psi_shifted_demo.csv", index=False)
    stability_md = [
        "# 特征稳定性（PSI）报告",
        "",
        "PSI 判读标准：<0.1 稳定；0.1-0.25 中度漂移；>0.25 显著漂移。",
        "",
        "## train vs test（同分布随机切分，预期稳定）",
        "",
        psi_train_test.to_markdown(index=False),
        "",
        f"最大 PSI：**{psi_train_test['psi'].max():.4f}** → "
        f"{'整体稳定' if psi_train_test['psi'].max() < 0.1 else '存在漂移'}",
        "",
        "## train vs 人为漂移的 test（age+15 岁、credit_amount×1.5、duration+12 月）",
        "",
        "用于验证 PSI 工具确实能检出漂移（自实现模块的在线对照实验）：",
        "",
        psi_shifted.head(8).to_markdown(index=False),
        "",
        f"最大 PSI：**{psi_shifted['psi'].max():.4f}** → 应显著大于 0.25。",
    ]
    (REPORTS_DIR / "stability_report.md").write_text("\n".join(stability_md), encoding="utf-8")
    log(f"PSI train-vs-test 最大值 {psi_train_test['psi'].max():.4f}；漂移演示最大值 {psi_shifted['psi'].max():.4f}")

    # ------------------------------------------------- training report
    top_iv = iv_table.head(8)
    top_shap = imp.head(8)
    report = [
        "# 训练报告（credit-g 端到端）",
        "",
        f"- 训练时间：{meta['trained_at']}；数据：{metrics_payload['data_source']}",
        f"- 切分：分层 train/test = {len(X_train)}/{len(X_test)}（seed={RANDOM_STATE}）",
        "",
        "## 实验矩阵（2 模型 × 3 不平衡策略，5 折分层 CV + 独立测试集）",
        "",
        results.to_markdown(index=False),
        "",
        f"**最优组合：{best_model} + {best_strategy}**；测试集 AUC={final_metrics['auc']:.4f}，"
        f"KS={final_metrics['ks']:.4f}，Gini={final_metrics['gini']:.4f}。",
        "",
        "## 不平衡处理结论",
        "",
        _imbalance_conclusion(results),
        "",
        "## IV 前 8（训练集 WOE/IV）",
        "",
        top_iv.to_markdown(index=False),
        "",
        f"## 全局重要性前 8（方法：{shap_method}）",
        "",
        top_shap.to_markdown(index=False),
        "",
        "## 单样本解释示例（测试集第 1 条）",
        "",
        f"部署模型（{best_model}）归因方法：`{metrics_payload['deployed_model_local_explanation']}`",
        "",
        f"```json\n{json.dumps({'pd': float(test_proba[0]), 'y_true': int(y_test.iloc[0]), 'top_features': local}, ensure_ascii=False, indent=2)}\n```",
        "",
        f"LightGBM 的 SHAP 归因（供对照）：`{json.dumps(local_lgbm, ensure_ascii=False)}`",
        "",
        "## 风险分档阈值（测试集 PD 分位）",
        "",
        f"- 低风险：PD < {t_low:.3f}；中风险：{t_low:.3f} ≤ PD < {t_high:.3f}；高风险：PD ≥ {t_high:.3f}",
        "",
        "## 概率校准（独立测试集，等频 10 桶）",
        "",
        "排序指标（AUC/KS）不约束 PD 的绝对值；下面逐桶对比平均预测 PD 与实际违约率，",
        "对角线代表完美校准。ECE = 按桶样本量加权的 |平均预测 PD − 实际违约率| 之和。",
        "",
        f"### 部署模型（{deployed_label}）",
        "",
        cal_deployed_table.to_markdown(index=False),
        "",
        f"Brier={calibration['deployed']['brier']:.4f}，ECE={calibration['deployed']['ece']:.4f}，"
        f"平均预测 PD={calibration['deployed']['mean_predicted_pd']:.3f}，"
        f"实际违约率={calibration['deployed']['observed_bad_rate']:.3f}",
        "",
        f"### 参照（{reference_label}）",
        "",
        cal_ref_table.to_markdown(index=False),
        "",
        f"Brier={calibration['reference']['brier']:.4f}，ECE={calibration['reference']['ece']:.4f}，"
        f"平均预测 PD={calibration['reference']['mean_predicted_pd']:.3f}",
        "",
        _calibration_conclusion(calibration),
        "",
        "![calibration](calibration_curve.png)",
        "",
    ]
    (REPORTS_DIR / "training_report.md").write_text("\n".join(report), encoding="utf-8")

    log(f"完成，用时 {time.time() - started:.1f}s；产物见 artifacts/ 与 reports/")


def _calibration_conclusion(calibration: dict) -> str:
    """按实测数字给出校准结论（数据驱动，不预写方向）。"""
    dep = calibration["deployed"]
    ref = calibration["reference"]
    bias = dep["mean_predicted_pd"] - dep["observed_bad_rate"]
    if bias > 0.05:
        direction = "系统性偏高"
    elif bias < -0.05:
        direction = "系统性偏低"
    else:
        direction = "基本一致"
    return (
        f"**结论**：部署模型平均预测 PD {dep['mean_predicted_pd']:.3f} vs 实际违约率 "
        f"{dep['observed_bad_rate']:.3f}（偏移 {bias:+.3f}）→ PD 绝对值{direction}；"
        f"参照（未加权 LR）平均预测 PD {ref['mean_predicted_pd']:.3f}，ECE {ref['ece']:.4f} vs "
        f"部署模型 {dep['ece']:.4f}。balanced 加权改善排序（AUC/KS）但把 PD 绝对值抬高，"
        f"PD 不能直接当作真实违约概率用于定价或资本计算；风险分档阈值基于 PD 相对排序，"
        f"不受此偏移影响。如需绝对校准可在验证集上做 Platt/isotonic 修正（本项目未实现）。"
    )


def _imbalance_conclusion(results: pd.DataFrame) -> str:
    lines = []
    for model in results["model"].unique():
        sub = results[results["model"] == model].set_index("strategy")
        none_auc = sub.loc["none", "cv_auc_mean"]
        weight_auc = sub.loc["weight", "cv_auc_mean"]
        smote_auc = sub.loc["smote", "cv_auc_mean"]
        best = max([("none", none_auc), ("weight", weight_auc), ("smote", smote_auc)], key=lambda kv: kv[1])
        gap = abs(weight_auc - smote_auc)
        lines.append(
            f"- **{model}**：none={none_auc:.4f}，weight={weight_auc:.4f}，smote={smote_auc:.4f}（CV AUC）。"
            f"最优策略为 {best[0]}；weight 与 smote 差距 {gap:.4f}。"
            + ("两者几乎无差异——credit-g 不平衡程度温和（约 30% 坏样本），样本加权以更低复杂度达到同等效果，作为部署默认策略。"
               if gap < 0.01 else
               "存在可见差异，选择 CV AUC 更高者并如实记录。")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
