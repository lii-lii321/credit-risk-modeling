# -*- coding: utf-8 -*-
"""端到端训练入口：实验对比 → 最优模型 → 校准修复 → 产物落盘 → 可解释性 → 稳定性报告。

运行：python scripts/run_training.py
产物：
- artifacts/pipeline.joblib、deploy_bundle.joblib（模型+校准器+schema 元数据）、
  model_meta.json、metrics.json
- reports/model_comparison.csv、iv_table.csv、shap_*.csv/png、
  calibration_table.csv（校准前/后/参照三组）、calibration_curve.png、
  fairness.csv / fairness.md（personal_status 分组公平性审计）、
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
from creditrisk.calibration_repair import (  # noqa: E402
    build_bundle,
    calibrate_pipeline,
    save_bundle,
)
from creditrisk.data import load_credit_data  # noqa: E402
from creditrisk.evaluate import evaluate_predictions  # noqa: E402
from creditrisk.explain import SHAP_AVAILABLE, explain_instance, global_importance  # noqa: E402
from creditrisk.fairness import fairness_audit, fairness_markdown, fairness_table  # noqa: E402
from creditrisk.models import (  # noqa: E402
    fit_deployable_pipeline,
    make_pipeline,
    run_all_experiments,
    select_best,
)
from creditrisk.psi import psi_table  # noqa: E402
from creditrisk.thresholds import (  # noqa: E402
    plot_tradeoff_curves,
    thresholds_for_approval_rates,
    tradeoff_table,
)
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
    test_proba = pipeline.predict_proba(X_test)[:, 1]  # 校准前（原始）PD
    final_metrics = evaluate_predictions(y_test, test_proba)
    log(f"最终模型测试集指标：{final_metrics}")
    joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.joblib")

    # ------------------------------------------------ calibration repair
    # 校准修复：balanced 加权使 PD 系统性偏高（校准前 ECE 见下）。协议（防泄漏）：
    # 1) 训练集 5 折分层 CV 得到每条样本的 held-out（OOF）预测概率；
    # 2) OOF 分层切两半——sel_fit 上分别拟合 Platt(sigmoid) 与 isotonic，
    #    sel_val 上对比两者 ECE 择优；择优者用全部 OOF 重拟合为部署校准器；
    # 3) 测试集只做单调变换，绝不参与校准器拟合。
    log("校准修复：在训练集 OOF held-out 预测上拟合 sigmoid/isotonic 并按验证 ECE 择优 …")
    calibrator, calib_selection, calib_test = calibrate_pipeline(
        pipeline, X_train, y_train, X_test, y_test,
        make_pipeline_fn=lambda: make_pipeline(best_model, best_strategy),
        n_splits=5, random_state=RANDOM_STATE,
    )
    calibrated_test_proba = calibrator.apply(test_proba)
    calibration_method = calibrator.method
    log(f"校准修复：择优方法={calibration_method}，测试集 ECE "
        f"{calib_test['test_ece_before']:.4f} → {calib_test['test_ece_after']:.4f}，"
        f"Brier {calib_test['test_brier_before']:.4f} → {calib_test['test_brier_after']:.4f}，"
        f"AUC {calib_test['test_auc_before']:.4f} → {calib_test['test_auc_after']:.4f}（排序保持）")

    # risk band 阈值：测试集（校准后）PD 的 60% / 85% 分位。
    # API/线上一律返回校准后 PD，分档阈值必须与之同尺度；校准为单调变换，
    # 与校准前阈值一一对应（排序不变，风险分档归属不受影响）。
    t_low, t_high = np.quantile(calibrated_test_proba, [0.60, 0.85])
    t_low_raw, t_high_raw = np.quantile(test_proba, [0.60, 0.85])

    # ------------------------------------------------------- calibration
    # 排序指标（AUC/KS）不约束 PD 的绝对值；class_weight="balanced" 会系统性抬高
    # 预测 PD。这里用未加权 LR 作参照量化该副作用：Brier / ECE / 可靠性曲线。
    ref_pipe = fit_deployable_pipeline(X_train, y_train, "logistic_regression", "none")
    ref_proba = ref_pipe.predict_proba(X_test)[:, 1]
    cal_deployed_table = reliability_data(y_test, test_proba, n_bins=10)
    cal_ref_table = reliability_data(y_test, ref_proba, n_bins=10)
    deployed_label = f"{best_model}+{best_strategy} (deployed, before calibration)"
    reference_label = "logistic_regression+none (reference)"
    repaired_label = f"{deployed_label} + {calibration_method} (deployed, after calibration)"
    cal_repaired_table = reliability_data(y_test, calibrated_test_proba, n_bins=10)
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
        "repair": {
            "method": calibration_method,
            "protocol": ("5-fold stratified OOF predictions on train (held-out per fold); "
                         "sigmoid vs isotonic fit on sel_fit half, compared by ECE on sel_val half; "
                         "winner refit on full OOF; test set never used for fitting"),
            "selection": calib_selection,
            "test": calib_test,
            "sigmoid_coefficients": calibrator.coefficients_,
            "model": repaired_label,
            "brier": calib_test["test_brier_after"],
            "ece": calib_test["test_ece_after"],
            "mean_predicted_pd": float(np.mean(calibrated_test_proba)),
            "observed_bad_rate": float(np.mean(y_test)),
            "reliability": cal_repaired_table.to_dict("records"),
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
        cal_repaired_table.assign(model=repaired_label),
        cal_ref_table.assign(model=reference_label),
    ])[["model", "bin", "n", "mean_predicted_pd", "observed_bad_rate"]]
    cal_csv.to_csv(REPORTS_DIR / "calibration_table.csv", index=False)
    plot_reliability_diagram(
        [(deployed_label, y_test, test_proba),
         (repaired_label, y_test, calibrated_test_proba),
         (reference_label, y_test, ref_proba)],
        REPORTS_DIR / "calibration_curve.png",
        n_bins=10,
    )
    log(f"校准：部署模型（校准前）Brier={calibration['deployed']['brier']:.4f} "
        f"ECE={calibration['deployed']['ece']:.4f}，平均预测 PD "
        f"{calibration['deployed']['mean_predicted_pd']:.3f} vs 实际违约率 "
        f"{calibration['deployed']['observed_bad_rate']:.3f}；校准后（{calibration_method}）"
        f"Brier={calib_test['test_brier_after']:.4f} ECE={calib_test['test_ece_after']:.4f}")

    # -------------------------------------------------- threshold trade-off
    # 把 PD 排序翻译成业务口径：不同批准率目标下的阈值与批内/拒件坏账率。
    # 阈值取测试集 PD 分位数，属排序型阈值，不受上面校准偏移影响。
    tradeoff_grid = np.quantile(test_proba, np.linspace(0.1, 0.9, 9))
    tradeoff = tradeoff_table(y_test, test_proba, tradeoff_grid)
    approval_targets = (0.7, 0.8, 0.9)
    tradeoff_targets = thresholds_for_approval_rates(y_test, test_proba, approval_targets)
    tradeoff.to_csv(REPORTS_DIR / "threshold_tradeoff.csv", index=False)
    tradeoff_targets.to_csv(REPORTS_DIR / "threshold_targets.csv", index=False)
    plot_tradeoff_curves(y_test, test_proba, tradeoff_grid,
                         REPORTS_DIR / "threshold_tradeoff.png")
    threshold_tradeoff = {
        "decision_rule": "approve if PD < threshold",
        "test_n": int(len(y_test)),
        "grid_quantiles": [float(q) for q in np.linspace(0.1, 0.9, 9)],
        "table": tradeoff.to_dict("records"),
        "targets": tradeoff_targets.to_dict("records"),
    }
    log("阈值-业务指标（目标批准率 70/80/90%）：")
    log(tradeoff_targets.to_string(index=False))

    # ------------------------------------------------------- fairness audit
    # 公平性审计（描述性，只测量不修正）：在测试集（n=200，复用训练管线现有切分
    # 与预测，不重新建模）上按 personal_status（含性别编码）分组，统计部署阈值
    # （校准后 PD < t_low，即低风险档=批准，与 API/线上同尺度）下的：
    # 分组样本量与实际违约率、选择率与 demographic parity 差距、
    # 等机会差距（各组 TPR = 批准|实际正常）、分组 AUC。
    # 未做再平衡/去偏/受保护属性移除；小样本组读数噪声大，不构成合规结论。
    log("公平性审计：按 personal_status 分组（部署阈值=校准后 PD < t_low）…")
    fairness = fairness_audit(
        y_test, calibrated_test_proba, X_test["personal_status"],
        threshold=float(t_low), group_column="personal_status",
    )
    fairness_tbl = fairness_table(fairness)
    fairness_tbl.to_csv(REPORTS_DIR / "fairness.csv", index=False)
    (REPORTS_DIR / "fairness.md").write_text(fairness_markdown(fairness), encoding="utf-8")
    log("公平性分组指标：")
    log(fairness_tbl.to_string(index=False))
    log(f"demographic parity 差距={fairness['demographic_parity_gap']:.4f}，"
        f"等机会差距={fairness['equal_opportunity_gap']:.4f}")


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
        "threshold_tradeoff": threshold_tradeoff,
        "fairness": fairness,
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
        "risk_band_thresholds_raw_pd": {"low_below": float(t_low_raw), "medium_below": float(t_high_raw)},
        "calibration": {
            "applied": True,
            "method": calibration_method,
            "test_ece_before": calib_test["test_ece_before"],
            "test_ece_after": calib_test["test_ece_after"],
            "test_brier_before": calib_test["test_brier_before"],
            "test_brier_after": calib_test["test_brier_after"],
        },
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

    # 部署 bundle：模型管线 + 校准器 + 特征 schema 元数据 + 风险分档阈值，
    # 供 FastAPI 与 Streamlit Demo 加载（单一事实来源）。
    bundle_meta = {
        "model_version": meta["model_version"],
        "model": best_model,
        "strategy": best_strategy,
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "category_values": CATEGORY_VALUES,
        "risk_band_thresholds": meta["risk_band_thresholds"],
        "calibration": meta["calibration"],
    }
    save_bundle(
        ARTIFACTS_DIR / "deploy_bundle.joblib",
        build_bundle(pipeline, calibrator, bundle_meta),
    )
    log(f"部署 bundle 已导出：{ARTIFACTS_DIR / 'deploy_bundle.joblib'}（校准器={calibration_method}）")

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
        "## 风险分档阈值（测试集校准后 PD 的 60%/85% 分位）",
        "",
        f"- 低风险：PD < {t_low:.3f}；中风险：{t_low:.3f} ≤ PD < {t_high:.3f}；高风险：PD ≥ {t_high:.3f}"
        f"（尺度：校准后 PD，与 API/线上输出一致）",
        f"- 校准前（原始 PD）尺度的等价阈值：{t_low_raw:.3f} / {t_high_raw:.3f}——",
        "校准为单调变换，两套阈值在排序意义下一一对应，样本风险分档归属完全一致",
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
        "### 校准修复（OOF held-out 预测上择优 sigmoid / isotonic）",
        "",
        "协议：训练集 5 折分层 CV 产出每条样本的 held-out（OOF）预测 → OOF 分层切两半，",
        "sel_fit 上分别拟合 Platt(sigmoid) 与 isotonic，sel_val 上对比 ECE 择优，",
        "择优者用全部 OOF 重拟合为部署校准器；测试集只做单调变换，绝不参与校准器拟合。",
        "",
        _repair_selection_table(calibration).to_markdown(index=False),
        "",
        f"择优结果：**{calibration_method}**"
        + (f"（σ(a·logit(p)+b)，a={calibrator.coefficients_.get('a', float('nan')):.4f}，"
           f"b={calibrator.coefficients_.get('b', float('nan')):.4f}）" if calibration_method == "sigmoid" else ""),
        "",
        "#### 测试集复测（校准前 → 校准后，n=200）",
        "",
        _repair_test_table(calib_test).to_markdown(index=False),
        "",
        _calibration_conclusion(calibration),
        "",
        "![calibration](calibration_curve.png)",
        "",
        "## 阈值-业务指标（批准率 / 坏账率，独立测试集）",
        "",
        f"决策规则：PD < 阈值 → 批准，PD ≥ 阈值 → 拒绝（n={len(y_test)}）。",
        "阈值取测试集 PD 的十分位分位数，属排序型阈值，不受 PD 绝对值校准偏移影响。",
        "",
        "### 阈值扫描（PD 十分位）",
        "",
        tradeoff.to_markdown(index=False),
        "",
        "### 按目标批准率反查阈值",
        "",
        tradeoff_targets.to_markdown(index=False),
        "",
        _threshold_conclusion(tradeoff_targets, len(y_test)),
        "",
        "![threshold trade-off](threshold_tradeoff.png)",
        "",
    ]
    (REPORTS_DIR / "training_report.md").write_text("\n".join(report), encoding="utf-8")

    log(f"完成，用时 {time.time() - started:.1f}s；产物见 artifacts/ 与 reports/")


def _threshold_conclusion(targets_table: pd.DataFrame, test_n: int) -> str:
    """按实测数字给出阈值-业务结论（数据驱动）：收紧批准率换来多少坏账率下降。"""
    tightest = targets_table.iloc[0]        # 目标批准率最低（最严）一行
    loosest = targets_table.iloc[-1]        # 目标批准率最高（最松）一行
    drop = loosest["bad_rate_approved"] - tightest["bad_rate_approved"]
    relative = drop / loosest["bad_rate_approved"] if loosest["bad_rate_approved"] else float("nan")
    smallest_cell = int(targets_table[["n_approved", "n_rejected"]].to_numpy().min())
    return (
        f"**结论**：批准率从 {loosest['approval_rate']:.0%} 收紧到 {tightest['approval_rate']:.0%}"
        f"（阈值 {loosest['threshold']:.3f} → {tightest['threshold']:.3f}），"
        f"批内坏账率由 {loosest['bad_rate_approved']:.1%} 降至 {tightest['bad_rate_approved']:.1%}"
        f"（相对下降 {relative:.0%}），被拒人群坏账率 {tightest['bad_rate_rejected']:.1%}。"
        f"本分析为描述性扫描，未引入利润/损失矩阵做阈值最优化；测试集仅 {test_n} 条，"
        f"最细分组（十分位扫描每档、或最松目标下的拒绝组）约 {smallest_cell} 条，"
        f"坏账率读数对单样本波动敏感（±1 个坏样本 ≈ ±{100 / smallest_cell:.0f} 个百分点）。"
    )


def _calibration_conclusion(calibration: dict) -> str:
    """三段式校准结论：发现（保留校准前数字）→ 修复（方法与择优依据）→ 复测（测试集实测）。"""
    dep = calibration["deployed"]
    ref = calibration["reference"]
    rep = calibration["repair"]
    t = rep["test"]
    sel = rep["selection"]
    bias = dep["mean_predicted_pd"] - dep["observed_bad_rate"]
    fixed = t["test_ece_after"] < t["test_ece_before"]
    verdict = (
        f"校准后平均 PD 与实际违约率偏差收窄至 "
        f"{abs(t['test_mean_pd_after'] - t['observed_bad_rate']):.3f}，"
        f"可谨慎用于需要绝对 PD 的场景（测试集仅 {t['test_n']} 条，逐桶读数仍有抽样噪声）。"
        if fixed else
        f"本次修复未能在测试集上降低 ECE（{t['test_ece_after']:.4f} ≥ {t['test_ece_before']:.4f}），"
        "如实记录：校准后 PD 仍不能当作真实违约概率使用。"
    )
    return (
        f"**发现**：balanced 加权改善排序（AUC/KS）但把 PD 绝对值系统性抬高——"
        f"部署模型平均预测 PD {dep['mean_predicted_pd']:.3f} vs 实际违约率 "
        f"{dep['observed_bad_rate']:.3f}（偏移 {bias:+.3f}），校准前 ECE {dep['ece']:.4f}"
        f"（未加权参照 {ref['ece']:.4f}），PD 不能直接当作真实违约概率。\n\n"
        f"**修复**：在训练集 OOF held-out 预测上择优校准器——验证段（n={sel['selection_valid_n']}）"
        f"ECE：raw={sel['ece_valid_raw']:.4f}，sigmoid={sel['ece_valid_sigmoid']:.4f}，"
        f"isotonic={sel['ece_valid_isotonic']:.4f}，选中 **{rep['method']}** 并以全部 OOF 重拟合；"
        f"测试集只做单调变换，未参与校准器拟合。\n\n"
        f"**复测**：测试集 ECE {t['test_ece_before']:.4f} → {t['test_ece_after']:.4f}，"
        f"Brier {t['test_brier_before']:.4f} → {t['test_brier_after']:.4f}，"
        f"平均预测 PD {t['test_mean_pd_before']:.3f} → {t['test_mean_pd_after']:.3f}"
        f"（实际违约率 {t['observed_bad_rate']:.3f}）；AUC {t['test_auc_before']:.4f} → "
        f"{t['test_auc_after']:.4f}（单调校准，排序能力保持）。{verdict}"
    )


def _repair_selection_table(calibration: dict) -> pd.DataFrame:
    """择优过程表：三方（raw/sigmoid/isotonic）在验证段的 ECE 与 Brier。"""
    sel = calibration["repair"]["selection"]
    rows = [
        {"candidate": "raw (uncalibrated)", "valid_ece": sel["ece_valid_raw"], "valid_brier": sel["brier_valid_raw"]},
        {"candidate": "sigmoid (platt)", "valid_ece": sel["ece_valid_sigmoid"], "valid_brier": sel["brier_valid_sigmoid"]},
        {"candidate": "isotonic", "valid_ece": sel["ece_valid_isotonic"], "valid_brier": sel["brier_valid_isotonic"]},
    ]
    table = pd.DataFrame(rows)
    table["selected"] = ["", "✔" if calibration["repair"]["method"] == "sigmoid" else "",
                         "✔" if calibration["repair"]["method"] == "isotonic" else ""]
    return table


def _repair_test_table(t: dict) -> pd.DataFrame:
    """测试集校准前后对比表：ECE/Brier/平均 PD/AUC/实际违约率。"""
    return pd.DataFrame([
        {"metric": "ECE ↓", "before": round(t["test_ece_before"], 4), "after": round(t["test_ece_after"], 4)},
        {"metric": "Brier ↓", "before": round(t["test_brier_before"], 4), "after": round(t["test_brier_after"], 4)},
        {"metric": "mean predicted PD", "before": round(t["test_mean_pd_before"], 4), "after": round(t["test_mean_pd_after"], 4)},
        {"metric": "AUC（排序保持）", "before": round(t["test_auc_before"], 4), "after": round(t["test_auc_after"], 4)},
    ])


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
