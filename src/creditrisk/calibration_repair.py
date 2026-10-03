"""PD 校准修正层：在 held-out 预测上拟合 Platt scaling（sigmoid）与 isotonic，
按验证集 ECE 择优，产出含校准器的部署 bundle。

背景（发现过程，见 calibration.py 与 README「概率校准」）：
class_weight="balanced" 使部署模型 PD 系统性偏高（测试集平均预测 0.437 vs 实际 0.300，
ECE 0.142）。本模块把该发现修复为部署能力：

协议（防止校准器偷看测试集）：
1. out_of_fold_proba：对训练集做 5 折分层 CV，逐折 fit 管线、预测本折，
   得到每条训练样本的 held-out（out-of-fold, OOF）预测概率——
   每个预测都来自未见过该样本的模型（与 sklearn CalibratedClassifierCV 同语义）；
2. OOF 预测再分层切两半：sel_fit 上分别拟合 sigmoid 与 isotonic，
   sel_val 上对比两者 ECE（择优在独立验证段上做，避免 isotonic 靠插值
   在拟合段自吹自擂）；择优后用全部 OOF 重拟合为部署校准器；
3. 部署校准器只对测试/线上 PD 做单调变换（sigmoid 严格单调，isotonic 单调不减），
   AUC/KS 排序能力不受影响。

两种候选方法：
- sigmoid（Platt scaling）：对 logit(p) 拟合一个无正则逻辑回归，
  calibrated = σ(a·logit(p) + b)，参数化光滑、小样本稳健；
- isotonic：自由单调阶梯拟合，样本充足时更贴近真实映射，但可能过拟合产生台阶。

部署 bundle（joblib）：模型管线 + 校准器 + 特征 schema 元数据 + 风险分档阈值，
供 FastAPI 与 Streamlit Demo 加载；序列化往返在 tests/test_calibration_repair.py 覆盖。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from .calibration import _validate
from .config import RANDOM_STATE

CALIBRATOR_METHODS = ("sigmoid", "isotonic")
_LOGIT_EPS = 1e-6


def logit(p, eps: float = _LOGIT_EPS) -> np.ndarray:
    """概率 → log-odds：裁剪到 [eps, 1-eps] 避免 0/1 处发散。"""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class ProbabilityCalibrator:
    """单调概率校准器：sigmoid（Platt）或 isotonic，fit 后经 apply 变换 PD。

    内部保存拟合好的 sklearn 对象，可被 joblib 直接序列化；
    apply 输出裁剪到 [0, 1]，对输入保持单调不减。
    """

    def __init__(self, method: str):
        if method not in CALIBRATOR_METHODS:
            raise ValueError(f"未知校准方法: {method}，可选 {CALIBRATOR_METHODS}")
        self.method = method
        self.model_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, raw_proba, y_true) -> ProbabilityCalibrator:
        y, p = _validate(y_true, raw_proba)
        if len(set(y)) < 2:
            raise ValueError("校准器拟合要求 y_true 同时包含 0/1 两类")
        if self.method == "sigmoid":
            # Platt scaling：在 log-odds 特征上拟合无正则逻辑回归（C 取大值近似无惩罚）
            model = LogisticRegression(C=1e6, max_iter=10000, solver="lbfgs")
            model.fit(logit(p).reshape(-1, 1), y)
        else:
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(p, y)
        self.model_ = model
        return self

    # ---------------------------------------------------------------- apply
    def apply(self, raw_proba) -> np.ndarray:
        """校准概率输出：输入校验同 _validate（仅概率侧），输出裁剪到 [0, 1]。"""
        if self.model_ is None:
            raise RuntimeError("ProbabilityCalibrator 尚未 fit")
        p = np.asarray(raw_proba, dtype=float)
        if p.size == 0:
            raise ValueError("输入为空")
        if not ((p >= 0) & (p <= 1)).all() or not np.isfinite(p).all():
            raise ValueError("raw_proba 必须为 [0, 1] 内的有限值")
        if self.method == "sigmoid":
            calibrated = self.model_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        else:
            calibrated = self.model_.predict(p)
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    # ---------------------------------------------------------------- utils
    @property
    def coefficients_(self) -> dict[str, float]:
        """sigmoid 的 (a, b)（calibrated = σ(a·logit(p)+b)），isotonic 无此参数。"""
        if self.method != "sigmoid" or self.model_ is None:
            return {}
        return {"a": float(self.model_.coef_[0, 0]), "b": float(self.model_.intercept_[0])}


def out_of_fold_proba(X: pd.DataFrame, y: pd.Series, make_pipeline_fn,
                      n_splits: int = 5, random_state: int = RANDOM_STATE) -> np.ndarray:
    """5 折分层 CV 的 held-out（OOF）预测概率：每条样本的预测来自未见过它的折模型。"""
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    for train_idx, valid_idx in skf.split(X, y):
        pipe = make_pipeline_fn()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("OOF 预测存在空洞：折划分未覆盖全部样本")
    return oof


def select_and_fit_calibrator(raw_fit, y_fit, raw_val, y_val,
                              n_bins: int = 10) -> tuple[ProbabilityCalibrator, dict]:
    """在 sel_fit 上分别拟合两种校准器，在 sel_val 上对比 ECE 择优，再用全部数据重拟合。

    返回 (部署校准器, 选择报告)。报告记录三方（raw/sigmoid/isotonic）在验证段的
    ECE 与 Brier，择优规则：验证 ECE 最小者；并列取 sigmoid（参数化更简单）。
    """
    from .calibration import brier_score, expected_calibration_error

    y_v, p_v = _validate(y_val, raw_val)
    y_f, p_f = _validate(y_fit, raw_fit)

    candidates: dict[str, ProbabilityCalibrator] = {}
    report: dict = {
        "selection_fit_n": int(len(y_f)),
        "selection_valid_n": int(len(y_v)),
        "n_bins": int(n_bins),
        "ece_valid_raw": expected_calibration_error(y_v, p_v, n_bins=n_bins),
        "brier_valid_raw": brier_score(y_v, p_v),
    }
    for method in CALIBRATOR_METHODS:
        calibrator = ProbabilityCalibrator(method).fit(p_f, y_f)
        calibrated_val = calibrator.apply(p_v)
        candidates[method] = calibrator
        report[f"ece_valid_{method}"] = expected_calibration_error(y_v, calibrated_val, n_bins=n_bins)
        report[f"brier_valid_{method}"] = brier_score(y_v, calibrated_val)

    chosen = min(CALIBRATOR_METHODS, key=lambda m: (report[f"ece_valid_{m}"], CALIBRATOR_METHODS.index(m)))
    report["method"] = chosen

    # 部署校准器：择优方法在全部 OOF 数据上重拟合（sel_fit + sel_val 合并）
    y_all, p_all = _validate(np.concatenate([y_f, y_v]), np.concatenate([p_f, p_v]))
    deploy_calibrator = ProbabilityCalibrator(chosen).fit(p_all, y_all)
    return deploy_calibrator, report


def calibrate_pipeline(pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       make_pipeline_fn, n_splits: int = 5,
                       random_state: int = RANDOM_STATE,
                       selection_frac: float = 0.5,
                       n_bins: int = 10) -> tuple[ProbabilityCalibrator, dict, dict]:
    """端到端校准修复：OOF 预测 → 择优 → 测试集校准前后指标。

    返回 (部署校准器, selection 报告, test 对比指标)。
    测试集对比指标包含校准前后的 ECE/Brier、平均 PD 与 AUC（验证排序能力保持）。
    """
    from .calibration import brier_score, expected_calibration_error
    from .evaluate import roc_auc

    oof = out_of_fold_proba(X_train, y_train, make_pipeline_fn,
                            n_splits=n_splits, random_state=random_state)
    y_train = pd.Series(y_train).reset_index(drop=True)
    raw_fit, raw_val, y_fit, y_val = train_test_split(
        oof, y_train, test_size=1.0 - selection_frac,
        stratify=y_train, random_state=random_state,
    )
    calibrator, selection = select_and_fit_calibrator(raw_fit, y_fit, raw_val, y_val, n_bins=n_bins)

    y_test_arr = np.asarray(y_test).astype(int)
    test_raw = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
    test_calibrated = calibrator.apply(test_raw)
    test_report = {
        "n_bins": int(n_bins),
        "test_n": int(len(y_test_arr)),
        "test_ece_before": expected_calibration_error(y_test_arr, test_raw, n_bins=n_bins),
        "test_ece_after": expected_calibration_error(y_test_arr, test_calibrated, n_bins=n_bins),
        "test_brier_before": brier_score(y_test_arr, test_raw),
        "test_brier_after": brier_score(y_test_arr, test_calibrated),
        "test_mean_pd_before": float(np.mean(test_raw)),
        "test_mean_pd_after": float(np.mean(test_calibrated)),
        "test_auc_before": roc_auc(y_test_arr, test_raw),
        "test_auc_after": roc_auc(y_test_arr, test_calibrated),
        "observed_bad_rate": float(np.mean(y_test_arr)),
    }
    return calibrator, selection, test_report


# ------------------------------------------------------------------- bundle
def apply_calibrated_proba(calibrator: ProbabilityCalibrator | None, raw_proba) -> np.ndarray:
    """部署口径：有校准器则变换，无则原样返回（float / ndarray 输入皆可）。"""
    raw = np.atleast_1d(np.asarray(raw_proba, dtype=float))
    if calibrator is None:
        return raw
    return calibrator.apply(raw)


def build_bundle(pipeline, calibrator: ProbabilityCalibrator | None, meta: dict) -> dict:
    """组装部署 bundle：模型管线 + 校准器 + 特征 schema 元数据。"""
    required = ("features",)
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"bundle meta 缺少字段: {missing}")
    return {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "meta": dict(meta),
    }


def save_bundle(path: Path, bundle: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path) -> dict:
    """加载部署 bundle；文件缺失时抛出带修复指引的 RuntimeError（供 UI 呈现）。"""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"缺少部署产物 {path}。请先在仓库目录运行 python scripts/run_training.py "
            "生成（或改用包含 artifacts/ 的完整仓库）。"
        )
    bundle = joblib.load(path)
    if "pipeline" not in bundle or "meta" not in bundle:
        raise RuntimeError(f"{path} 不是有效的部署 bundle（缺少 pipeline/meta 字段）。")
    return bundle
