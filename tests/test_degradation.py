# -*- coding: utf-8 -*-
"""诚实降级的真实可达性测试：在子进程中屏蔽 shap，验证

1. creditrisk.explain 的 SHAP_AVAILABLE 变为 False，global_importance 自动
   走 permutation importance（而非崩溃）；
2. scripts/run_training.py 模块级导入不再无条件依赖 shap
   （回归：曾因顶部硬 import shap 导致降级分支不可达）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPLAIN_WITHOUT_SHAP = r"""
import sys
sys.modules["shap"] = None  # 使 import shap 抛 ImportError，模拟未安装
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import creditrisk.explain as explain

assert explain.SHAP_AVAILABLE is False, "shap 被屏蔽后 SHAP_AVAILABLE 应为 False"
assert explain.shap is None

from creditrisk.features import build_pipeline
from lightgbm import LGBMClassifier

rng = np.random.default_rng(0)
X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.choice(list("xyz"), size=120)})
y = pd.Series((rng.random(120) < 0.35).astype(int))
pipe = build_pipeline(LGBMClassifier(n_estimators=5, verbose=-1, random_state=0), n_bins=3)
pipe.fit(X, y)

table = explain.global_importance(pipe, X, y=y)  # use_shap 默认 True → 必须自动降级
assert (table["method"] == "permutation_auc_drop").all(), table["method"].unique()

# 树模型在无 shap 时单样本解释必须明确报错（而非静默给出错误归因）
try:
    explain.explain_instance(pipe, X.iloc[[0]], top_k=2)
    raise SystemExit("FAIL: expected ValueError for tree model without shap")
except ValueError as exc:
    assert "无法做单样本解释" in str(exc)

# 线性模型的单样本解释走 coef 降级路径，无 shap 依然可用
from sklearn.linear_model import LogisticRegression

lr_pipe = build_pipeline(LogisticRegression(max_iter=2000), n_bins=3).fit(X, y)
local = explain.explain_instance(lr_pipe, X.iloc[[0]], top_k=2)
assert len(local) == 2
print("DEGRADATION_OK")
"""

TRAINING_IMPORT_WITHOUT_SHAP = r"""
import runpy
import sys

sys.modules["shap"] = None
runpy.run_path("scripts/run_training.py", run_name="import_smoke")
print("TRAINING_IMPORT_OK")
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_explain_degrades_to_permutation_without_shap():
    proc = _run(EXPLAIN_WITHOUT_SHAP)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "DEGRADATION_OK" in proc.stdout


def test_training_entry_imports_without_shap():
    proc = _run(TRAINING_IMPORT_WITHOUT_SHAP)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "TRAINING_IMPORT_OK" in proc.stdout
