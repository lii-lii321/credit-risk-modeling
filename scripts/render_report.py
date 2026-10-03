"""免训练重渲染一页式 HTML 训练报告（reports/report.html）。

直接读取已提交/已有的 artifacts/metrics.json、artifacts/model_meta.json 与
reports/ 下既有图表、CSV 产物与 README（已知限制章节），重新组装单页自包含
HTML；不加载模型、不重新训练。与 scripts/run_training.py 训练末尾自动生成
的是同一渲染函数（src/creditrisk/html_report.render_html_report）。

运行：python scripts/render_report.py
可选：python scripts/render_report.py <project_root>（默认取本脚本上级目录）
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditrisk.html_report import rerender_from_artifacts  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT
    out = rerender_from_artifacts(root)
    print(f"[render_report] 已重渲染：{out}")


if __name__ == "__main__":
    main()
