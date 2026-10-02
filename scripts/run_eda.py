# -*- coding: utf-8 -*-
"""运行 EDA：python scripts/run_eda.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk.config import REPORTS_DIR  # noqa: E402
from creditrisk.data import load_credit_data  # noqa: E402
from creditrisk.eda import run_eda  # noqa: E402


def main() -> None:
    data = load_credit_data()
    stats = run_eda(data.df, REPORTS_DIR)
    print("EDA 完成：", stats)
    if data.is_synthetic:
        print("警告：当前数据为合成降级数据，图表结论仅供管线演示。")


if __name__ == "__main__":
    main()
