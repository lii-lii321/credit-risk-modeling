# -*- coding: utf-8 -*-
"""开发期校验：合成数据坏样本率校准。"""
import sys

sys.path.insert(0, "src")
from creditrisk.data import generate_synthetic_credit  # noqa: E402


def main() -> None:
    for seed in (1, 42, 123):
        for n in (500, 1000):
            d = generate_synthetic_credit(n=n, seed=seed)
            sys.stdout.write(f"seed={seed} n={n} bad_rate={d.y.mean():.3f}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
