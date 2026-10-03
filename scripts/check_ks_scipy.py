"""开发期校验脚本：自实现 KS 与 scipy.stats.ks_2samp 交叉验证。"""
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, "src")
from creditrisk.evaluate import ks_statistic  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    ok = True
    for _ in range(20):
        y = rng.binomial(1, 0.35, 800)
        s = rng.random(800) + 0.4 * y
        mine = ks_statistic(y, s)
        ref = ks_2samp(s[y == 1], s[y == 0]).statistic
        if abs(mine - ref) > 1e-12:
            ok = False
            sys.stdout.write(f"MISMATCH mine={mine} ref={ref}\n")
    sys.stdout.write("KS vs scipy ks_2samp over 20 random cases: %s\n" % ("MATCH" if ok else "MISMATCH"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
