"""
run_all.py  (data_exploration)
==============================
End-to-end: census -> families -> stats -> figures.

    python -m data_exploration.run_all                  # everything, resuming
    python -m data_exploration.run_all --skip census    # reuse the scan

The census is the only expensive stage (~5 min on 8 workers for all 3,986,754
structures) and it resumes from whatever shard parts already exist, so re-running
this module after a code change costs a couple of minutes, not a re-scan.
"""

from __future__ import annotations

import argparse
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="train_4M")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["census", "families", "stats", "figures"])
    a = p.parse_args()

    t0 = time.time()
    if "census" not in a.skip:
        from . import census
        census.run(src=a.src, workers=a.workers)
    if "families" not in a.skip:
        from . import families
        families.run()
    if "stats" not in a.skip:
        from . import stats
        stats.run()
    if "figures" not in a.skip:
        from . import figures
        for name, fn in figures.FIGURES.items():
            fn()
    print(f"[run_all] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
