"""Draw every figure that has results behind it:  python experiments/make_plots.py [--results results]"""

import argparse

from neurotext.plots import make_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    made = make_all(ap.parse_args().results)
    print(f"\n{len(made)} figure(s) written")
