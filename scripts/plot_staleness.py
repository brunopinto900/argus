#!/usr/bin/env python3
"""Plot staleness error vs. gradient change, from logs/staleness_log.csv
(see examples/online_mapping_demo.cpp and design_decisions.md's
"Staleness error" section)."""
import argparse
import csv
import math

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="logs/staleness_log.csv")
    parser.add_argument("--out", default="logs/staleness_error_vs_grad_change.png")
    parser.add_argument(
        "--skip-first", type=int, default=1,
        help="drop the first N rows (startup transient: own_map still kUnknown "
             "near t=0, d0/grad0 are conservative defaults, not a real "
             "linearization error -- see design_decisions.md)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))[args.skip_first:]
    grad_diff = []
    error = []
    step_dist = []
    for r in rows:
        dgx = float(r["grad1_x"]) - float(r["grad0_x"])
        dgy = float(r["grad1_y"]) - float(r["grad0_y"])
        grad_diff.append(math.hypot(dgx, dgy))
        error.append(float(r["error"]))
        step_dist.append(float(r["step_dist"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(grad_diff, error, c=step_dist, cmap="viridis", s=18)
    ax.set_xlabel("|grad1 - grad0|  (how much the ESDF gradient rotated/changed)")
    ax.set_ylabel("staleness error = d_lin(p1) - true_dist(p1)  [m]")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_title("Staleness error vs. gradient change")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("step_dist (|p1 - p0|) [m]")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out} ({len(rows)} points)")


if __name__ == "__main__":
    main()
