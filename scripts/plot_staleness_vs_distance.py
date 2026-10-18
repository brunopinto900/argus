#!/usr/bin/env python3
"""Plot staleness error vs. distance to the obstacle, from
logs/staleness_log.csv (see examples/online_mapping_demo.cpp and
design_decisions.md's "Staleness error" section)."""
import argparse
import csv

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="logs/staleness_log.csv")
    parser.add_argument("--out", default="logs/staleness_error_vs_distance.png")
    parser.add_argument(
        "--skip-first", type=int, default=1,
        help="drop the first N rows (startup transient: own_map still kUnknown "
             "near t=0, d0/grad0 are conservative defaults, not a real "
             "linearization error -- see design_decisions.md)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))[args.skip_first:]
    dist = [float(r["true_dist_p1"]) for r in rows]
    error = [float(r["error"]) for r in rows]
    step_dist = [float(r["step_dist"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(dist, error, c=step_dist, cmap="viridis", s=18)
    axes[0].set_xlabel("true_dist(p1)  --  distance to the obstacle [m]")
    axes[0].set_ylabel("staleness error = d_lin(p1) - true_dist(p1)  [m]")
    axes[0].axhline(0.0, color="grey", linewidth=0.8)
    axes[0].set_title("Full range")

    # Zoom: drop the handful of largest-magnitude outliers so the real
    # near-zero cluster isn't flattened by a couple of points at -0.02/-0.065.
    sorted_err = sorted(error, key=abs)
    cutoff = sorted_err[int(0.95 * len(sorted_err))] if len(sorted_err) > 5 else max(sorted_err, key=abs)
    cutoff = abs(cutoff) * 1.2 if cutoff != 0 else 0.01
    zoom_dist = [d for d, e in zip(dist, error) if abs(e) <= cutoff]
    zoom_err = [e for e in error if abs(e) <= cutoff]
    zoom_step = [s for s, e in zip(step_dist, error) if abs(e) <= cutoff]
    n_dropped = len(error) - len(zoom_err)

    sc2 = axes[1].scatter(zoom_dist, zoom_err, c=zoom_step, cmap="viridis", s=18)
    axes[1].set_xlabel("true_dist(p1)  --  distance to the obstacle [m]")
    axes[1].axhline(0.0, color="grey", linewidth=0.8)
    axes[1].set_title(f"Zoomed (|error| <= {cutoff:.4f}, {n_dropped} outlier(s) dropped)")

    cbar = fig.colorbar(sc2, ax=axes, label="step_dist (|p1 - p0|) [m]")
    fig.suptitle("Staleness error vs. distance to obstacle")
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out} ({len(rows)} points, {n_dropped} dropped from zoom panel)")


if __name__ == "__main__":
    main()
