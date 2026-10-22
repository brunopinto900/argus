"""
Contact sheet: one small top-down map per scenario (obstacles + drone path +
target path + closest-approach/worst-occlusion markers), laid out in a grid
with a short text summary under each -- a single-glance spatial overview
across the whole curated suite, instead of opening each scenario's full
9-panel dashboard one at a time. Those dashboards (scripts/compute_metrics.py,
already saved per scenario under logs/suite/<name>_dashboard.png by
scripts/run_scenario_suite.py) stay the drill-down tool for whichever
thumbnail here looks interesting -- this script only draws the overview.

Reads scripts/run_scenario_suite.py's SCENARIOS list and its already-
archived logs/suite/<name>_trajectory.csv (run run_scenario_suite.py first
if those don't exist yet) -- no re-simulation here.

Run:
    python3 scripts/plot_scenario_contact_sheet.py
    python3 scripts/plot_scenario_contact_sheet.py --save-png logs/suite/contact_sheet.png
"""

import argparse
import math
import os

import matplotlib
import matplotlib.patches
import numpy as np
import pandas as pd
import yaml

import run_scenario_suite as suite

REPO_ROOT = suite.REPO_ROOT
SUITE_DIR = suite.SUITE_DIR

N_COLS = 5


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-png", default=os.path.join(SUITE_DIR, "contact_sheet.png"),
                    help="Output PNG path (default: %(default)s)")
    return p.parse_args()


def _blur_threshold_deg_s(cfg):
    """Same derivation as compute_metrics.py's BLUR_THRESHOLD_DEG_S --
    duplicated here for the same reason as run_scenario_suite.py's other
    standalone formulas."""
    fov_half_angle_deg = cfg["camera"]["fov_half_angle_deg"]
    rate_hz = cfg["mapping_sensor"]["update_rate_hz"]
    width_px, tol_px = 640, 1.0
    f_px = width_px / (2.0 * math.tan(math.radians(fov_half_angle_deg)))
    t_exposure = 1.0 / rate_hz
    return math.degrees(tol_px / (t_exposure * f_px))


def _draw_obstacles(ax, obstacles):
    for o in obstacles:
        cx, cy = o["center"]
        if o["shape"] == "circle":
            ax.add_patch(matplotlib.patches.Circle((cx, cy), o["radius"], facecolor="dimgray",
                                                     alpha=0.5, edgecolor="black", linewidth=0.5))
        else:
            h = o["half_size"]
            ax.add_patch(matplotlib.patches.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                                        facecolor="dimgray", alpha=0.5,
                                                        edgecolor="black", linewidth=0.5))


def plot_cell(ax, name, scenario_file, df, cfg, blur_threshold_deg_s):
    m = suite.compute_gate_and_stats(df, cfg)

    try:
        obstacles = yaml.safe_load(open(os.path.join(REPO_ROOT, "config", scenario_file)))["obstacles"]
    except (FileNotFoundError, KeyError, TypeError):
        obstacles = []
    _draw_obstacles(ax, obstacles)

    ax.plot(df["x"], df["y"], color="tab:blue", linewidth=1.0, alpha=0.8, label="drone")
    ax.plot(df["fx"], df["fy"], "--", color="tab:red", linewidth=1.0, alpha=0.8, label="target")

    # No obstacles -> clearance_true/occlusion_clearance are both the
    # ESDF's "nothing nearby" sentinel, so their argmin is an arbitrary
    # index, not a real event -- skip the markers rather than plot
    # something misleading.
    has_obstacles = m["min_clearance_true_m"] < 1000.0
    clear_color = "tab:green" if m["collision_free"] else "tab:red"
    if has_obstacles:
        worst_clear_i = int(df["clearance_true"].values.argmin())
        wc = df.loc[worst_clear_i]
        ax.scatter([wc["x"]], [wc["y"]], color=clear_color, marker="X", s=40, zorder=5,
                   label="closest approach")

        worst_occ_i = int(df["occlusion_clearance"].values.argmin())
        wo = df.loc[worst_occ_i]
        ax.scatter([wo["x"]], [wo["y"]], color="tab:purple", marker="o", s=18, zorder=5,
                   label="worst occlusion")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:green" if m["gates_pass"] else "tab:red")
        spine.set_linewidth(2.5)
    ax.set_title(name, fontsize=8, fontweight="bold")

    onboard_color = "tab:red" if m["pipeline_onboard_time_max_ms"] > cfg["mpc"]["Ts"] * 1000.0 else "black"
    blur_color = "tab:red" if m["body_rate_magnitude_max_deg_s"] > blur_threshold_deg_s else "black"
    clear_flag = "collision-free" if m["collision_free"] else "COLLISION"
    # No-obstacle scenarios report the ESDF's "nothing nearby" sentinel
    # (a huge distance, not a real clearance) -- show that plainly instead
    # of a meaningless 50000000.00m.
    clear_str = ("n/a (no obstacles)" if m["min_clearance_true_m"] > 1000.0
                 else f"{m['min_clearance_true_m']:.2f}m")

    lines = [
        (f"clear {clear_str} [{clear_flag}]", clear_color),
        (f"track {m['drone_target_dist_mean_m']:.2f}±{m['drone_target_dist_std_m']:.2f}m  "
         f"reacq {m['reacquisition_events']}", "black"),
        (f"FOV {m['fov_exceedance_pct']:.0f}%  onboard {m['pipeline_onboard_time_mean_ms']:.1f}/"
         f"{m['pipeline_onboard_time_max_ms']:.1f}ms", onboard_color),
        (f"blur max {m['body_rate_magnitude_max_deg_s']:.0f}°/s "
         f"(thr {blur_threshold_deg_s:.0f})", blur_color),
    ]
    y = -0.12
    for text, color in lines:
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
                 fontsize=6, color=color)
        y -= 0.11


def main():
    args = _parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt

    with open(os.path.join(REPO_ROOT, "config", "quadrotor.yaml")) as f:
        cfg = yaml.safe_load(f)
    blur_threshold_deg_s = _blur_threshold_deg_s(cfg)

    n = len(suite.SCENARIOS)
    n_rows = math.ceil(n / N_COLS)
    fig = plt.figure(figsize=(3.1 * N_COLS, 3.9 * n_rows))
    gs = fig.add_gridspec(n_rows, N_COLS, hspace=0.6, wspace=0.3)

    handles_labels = None
    for i, (name, scenario_file, _) in enumerate(suite.SCENARIOS):
        ax = fig.add_subplot(gs[i // N_COLS, i % N_COLS])
        csv_path = os.path.join(SUITE_DIR, f"{name}_trajectory.csv")
        if not os.path.exists(csv_path):
            ax.axis("off")
            ax.set_title(f"{name}\n(no logs/suite/*_trajectory.csv -- run "
                         f"run_scenario_suite.py first)", fontsize=7)
            continue
        df = pd.read_csv(csv_path)
        plot_cell(ax, name, scenario_file, df, cfg, blur_threshold_deg_s)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    if handles_labels is not None:
        fig.legend(*handles_labels, loc="lower center", ncol=4, fontsize=9,
                    bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Curated scenario suite -- contact sheet "
                  "(green/red border = hard gates pass/fail)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(args.save_png, dpi=150, bbox_inches="tight")
    print(f"Saved {args.save_png}")


if __name__ == "__main__":
    main()
