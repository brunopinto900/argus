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


def stress_caption(m, blur_threshold_deg_s):
    """Very brief "what's actually stressed here" caption, computed from
    this scenario's own metrics (collision clearance, blur, tracking
    stability, FOV exceedance) rather than a hand-written description of
    scenario intent -- the user asked for what the numbers show, not a
    restatement of the design goal. Picks the top 2 standouts by a
    normalized severity score per metric, each scaled so ~1.0 is "notably
    stressed" for that metric:
      - collision: how close min_clearance_true_m got to 0, relative to
        the 0.5m band every curated scenario's margin was designed around
      - blur: body_rate_magnitude_max_deg_s relative to the derived
        9 deg/s threshold
      - FOV exceedance: scaled against a 10% nominal band
      - tracking stability: drone<->target distance std, scaled against
        a 0.1m nominal band
    """
    has_obstacles = m["min_clearance_true_m"] < 1000.0
    candidates = []
    if has_obstacles:
        collision_sev = max(0.0, 1.0 - m["min_clearance_true_m"] / 0.5)
        label = f"clearance {m['min_clearance_true_m']:.2f}m"
        if not m["collision_free"]:
            label += " (COLLISION)"
        candidates.append((collision_sev, label))
    blur_sev = m["body_rate_magnitude_max_deg_s"] / blur_threshold_deg_s
    candidates.append((blur_sev, f"blur max {m['body_rate_magnitude_max_deg_s']:.0f}°/s"))
    fov_sev = m["fov_exceedance_pct"] / 10.0
    candidates.append((fov_sev, f"FOV exceed {m['fov_exceedance_pct']:.0f}%"))
    track_sev = m["drone_target_dist_std_m"] / 0.1
    candidates.append((track_sev, f"tracking ±{m['drone_target_dist_std_m']:.2f}m"))

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = [label for sev, label in candidates[:2] if sev > 0.3]
    return " · ".join(top) if top else "no standout metric"


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

    # FOV cone at the worst (max-gamma) bearing moment of the run -- same
    # wedge geometry as compute_metrics.py's occlusion_map_panel, drawn
    # here for every scenario (not just the ones with an occlusion event)
    # so the suite always shows where the target came closest to leaving
    # the cone, not just whether it did.
    fov_half_angle_deg = cfg["camera"]["fov_half_angle_deg"]
    gamma = suite._bearing_gamma_deg(df)
    worst_fov_i = int(gamma.values.argmax())
    wf = df.loc[worst_fov_i]
    yaw_deg = math.degrees(wf["psi"])
    sightline_len = math.hypot(wf["fx"] - wf["x"], wf["fy"] - wf["y"])
    wedge = matplotlib.patches.Wedge((wf["x"], wf["y"]), sightline_len * 1.3,
                                      yaw_deg - fov_half_angle_deg, yaw_deg + fov_half_angle_deg,
                                      color="orange", alpha=0.22, zorder=0.5, label="FOV cone (worst moment)")
    ax.add_patch(wedge)
    ax.scatter([wf["x"]], [wf["y"]], color="orange", marker="^", s=22, zorder=5,
               edgecolor="black", linewidth=0.3, label="worst FOV moment")

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

    # Ticks left on (small font) rather than stripped -- gives each map an
    # actual x/y distance reference instead of an unscaled sketch.
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:green" if m["gates_pass"] else "tab:red")
        spine.set_linewidth(2.5)
    ax.set_title(name, fontsize=8, fontweight="bold", pad=26)
    ax.text(0.5, 1.22, stress_caption(m, blur_threshold_deg_s), transform=ax.transAxes,
             ha="center", va="bottom", fontsize=6.5, style="italic", color="dimgray")

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
        (f"blur {m['body_rate_magnitude_mean_deg_s']:.0f}±{m['body_rate_magnitude_std_deg_s']:.0f} / "
         f"max {m['body_rate_magnitude_max_deg_s']:.0f}°/s (thr {blur_threshold_deg_s:.0f})", blur_color),
        (f"target speed {m['target_speed_min_m_s']:.2f}/{m['target_speed_mean_m_s']:.2f}/"
         f"{m['target_speed_max_m_s']:.2f} m/s (min/mean/max)", "black"),
        (f"target accel mean|.| {m['target_accel_mean_abs_m_s2']:.2f} m/s²", "black"),
    ]
    y = -0.14
    for text, color in lines:
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
                 fontsize=6, color=color)
        y -= 0.115


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
    fig = plt.figure(figsize=(3.3 * N_COLS, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, N_COLS, hspace=1.3, wspace=0.35)

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
                    bbox_to_anchor=(0.5, -0.035))

    fig.suptitle("Curated scenario suite -- contact sheet "
                  "(green/red border = hard gates pass/fail)", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    fig.savefig(args.save_png, dpi=150, bbox_inches="tight")
    print(f"Saved {args.save_png}")


if __name__ == "__main__":
    main()
