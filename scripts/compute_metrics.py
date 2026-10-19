"""
Single-run metrics: collision/safety, tracking, FOV/bearing, occlusion,
solver health, and smoothness -- computed from one trajectory CSV (currently
only examples/online_mapping_demo.cpp's logs/online_mapping_trajectory.csv
has the clearance_true/occlusion_clearance/solve_time_ms columns this script
needs; trailer_mpc_demo/argus_mpc logs will just skip the collision/
occlusion/solver-time panels).

This is the single-run drill-down tool (prints a summary table, draws one
multi-panel figure) -- the eventual Monte Carlo sweep runner will call the
same metric functions per-run and aggregate across scenarios instead of
plotting one run in isolation.

Run after examples/online_mapping_demo.cpp:
    ./build/online_mapping_demo
    python3 scripts/compute_metrics.py

Headless export:
    python3 scripts/compute_metrics.py --save-png docs/metrics.png
"""

import argparse

import numpy as np
import pandas as pd
import yaml


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", default="logs/online_mapping_trajectory.csv",
                    help="Trajectory CSV to analyze (default: %(default)s)")
    p.add_argument("--config-path", default="config/quadrotor.yaml",
                    help="Source of d_safe_m, fov_half_angle_deg, rate_max, "
                         "trailer.distance_to_target")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    p.add_argument("--occlusion-threshold-m", type=float, default=0.1,
                    help="occlusion_clearance below this counts as the sightline "
                         "actually being blocked, not just close to an obstacle "
                         "(default: %(default)s m, a heuristic -- see the module "
                         "docstring)")
    p.add_argument("--real-target-m", type=float, default=0.27,
                    help="Actual physical safety requirement, distinct from "
                         "--config-path's d_safe_m knob (which is already "
                         "inflated 2.7x to compensate for the soft hinge cost's "
                         "own undershoot, see design_decisions.md's 'd_safe "
                         "inflation') -- collision/d_safe_violation metrics are "
                         "measured against THIS, same convention as "
                         "plot_telemetry_dashboard.py's --real-target-m "
                         "(default: %(default)s m)")
    p.add_argument("--scenario-config", default="config/forest.yaml",
                    help="Scenario file (online_mapping_demo's obstacles: "
                         "section) to draw obstacle outlines on the occlusion "
                         "map panel. Missing file is not an error, the "
                         "outlines are just skipped.")
    p.add_argument("--summary-path", default="logs/online_mapping_summary.csv",
                    help="online_mapping_demo's one-row mapping-coverage "
                         "summary (resolved_fraction, observable/matched/"
                         "mismatched voxels). Missing file is not an error, "
                         "the mapping-coverage metrics are just skipped.")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv(_args.csv_path)
has_collision_cols = {"clearance_true", "occlusion_clearance", "solve_time_ms"} <= set(df.columns)

try:
    _mapping_summary = pd.read_csv(_args.summary_path).iloc[0]
except (FileNotFoundError, IndexError):
    _mapping_summary = None

try:
    _cfg = yaml.safe_load(open(_args.config_path))
except FileNotFoundError:
    _cfg = {}
D_SAFE_M = (_cfg.get("obstacle_avoidance") or {}).get("d_safe_m")
FOV_HALF_ANGLE_DEG = (_cfg.get("camera") or {}).get("fov_half_angle_deg")
RATE_MAX = (_cfg.get("constraints") or {}).get("rate_max")
DISTANCE_TO_TARGET = (_cfg.get("trailer") or {}).get("distance_to_target")
# Design target for pitch bearing -- NOT 0 (see config/quadrotor.yaml's
# trailer.target_bearing_deg comment). yaw has no such intentional offset
# (0 really is its design target), only pitch needs this.
TARGET_BEARING_DEG = (_cfg.get("trailer") or {}).get("target_bearing_deg")
TS_MS = (_cfg.get("mpc") or {}).get("Ts")
TS_MS = TS_MS * 1000.0 if TS_MS is not None else None

dt = float(df["t"].diff().median())
has_pipeline_time = "pipeline_time_ms" in df.columns


def bearing_angles_deg():
    """Full combined bearing angle gamma (angle between boresight and the
    true direction to the FOV target) plus its yaw/pitch decomposition --
    same ZYX-rotation convention as scripts/plot_telemetry_dashboard.py's
    _target_bearing_errors() and scripts/animate_xy.py's bearing_errors_deg(),
    extended here to the combined angle since that's what r_fov's cone
    (cos(alpha) vs cos(gamma)) actually penalizes, not yaw/pitch separately.

    Named "offset" (not "error"): these are raw angular offsets from dead-
    center of the boresight, and 0 is NOT always the design target -- pitch
    is deliberately held near trailer.target_bearing_deg (looking down at a
    ground target from a standoff altitude), only yaw's design target is
    genuinely 0. See pitch_bearing_deviation_from_design_deg below for the
    metric that's actually an "error" in the pitch axis.
    """
    cp, sp = np.cos(df["phi"]), np.sin(df["phi"])
    ct, st = np.cos(df["theta"]), np.sin(df["theta"])
    cy, sy = np.cos(df["psi"]), np.sin(df["psi"])
    dx = df["fx"] - df["x"]
    dy = df["fy"] - df["y"]
    dz = df["fz"] - df["z"]
    db_x = cy * ct * dx + sy * ct * dy - st * dz
    db_y = (cy * st * sp - sy * cp) * dx + (sy * st * sp + cy * cp) * dy + ct * sp * dz
    db_z = (cy * st * cp + sy * sp) * dx + (sy * st * cp - cy * sp) * dy + ct * cp * dz
    yaw_offset = np.asarray(np.degrees(np.arctan2(db_y, db_x)))
    pitch_offset = np.asarray(np.degrees(np.arctan2(-db_z, db_x)))
    rng = np.sqrt(db_x**2 + db_y**2 + db_z**2)
    gamma = np.asarray(np.degrees(np.arccos(np.clip(db_x / rng, -1.0, 1.0))))
    return yaw_offset, pitch_offset, gamma


def contiguous_true_runs(mask):
    """[(start_idx, end_idx_inclusive), ...] for each run of consecutive True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def compute_summary():
    s = {}
    t = df["t"].values

    # -- Collision / safety -------------------------------------------------
    # Measured against --real-target-m (the actual physical requirement),
    # NOT config's d_safe_m -- that knob is already inflated 2.7x to
    # compensate for the soft hinge cost's own undershoot (see
    # design_decisions.md's "d_safe inflation"), so clearance dropping below
    # d_safe_m is expected/by design, not a violation. d_safe_m is still
    # plotted as a reference line (how much margin the cost itself was
    # defending), it's just not the collision bar.
    if has_collision_cols:
        clearance = df["clearance_true"].values
        s["min_clearance_true_m"] = float(np.min(clearance))
        violation = clearance < _args.real_target_m
        s["real_target_violation_steps"] = int(violation.sum())
        s["collision_free"] = bool(not violation.any())
        margin = np.maximum(0.0, _args.real_target_m - clearance)
        s["clearance_margin_integral_m_s"] = float(np.sum(margin) * dt)

    # -- Tracking -------------------------------------------------------------
    xy_err = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])
    s["xy_tracking_error_mean_m"] = float(xy_err.mean())
    s["xy_tracking_error_max_m"] = float(xy_err.max())
    z_err = (df["z"] - df["zt"]).values
    s["z_tracking_error_mean_m"] = float(z_err.mean())
    s["z_tracking_error_max_abs_m"] = float(np.max(np.abs(z_err)))
    if DISTANCE_TO_TARGET is not None:
        standoff_err = np.hypot(df["x"] - df["fx"], df["y"] - df["fy"]) - DISTANCE_TO_TARGET
        s["standoff_error_mean_m"] = float(standoff_err.mean())
        s["standoff_error_max_abs_m"] = float(np.max(np.abs(standoff_err)))
    drone_path_len = float(np.sum(np.hypot(np.diff(df["x"]), np.diff(df["y"]))))
    target_path_len = float(np.sum(np.hypot(np.diff(df["fx"]), np.diff(df["fy"]))))
    s["path_length_ratio"] = drone_path_len / target_path_len if target_path_len > 0 else float("nan")

    # -- FOV / bearing ----------------------------------------------------------
    yaw_offset, pitch_offset, gamma = bearing_angles_deg()
    s["yaw_bearing_offset_mean_deg"] = float(yaw_offset.mean())
    s["yaw_bearing_offset_std_deg"] = float(yaw_offset.std())
    s["pitch_bearing_offset_mean_deg"] = float(pitch_offset.mean())
    s["pitch_bearing_offset_std_deg"] = float(pitch_offset.std())
    s["pitch_bearing_offset_max_deg"] = float(np.max(np.abs(pitch_offset)))
    # The actual "error" in the pitch axis: deviation from the configured
    # design bearing (trailer.target_bearing_deg), not from 0 -- see
    # bearing_angles_deg()'s docstring.
    if TARGET_BEARING_DEG is not None:
        pitch_deviation = pitch_offset - TARGET_BEARING_DEG
        s["pitch_bearing_deviation_from_design_deg_mean"] = float(pitch_deviation.mean())
        s["pitch_bearing_deviation_from_design_deg_max_abs"] = float(np.max(np.abs(pitch_deviation)))
    if FOV_HALF_ANGLE_DEG is not None:
        out_of_fov = gamma > FOV_HALF_ANGLE_DEG
        s["fov_exceedance_pct"] = float(100.0 * out_of_fov.mean())
        s["out_of_fov_duration_total_s"] = float(out_of_fov.sum() * dt)
        # The first event is "initial acquisition" (nothing was ever in FOV
        # yet), not a "re"-acquisition -- only matters once initial
        # pose/heading gets randomized (today's fixed initial condition
        # always starts in-FOV, so this is 0.0 in practice, but the
        # distinction has to be right before that assumption changes).
        events = contiguous_true_runs(out_of_fov)
        if events and events[0][0] == 0:
            ib, ie = events[0]
            s["initial_acquisition_latency_s"] = float((t[ie] - t[ib]) + dt)
            reacq_events = events[1:]
        else:
            s["initial_acquisition_latency_s"] = 0.0
            reacq_events = events
        s["reacquisition_events"] = len(reacq_events)
        s["reacquisition_latencies_s"] = [float((t[e] - t[b]) + dt) for b, e in reacq_events]

        # "Settled" means inside the FOV cone (gamma <= fov_half_angle_deg),
        # same bar as fov_exceedance_pct above -- NOT centered/near-zero
        # bearing, which this architecture's static standoff bearing never
        # approaches by design (see kTargetBearingDeg). First time gamma
        # enters the cone and never leaves it again for the rest of the run.
        in_fov = ~out_of_fov
        still_out = np.where(~in_fov)[0]
        s["initial_settle_time_s"] = float(t[still_out[-1]] + dt) if len(still_out) else 0.0

    # -- Occlusion ------------------------------------------------------------
    if has_collision_cols:
        occ = df["occlusion_clearance"].values
        s["occlusion_clearance_min_m"] = float(np.min(occ))
        s["occlusion_clearance_mean_m"] = float(np.mean(occ))
        occluded = occ < _args.occlusion_threshold_m
        s["occluded_pct"] = float(100.0 * occluded.mean())

    # -- Solver health ----------------------------------------------------------
    fail = (df["solve_status"] != 0).values
    s["solve_fail_rate_pct"] = float(100.0 * fail.mean())
    if has_collision_cols:
        st = df["solve_time_ms"].values
        s["solve_time_mean_ms"] = float(np.mean(st))
        s["solve_time_max_ms"] = float(np.max(st))
    # pipeline_time_ms is the whole control-tick wall-clock cost (follower
    # goal points, obstacle linearization, the MPC solve, and every sensor
    # sub-tick's lidar integration + own_map.computeEsdf()) -- solve_time_ms
    # alone only covers acados_solve() and understates real-time risk by
    # ~50x in practice (the ESDF recompute, not the solve, is the real cost
    # driver). See design_decisions.md's "Metrics dashboard" section.
    if has_pipeline_time:
        pt = df["pipeline_time_ms"].values
        s["pipeline_time_mean_ms"] = float(np.mean(pt))
        s["pipeline_time_max_ms"] = float(np.max(pt))
        if TS_MS is not None:
            over_budget = pt > TS_MS
            s["pipeline_over_budget_pct"] = float(100.0 * over_budget.mean())
            s["pipeline_real_time_safe"] = bool(not over_budget.any())
    if RATE_MAX is not None:
        eps = 0.02 * RATE_MAX
        sat = ((df["p_cmd"].abs() > RATE_MAX - eps) |
               (df["q_cmd"].abs() > RATE_MAX - eps) |
               (df["r_cmd"].abs() > RATE_MAX - eps)).values
        s["rate_saturation_pct"] = float(100.0 * sat.mean())

    # -- Mapping coverage -------------------------------------------------------
    # From online_mapping_demo.cpp's logs/online_mapping_summary.csv -- how
    # much of the relevant space own_map actually resolved, not a controller
    # metric. Low coverage (e.g. outrunning the sensor at high speed) can
    # explain collision/occlusion results that otherwise look like pure
    # controller failures.
    if _mapping_summary is not None:
        s["own_map_resolved_fraction_pct"] = float(_mapping_summary["resolved_fraction"]) * 100.0
        total = _mapping_summary["total_voxels"]
        observable = _mapping_summary["observable_voxels"]
        s["observable_voxels_pct"] = float(100.0 * observable / total) if total > 0 else float("nan")
        s["own_map_matched_pct"] = (float(100.0 * _mapping_summary["matched_voxels"] / observable)
                                     if observable > 0 else float("nan"))
        s["own_map_mismatched_voxels"] = int(_mapping_summary["mismatched_voxels"])

    # -- Smoothness -------------------------------------------------------------
    dp = np.diff(df["p"]) / dt
    dq = np.diff(df["q"]) / dt
    dr = np.diff(df["r"]) / dt
    s["body_rate_accel_rms_rad_s2"] = float(np.sqrt(np.mean(dp**2 + dq**2 + dr**2)))

    return s


def print_summary(s):
    print(f"Metrics -- {_args.csv_path}\n" + "=" * 60)
    for k, v in s.items():
        if isinstance(v, list):
            if v:
                print(f"{k:38s} n={len(v)}  mean={np.mean(v):.3f}s  max={np.max(v):.3f}s")
            else:
                print(f"{k:38s} (none)")
        elif isinstance(v, bool):
            print(f"{k:38s} {v}")
        elif isinstance(v, float):
            print(f"{k:38s} {v:.4f}")
        else:
            print(f"{k:38s} {v}")


def plot_results(s):
    panels = []

    def clearance_panel(a):
        if not has_collision_cols:
            a.axis("off")
            a.set_title("Clearance (no clearance_true column)")
            return
        a.plot(df["t"], df["clearance_true"], color="tab:blue", label="clearance_true (ground truth)")
        a.plot(df["t"], df["dist_obstacle"], color="tab:cyan", linestyle=":", alpha=0.7,
               label="dist_obstacle (own_map belief)")
        a.axhline(_args.real_target_m, color="tab:green", linestyle="--",
                   label=f"real target = {_args.real_target_m:.3f} m")
        if D_SAFE_M is not None:
            a.axhline(D_SAFE_M, color="tab:red", linestyle=":",
                       label=f"d_safe knob (config) = {D_SAFE_M:.3f} m")
        a.set_xlabel("t [s]"); a.set_ylabel("distance [m]")
        a.set_title("Clearance to nearest obstacle")
        a.legend(fontsize=8)
    panels.append(clearance_panel)

    def occlusion_panel(a):
        if not has_collision_cols:
            a.axis("off")
            a.set_title("Occlusion (no occlusion_clearance column)")
            return
        a.plot(df["t"], df["occlusion_clearance"], color="tab:purple")
        a.axhline(_args.occlusion_threshold_m, color="tab:red", linestyle="--",
                   label=f"threshold = {_args.occlusion_threshold_m:.2f} m")
        occluded = (df["occlusion_clearance"] < _args.occlusion_threshold_m).values
        for b, e in contiguous_true_runs(occluded):
            a.axvspan(df["t"].iloc[b], df["t"].iloc[e], color="red", alpha=0.2,
                       label="occluded" if (b, e) == contiguous_true_runs(occluded)[0] else None)
        a.set_xlabel("t [s]"); a.set_ylabel("distance [m]")
        a.set_title(f"Sightline clearance (occlusion proxy) -- "
                    f"occluded {s['occluded_pct']:.1f}% of the run")
        a.legend(fontsize=8)
    panels.append(occlusion_panel)

    def occlusion_map_panel(a):
        """Spatial view of the occlusion metric: where the sightline actually
        ran, not just how close it got over time -- the time-series panel
        above answers "when," this answers "where/why." Obstacle outlines +
        drone/target paths + the sightline at the single worst (minimum
        occlusion_clearance) moment, so a close call is a visible ray
        grazing a visible obstacle, not just a dip in a line plot."""
        if not has_collision_cols:
            a.axis("off")
            a.set_title("Occlusion map (no occlusion_clearance column)")
            return
        try:
            obstacles = yaml.safe_load(open(_args.scenario_config))["obstacles"]
        except (FileNotFoundError, KeyError, TypeError):
            obstacles = []
        for i, o in enumerate(obstacles):
            cx, cy = o["center"]
            label = "obstacle" if i == 0 else None
            if o["shape"] == "circle":
                a.add_patch(plt.Circle((cx, cy), o["radius"], facecolor="dimgray",
                                        alpha=0.4, edgecolor="black", label=label))
            else:
                h = o["half_size"]
                a.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                           facecolor="dimgray", alpha=0.4,
                                           edgecolor="black", label=label))

        a.plot(df["x"], df["y"], color="tab:blue", alpha=0.5, label="drone path")
        a.plot(df["fx"], df["fy"], "--", color="tab:red", alpha=0.5, label="target path")

        worst_idx = int(df["occlusion_clearance"].idxmin())
        worst = df.loc[worst_idx]

        if FOV_HALF_ANGLE_DEG is not None:
            yaw_deg = np.degrees(worst["psi"])
            sightline_len = np.hypot(worst["fx"] - worst["x"], worst["fy"] - worst["y"])
            wedge = mpatches.Wedge((worst["x"], worst["y"]), sightline_len * 1.3,
                                    yaw_deg - FOV_HALF_ANGLE_DEG, yaw_deg + FOV_HALF_ANGLE_DEG,
                                    color="orange", alpha=0.25, zorder=1,
                                    label=f"camera FOV (yaw only, ±{FOV_HALF_ANGLE_DEG:.0f}°)")
            a.add_patch(wedge)

        a.plot([worst["x"], worst["fx"]], [worst["y"], worst["fy"]],
               color="black", linewidth=2, zorder=5,
               label=f"worst sightline (clearance={worst['occlusion_clearance']:.3f} m, "
                     f"t={worst['t']:.2f}s)")
        a.scatter([worst["x"]], [worst["y"]], color="tab:blue", zorder=6, marker="o")
        a.scatter([worst["fx"]], [worst["fy"]], color="tab:red", zorder=6, marker="*")

        occluded = df["occlusion_clearance"] < _args.occlusion_threshold_m
        if occluded.any():
            a.scatter(df.loc[occluded, "x"], df.loc[occluded, "y"], color="orange", s=10,
                       zorder=4, label=f"occluded (<{_args.occlusion_threshold_m:.2f} m)")

        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
        a.set_title("Occlusion map")
        a.set_aspect("equal", adjustable="datalim")
        a.legend(fontsize=7)
    panels.append(occlusion_map_panel)

    def bearing_panel(a):
        yaw_offset, pitch_offset, gamma = bearing_angles_deg()
        a.plot(df["t"], yaw_offset, label="yaw offset", alpha=0.7)
        a.plot(df["t"], pitch_offset, label="pitch offset", alpha=0.7)
        a.plot(df["t"], gamma, color="black", label="combined (gamma)")
        if TARGET_BEARING_DEG is not None:
            a.axhline(TARGET_BEARING_DEG, color="tab:orange", linestyle="--", linewidth=1,
                       label=f"pitch design target ({TARGET_BEARING_DEG:.0f}°)")
        if FOV_HALF_ANGLE_DEG is not None:
            a.axhline(FOV_HALF_ANGLE_DEG, color="tab:red", linestyle=":", linewidth=1,
                       label=f"FOV edge ({FOV_HALF_ANGLE_DEG:.0f}°)")
            a.axhline(-FOV_HALF_ANGLE_DEG, color="tab:red", linestyle=":", linewidth=1)
        a.set_xlabel("t [s]"); a.set_ylabel("angle [deg]")
        a.set_title("Bearing offset from boresight")
        a.legend(fontsize=8)
    panels.append(bearing_panel)

    def tracking_panel(a):
        xy_err = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])
        a.plot(df["t"], xy_err, color="tab:green", label="XY tracking error")
        a.plot(df["t"], (df["z"] - df["zt"]).abs(), color="tab:orange", label="|Z tracking error|")
        a.set_xlabel("t [s]"); a.set_ylabel("error [m]")
        a.set_title("Tracking error")
        a.legend(fontsize=8)
    panels.append(tracking_panel)

    def standoff_panel(a):
        dist = np.hypot(df["x"] - df["fx"], df["y"] - df["fy"])
        a.plot(df["t"], dist, color="tab:blue", label="|drone - target|")
        if DISTANCE_TO_TARGET is not None:
            a.axhline(DISTANCE_TO_TARGET, color="tab:red", linestyle="--",
                       label=f"standoff = {DISTANCE_TO_TARGET:.2f} m")
        a.set_xlabel("t [s]"); a.set_ylabel("distance [m]")
        a.set_title("Standoff distance")
        a.legend(fontsize=8)
    panels.append(standoff_panel)

    def solver_panel(a):
        if not has_collision_cols:
            a.axis("off")
            a.set_title("Solve time (no solve_time_ms column)")
            return
        a.plot(df["t"], df["solve_time_ms"], color="tab:brown", label="acados solve_time_ms")
        if has_pipeline_time:
            a.plot(df["t"], df["pipeline_time_ms"], color="tab:purple",
                   label="whole-pipeline pipeline_time_ms")
        if TS_MS is not None:
            a.axhline(TS_MS, color="tab:red", linestyle="--",
                       label=f"real-time deadline (Ts = {TS_MS:.0f} ms)")
        fails = df[df["solve_status"] != 0]
        if len(fails):
            a.scatter(fails["t"], fails["solve_time_ms"], color="red", zorder=5, label="solve failed")
        a.set_xlabel("t [s]"); a.set_ylabel("time [ms]")
        a.set_title("Solver time vs. whole-pipeline time per step")
        a.legend(fontsize=8)
    panels.append(solver_panel)

    def jerk_panel(a):
        dp = np.degrees(np.diff(df["p"]) / dt)
        dq = np.degrees(np.diff(df["q"]) / dt)
        dr = np.degrees(np.diff(df["r"]) / dt)
        a.plot(df["t"][1:], dp, label="d(p)/dt", alpha=0.7)
        a.plot(df["t"][1:], dq, label="d(q)/dt", alpha=0.7)
        a.plot(df["t"][1:], dr, label="d(r)/dt", alpha=0.7)
        a.set_xlabel("t [s]"); a.set_ylabel("body-rate angular accel [deg/s²]")
        a.set_title("Body-rate angular acceleration (smoothness proxy)")
        a.legend(fontsize=8)
    panels.append(jerk_panel)

    def fnum(key, fmt="{:.3f}"):
        v = s.get(key)
        if v is None:
            return "--"
        return fmt.format(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else str(v)

    # (label, value, status) -- status is "ok"/"warn"/None (None = plain,
    # purely informational, no pass/fail judgement implied).
    ok, warn = "tab:green", "tab:red"
    groups = [
        ("Collision & safety", [
            ("min clearance (true)", fnum("min_clearance_true_m") + " m", None),
            ("real-target violations", fnum("real_target_violation_steps", "{:.0f}") + " steps",
             ok if s.get("collision_free") else warn),
            ("collision free", str(s.get("collision_free")), ok if s.get("collision_free") else warn),
            ("clearance margin integral", fnum("clearance_margin_integral_m_s") + " m·s", None),
        ]),
        ("Tracking", [
            ("XY error (mean / max)", f"{fnum('xy_tracking_error_mean_m')} / "
                                       f"{fnum('xy_tracking_error_max_m')} m", None),
            ("Z error (mean / max|.|)", f"{fnum('z_tracking_error_mean_m')} / "
                                         f"{fnum('z_tracking_error_max_abs_m')} m", None),
            ("standoff error (mean / max|.|)", f"{fnum('standoff_error_mean_m')} / "
                                                f"{fnum('standoff_error_max_abs_m')} m", None),
            ("path length ratio", fnum("path_length_ratio"), None),
        ]),
        ("FOV & bearing", [
            ("yaw offset (mean ± std)", f"{fnum('yaw_bearing_offset_mean_deg')} ± "
                                              f"{fnum('yaw_bearing_offset_std_deg')} °", None),
            ("pitch offset (mean ± std)", f"{fnum('pitch_bearing_offset_mean_deg')} ± "
                                                f"{fnum('pitch_bearing_offset_std_deg')} °", None),
            ("pitch deviation from design", f"{fnum('pitch_bearing_deviation_from_design_deg_mean')} "
                                             f"(max|.| {fnum('pitch_bearing_deviation_from_design_deg_max_abs')}) °", None),
            ("FOV exceedance", fnum("fov_exceedance_pct", "{:.1f}") + " %",
             ok if s.get("fov_exceedance_pct", 1) == 0 else warn),
            ("initial acquisition latency", fnum("initial_acquisition_latency_s") + " s", None),
            ("reacquisition events", fnum("reacquisition_events", "{:.0f}"), None),
            ("initial settle time", fnum("initial_settle_time_s") + " s", None),
        ]),
        ("Occlusion", [
            ("sightline clearance (min / mean)", f"{fnum('occlusion_clearance_min_m')} / "
                                                  f"{fnum('occlusion_clearance_mean_m')} m", None),
            ("occluded", fnum("occluded_pct", "{:.1f}") + " %",
             ok if s.get("occluded_pct", 1) == 0 else warn),
        ]),
        ("Solver health", [
            ("solve fail rate", fnum("solve_fail_rate_pct", "{:.1f}") + " %",
             ok if s.get("solve_fail_rate_pct", 1) == 0 else warn),
            ("solve time (mean / max)", f"{fnum('solve_time_mean_ms')} / "
                                         f"{fnum('solve_time_max_ms')} ms", None),
            ("pipeline time (mean / max)", f"{fnum('pipeline_time_mean_ms')} / "
                                            f"{fnum('pipeline_time_max_ms')} ms", None),
            ("real-time safe (pipeline < Ts)", str(s.get("pipeline_real_time_safe", "--")),
             ok if s.get("pipeline_real_time_safe") else warn),
            ("rate saturation", fnum("rate_saturation_pct", "{:.1f}") + " %",
             ok if s.get("rate_saturation_pct", 1) == 0 else warn),
        ]),
        ("Smoothness", [
            ("body-rate accel (rms)", fnum("body_rate_accel_rms_rad_s2") + " rad/s²", None),
        ]),
        ("Mapping coverage", [
            ("own_map resolved", fnum("own_map_resolved_fraction_pct", "{:.1f}") + " %", None),
            ("observable voxels", fnum("observable_voxels_pct", "{:.1f}") + " % of grid", None),
            ("matched vs. ground truth", fnum("own_map_matched_pct", "{:.4f}") + "% "
             f"({fnum('own_map_mismatched_voxels', '{:.0f}')} mismatched)",
             ok if s.get("own_map_mismatched_voxels") == 0 else warn),
        ]),
    ]

    def render_column(a, group_names, x0):
        y = 1.0
        for name in group_names:
            lines = dict(groups)[name]
            a.text(x0, y, name, transform=a.transAxes, va="top", ha="left",
                   fontsize=9, fontweight="bold")
            y -= 0.085
            for label, value, status in lines:
                a.text(x0 + 0.02, y, label, transform=a.transAxes, va="top", ha="left",
                       fontsize=8)
                color = status if status is not None else "black"
                a.text(x0 + 0.46, y, value, transform=a.transAxes, va="top", ha="left",
                       fontsize=8, color=color,
                       fontweight="bold" if status is not None else "normal")
                y -= 0.065
            y -= 0.025

    def summary_panel(a):
        a.axis("off")
        render_column(a, ["Collision & safety", "Tracking", "FOV & bearing"], 0.0)
        render_column(a, ["Occlusion", "Solver health", "Smoothness", "Mapping coverage"], 0.52)
        a.set_title("Summary (green/red = pass/fail against this run's own thresholds)")
    panels.append(summary_panel)

    data_panels = panels[:-1]
    n_data_rows = (len(data_panels) + 1) // 2
    fig = plt.figure(figsize=(14, 4 * n_data_rows + 4.5))
    gs = fig.add_gridspec(n_data_rows + 1, 2, height_ratios=[1] * n_data_rows + [1.3])
    fig.suptitle(f"Metrics -- {_args.csv_path}")

    for i, panel_fn in enumerate(data_panels):
        a = fig.add_subplot(gs[i // 2, i % 2])
        panel_fn(a)
        a.grid(True, alpha=0.3)

    summary_ax = fig.add_subplot(gs[n_data_rows, :])
    summary_panel(summary_ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if _args.save_png:
        fig.savefig(_args.save_png, dpi=150)
        print(f"\nSaved {_args.save_png}")
    else:
        plt.show()


if __name__ == "__main__":
    summary = compute_summary()
    print_summary(summary)
    plot_results(summary)
