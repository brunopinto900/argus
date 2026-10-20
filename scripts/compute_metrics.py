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
    p.add_argument("--d-fov", type=float, default=4.0,
                    help="Occlusion falloff scale [m]: occlusion_clearance (the "
                         "ground-truth min ESDF distance along the drone->target "
                         "sightline, already logged by online_mapping_demo.cpp) is "
                         "converted to a percentage via occ_pct = 100 * max(0, 1 - "
                         "occlusion_clearance/d_FOV) -- 0 m clearance = 100% "
                         "occluded, >= d_FOV clearance = 0% occluded. Same formula "
                         "and default as scripts/occlusion_metric_demo.py (default: "
                         "%(default)s m)")
    p.add_argument("--occlusion-event-pct", type=float, default=80.0,
                    help="occ_pct above this counts as an occlusion EVENT worth "
                         "flagging red, not just some nonzero degradation "
                         "(default: %(default)s %%)")
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

# Blur threshold -- derived from the actual depth camera this project is
# built around (OAK-D-Lite; see ros2/argus_px4_bridge/doc/README.md's
# "Depth camera & world" section), not a generic rule of thumb. See
# design_decisions.md's "Blur threshold" section for the full derivation.
# blur_px = omega[rad/s] * t_exposure[s] * f_px, f_px = width_px/(2*tan(half_angle)).
# width_px=640 and the 1/update_rate_hz worst-case exposure both come from
# this project's own depth-camera topic (640x480) and
# mapping_sensor.update_rate_hz -- only the "<=1px is sharp" tolerance is a
# generic imaging convention, not derived from this hardware.
_DEPTH_CAM_WIDTH_PX = 640
_BLUR_TOLERANCE_PX = 1.0
_mapping_rate_hz = (_cfg.get("mapping_sensor") or {}).get("update_rate_hz")
if FOV_HALF_ANGLE_DEG is not None and _mapping_rate_hz:
    _f_px = _DEPTH_CAM_WIDTH_PX / (2.0 * np.tan(np.radians(FOV_HALF_ANGLE_DEG)))
    _t_exposure_worst_s = 1.0 / _mapping_rate_hz
    BLUR_THRESHOLD_DEG_S = float(np.degrees(_BLUR_TOLERANCE_PX / (_t_exposure_worst_s * _f_px)))
else:
    BLUR_THRESHOLD_DEG_S = None

dt = float(df["t"].diff().median())
has_pipeline_time = "pipeline_time_ms" in df.columns
has_pipeline_onboard_time = "pipeline_onboard_time_ms" in df.columns
has_esdf_time = "esdf_time_ms" in df.columns


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
    # occlusion_clearance is the ground-truth min ESDF distance along the
    # drone->target sightline (see online_mapping_demo.cpp's
    # occlusionClearance()) -- already exactly the "lateral clearance to the
    # bearing vector, depth-gated to the segment between drone and target"
    # quantity the occlusion metric needs (design_decisions.md's "Occlusion
    # metric" section), so no new CSV column is needed: just convert the
    # existing clearance into the same percentage/falloff scripts/
    # occlusion_metric_demo.py validates.
    if has_collision_cols:
        occ = df["occlusion_clearance"].values
        occ_pct = 100.0 * np.clip(1.0 - occ / _args.d_fov, 0.0, 1.0)
        s["occlusion_clearance_min_m"] = float(np.min(occ))
        s["occlusion_clearance_mean_m"] = float(np.mean(occ))
        s["occlusion_pct_mean"] = float(np.mean(occ_pct))
        s["occlusion_pct_max"] = float(np.max(occ_pct))
        occluded = occ_pct > _args.occlusion_event_pct
        s["occluded_pct"] = float(100.0 * occluded.mean())

    # -- Solver health ----------------------------------------------------------
    fail = (df["solve_status"] != 0).values
    s["solve_fail_rate_pct"] = float(100.0 * fail.mean())
    if has_collision_cols:
        st = df["solve_time_ms"].values
        s["solve_time_mean_ms"] = float(np.mean(st))
        s["solve_time_max_ms"] = float(np.max(st))
    # pipeline_onboard_time_ms is the whole control-tick wall-clock cost a
    # real flight computer would actually pay (follower goal points,
    # obstacle linearization, the MPC solve, and every sensor sub-tick's
    # lidar integration + own_map.computeEsdf()), with
    # ground_truth.raycast() subtracted out -- that call stands in for a
    # real depth sensor's own hardware resolving hits/misses, not something
    # an onboard pipeline would ever pay for. solve_time_ms alone only
    # covers acados_solve() and understates real-time risk by ~50x in
    # practice (the ESDF recompute, not the solve, is the real cost
    # driver). The raw wall-clock (pipeline_time_ms, including that
    # raycast) is still computed below for s[], just not surfaced in the
    # plot/summary -- see design_decisions.md's "Metrics dashboard"
    # section.
    if has_pipeline_time:
        pt = df["pipeline_time_ms"].values
        s["pipeline_time_mean_ms"] = float(np.mean(pt))
        s["pipeline_time_max_ms"] = float(np.max(pt))
    if has_pipeline_onboard_time:
        pot = df["pipeline_onboard_time_ms"].values
        s["pipeline_onboard_time_mean_ms"] = float(np.mean(pot))
        s["pipeline_onboard_time_max_ms"] = float(np.max(pot))
    if has_esdf_time:
        et = df["esdf_time_ms"].values
        s["esdf_time_mean_ms"] = float(np.mean(et))
        s["esdf_time_max_ms"] = float(np.max(et))
    real_time_series = pot if has_pipeline_onboard_time else (pt if has_pipeline_time else None)
    if real_time_series is not None and TS_MS is not None:
        over_budget = real_time_series > TS_MS
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
    # Two different physical questions, not the same metric twice --
    # see design_decisions.md's "Smoothness" section:
    #  - body_rate_accel (d(p,q,r)/dt): is the COMMANDED motion smooth, or
    #    chattering (large reversals)? A single large, purposeful turn has
    #    high rate but low jerk; an oscillation has both. This is what
    #    flagged the softplus_beta=5 chattering pathology.
    #  - body_rate_magnitude (p,q,r themselves): the actual blur proxy --
    #    image-plane smear during one exposure scales with angular rate
    #    integrated over the exposure window, not with how fast that rate
    #    is changing. A sustained high rate blurs every frame even with
    #    zero jerk; a jerky-but-small rate never blurs at all.
    dp = np.diff(df["p"]) / dt
    dq = np.diff(df["q"]) / dt
    dr = np.diff(df["r"]) / dt
    s["body_rate_accel_rms_rad_s2"] = float(np.sqrt(np.mean(dp**2 + dq**2 + dr**2)))

    rate_mag_deg_s = np.degrees(np.sqrt(df["p"]**2 + df["q"]**2 + df["r"]**2)).values
    s["body_rate_magnitude_mean_deg_s"] = float(np.mean(rate_mag_deg_s))
    s["body_rate_magnitude_max_deg_s"] = float(np.max(rate_mag_deg_s))

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
        occ_pct = 100.0 * np.clip(1.0 - df["occlusion_clearance"].values / _args.d_fov, 0.0, 1.0)
        a.plot(df["t"], occ_pct, color="tab:purple")
        a.axhline(_args.occlusion_event_pct, color="tab:red", linestyle="--",
                   label=f"event threshold = {_args.occlusion_event_pct:.0f} %")
        occluded = occ_pct > _args.occlusion_event_pct
        for b, e in contiguous_true_runs(occluded):
            a.axvspan(df["t"].iloc[b], df["t"].iloc[e], color="red", alpha=0.2,
                       label="occluded" if (b, e) == contiguous_true_runs(occluded)[0] else None)
        a.set_xlabel("t [s]"); a.set_ylabel("occlusion [%]")
        a.set_ylim(-5, 105)
        a.set_title(f"Occlusion -- 100*max(0, 1 - clearance/d_FOV={_args.d_fov:.1f}m) -- "
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
        worst_pct = 100.0 * np.clip(1.0 - worst["occlusion_clearance"] / _args.d_fov, 0.0, 1.0)

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
               label=f"worst sightline (occ={worst_pct:.0f}%, clearance={worst['occlusion_clearance']:.3f} m, "
                     f"t={worst['t']:.2f}s)")
        a.scatter([worst["x"]], [worst["y"]], color="tab:blue", zorder=6, marker="o")
        a.scatter([worst["fx"]], [worst["fy"]], color="tab:red", zorder=6, marker="*")

        occ_pct = 100.0 * np.clip(1.0 - df["occlusion_clearance"].values / _args.d_fov, 0.0, 1.0)
        occluded = occ_pct > _args.occlusion_event_pct
        if occluded.any():
            a.scatter(df.loc[occluded, "x"], df.loc[occluded, "y"], color="orange", s=10,
                       zorder=4, label=f"occluded (>{_args.occlusion_event_pct:.0f}%)")

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
        if has_esdf_time:
            a.plot(df["t"], df["esdf_time_ms"], color="tab:orange",
                   label="esdf_time_ms (own_map.computeEsdf, all sub-ticks)")
        if has_pipeline_onboard_time:
            a.plot(df["t"], df["pipeline_onboard_time_ms"], color="tab:purple",
                   label="pipeline_onboard_time_ms (real onboard cost only)")
        if TS_MS is not None:
            a.axhline(TS_MS, color="tab:red", linestyle="--",
                       label=f"real-time deadline (Ts = {TS_MS:.0f} ms)")
        fails = df[df["solve_status"] != 0]
        if len(fails):
            a.scatter(fails["t"], fails["solve_time_ms"], color="red", zorder=5, label="solve failed")
        a.set_xlabel("t [s]"); a.set_ylabel("time [ms]")
        a.set_title("Onboard pipeline time per step (sensor-sim raycast excluded)")
        a.legend(fontsize=8)
    panels.append(solver_panel)

    def jerk_panel(a):
        dp = np.degrees(np.diff(df["p"]) / dt)
        dq = np.degrees(np.diff(df["q"]) / dt)
        dr = np.degrees(np.diff(df["r"]) / dt)
        a.plot(df["t"][1:], dp, label="roll rate accel (d(p)/dt)", alpha=0.7)
        a.plot(df["t"][1:], dq, label="pitch rate accel (d(q)/dt)", alpha=0.7)
        a.plot(df["t"][1:], dr, label="yaw rate accel (d(r)/dt)", alpha=0.7)
        a.set_xlabel("t [s]"); a.set_ylabel("body-rate angular accel [deg/s²]")
        a.set_title("Body-rate angular acceleration (control-smoothness / chatter proxy)")
        a.legend(fontsize=8)
    panels.append(jerk_panel)

    def rate_magnitude_panel(a):
        """Blur proxy, distinct from the jerk panel above: image-plane smear
        during one exposure scales with angular RATE integrated over the
        exposure window, not with how fast that rate is changing -- a
        sustained high rate blurs every frame even with zero jerk."""
        mag = np.degrees(np.sqrt(df["p"]**2 + df["q"]**2 + df["r"]**2))
        a.plot(df["t"], mag, color="tab:red")
        if BLUR_THRESHOLD_DEG_S is not None:
            a.axhline(BLUR_THRESHOLD_DEG_S, color="black", linestyle="--", linewidth=1,
                       label=f"blur threshold ({_BLUR_TOLERANCE_PX:.0f}px @ OAK-D-Lite "
                             f"640px/{_mapping_rate_hz:.0f}fps) = {BLUR_THRESHOLD_DEG_S:.1f} °/s")
            a.legend(fontsize=7)
        a.set_xlabel("t [s]"); a.set_ylabel("combined body-rate magnitude [deg/s]")
        a.set_title("Body-rate magnitude (blur proxy)")
    panels.append(rate_magnitude_panel)

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
            ("pitch dev. from design", f"{fnum('pitch_bearing_deviation_from_design_deg_mean', '{:.2f}')} "
                                        f"(max {fnum('pitch_bearing_deviation_from_design_deg_max_abs', '{:.2f}')}) °", None),
            ("FOV exceedance", fnum("fov_exceedance_pct", "{:.1f}") + " %",
             ok if s.get("fov_exceedance_pct", 1) == 0 else warn),
            ("initial acquisition latency", fnum("initial_acquisition_latency_s") + " s", None),
            ("reacquisition events", fnum("reacquisition_events", "{:.0f}"), None),
            ("initial settle time", fnum("initial_settle_time_s") + " s", None),
        ]),
        ("Occlusion", [
            ("occlusion (mean / max)", f"{fnum('occlusion_pct_mean', '{:.1f}')} / "
                                        f"{fnum('occlusion_pct_max', '{:.1f}')} %", None),
            ("sightline clearance (min / mean)", f"{fnum('occlusion_clearance_min_m')} / "
                                                  f"{fnum('occlusion_clearance_mean_m')} m", None),
            (f"occlusion events (>{_args.occlusion_event_pct:.0f}%)", fnum("occluded_pct", "{:.1f}") + " %",
             ok if s.get("occluded_pct", 1) == 0 else warn),
        ]),
        ("Solver health", [
            ("solve fail rate", fnum("solve_fail_rate_pct", "{:.1f}") + " %",
             ok if s.get("solve_fail_rate_pct", 1) == 0 else warn),
            ("solve time (mean / max)", f"{fnum('solve_time_mean_ms')} / "
                                         f"{fnum('solve_time_max_ms')} ms", None),
            ("esdf compute (mean / max)", f"{fnum('esdf_time_mean_ms')} / "
                                            f"{fnum('esdf_time_max_ms')} ms", None),
            ("onboard pipeline (mean / max)", f"{fnum('pipeline_onboard_time_mean_ms')} / "
                                            f"{fnum('pipeline_onboard_time_max_ms')} ms", None),
            ("real-time safe (< Ts)", str(s.get("pipeline_real_time_safe", "--")),
             ok if s.get("pipeline_real_time_safe") else warn),
            ("rate saturation", fnum("rate_saturation_pct", "{:.1f}") + " %",
             ok if s.get("rate_saturation_pct", 1) == 0 else warn),
        ]),
        ("Smoothness", [
            ("body-rate accel, chatter proxy (rms)", fnum("body_rate_accel_rms_rad_s2") + " rad/s²", None),
            ("body-rate magnitude, blur proxy (mean / max)", f"{fnum('body_rate_magnitude_mean_deg_s', '{:.1f}')} / "
                                                              f"{fnum('body_rate_magnitude_max_deg_s', '{:.1f}')} °/s", None),
        ]),
        ("Mapping coverage", [
            ("own_map resolved", fnum("own_map_resolved_fraction_pct", "{:.1f}") + " %", None),
            ("observable voxels", fnum("observable_voxels_pct", "{:.1f}") + " % of grid", None),
            ("matched vs. ground truth", fnum("own_map_matched_pct", "{:.2f}") + "% "
             f"({fnum('own_map_mismatched_voxels', '{:.0f}')} mismatched)",
             ok if s.get("own_map_mismatched_voxels") == 0 else warn),
        ]),
    ]

    # Base (unscaled) vertical steps, in axes fraction -- header_step for a
    # group's title line, line_step per metric row, gap_step between groups.
    header_step, line_step, gap_step = 0.085, 0.065, 0.025

    def group_weight(name):
        n = len(dict(groups)[name])
        return header_step + n * line_step + gap_step

    # Each column is a half-width [x0, x0+0.5) slot: label starts right after
    # x0, value starts at a fixed offset inside the slot -- wide enough that
    # the longest label in `groups` above doesn't run into the value column,
    # and the value column doesn't run into the next column's label.
    col_width, value_offset = 0.5, 0.32

    def render_column(a, group_names, x0, scale):
        y = 0.98
        for name in group_names:
            lines = dict(groups)[name]
            a.text(x0, y, name, transform=a.transAxes, va="top", ha="left",
                   fontsize=9, fontweight="bold")
            y -= header_step * scale
            for label, value, status in lines:
                a.text(x0 + 0.02, y, label, transform=a.transAxes, va="top", ha="left",
                       fontsize=7.5)
                color = status if status is not None else "black"
                a.text(x0 + value_offset, y, value, transform=a.transAxes, va="top", ha="left",
                       fontsize=7.5, color=color,
                       fontweight="bold" if status is not None else "normal")
                y -= line_step * scale
            y -= gap_step * scale

    def summary_panel(a):
        a.axis("off")
        # Groups are packed into two columns by a greedy balance on their
        # rendered weight (not a fixed hand-picked split) so a future
        # metric/group added to `groups` above never silently pushes a
        # column's text below the axes -- each column is then scaled down
        # (tighter line spacing) if its natural weight still would have.
        all_names = [name for name, _ in groups]
        columns = [[], []]
        totals = [0.0, 0.0]
        for name in sorted(all_names, key=group_weight, reverse=True):
            col = 0 if totals[0] <= totals[1] else 1
            columns[col].append(name)
            totals[col] += group_weight(name)
        # Column order follows `groups`' own order, not the balancing sort.
        for col in columns:
            col.sort(key=all_names.index)
        budget = 0.92  # leaves top/bottom margin inside the axes
        scale = min(1.0, budget / max(totals[0], totals[1], budget))
        render_column(a, columns[0], 0.0, scale)
        render_column(a, columns[1], col_width, scale)
        a.set_title("Summary (green/red = pass/fail against this run's own thresholds)")
    panels.append(summary_panel)

    data_panels = panels[:-1]
    n_data_rows = (len(data_panels) + 1) // 2
    fig = plt.figure(figsize=(15.5, 4 * n_data_rows + 5.5))
    gs = fig.add_gridspec(n_data_rows + 1, 2, height_ratios=[1] * n_data_rows + [1.6])
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
