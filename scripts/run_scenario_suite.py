"""
Curated edge-case scenario suite: runs examples/online_mapping_demo.cpp
against a small, hand-designed set of scenarios (not a Monte Carlo sweep --
see design_decisions.md's "Curated edge-case scenario suite") and reports,
per scenario:
  - the hard pass/fail gates established for this project: collision
    (clearance_true < --real-target-m), solver failure (solve_status != 0),
    real-time budget (pipeline_onboard_time_ms > Ts), rate saturation
    (|p/q/r_cmd| pinned at rate_max) -- any one of these failing is a
    genuine "this run is broken," not a matter of degree.
  - a handful of soft metrics (tracking error, FOV exceedance, occlusion,
    blur-proxy rate magnitude) with no absolute threshold, just reported and
    compared against a stored baseline (logs/suite/baseline.json) with a
    provisional relative tolerance -- see design_decisions.md's "Threshold
    philosophy" for why these don't get hand-picked absolute numbers.

Each scenario's own full trajectory CSV/summary CSV/dashboard PNG are kept
under logs/suite/<name>_* for drill-down with scripts/compute_metrics.py,
same convention as a single scenario run.

Run (needs the acados venv + env vars; this script exports them itself):
    python3 scripts/run_scenario_suite.py
    python3 scripts/run_scenario_suite.py --save-baseline   # overwrite baseline
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PYTHON = os.path.expanduser("~/.venv/bin/python3")
ACADOS_DIR = os.path.expanduser("~/acados")
SUITE_DIR = os.path.join(REPO_ROOT, "logs", "suite")

_ENV = dict(os.environ)
_ENV["ACADOS_SOURCE_DIR"] = ACADOS_DIR
_ENV["LD_LIBRARY_PATH"] = os.path.join(ACADOS_DIR, "lib") + ":" + _ENV.get("LD_LIBRARY_PATH", "")

# (name, scenario yaml, brief rationale) -- Forest Clutter is forest.yaml
# itself, already used throughout design_decisions.md; the other five are
# new. Brief descriptions only -- the full rationale for each lives as a
# comment block at the top of its own config/*.yaml.
SCENARIOS = [
    ("forest_clutter", "forest.yaml",
     "Scattered small/medium obstacles + one narrow passage -- the general "
     "multi-obstacle baseline this whole project has been validated against."),
    ("wall_slowdown", "wall_slowdown.yaml",
     "One large obstacle dead ahead -- does avoidance react early enough to "
     "a big, slow-closing obstacle instead of forest.yaml's small/fast ones?"),
    ("sharp_corner", "sharp_corner.yaml",
     "Near-right-angle turn taken at full cruise speed -- aggressive "
     "cornering, not obstacle-forced deceleration."),
    ("stop_and_go", "stop_and_go.yaml",
     "No obstacles: accelerate, cruise, full stop, hold, accelerate again -- "
     "isolates the motion-profile transient from any avoidance cost."),
    ("pedestrian_walk", "pedestrian_walk.yaml",
     "Walk, full stop, lateral sidestep, small twist, walk again -- one "
     "continuous Skydio-style clip chaining behaviors, not one isolated event."),
    ("occlusion_cut", "occlusion_cut.yaml",
     "Leader sidesteps an obstacle at cruise speed -- follower lag should "
     "momentarily foul the bearing sightline even though the target itself "
     "is never behind the obstacle."),
    ("compound_stress", "compound_stress.yaml",
     "Narrow passage AND a sharp turn, both at full cruise speed -- combines "
     "geometry tightness, heading change, and speed instead of isolating one."),
    ("target_reversal", "target_reversal.yaml",
     "Leader reverses direction almost 180 degrees -- stresses the "
     "follower's heading-tracking under a near-full reversal, no obstacles."),
    ("standoff_conflict", "standoff_conflict.yaml",
     "Leader's dodge margin is tighter than d_safe_m -- forces sustained "
     "tension between the drone's standoff tracking and its own avoidance margin."),
    ("sensor_range_edge", "sensor_range_edge.yaml",
     "Obstacle starts beyond mapping_sensor.max_range_m, approached at full "
     "cruise with no early slowdown -- tests reacting fast once finally seen."),
    ("high_curvature", "high_curvature.yaml",
     "Sustained S-curve/slalom at cruise speed, no obstacles -- continuous "
     "heading change instead of sharp_corner.yaml's single turn event."),
    ("cold_start_acquisition", "cold_start_acquisition.yaml",
     "Drone starts with yaw offset so the target begins outside the FOV "
     "cone -- exercises initial_acquisition_latency_s for the first time."),
    ("serpentine_clutter", "serpentine_clutter.yaml",
     "Continuous weave through closely-spaced obstacles at cruise speed -- "
     "sustained chattering stress, not occasional dodges."),
    ("stop_near_obstacle", "stop_near_obstacle.yaml",
     "Leader stops and holds right beside an obstacle -- sustained "
     "near-contact dwell, not a brief closest-approach."),
    ("near_miss_cascade", "near_miss_cascade.yaml",
     "Several obstacles dodged back-to-back with little recovery distance "
     "between them, at cruise speed -- tests whether transients settle in time."),
]

REAL_TARGET_M = 0.27
D_FOV = 4.0
OCCLUSION_EVENT_PCT = 80.0

# Soft metrics tracked in the comparison table/baseline. "higher_is_worse"
# picks the regression direction -- these are all "closer to the problem
# this metric exists to catch" when they go up.
SOFT_METRICS = [
    "xy_tracking_error_mean_m",
    "xy_tracking_error_max_m",
    "fov_exceedance_pct",
    "occlusion_pct_max",
    "body_rate_magnitude_max_deg_s",
]

# Provisional regression tolerance: flag only if a soft metric got worse by
# more than this fraction relative to baseline (not an absolute threshold --
# see design_decisions.md's "Threshold philosophy"). Picked as a round
# number, not derived from any noise measurement yet.
REGRESSION_TOLERANCE_FRAC = 0.20


def contiguous_true_runs(mask):
    """[(start_idx, end_idx_inclusive), ...] for each run of consecutive True
    -- same as compute_metrics.py's helper, duplicated for the same reason
    as the rest of this module's standalone metric extraction."""
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


def _bearing_gamma_deg(df):
    """Same ZYX-rotation formula as compute_metrics.py's bearing_angles_deg(),
    reduced to just gamma (combined bearing angle)."""
    cp, sp = np.cos(df["phi"]), np.sin(df["phi"])
    ct, st = np.cos(df["theta"]), np.sin(df["theta"])
    cy, sy = np.cos(df["psi"]), np.sin(df["psi"])
    dx = df["fx"] - df["x"]
    dy = df["fy"] - df["y"]
    dz = df["fz"] - df["z"]
    db_x = cy * ct * dx + sy * ct * dy - st * dz
    rng = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.degrees(np.arccos(np.clip(db_x / rng, -1.0, 1.0)))


def compute_gate_and_stats(df, cfg):
    """Lean, standalone metric extraction -- just the hard-gate fields plus
    SOFT_METRICS, duplicated from compute_metrics.py's compute_summary()
    rather than importing it (that script is a single-run CLI tool coupled
    to its own argparse globals, same reasoning as scripts/
    sweep_softplus_beta.py's own standalone analyze())."""
    m = {}

    clearance = df["clearance_true"].values
    m["collision_free"] = bool(not (clearance < REAL_TARGET_M).any())
    m["min_clearance_true_m"] = float(np.min(clearance))

    m["solve_fail_rate_pct"] = float(100.0 * (df["solve_status"] != 0).mean())

    ts_ms = cfg["mpc"]["Ts"] * 1000.0
    pot = df["pipeline_onboard_time_ms"].values
    m["pipeline_real_time_safe"] = bool(not (pot > ts_ms).any())
    m["pipeline_onboard_time_mean_ms"] = float(np.mean(pot))
    m["pipeline_onboard_time_max_ms"] = float(np.max(pot))

    rate_max = cfg["constraints"]["rate_max"]
    eps = 0.02 * rate_max
    sat = ((df["p_cmd"].abs() > rate_max - eps) |
           (df["q_cmd"].abs() > rate_max - eps) |
           (df["r_cmd"].abs() > rate_max - eps))
    m["rate_saturation_pct"] = float(100.0 * sat.mean())

    xy_err = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])
    m["xy_tracking_error_mean_m"] = float(xy_err.mean())
    m["xy_tracking_error_max_m"] = float(xy_err.max())

    # Raw drone<->target (leader) distance -- hovers around the design
    # standoff distance. Mean +/- std, NOT max: every scenario's initial
    # transient (closing from a cold start) would otherwise dominate a max
    # and say nothing about steady-state behavior, same reasoning as the
    # "separate initial vs. remaining instants" split earlier in
    # design_decisions.md's softplus_beta investigation.
    drone_target_dist = np.hypot(df["x"] - df["fx"], df["y"] - df["fy"])
    m["drone_target_dist_mean_m"] = float(drone_target_dist.mean())
    m["drone_target_dist_std_m"] = float(drone_target_dist.std())

    # Target (leader) speed/acceleration -- from the logged fx,fy path
    # itself (the realized spline), not the scenario yaml's speed_profile
    # keyframes, so this reflects what the leader actually did.
    t = df["t"].values
    target_vx = np.gradient(df["fx"].values, t)
    target_vy = np.gradient(df["fy"].values, t)
    target_speed = np.hypot(target_vx, target_vy)
    m["target_speed_min_m_s"] = float(np.min(target_speed))
    m["target_speed_mean_m_s"] = float(np.mean(target_speed))
    m["target_speed_max_m_s"] = float(np.max(target_speed))
    target_accel = np.gradient(target_speed, t)
    m["target_accel_mean_abs_m_s2"] = float(np.mean(np.abs(target_accel)))

    fov_half_angle_deg = cfg["camera"]["fov_half_angle_deg"]
    gamma = _bearing_gamma_deg(df)
    out_of_fov = gamma > fov_half_angle_deg
    m["fov_exceedance_pct"] = float(100.0 * out_of_fov.mean())
    # Reacquisition count: same convention as compute_metrics.py's
    # compute_summary() -- the first out-of-FOV run (if it starts at index
    # 0) is "initial acquisition," not a "re"-acquisition.
    events = contiguous_true_runs(out_of_fov)
    reacq_events = events[1:] if (events and events[0][0] == 0) else events
    m["reacquisition_events"] = len(reacq_events)

    occ = df["occlusion_clearance"].values
    occ_pct = 100.0 * np.clip(1.0 - occ / D_FOV, 0.0, 1.0)
    m["occlusion_pct_max"] = float(np.max(occ_pct))
    m["occluded_pct"] = float(100.0 * (occ_pct > OCCLUSION_EVENT_PCT).mean())

    rate_mag_deg_s = np.degrees(np.sqrt(df["p"]**2 + df["q"]**2 + df["r"]**2))
    m["body_rate_magnitude_mean_deg_s"] = float(np.mean(rate_mag_deg_s))
    m["body_rate_magnitude_std_deg_s"] = float(np.std(rate_mag_deg_s))
    m["body_rate_magnitude_max_deg_s"] = float(np.max(rate_mag_deg_s))

    m["gates_pass"] = bool(m["collision_free"] and m["solve_fail_rate_pct"] == 0.0
                            and m["pipeline_real_time_safe"] and m["rate_saturation_pct"] == 0.0)
    return m


def run_scenario(name, scenario_file):
    subprocess.run(["./build/online_mapping_demo", f"config/{scenario_file}"],
                    cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    df = pd.read_csv(os.path.join(REPO_ROOT, "logs", "online_mapping_trajectory.csv"))

    os.makedirs(SUITE_DIR, exist_ok=True)
    traj_dst = os.path.join(SUITE_DIR, f"{name}_trajectory.csv")
    shutil.copy(os.path.join(REPO_ROOT, "logs", "online_mapping_trajectory.csv"), traj_dst)
    summary_src = os.path.join(REPO_ROOT, "logs", "online_mapping_summary.csv")
    if os.path.exists(summary_src):
        shutil.copy(summary_src, os.path.join(SUITE_DIR, f"{name}_summary.csv"))

    subprocess.run(
        [VENV_PYTHON, "scripts/compute_metrics.py",
         "--csv-path", traj_dst,
         "--scenario-config", f"config/{scenario_file}",
         "--summary-path", os.path.join(SUITE_DIR, f"{name}_summary.csv"),
         "--save-png", os.path.join(SUITE_DIR, f"{name}_dashboard.png")],
        cwd=REPO_ROOT, env=_ENV, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return df


def load_baseline():
    path = os.path.join(SUITE_DIR, "baseline.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_baseline(results):
    os.makedirs(SUITE_DIR, exist_ok=True)
    with open(os.path.join(SUITE_DIR, "baseline.json"), "w") as f:
        json.dump(results, f, indent=2)


def regressions(name, metrics, baseline):
    if baseline is None or name not in baseline:
        return []
    flagged = []
    base = baseline[name]
    for key in SOFT_METRICS:
        if key not in base or base[key] is None:
            continue
        old, new = base[key], metrics[key]
        tol = max(abs(old) * REGRESSION_TOLERANCE_FRAC, 1e-6)
        if new > old + tol:
            flagged.append(f"{key}: {old:.3f} -> {new:.3f}")
    return flagged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-baseline", action="store_true",
                    help="Overwrite logs/suite/baseline.json with this run's results "
                         "instead of just comparing against it")
    args = p.parse_args()

    with open(os.path.join(REPO_ROOT, "config", "quadrotor.yaml")) as f:
        cfg = yaml.safe_load(f)

    baseline = None if args.save_baseline else load_baseline()
    results = {}
    for name, scenario_file, _ in SCENARIOS:
        print(f"--- {name} ({scenario_file}) ---")
        df = run_scenario(name, scenario_file)
        results[name] = compute_gate_and_stats(df, cfg)

    print("\n" + "=" * 100)
    header = f"{'scenario':<18}{'gates':<8}{'xy_err mean/max':<20}{'fov_exc%':<10}{'occ_max%':<10}{'rate_mag_max':<14}"
    print(header)
    print("-" * len(header))
    any_regressions = False
    for name, _, _ in SCENARIOS:
        m = results[name]
        gate_str = "PASS" if m["gates_pass"] else "FAIL"
        print(f"{name:<18}{gate_str:<8}"
              f"{m['xy_tracking_error_mean_m']:.3f}/{m['xy_tracking_error_max_m']:.3f}".ljust(18) + "  "
              f"{m['fov_exceedance_pct']:<10.1f}{m['occlusion_pct_max']:<10.1f}"
              f"{m['body_rate_magnitude_max_deg_s']:<14.1f}")
        flags = regressions(name, m, baseline)
        if flags:
            any_regressions = True
            print(f"  REGRESSION vs baseline: {'; '.join(flags)}")

    if baseline is None:
        save_baseline(results)
        print(f"\nNo baseline found -- saved this run as logs/suite/baseline.json")
    elif args.save_baseline:
        save_baseline(results)
        print(f"\nOverwrote logs/suite/baseline.json with this run")
    elif not any_regressions:
        print("\nNo regressions vs baseline.")


if __name__ == "__main__":
    main()
