"""
Single-figure, multi-panel telemetry dashboard for any trajectory CSV
sharing the canonical step,t,x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,T,p_cmd,
q_cmd,r_cmd,xt,yt,zt,fx,fy,fz,solve_status layout -- src/mpc_controller.cpp's
logs/trajectory.csv, examples/trailer_mpc_demo.cpp's
logs/trailer_mpc_trajectory.csv, and examples/online_mapping_demo.cpp's
logs/online_mapping_trajectory.csv all write this same layout (the last one
adds a trailing dist_obstacle column -- see design_decisions.md) specifically
so this one script works on any of them. Not just the XY path
scripts/plot_trailer_mpc_xy.py and scripts/animate_xy.py already cover.

Run from the repo root after executing any of the three C++ demos:
    ./build/argus_mpc                      # or trailer_mpc_demo / online_mapping_demo
    python3 scripts/plot_telemetry_dashboard.py --csv-path logs/trailer_mpc_trajectory.csv

Headless export (no display required):
    python3 scripts/plot_telemetry_dashboard.py --save-png docs/telemetry.png

Panels (one figure, always 2 columns):
  - XY path:             drone (actual) vs. position reference vs. FOV target
  - Yaw bearing error:     target's azimuth offset from dead-center of the
                            camera boresight (0 = target centered)
  - Pitch bearing error:   target's elevation offset from dead-center of the
                            camera boresight (0 = target centered) -- both
                            computed via the full ZYX rotation (not psi/theta
                            against a flat/approximate reference), same
                            convention as scripts/animate_xy.py's
                            bearing_errors_deg(); the FOV cone's
                            fov_half_angle_deg is drawn as a reference band
                            when --config-path's camera section has it
  - Velocity:              vx, vy, vz vs. time
  - Attitude:              roll, pitch, yaw vs. time (deg)
  - Body rates:            p, q, r vs. time (deg/s)
  - Height tracking:       MPC height (z, actual) vs. reference height (zt)
  - XY tracking error:      |drone - reference| vs. time
  - Distance to FOV target:  |drone - target| vs. time, always shown
                            (--keepaway-m draws a constant-distance
                            reference line, e.g. trailer.distance_to_target)
  - Distance to obstacle:    only if the CSV has a dist_obstacle column
                            (online_mapping_demo only) -- d_safe knob,
                            --real-target-m (default 0.27, the physically
                            derived margin), closest-approach marker
"""

import argparse

import numpy as np
import pandas as pd
import yaml


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", default="logs/trajectory.csv",
                    help="Trajectory CSV to plot (default: %(default)s)")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    p.add_argument("--keepaway-m", type=float, default=None,
                    help="Constant intended distance from the FOV target (e.g. "
                         "trailer.distance_to_target, or an orbit radius) -- "
                         "drawn as a reference line on the distance-to-target "
                         "panel. Not read from config automatically since its "
                         "meaning is scenario-specific; omit to skip the line.")
    p.add_argument("--config-path", default="config/quadrotor.yaml",
                    help="Source of obstacle_avoidance.d_safe_m, used only when "
                         "the CSV has a dist_obstacle column")
    p.add_argument("--real-target-m", type=float, default=0.27,
                    help="Actual physical safety requirement, distinct from "
                         "--config-path's d_safe_m knob -- see "
                         "design_decisions.md's 'd_safe inflation' section "
                         "(default: 0.27, the drone-size + voxel + lidar-"
                         "noise derived margin from that section). Only "
                         "drawn when the CSV has a dist_obstacle column.")
    p.add_argument("--scenario-config", default="config/forest.yaml",
                    help="Scenario file (online_mapping_demo's obstacles: "
                         "section -- same file passed as that demo's CLI arg) "
                         "to draw on the XY path panel -- only used when the "
                         "CSV has a dist_obstacle column. Missing file is not "
                         "an error, the outline is just skipped.")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

df = pd.read_csv(_args.csv_path)
has_obstacle = "dist_obstacle" in df.columns


def _xy_path(a):
    a.plot(df["fx"], df["fy"], "--", color="tab:red", label="FOV target")
    a.plot(df["xt"], df["yt"], ":", color="tab:green", label="position reference")
    a.plot(df["x"], df["y"], "-", color="tab:blue", label="drone (actual)")
    if has_obstacle:
        try:
            obstacles = yaml.safe_load(open(_args.scenario_config))["obstacles"]
        except (FileNotFoundError, KeyError, TypeError):
            obstacles = []
        for i, o in enumerate(obstacles):
            cx, cy = o["center"]
            label = "obstacle" if i == 0 else None
            if o["shape"] == "circle":
                a.add_patch(plt.Circle((cx, cy), o["radius"], facecolor="black",
                                        alpha=0.3, edgecolor="black", label=label))
            else:
                h = o["half_size"]
                a.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                           facecolor="black", alpha=0.3,
                                           edgecolor="black", label=label))
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("XY path")
    a.set_aspect("equal", adjustable="datalim")
    a.legend()


def _target_bearing_errors():
    """Signed yaw/pitch offset of the FOV target from dead-center of the
    camera boresight (body +x axis), via the full ZYX rotation -- same
    convention as scripts/animate_xy.py's bearing_errors_deg(), just
    vectorized over the whole trajectory instead of one frame at a time.

    Deliberately NOT "theta/psi vs. a flat/approximate reference": the MPC
    has no decomposed per-axis FOV reference (r_fov only penalizes the
    combined boresight-cone residual, see generate_mpc.py), so the only
    reference that actually means "target is centered in frame" is the
    target's own bearing projected into the body frame -- 0 in both axes
    means the target sits exactly on the boresight, regardless of what phi/
    theta/psi individually are doing to get it there.
    """
    cp, sp = np.cos(df["phi"]), np.sin(df["phi"])
    ct, st = np.cos(df["theta"]), np.sin(df["theta"])
    cy, sy = np.cos(df["psi"]), np.sin(df["psi"])
    dx = df["fx"] - df["x"]
    dy = df["fy"] - df["y"]
    dz = df["fz"] - df["z"]
    # d_body = R_zyx(phi,theta,psi)^T @ [dx,dy,dz]
    db_x = cy * ct * dx + sy * ct * dy - st * dz
    db_y = (cy * st * sp - sy * cp) * dx + (sy * st * sp + cy * cp) * dy + ct * sp * dz
    db_z = (cy * st * cp + sy * sp) * dx + (sy * st * cp - cy * sp) * dy + ct * cp * dz
    yaw_err = np.degrees(np.arctan2(db_y, db_x))
    pitch_err = np.degrees(np.arctan2(-db_z, db_x))
    return yaw_err, pitch_err


def _fov_half_angle_deg():
    try:
        return yaml.safe_load(open(_args.config_path))["camera"]["fov_half_angle_deg"]
    except (FileNotFoundError, KeyError):
        return None


def _draw_fov_band(a):
    half_angle = _fov_half_angle_deg()
    if half_angle is not None:
        a.axhline(half_angle, color="tab:red", linestyle=":", linewidth=1,
                   label=f"FOV edge (±{half_angle:.0f}°)")
        a.axhline(-half_angle, color="tab:red", linestyle=":", linewidth=1)
        a.legend()


def _yaw_error(a):
    yaw_err, _ = _target_bearing_errors()
    a.plot(df["t"], yaw_err, color="tab:orange")
    _draw_fov_band(a)
    a.set_xlabel("t [s]")
    a.set_ylabel("yaw bearing error [deg]")
    a.set_title("Yaw bearing error (target vs. boresight)")


def _pitch_error(a):
    _, pitch_err = _target_bearing_errors()
    a.plot(df["t"], pitch_err, color="tab:orange")
    _draw_fov_band(a)
    a.set_xlabel("t [s]")
    a.set_ylabel("pitch bearing error [deg]")
    a.set_title("Pitch bearing error (target vs. boresight)")


def _velocity(a):
    a.plot(df["t"], df["vx"], label="vx")
    a.plot(df["t"], df["vy"], label="vy")
    a.plot(df["t"], df["vz"], label="vz")
    a.set_xlabel("t [s]")
    a.set_ylabel("velocity [m/s]")
    a.set_title("Velocity")
    a.legend()


def _attitude(a):
    a.plot(df["t"], np.degrees(df["phi"]), label="roll (phi)")
    a.plot(df["t"], np.degrees(df["theta"]), label="pitch (theta)")
    a.plot(df["t"], np.degrees(df["psi"]), label="yaw (psi)")
    a.set_xlabel("t [s]")
    a.set_ylabel("angle [deg]")
    a.set_title("Attitude")
    a.legend()


def _body_rates(a):
    a.plot(df["t"], np.degrees(df["p"]), label="p (roll rate)")
    a.plot(df["t"], np.degrees(df["q"]), label="q (pitch rate)")
    a.plot(df["t"], np.degrees(df["r"]), label="r (yaw rate)")
    a.set_xlabel("t [s]")
    a.set_ylabel("rate [deg/s]")
    a.set_title("Body rates")
    a.legend()


def _height(a):
    a.plot(df["t"], df["zt"], "--", color="tab:green", label="reference height (zt)")
    a.plot(df["t"], df["z"], color="tab:blue", label="MPC height (actual)")
    a.set_xlabel("t [s]")
    a.set_ylabel("z [m]")
    a.set_title("Height tracking")
    a.legend()


def _tracking_error(a):
    tracking_error = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])
    a.plot(df["t"], tracking_error, color="tab:purple")
    a.set_xlabel("t [s]")
    a.set_ylabel("|drone - reference| [m]")
    a.set_title("XY tracking error")


def _target_distance(a):
    dist = np.hypot(df["x"] - df["fx"], df["y"] - df["fy"])
    a.plot(df["t"], dist, color="tab:green", label="|drone - FOV target|")
    if _args.keepaway_m is not None:
        a.axhline(_args.keepaway_m, color="tab:red", linestyle="--",
                   label=f"keepaway = {_args.keepaway_m:.2f} m")
        a.legend()
    a.set_xlabel("t [s]")
    a.set_ylabel("distance [m]")
    a.set_title("Distance to FOV target")


def _obstacle_distance(a):
    a.plot(df["t"], df["dist_obstacle"], color="tab:blue", label="dist to obstacle (own_map)")
    if _args.real_target_m is not None:
        a.axhline(_args.real_target_m, color="tab:green", linestyle="--",
                   label=f"real target = {_args.real_target_m:.3f} m")
    try:
        d_safe = yaml.safe_load(open(_args.config_path))["obstacle_avoidance"]["d_safe_m"]
        a.axhline(d_safe, color="tab:red", linestyle=":",
                   label=f"d_safe knob (config) = {d_safe:.3f} m")
    except (FileNotFoundError, KeyError):
        pass
    closest = df["dist_obstacle"].min()
    closest_t = df.loc[df["dist_obstacle"].idxmin(), "t"]
    a.scatter([closest_t], [closest], color="black", zorder=5,
              label=f"closest approach = {closest:.3f} m")
    a.set_xlabel("t [s]")
    a.set_ylabel("distance [m]")
    a.set_title("Distance to obstacle")
    a.legend()


panels = [_xy_path, _yaw_error, _pitch_error, _velocity, _attitude, _body_rates,
          _height, _tracking_error, _target_distance]
if has_obstacle:
    panels.append(_obstacle_distance)

n_rows = (len(panels) + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
fig.suptitle(f"Telemetry dashboard -- {_args.csv_path}")
axes_flat = axes.flatten()
for a, panel_fn in zip(axes_flat, panels):
    panel_fn(a)
    a.grid(True, alpha=0.3)
for a in axes_flat[len(panels):]:
    a.axis("off")

fig.tight_layout()

if _args.save_png:
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
else:
    plt.show()
