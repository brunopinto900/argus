#!/usr/bin/env python3
"""Standalone test of the leader motion-generation module (waypoints +
speed_profile -> argus_follower::waypointTarget's realized trajectory),
independent of the MPC/follower/sensor pipeline.

Reimplements TargetMotionSimulator.cpp / CubicSpline.cpp exactly:
  - speed_profile keyframes: piecewise-linear interpolation (interpolateSpeed)
  - arc-length s(t): trapezoidal integration of speed(t), sampled every
    sample_dt = Ts = 0.05s (same as control_dt in online_mapping_demo.cpp)
  - s -> (x,y): linear walk along the waypoint polyline (pointAtArcLength)
  - (t_i, x_i) / (t_i, y_i) -> natural cubic spline (bc_type='natural'),
    matching CubicSpline.hpp's explicit note that this reproduces its
    numbers exactly.
  - velocity = spline.derivative(t), acceleration = spline 2nd derivative.

This lets every scenario's motion profile be vetted for plausible
kinematics (speed, tangential accel, centripetal accel / turn radius)
before it ever touches the MPC -- catches both bug classes already found
the hard way: path-length/speed-profile integral mismatches, and the
spline cutting corners tighter than the waypoints suggest.
"""
import argparse
import pathlib
import sys

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
OUT_DIR = REPO_ROOT / "logs" / "motion_profile"

SAMPLE_DT = 0.05  # Ts, matches control_dt used in online_mapping_demo.cpp


def load_leader_config(yaml_path: pathlib.Path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    leader = cfg["leader"]
    waypoints = [tuple(p) for p in leader["waypoints"]]
    speed_profile = [tuple(p) for p in leader["speed_profile"]]
    t_end = float(leader["t_end"])
    obstacles = cfg.get("obstacles", [])
    return waypoints, speed_profile, t_end, obstacles


def path_length(waypoints):
    wp = np.array(waypoints, dtype=float)
    return float(np.sum(np.hypot(*np.diff(wp, axis=0).T)))


def speed_profile_integral(speed_profile):
    t = np.array([p[0] for p in speed_profile], dtype=float)
    v = np.array([p[1] for p in speed_profile], dtype=float)
    return float(np.trapz(v, t))


def point_at_arclength(wp: np.ndarray, s: float):
    if len(wp) == 1 or s <= 0.0:
        return wp[0]
    remaining = s
    for i in range(len(wp) - 1):
        seg = wp[i + 1] - wp[i]
        seg_len = float(np.hypot(*seg))
        last_segment = (i + 2 == len(wp))
        if remaining <= seg_len or last_segment:
            frac = min(1.0, remaining / seg_len) if seg_len > 0.0 else 0.0
            return wp[i] + frac * seg
        remaining -= seg_len
    return wp[-1]


def waypoint_target(waypoints, speed_profile, t_end, sample_dt=SAMPLE_DT):
    """Exact port of argus_follower::waypointTarget. Returns (t, csx, csy)."""
    n = round(t_end / sample_dt) + 1
    t = np.arange(n) * sample_dt

    kf_t = np.array([p[0] for p in speed_profile], dtype=float)
    kf_v = np.array([p[1] for p in speed_profile], dtype=float)
    v_at = lambda tt: np.interp(tt, kf_t, kf_v)  # clamps outside range

    wp = np.array(waypoints, dtype=float)

    s = 0.0
    xs = np.empty(n)
    ys = np.empty(n)
    for i in range(n):
        if i > 0:
            dt = t[i] - t[i - 1]
            s += 0.5 * (v_at(t[i - 1]) + v_at(t[i])) * dt
        x, y = point_at_arclength(wp, s)
        xs[i] = x
        ys[i] = y

    csx = CubicSpline(t, xs, bc_type="natural")
    csy = CubicSpline(t, ys, bc_type="natural")
    return t, csx, csy


def analyze(name, yaml_path):
    waypoints, speed_profile, t_end, obstacles = load_leader_config(yaml_path)
    t, csx, csy = waypoint_target(waypoints, speed_profile, t_end)

    # Evaluate on a finer grid than the spline knots to see the true
    # continuous shape between them (the spline itself is smooth, but a
    # knot-only sampling could still miss intra-segment extrema).
    t_fine = np.linspace(0.0, t_end, max(4000, 20 * len(t)))
    x, y = csx(t_fine), csy(t_fine)
    vx, vy = csx(t_fine, 1), csy(t_fine, 1)
    ax, ay = csx(t_fine, 2), csy(t_fine, 2)

    speed = np.hypot(vx, vy)
    accel_mag = np.hypot(ax, ay)
    speed_safe = np.where(speed > 1e-6, speed, 1e-6)
    tangential = (vx * ax + vy * ay) / speed_safe
    # normal (centripetal) component magnitude
    normal = np.sqrt(np.clip(accel_mag**2 - tangential**2, 0.0, None))
    curvature = (vx * ay - vy * ax) / speed_safe**3
    radius = np.where(np.abs(curvature) > 1e-6, 1.0 / np.abs(curvature), np.inf)

    L = path_length(waypoints)
    S = speed_profile_integral(speed_profile)

    i_peak = int(np.argmax(accel_mag))
    i_peak_n = int(np.argmax(normal))
    i_peak_tan = int(np.argmax(np.abs(tangential)))

    report = {
        "name": name,
        "path_length_m": L,
        "speed_profile_integral_m": S,
        "length_mismatch_m": S - L,
        "speed_min": float(speed.min()),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "accel_mag_max": float(accel_mag[i_peak]),
        "accel_mag_max_t": float(t_fine[i_peak]),
        "accel_mag_max_xy": (float(x[i_peak]), float(y[i_peak])),
        "tangential_at_peak": float(tangential[i_peak]),
        "normal_at_peak": float(normal[i_peak]),
        "normal_max": float(normal[i_peak_n]),
        "normal_max_t": float(t_fine[i_peak_n]),
        "radius_at_normal_max_m": float(radius[i_peak_n]),
        "tangential_max_abs": float(np.abs(tangential[i_peak_tan])),
        "tangential_max_abs_t": float(t_fine[i_peak_tan]),
    }
    data = dict(t=t_fine, x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay,
                speed=speed, accel_mag=accel_mag, tangential=tangential,
                normal=normal, radius=radius, waypoints=np.array(waypoints),
                obstacles=obstacles)
    return report, data


def print_report(r):
    print(f"--- {r['name']} ---")
    print(f"  path length (waypoints):     {r['path_length_m']:.3f} m")
    print(f"  speed_profile integral:      {r['speed_profile_integral_m']:.3f} m"
          f"   (mismatch {r['length_mismatch_m']:+.3f} m)")
    print(f"  speed   min/mean/max:        {r['speed_min']:.2f} / "
          f"{r['speed_mean']:.2f} / {r['speed_max']:.2f} m/s")
    print(f"  |accel| max:                 {r['accel_mag_max']:.2f} m/s^2 "
          f"@ t={r['accel_mag_max_t']:.2f}s  xy={r['accel_mag_max_xy']}")
    print(f"    -> tangential/normal there: {r['tangential_at_peak']:+.2f} / "
          f"{r['normal_at_peak']:.2f} m/s^2")
    print(f"  normal (centripetal) max:    {r['normal_max']:.2f} m/s^2 "
          f"@ t={r['normal_max_t']:.2f}s  (implied radius {r['radius_at_normal_max_m']:.2f} m)")
    print(f"  tangential max |.|:          {r['tangential_max_abs']:.2f} m/s^2 "
          f"@ t={r['tangential_max_abs_t']:.2f}s")
    print()


def plot_scenario(name, data, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax0 = axes[0]
    sc = ax0.scatter(data["x"], data["y"], c=data["accel_mag"], cmap="inferno",
                      s=4)
    ax0.plot(data["waypoints"][:, 0], data["waypoints"][:, 1], "o--",
              color="tab:blue", alpha=0.4, ms=4, lw=1, label="waypoints")
    for obs in data["obstacles"]:
        c = obs["center"]
        if obs["shape"] == "circle":
            patch = plt.Circle(c, obs["radius"], fill=False, color="tab:green")
        else:
            hs = obs["half_size"]
            patch = plt.Rectangle((c[0] - hs, c[1] - hs), 2 * hs, 2 * hs,
                                   fill=False, color="tab:green")
        ax0.add_patch(patch)
    ax0.set_aspect("equal")
    ax0.set_title(f"{name}: realized path, colored by |accel|")
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]")
    fig.colorbar(sc, ax=ax0, label="|accel| [m/s^2]")
    ax0.legend(loc="best", fontsize=8)

    ax1 = axes[1]
    ax1.plot(data["t"], data["speed"], color="tab:blue")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("speed [m/s]", color="tab:blue")
    ax1.set_title("speed(t)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = axes[2]
    ax2.plot(data["t"], data["tangential"], color="tab:orange",
              label="tangential (speed change)")
    ax2.plot(data["t"], data["normal"], color="tab:red",
              label="normal (centripetal/turning)")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("accel [m/s^2]")
    ax2.set_title("tangential vs normal accel")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*",
                         help="scenario yaml filenames under config/ "
                              "(default: all *.yaml in config/)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.scenarios:
        yaml_files = [CONFIG_DIR / s for s in args.scenarios]
    else:
        yaml_files = sorted(CONFIG_DIR.glob("*.yaml"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for yp in yaml_files:
        name = yp.stem
        report, data = analyze(name, yp)
        print_report(report)
        reports.append(report)
        if not args.no_plots:
            out_path = plot_scenario(name, data, OUT_DIR)
            print(f"  plot saved: {out_path}")
            print()

    print("=" * 100)
    header = (f"{'scenario':<22} {'len_mismatch_m':>15} {'speed_max':>10} "
              f"{'accel_max':>10} {'tangential@peak':>16} {'normal@peak':>12} "
              f"{'min_radius_m':>13}")
    print(header)
    print("-" * len(header))
    for r in reports:
        print(f"{r['name']:<22} {r['length_mismatch_m']:>+15.3f} "
              f"{r['speed_max']:>10.2f} {r['accel_mag_max']:>10.2f} "
              f"{r['tangential_at_peak']:>+16.2f} {r['normal_at_peak']:>12.2f} "
              f"{r['radius_at_normal_max_m']:>13.2f}")


if __name__ == "__main__":
    main()
