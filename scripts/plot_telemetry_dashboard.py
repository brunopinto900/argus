"""
Single-figure, multi-panel telemetry dashboard for any trajectory CSV
sharing src/mpc_controller.cpp's log layout (argus_mpc's own
logs/trajectory.csv, or examples/trailer_mpc_demo.cpp's
logs/trailer_mpc_trajectory.csv) -- the full simulated drone state over
time, not just the XY path scripts/plot_trailer_mpc_xy.py and
scripts/animate_xy.py already cover.

Run from the repo root after executing either C++ demo:
    ./build/argus_mpc                     # or ./build/trailer_mpc_demo ...
    python3 scripts/plot_telemetry_dashboard.py --csv-path logs/trailer_mpc_trajectory.csv

Headless export (no display required):
    python3 scripts/plot_telemetry_dashboard.py --save-png docs/telemetry.png

Panels (one figure, 3x2 grid):
  - XY path:        drone (actual) vs. MPC reference vs. tracked target
  - Position:        x, y, z vs. time
  - Velocity:        vx, vy, vz vs. time
  - Attitude:        roll, pitch, yaw vs. time (deg)
  - Body rates:       p, q, r vs. time (deg/s)
  - XY tracking error: |drone - reference| vs. time
"""

import argparse

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-path", default="logs/trajectory.csv",
                    help="Trajectory CSV to plot (default: %(default)s)")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

df = pd.read_csv(_args.csv_path)

fig, ax = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f"Telemetry dashboard -- {_args.csv_path}")

# ── XY path ──────────────────────────────────────────────────────────────────
a = ax[0, 0]
a.plot(df["fx"], df["fy"], "--", color="tab:red", label="target")
a.plot(df["xt"], df["yt"], ":", color="tab:green", label="MPC reference")
a.plot(df["x"], df["y"], "-", color="tab:blue", label="drone (actual)")
a.set_xlabel("x [m]")
a.set_ylabel("y [m]")
a.set_title("XY path")
a.set_aspect("equal", adjustable="datalim")
a.legend()
a.grid(True, alpha=0.3)

# ── Position ─────────────────────────────────────────────────────────────────
a = ax[0, 1]
a.plot(df["t"], df["x"], label="x")
a.plot(df["t"], df["y"], label="y")
a.plot(df["t"], df["z"], label="z")
a.set_xlabel("t [s]")
a.set_ylabel("position [m]")
a.set_title("Position")
a.legend()
a.grid(True, alpha=0.3)

# ── Velocity ─────────────────────────────────────────────────────────────────
a = ax[1, 0]
a.plot(df["t"], df["vx"], label="vx")
a.plot(df["t"], df["vy"], label="vy")
a.plot(df["t"], df["vz"], label="vz")
a.set_xlabel("t [s]")
a.set_ylabel("velocity [m/s]")
a.set_title("Velocity")
a.legend()
a.grid(True, alpha=0.3)

# ── Attitude ─────────────────────────────────────────────────────────────────
a = ax[1, 1]
a.plot(df["t"], np.degrees(df["phi"]), label="roll (phi)")
a.plot(df["t"], np.degrees(df["theta"]), label="pitch (theta)")
a.plot(df["t"], np.degrees(df["psi"]), label="yaw (psi)")
a.set_xlabel("t [s]")
a.set_ylabel("angle [deg]")
a.set_title("Attitude")
a.legend()
a.grid(True, alpha=0.3)

# ── Body rates ───────────────────────────────────────────────────────────────
a = ax[2, 0]
a.plot(df["t"], np.degrees(df["p"]), label="p (roll rate)")
a.plot(df["t"], np.degrees(df["q"]), label="q (pitch rate)")
a.plot(df["t"], np.degrees(df["r"]), label="r (yaw rate)")
a.set_xlabel("t [s]")
a.set_ylabel("rate [deg/s]")
a.set_title("Body rates")
a.legend()
a.grid(True, alpha=0.3)

# ── XY tracking error ────────────────────────────────────────────────────────
a = ax[2, 1]
tracking_error = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])
a.plot(df["t"], tracking_error, color="tab:purple")
a.set_xlabel("t [s]")
a.set_ylabel("|drone - reference| [m]")
a.set_title("XY tracking error")
a.grid(True, alpha=0.3)

fig.tight_layout()

if _args.save_png:
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
else:
    plt.show()
