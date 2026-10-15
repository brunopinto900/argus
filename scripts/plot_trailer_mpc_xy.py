"""
2-D XY path and MPC tracking-error plots for examples/trailer_mpc_demo.cpp
-- the quick, top-down counterpart to scripts/animate_xy.py's 3-D view
(camera frustum, FOV angle errors) for when only the XY tracking picture
is needed.

Run from the repo root after executing the C++ demo:
    ./build/trailer_mpc_demo [line|circle|zigzag] [t_end_seconds]
    python3 scripts/plot_trailer_mpc_xy.py

Headless export (no display required):
    python3 scripts/plot_trailer_mpc_xy.py --save-png docs/trailer_mpc_xy.png

What is shown:
  - Top plot:    leader (dashed), the follower's goal-point reference fed
                  to the MPC (dotted), and the drone's actual simulated
                  position (solid) overlaid -- the gap between dotted and
                  solid is what the MPC solver is doing; the gap between
                  dashed and dotted is what TrailerFollower is doing.
  - Bottom plot: |drone - reference| tracking error over time -- the same
                  quantity trailer_mpc_demo prints periodically to the
                  console, plotted continuously instead of sampled.
"""

import argparse

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="logs/trailer_mpc_trajectory.csv",
                    help="Path to the CSV produced by trailer_mpc_demo (default: %(default)s)")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

df = pd.read_csv(_args.csv)
tracking_error = np.hypot(df["x"] - df["xt"], df["y"] - df["yt"])

fig, (ax_xy, ax_err) = plt.subplots(2, 1, figsize=(7, 10))

ax_xy.plot(df["fx"], df["fy"], "--", color="tab:red", label="leader (target)")
ax_xy.plot(df["xt"], df["yt"], ":", color="tab:green", label="follower reference (to MPC)")
ax_xy.plot(df["x"], df["y"], "-", color="tab:blue", label="drone (actual)")
ax_xy.scatter(df["fx"].iloc[0], df["fy"].iloc[0], color="tab:red", marker="o", s=40)
ax_xy.scatter(df["x"].iloc[0], df["y"].iloc[0], color="tab:blue", marker="o", s=40)
ax_xy.set_xlabel("x [m]")
ax_xy.set_ylabel("y [m]")
ax_xy.set_title("Leader vs. follower reference vs. actual drone path (XY)")
ax_xy.set_aspect("equal", adjustable="datalim")
ax_xy.legend()
ax_xy.grid(True, alpha=0.3)

ax_err.plot(df["t"], tracking_error, "-", color="tab:purple")
ax_err.set_xlabel("t [s]")
ax_err.set_ylabel("|drone - reference| [m]")
ax_err.set_title("MPC XY tracking error")
ax_err.grid(True, alpha=0.3)

fig.tight_layout()

if _args.save_png:
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
else:
    plt.show()
