"""
2-D XY path and velocity-overlap plots for the trailer-like leader-follower.

Run from the repo root after executing the C++ demo (produces
logs/follower_trajectory.csv):
    ./build/follower_demo [line|circle|zigzag] [t_end_seconds]
    python3 argus_follower/scripts/plot_follower.py

Headless export (no display required):
    python3 argus_follower/scripts/plot_follower.py --save-png docs/follower.png

What is shown:
  - Top plot:    leader (dashed) and follower (solid) XY paths overlaid,
                  equal aspect so geometry (e.g. circle radius, hitch
                  offset) isn't visually distorted.
  - Bottom plot: leader and follower velocity components (vx, vy) over
                  time, overlaid -- this is what actually shows the
                  follower lagging/leading the leader in speed, which the
                  XY path alone doesn't make obvious.
"""

import argparse

import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="logs/follower_trajectory.csv",
                    help="Path to the CSV produced by follower_demo (default: %(default)s)")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    return p.parse_args()


_args = _parse_args()

# Headless (Agg) backend for --save-png — must be set before importing
# pyplot. Default interactive use (no flag) is unaffected.
import matplotlib
if _args.save_png:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

df = pd.read_csv(_args.csv)

fig, (ax_xy, ax_vel) = plt.subplots(2, 1, figsize=(7, 10))

ax_xy.plot(df["leader_x"], df["leader_y"], "--", color="tab:red", label="leader (target)")
ax_xy.plot(df["follower_x"], df["follower_y"], "-", color="tab:blue", label="follower")
ax_xy.scatter(df["leader_x"].iloc[0], df["leader_y"].iloc[0], color="tab:red", marker="o", s=40)
ax_xy.scatter(df["follower_x"].iloc[0], df["follower_y"].iloc[0], color="tab:blue", marker="o",
              s=40)
ax_xy.set_xlabel("x [m]")
ax_xy.set_ylabel("y [m]")
ax_xy.set_title("Leader vs. follower path (XY)")
ax_xy.set_aspect("equal", adjustable="datalim")
ax_xy.legend()
ax_xy.grid(True, alpha=0.3)

ax_vel.plot(df["t"], df["leader_vx"], "--", color="tab:red", label="leader vx")
ax_vel.plot(df["t"], df["leader_vy"], "--", color="tab:orange", label="leader vy")
ax_vel.plot(df["t"], df["follower_vx"], "-", color="tab:blue", label="follower vx")
ax_vel.plot(df["t"], df["follower_vy"], "-", color="tab:cyan", label="follower vy")
ax_vel.set_xlabel("t [s]")
ax_vel.set_ylabel("velocity [m/s]")
ax_vel.set_title("Leader vs. follower velocity components")
ax_vel.legend()
ax_vel.grid(True, alpha=0.3)

fig.tight_layout()

if _args.save_png:
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
else:
    plt.show()
