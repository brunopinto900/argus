"""
Real-time animation: a static obstacle and its ESDF (background heatmap,
red = close/occupied -> blue = far, same hue sweep as
examples/esdf_2d_demo.cpp and argus_mapping_node's RViz marker coloring),
with the target's path (straightLineWithBypassTarget) and the (currently
obstacle-unaware) TrailerFollower reference tracking it.

Run from the repo root after executing the C++ demo:
    ./build/follower_obstacle_demo
    python3 argus_follower/scripts/plot_follower_obstacle.py

Headless export (no display required):
    python3 argus_follower/scripts/plot_follower_obstacle.py --save-png docs/obstacle.png
    python3 argus_follower/scripts/plot_follower_obstacle.py --save-gif docs/obstacle.gif \
        --stride 2 --fps 20

This is a before/after baseline, not a demonstration of working obstacle
avoidance -- TrailerFollower has no obstacle awareness at all yet. What
this shows is whether/when its own corner-cutting behaviour alone would
carry the follower's reference through the obstacle.
"""
import argparse

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esdf-csv", default="logs/follower_obstacle_esdf.csv")
    p.add_argument("--traj-csv", default="logs/follower_obstacle_trajectory.csv")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render one frame headlessly to PATH instead of showing a window")
    p.add_argument("--save-gif", metavar="PATH",
                    help="Render an animated GIF headlessly to PATH instead of showing a window")
    p.add_argument("--frame", type=int, default=None,
                    help="Frame index for --save-png (default: midpoint of the run)")
    p.add_argument("--start-frame", type=int, default=0, help="First frame for --save-gif")
    p.add_argument("--num-frames", type=int, default=None,
                    help="Number of frames for --save-gif, before --stride (default: to the end)")
    p.add_argument("--stride", type=int, default=2,
                    help="Take every Nth frame for --save-gif (reduces file size)")
    p.add_argument("--fps", type=int, default=20, help="Playback fps for --save-gif")
    p.add_argument("--trail-seconds", type=float, default=6.0,
                    help="How much trailing path history to draw behind each marker")
    p.add_argument("--max-distance", type=float, default=2.0,
                    help="Distance (m) at which the ESDF colormap reaches pure blue")
    return p.parse_args()


_args = _parse_args()

# Headless (Agg) backend for --save-png/--save-gif — must be set before
# importing pyplot. Default interactive use (no flags) is unaffected.
import matplotlib
if _args.save_png or _args.save_gif:
    matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def load_esdf(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    nx, ny = int(header[0]), int(header[1])
    voxel_size, ox, oy = float(header[2]), float(header[3]), float(header[4])
    grid = np.loadtxt(path, delimiter=",", skiprows=1)
    assert grid.shape == (ny, nx), f"expected ({ny},{nx}), got {grid.shape}"
    extent = [ox, ox + nx * voxel_size, oy, oy + ny * voxel_size]
    return grid, extent


def esdf_to_rgb(grid, max_distance):
    # Same red(close/occupied)->blue(far) hue sweep as distanceToColor() in
    # examples/esdf_2d_demo.cpp / ros2/argus_mapping's RViz marker coloring
    # (0 deg hue = red, 240 deg = blue), just vectorized via matplotlib's
    # own HSV->RGB instead of the per-pixel branch-based version there.
    t = np.clip(grid / max_distance, 0.0, 1.0)
    hue = t * (240.0 / 360.0)
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(hsv)


esdf, extent = load_esdf(_args.esdf_csv)
rgb = esdf_to_rgb(esdf, _args.max_distance)
traj = pd.read_csv(_args.traj_csv)
n_frames = len(traj)
dt = traj["t"].iloc[1] - traj["t"].iloc[0]
trail_steps = max(1, int(_args.trail_seconds / dt))

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")

(leader_trail,) = ax.plot([], [], "-", color="red", linewidth=2, label="leader (target)")
(follower_trail,) = ax.plot([], [], "-", color="deepskyblue", linewidth=2,
                             label="follower (reference)")
(leader_dot,) = ax.plot([], [], "o", color="red", markersize=8)
(follower_dot,) = ax.plot([], [], "o", color="deepskyblue", markersize=8)
ax.legend(loc="upper right")


def update(frame):
    i0 = max(0, frame - trail_steps)
    leader_trail.set_data(traj["leader_x"].iloc[i0:frame + 1], traj["leader_y"].iloc[i0:frame + 1])
    follower_trail.set_data(traj["follower_x"].iloc[i0:frame + 1],
                             traj["follower_y"].iloc[i0:frame + 1])
    leader_dot.set_data([traj["leader_x"].iloc[frame]], [traj["leader_y"].iloc[frame]])
    follower_dot.set_data([traj["follower_x"].iloc[frame]], [traj["follower_y"].iloc[frame]])
    ax.set_title(f"Target bypassing an obstacle vs. follower reference  (t = {traj['t'].iloc[frame]:.1f}s)")
    return leader_trail, follower_trail, leader_dot, follower_dot


if _args.save_png:
    frame = _args.frame if _args.frame is not None else n_frames // 2
    update(frame)
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
elif _args.save_gif:
    start = _args.start_frame
    end = start + _args.num_frames if _args.num_frames else n_frames
    frames = range(start, min(end, n_frames), _args.stride)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(_args.save_gif, writer="pillow", fps=_args.fps)
    print(f"Saved {_args.save_gif}")
else:
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=dt * 1000, blit=True)
    plt.show()
