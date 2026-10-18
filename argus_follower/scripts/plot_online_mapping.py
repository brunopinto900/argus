"""
Animates the online/incremental mapping demo: own_map's distance snapshot
sequence (red->blue hue sweep by distance, same convention as
examples/esdf_2d_demo.cpp), with the leader/target, the follower's
open-loop reference, and the MPC-controlled drone's actual closed-loop
path all overlaid -- the mapping sensor rides the MPC path, not the
reference directly (see online_mapping_demo.cpp).

Run from the repo root after executing the C++ demo:
    ./build/online_mapping_demo
    python3 argus_follower/scripts/plot_online_mapping.py

Headless export (no display required):
    python3 argus_follower/scripts/plot_online_mapping.py --save-png docs/online_mapping.png
    python3 argus_follower/scripts/plot_online_mapping.py --save-gif docs/online_mapping.gif \
        --fps 10

Unlike plot_follower_obstacle.py (instant full access to the ground-truth
ESDF, background never changes), the background here evolves over the
run as the sensor sweep resolves voxels. own_map's conservative ESDF mode
(computeEsdf(treat_unknown_as_occupied=true)) renders kUnknown as distance
0, same red as a real obstacle -- so a grey ground-truth overlay (every
true obstacle cell, regardless of whether own_map has observed it yet) is
drawn on top, faded enough to still show the ESDF colour underneath once
a cell is both truly occupied and already mapped.
"""
import argparse
import struct

import numpy as np
import pandas as pd

K_OCCUPIED = 2


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-log", default="logs/online_mapping_esdf_snapshots.bin")
    p.add_argument("--traj-csv", default="logs/online_mapping_trajectory.csv")
    p.add_argument("--save-png", metavar="PATH",
                    help="Render one frame headlessly to PATH instead of showing a window")
    p.add_argument("--save-gif", metavar="PATH",
                    help="Render an animated GIF headlessly to PATH instead of showing a window")
    p.add_argument("--frame", type=int, default=None,
                    help="Snapshot index for --save-png (default: last frame)")
    p.add_argument("--fps", type=int, default=10, help="Playback fps for --save-gif")
    p.add_argument("--trail-seconds", type=float, default=6.0,
                    help="How much trailing path history to draw behind each marker")
    p.add_argument("--max-distance", type=float, default=2.0,
                    help="Distance (m) at which the ESDF colormap reaches pure blue")
    p.add_argument("--fov-half-angle-deg", type=float, default=40.0,
                    help="Camera FOV cone half-angle (deg) -- must match "
                         "config/quadrotor.yaml's camera.fov_half_angle_deg")
    p.add_argument("--frustum-length", type=float, default=3.0,
                    help="2-D camera frustum wedge length (m)")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png or _args.save_gif:
    matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Patch, Wedge


def load_snapshot_log(path):
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        voxel_size, origin_x, origin_y = struct.unpack("<ddd", f.read(24))
        (sensor_max_range,) = struct.unpack("<d", f.read(8))

        n_voxels = nx * ny
        ground_truth_state = np.frombuffer(f.read(n_voxels), dtype=np.uint8).reshape(ny, nx)

        (num_snapshots,) = struct.unpack("<i", f.read(4))

        times = np.empty(num_snapshots)
        distances = np.empty((num_snapshots, ny, nx), dtype=np.float32)

        # state is still logged per voxel (own_map's format is shared with
        # other consumers), just not needed here -- read and discard it.
        voxel_dtype = np.dtype([("distance", "<f4"), ("state", "u1")])
        for i in range(num_snapshots):
            (times[i],) = struct.unpack("<d", f.read(8))
            voxels = np.frombuffer(f.read(n_voxels * voxel_dtype.itemsize), dtype=voxel_dtype)
            distances[i] = voxels["distance"].reshape(ny, nx)

    extent = [origin_x, origin_x + nx * voxel_size, origin_y, origin_y + ny * voxel_size]
    return times, distances, extent, ground_truth_state, sensor_max_range


def esdf_to_rgb(distance, max_distance):
    # Same red(close/occupied)->blue(far) hue sweep as plot_follower_obstacle.py's
    # esdf_to_rgb() / examples/esdf_2d_demo.cpp's distanceToColor(), but
    # desaturated ("faded") so it reads as a soft backdrop the grey
    # ground-truth overlay and the path/obstacle lines stay legible against,
    # not a competing saturated layer.
    t = np.clip(distance / max_distance, 0.0, 1.0)
    hue = t * (240.0 / 360.0)
    saturation = 0.55
    hsv = np.stack([hue, np.full_like(t, saturation), np.ones_like(t)], axis=-1)
    return hsv_to_rgb(hsv)


times, distances, extent, ground_truth_state, sensor_max_range = load_snapshot_log(
    _args.snapshot_log)
n_frames = len(times)
traj = pd.read_csv(_args.traj_csv)
# Canonical schema (same as mpc_controller.cpp/trailer_mpc_demo.cpp, see
# design_decisions.md): fx,fy=FOV target (the leader), xt,yt=drone position
# reference (the follower's goal point), x,y,psi=drone actual. Aliased here
# instead of renaming every reference below, to keep this script's own
# leader/follower/mpc naming self-explanatory.
traj = traj.rename(columns={
    "fx": "leader_x", "fy": "leader_y",
    "xt": "follower_x", "yt": "follower_y",
    "x": "mpc_x", "y": "mpc_y", "psi": "mpc_psi",
})
control_dt = traj["t"].iloc[1] - traj["t"].iloc[0]
trail_steps = max(1, int(_args.trail_seconds / control_dt))


def nearest_control_index(t):
    return int(np.argmin(np.abs(traj["t"].values - t)))


fig, ax = plt.subplots(figsize=(7, 7))
# alpha<1 + bilinear interpolation: a soft, faded backdrop (not a competing
# saturated/blocky-voxel layer) that the grey ground-truth overlay and
# path/obstacle lines stay legible against.
im = ax.imshow(esdf_to_rgb(distances[0], _args.max_distance),
                origin="lower", extent=extent, interpolation="bilinear", alpha=0.75)

# Ground truth (privileged, simulator-only) obstacle shape -- drawn
# unconditionally for every true obstacle cell, regardless of whether
# own_map has observed it yet (own_map's belief is the ESDF hue sweep
# above; this is what's actually there). Low alpha so both layers stay
# visible where they overlap -- grey tints the ESDF colour rather than
# hiding it once a cell is both truly occupied and already mapped.
ax.contour(ground_truth_state == K_OCCUPIED, levels=[0.5], extent=extent, origin="lower",
           colors="dimgray", linewidths=2.0, linestyles="--")
ax.contourf(ground_truth_state == K_OCCUPIED, levels=[0.5, 1.5], extent=extent, origin="lower",
            colors="grey", alpha=0.3)
obstacle_proxy = Patch(facecolor="grey", alpha=0.3, edgecolor="dimgray",
                        linestyle="--", label="obstacle (unobserved)")

sensor_circle = plt.Circle((0, 0), sensor_max_range, fill=False, color="deepskyblue",
                            linestyle=":", linewidth=1.2, label=f"sensor range ({sensor_max_range:.0f}m)")
ax.add_patch(sensor_circle)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
margin = 0.5 * sensor_max_range
ax.set_xlim(extent[0] - margin, extent[1] + margin)
ax.set_ylim(extent[2] - margin, extent[3] + margin)

(leader_trail,) = ax.plot([], [], "-", color="red", linewidth=2, label="leader (target)")
(follower_trail,) = ax.plot([], [], "-", color="deepskyblue", linewidth=2,
                             label="follower (reference)")
(mpc_trail,) = ax.plot([], [], "--", color="lime", linewidth=2, label="MPC path (actual)")
(leader_dot,) = ax.plot([], [], "o", color="red", markersize=8)
(follower_dot,) = ax.plot([], [], "o", color="deepskyblue", markersize=8)
(mpc_dot,) = ax.plot([], [], "o", color="lime", markersize=8)

# Top-down camera yaw frustum -- forward-facing (body +x), same
# fov_half_angle_deg as config/quadrotor.yaml's camera cost, drawn from
# the actual MPC/drone position and yaw (psi). Purely a visual check of
# where the camera is pointing -- the FOV cost has no obstacle awareness
# yet (see design_decisions.md), so this frustum sweeping through the
# obstacle is expected, not validated against anything here.
frustum = Wedge((0, 0), _args.frustum_length, 0, 0, color="orange", alpha=0.3,
                 label="camera FOV (yaw only)")
ax.add_patch(frustum)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [obstacle_proxy], labels=labels + [obstacle_proxy.get_label()],
          loc="upper right")


def update(frame):
    im.set_data(esdf_to_rgb(distances[frame], _args.max_distance))

    ci = nearest_control_index(times[frame])
    i0 = max(0, ci - trail_steps)
    leader_trail.set_data(traj["leader_x"].iloc[i0:ci + 1], traj["leader_y"].iloc[i0:ci + 1])
    follower_trail.set_data(traj["follower_x"].iloc[i0:ci + 1], traj["follower_y"].iloc[i0:ci + 1])
    mpc_trail.set_data(traj["mpc_x"].iloc[i0:ci + 1], traj["mpc_y"].iloc[i0:ci + 1])
    leader_dot.set_data([traj["leader_x"].iloc[ci]], [traj["leader_y"].iloc[ci]])
    follower_dot.set_data([traj["follower_x"].iloc[ci]], [traj["follower_y"].iloc[ci]])
    mpc_dot.set_data([traj["mpc_x"].iloc[ci]], [traj["mpc_y"].iloc[ci]])
    # Sensor rides the actual closed-loop drone position, not the
    # open-loop follower reference -- see online_mapping_demo.cpp.
    mpc_x, mpc_y = traj["mpc_x"].iloc[ci], traj["mpc_y"].iloc[ci]
    sensor_circle.center = (mpc_x, mpc_y)

    yaw_deg = np.degrees(traj["mpc_psi"].iloc[ci])
    frustum.set_center((mpc_x, mpc_y))
    frustum.set_theta1(yaw_deg - _args.fov_half_angle_deg)
    frustum.set_theta2(yaw_deg + _args.fov_half_angle_deg)

    ax.set_title(f"Online mapping: own_map ESDF  (t = {times[frame]:.1f}s)")
    return (im, sensor_circle, frustum, leader_trail, follower_trail, mpc_trail,
            leader_dot, follower_dot, mpc_dot)


if _args.save_png:
    frame = _args.frame if _args.frame is not None else n_frames - 1
    update(frame)
    fig.savefig(_args.save_png, dpi=150)
    print(f"Saved {_args.save_png}")
elif _args.save_gif:
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    anim.save(_args.save_gif, writer="pillow", fps=_args.fps)
    print(f"Saved {_args.save_gif}")
else:
    dt_ms = (times[1] - times[0]) * 1000 if n_frames > 1 else 100
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=dt_ms, blit=True)
    plt.show()
