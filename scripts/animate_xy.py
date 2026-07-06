"""
Animated 3-D view of the quadrotor moving-target tracking MPC.

Run from the repo root after executing the C++ binary:
    ./build/argus_mpc          # produces logs/trajectory.csv
    python3 scripts/animate_xy.py

Headless export (e.g. for docs/README assets), no display required:
    python3 scripts/animate_xy.py --save-png docs/tracking.png
    python3 scripts/animate_xy.py --save-gif docs/tracking.gif \
        --start-frame 20 --num-frames 160 --stride 2 --fps 15

What is shown:
  - Grey dashed path    : complete logged target trajectory (ground plane)
  - Red star            : current ground target position (moves)
  - Blue trail          : recent drone history (last 4 s)
  - Blue dot            : current drone position
  - Green dot           : current MPC reference position (above target at z_ref)
  - Amber frustum       : camera frustum (actual yaw + pitch)
  - Dashed green frustum: ideal frustum (yaw + pitch required to see target)
  - Amber polygon       : camera footprint projected onto the ground plane (z=0)
  - Text                : XY tracking error, z error, yaw/pitch bearing errors
"""

import argparse
import sys

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-png", metavar="PATH",
                   help="Render one frame headlessly to PATH instead of showing a window")
    p.add_argument("--save-gif", metavar="PATH",
                   help="Render an animated GIF headlessly to PATH instead of showing a window")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame index for --save-png (default: midpoint of the run)")
    p.add_argument("--start-frame", type=int, default=0, help="First frame for --save-gif")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Number of frames for --save-gif, before --stride (default: to the end)")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth frame for --save-gif (reduces file size)")
    p.add_argument("--fps", type=int, default=15, help="Playback fps for --save-gif")
    return p.parse_args()


_args = _parse_args()

# Headless (Agg) backend for --save-png/--save-gif — must be set before
# importing pyplot. Default interactive use (no flags) is unaffected.
import matplotlib
if _args.save_png or _args.save_gif:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CSV_PATH      = "logs/trajectory.csv"
DRONE_Z_REF   = 1.5    # drone's z reference [m] — must match config/quadrotor.yaml mpc.hover_altitude
TRAIL_SECONDS = 4.0
FRUSTUM_LEN   = 0.70   # camera frustum length [m]
FRUSTUM_H_FOV = np.deg2rad(40.0)  # horizontal half-angle
FRUSTUM_V_FOV = np.deg2rad(30.0)  # vertical half-angle
XY_LIMIT      = 5.0    # axis half-extent [m]

# ── Load data ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: {CSV_PATH} not found.\nRun ./build/argus_mpc first.")
    sys.exit(1)

t_arr     = df["t"].to_numpy()
x_arr     = df["x"].to_numpy()
y_arr     = df["y"].to_numpy()
z_arr     = df["z"].to_numpy()
phi_arr   = df["phi"].to_numpy()
theta_arr = df["theta"].to_numpy()
psi_arr   = df["psi"].to_numpy()
xt_arr    = df["xt"].to_numpy()
yt_arr    = df["yt"].to_numpy()
zt_arr    = df["zt"].to_numpy()
fx_arr    = df["fx"].to_numpy()   # FOV target (origin in circle mode)
fy_arr    = df["fy"].to_numpy()
fz_arr    = df["fz"].to_numpy()

Ts          = float(t_arr[1] - t_arr[0])
trail_steps = int(TRAIL_SECONDS / Ts)
n_frames    = len(t_arr)

# ── Helpers ──────────────────────────────────────────────────────────────────

def rotation_matrix_zyx(phi, theta, psi):
    """ZYX body-to-world rotation matrix: v_world = R @ v_body."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ],
    ])


def frustum_edge_segs(pos, R):
    """Return (8, 2, 3) array of (start, end) line segment pairs for one frustum."""
    lh = np.tan(FRUSTUM_H_FOV) * FRUSTUM_LEN
    lv = np.tan(FRUSTUM_V_FOV) * FRUSTUM_LEN
    # 4 corners in body frame — camera along body +x axis
    corners_b = np.array([
        [FRUSTUM_LEN,  lh,  lv],
        [FRUSTUM_LEN, -lh,  lv],
        [FRUSTUM_LEN, -lh, -lv],
        [FRUSTUM_LEN,  lh, -lv],
    ])
    corners_w = pos + (R @ corners_b.T).T
    segs = np.empty((8, 2, 3))
    for i, c in enumerate(corners_w):
        segs[i, 0] = pos
        segs[i, 1] = c
    for i in range(4):
        segs[4 + i, 0] = corners_w[i]
        segs[4 + i, 1] = corners_w[(i + 1) % 4]
    return segs


def ref_attitude(drone_pos, target_pos):
    """Ideal (psi, theta) to aim body +x at target_pos with zero roll.

    Positive theta = nose-down in the ZYX/ENU convention used by this model.
    """
    d = target_pos - drone_pos
    xy = np.hypot(d[0], d[1])
    psi   = np.arctan2(d[1], d[0])
    theta = np.arctan2(-d[2], xy)   # + = nose down toward ground target
    return psi, theta


FOOTPRINT_MAX_RANGE = 8.0   # ground range cap for corners pointing toward horizon [m]

def frustum_ground_footprint(pos, R):
    """Ray-cast the 4 frustum corner rays to z=0.

    Corners pointing upward or horizontal are clamped to FOOTPRINT_MAX_RANGE
    so the polygon is always visible even when the camera is mostly level.
    Returns (4,3) array of ground hit points.
    """
    lh = np.tan(FRUSTUM_H_FOV) * FRUSTUM_LEN
    lv = np.tan(FRUSTUM_V_FOV) * FRUSTUM_LEN
    corners_b = np.array([
        [FRUSTUM_LEN,  lh,  lv],
        [FRUSTUM_LEN, -lh,  lv],
        [FRUSTUM_LEN, -lh, -lv],
        [FRUSTUM_LEN,  lh, -lv],
    ])
    hits = []
    for cb in corners_b:
        d = R @ cb
        if d[2] < -1e-4:
            hit = pos + (-pos[2] / d[2]) * d
        else:
            xy_len = np.hypot(d[0], d[1])
            if xy_len > 1e-6:
                t = FOOTPRINT_MAX_RANGE / xy_len
                hit = np.array([pos[0] + d[0] * t, pos[1] + d[1] * t, 0.0])
            else:
                hit = np.array([pos[0], pos[1], 0.0])
        hit = np.array([np.clip(hit[0], -XY_LIMIT, XY_LIMIT),
                        np.clip(hit[1], -XY_LIMIT, XY_LIMIT),
                        0.0])
        hits.append(hit)
    return np.array(hits)


def bearing_errors_deg(drone_pos, phi, theta, psi, target_pos):
    """Yaw and pitch bearing errors to target_pos in degrees.

    Uses the bearing vector d_body = R^T * (target - drone) in body frame.
    Yaw error  : atan2(d_body_y, d_body_x)  — target left/right of boresight
    Pitch error: atan2(-d_body_z, d_body_x) — target above/below boresight
    """
    d = target_pos - drone_pos
    if np.linalg.norm(d) < 1e-6:
        return 0.0, 0.0
    R  = rotation_matrix_zyx(phi, theta, psi)
    db = R.T @ d
    return (np.degrees(np.arctan2(db[1],  db[0])),
            np.degrees(np.arctan2(-db[2], db[0])))


# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor("#1a1a2e")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#16213e")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#2a2a4a")
ax.tick_params(colors="#aaaacc", labelsize=8)
for attr in ("xaxis", "yaxis", "zaxis"):
    getattr(ax, attr).label.set_color("#aaaacc")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Quadrotor Moving-Target Tracking MPC — 3-D View",
             color="#ccccff", fontsize=12, pad=12)

PAD = 1.2
ax.set_xlim(-XY_LIMIT, XY_LIMIT)
ax.set_ylim(-XY_LIMIT, XY_LIMIT)
ax.set_zlim(0.0, DRONE_Z_REF + PAD)
ax.view_init(elev=25, azim=-60)

# ── Static elements ───────────────────────────────────────────────────────────
# Full target trajectory (logged ground path) — shown as a faint dashed line
ax.plot(xt_arr, yt_arr, zt_arr,
        "--", color="#555577", linewidth=1.2, label="Reference trajectory")

# ── Animated artists ──────────────────────────────────────────────────────────
trail_line, = ax.plot([], [], [], color="#4488ff", linewidth=1.4, alpha=0.55,
                      label="Drone trail")
drone_dot,  = ax.plot([], [], [], "o", color="#66aaff", markersize=9,
                      markeredgecolor="#ffffff", markeredgewidth=0.7)
target_dot, = ax.plot([], [], [], "*", color="#e05252", markersize=14,
                      label="Ground target")
ref_dot,    = ax.plot([], [], [], "o", color="#55ee88", markersize=6, alpha=0.7,
                      label="MPC reference (above target)")

# 8 line objects per frustum (4 apex→corner + 4 base edges)
frustum_act = [ax.plot([], [], [], color="#ffaa00", linewidth=1.5)[0]
               for _ in range(8)]
frustum_ref = [ax.plot([], [], [], color="#55ee88", linewidth=1.2,
                        linestyle="--")[0]
               for _ in range(8)]

# Camera boresight ray: drone → ground-plane hit point
boresight_line, = ax.plot([], [], [], color="#00ddff", linewidth=1.8, alpha=0.85)
boresight_dot,  = ax.plot([], [], [], "x", color="#00ddff", markersize=10,
                           markeredgewidth=2.0)

# Ground footprint of the actual camera frustum (filled polygon at z=0).
_POLY_DUMMY = np.zeros((4, 3))
ground_poly = Poly3DCollection([_POLY_DUMMY], alpha=0.20,
                                facecolor="#ffaa00", edgecolor="#ffcc44",
                                linewidth=1.2, zorder=2)
ax.add_collection3d(ground_poly)

# Screen-space text overlays
time_text   = ax.text2D(0.03, 0.97, "", transform=ax.transAxes,
                        fontsize=10, color="#ddddff",  va="top")
xy_err_text = ax.text2D(0.03, 0.91, "", transform=ax.transAxes,
                        fontsize=9,  color="#88ffaa",  va="top")
z_err_text  = ax.text2D(0.03, 0.85, "", transform=ax.transAxes,
                        fontsize=9,  color="#aaddff",  va="top")
yaw_text    = ax.text2D(0.97, 0.97, "", transform=ax.transAxes,
                        fontsize=9,  color="#ffaa44",  va="top", ha="right")
pitch_text  = ax.text2D(0.97, 0.91, "", transform=ax.transAxes,
                        fontsize=9,  color="#ff88aa",  va="top", ha="right")

# ── Legend (proxy artists) ────────────────────────────────────────────────────
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
legend_handles = [
    mlines.Line2D([0], [0], color="#555577", linestyle="--", label="Target trajectory"),
    mlines.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e05252",
                  markersize=12, linestyle="None", label="Ground target"),
    mlines.Line2D([0], [0], color="#4488ff", label="Drone trail"),
    mlines.Line2D([0], [0], color="#ffaa00", linewidth=1.5,
                  label="Frustum — actual"),
    mlines.Line2D([0], [0], color="#55ee88", linewidth=1.2, linestyle="--",
                  label="Frustum — ideal (ψ+θ to target)"),
    mlines.Line2D([0], [0], color="#00ddff", linewidth=1.8,
                  label="Boresight ray (ground hit)"),
    mpatches.Patch(facecolor="#ffaa00", edgecolor="#ffcc44", alpha=0.5,
                   label="Camera footprint (ground)"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#55ee88",
                  markersize=7, linestyle="None", label="MPC reference"),
]
ax.legend(handles=legend_handles, loc="lower left",
          facecolor="#1a1a2e", edgecolor="#444466",
          labelcolor="#ccccee", fontsize=8)


# ── Animation helpers ─────────────────────────────────────────────────────────

def _apply_frustum(lines, pos, phi, theta, psi):
    R    = rotation_matrix_zyx(phi, theta, psi)
    segs = frustum_edge_segs(pos, R)
    for line, seg in zip(lines, segs):
        line.set_data_3d([seg[0, 0], seg[1, 0]],
                         [seg[0, 1], seg[1, 1]],
                         [seg[0, 2], seg[1, 2]])


def init():
    trail_line.set_data_3d([], [], [])
    drone_dot.set_data_3d([], [], [])
    target_dot.set_data_3d([], [], [])
    ref_dot.set_data_3d([], [], [])
    boresight_line.set_data_3d([], [], [])
    boresight_dot.set_data_3d([], [], [])
    ground_poly.set_verts([_POLY_DUMMY])
    for ln in frustum_act + frustum_ref:
        ln.set_data_3d([], [], [])
    for txt in (time_text, xy_err_text, z_err_text, yaw_text, pitch_text):
        txt.set_text("")
    return ([trail_line, drone_dot, target_dot, ref_dot,
             boresight_line, boresight_dot, ground_poly]
            + frustum_act + frustum_ref
            + [time_text, xy_err_text, z_err_text, yaw_text, pitch_text])


def update(frame):
    s = max(0, frame - trail_steps)
    trail_line.set_data_3d(x_arr[s:frame+1], y_arr[s:frame+1], z_arr[s:frame+1])

    pos     = np.array([x_arr[frame],  y_arr[frame],  z_arr[frame]])
    fov_pos = np.array([fx_arr[frame], fy_arr[frame], fz_arr[frame]])  # FOV target
    ref_pos = np.array([xt_arr[frame], yt_arr[frame], zt_arr[frame]])  # drone ref

    drone_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])
    target_dot.set_data_3d([fov_pos[0]], [fov_pos[1]], [fov_pos[2]])
    ref_dot.set_data_3d([ref_pos[0]], [ref_pos[1]], [ref_pos[2]])

    # Actual frustum — from logged attitude
    R_act = rotation_matrix_zyx(phi_arr[frame], theta_arr[frame], psi_arr[frame])
    _apply_frustum(frustum_act, pos,
                   phi_arr[frame], theta_arr[frame], psi_arr[frame])

    # Ideal frustum — yaw + pitch to point directly at FOV target, zero roll
    psi_r, theta_r = ref_attitude(pos, fov_pos)
    _apply_frustum(frustum_ref, pos, 0.0, theta_r, psi_r)

    # Ground footprint: project the 4 frustum corner rays onto z=0
    ground_poly.set_verts([frustum_ground_footprint(pos, R_act)])

    # Boresight ray: body +x in world frame, cast to z = 0 ground plane.
    # If pointing upward, fall back to a fixed-length line.
    boresight = R_act[:, 0]           # body x-axis in world frame
    if boresight[2] < -1e-4:          # ray hits ground (pointing down)
        t_hit = -pos[2] / boresight[2]
        hit = pos + t_hit * boresight
    else:
        hit = pos + 2.5 * boresight   # extend 2.5 m in boresight direction
    boresight_line.set_data_3d([pos[0], hit[0]], [pos[1], hit[1]], [pos[2], hit[2]])
    boresight_dot.set_data_3d([hit[0]], [hit[1]], [hit[2]])

    # Errors
    yaw_err, pitch_err = bearing_errors_deg(
        pos, phi_arr[frame], theta_arr[frame], psi_arr[frame], fov_pos)
    xy_err = np.hypot(pos[0] - ref_pos[0], pos[1] - ref_pos[1])
    z_err  = pos[2] - DRONE_Z_REF

    t = t_arr[frame]
    time_text.set_text(f"t = {t:.2f} s")
    xy_err_text.set_text(f"XY error: {xy_err:.3f} m")
    z_err_text.set_text(f"Z  error: {z_err:+.3f} m")
    yaw_text.set_text(f"ψ err to target: {yaw_err:+.0f}°")
    pitch_text.set_text(f"θ err to target: {pitch_err:+.0f}°")

    return ([trail_line, drone_dot, target_dot, ref_dot,
             boresight_line, boresight_dot, ground_poly]
            + frustum_act + frustum_ref
            + [time_text, xy_err_text, z_err_text, yaw_text, pitch_text])


plt.tight_layout()

if _args.save_png:
    frame_idx = _args.frame if _args.frame is not None else n_frames // 2
    init()
    update(frame_idx)
    fig.savefig(_args.save_png, dpi=130)
    print(f"Saved {_args.save_png} (frame {frame_idx}/{n_frames})")
elif _args.save_gif:
    end = _args.start_frame + _args.num_frames if _args.num_frames else n_frames
    frame_indices = list(range(_args.start_frame, min(end, n_frames), _args.stride))
    ani = animation.FuncAnimation(
        fig, update, frames=frame_indices, init_func=init, blit=False,
    )
    ani.save(_args.save_gif, writer="pillow", fps=_args.fps)
    print(f"Saved {_args.save_gif} ({len(frame_indices)} frames @ {_args.fps} fps)")
else:
    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=int(Ts * 1000),
        blit=False,
    )
    plt.show()
