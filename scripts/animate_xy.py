"""
Animated 3-D view of the quadrotor circle-tracking MPC.

Run from the repo root after executing the C++ binary:
    ./build/argus_mpc          # produces logs/trajectory.csv
    python3 scripts/animate_xy.py

What is shown:
  - Grey dashed circle  : reference trajectory at z = 1.5 m
  - Red star            : ground target at origin (0, 0, 0)
  - Blue trail          : recent drone history (last 4 s)
  - Blue dot            : current drone position
  - Green dot           : current reference position
  - Amber frustum       : camera frustum (actual yaw + pitch)
  - Dashed green frustum: ideal frustum (yaw + pitch required to see target)
  - Text                : XY tracking error, yaw bearing error, pitch bearing error
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

CSV_PATH      = "logs/trajectory.csv"
CIRCLE_R      = 2.0
CIRCLE_Z      = 1.5
CIRCLE_PERIOD = 4.19   # lap time [s] — must match config/quadrotor.yaml circle.period
TRAIL_SECONDS = 4.0
FRUSTUM_LEN   = 0.70   # camera frustum length [m]
FRUSTUM_H_FOV = np.deg2rad(40.0)  # horizontal half-angle
FRUSTUM_V_FOV = np.deg2rad(30.0)  # vertical half-angle
TARGET_POS    = np.array([0.0, 0.0, 0.0])  # ground target at circle centre

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
xr_arr    = df["x_ref"].to_numpy()
yr_arr    = df["y_ref"].to_numpy()
zr_arr    = df["z_ref"].to_numpy()

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


def ref_attitude(drone_pos):
    """Ideal (psi, theta) to aim body +x at TARGET_POS with zero roll.

    Positive theta = nose-down in the ZYX/ENU convention used by this model.
    """
    d = TARGET_POS - drone_pos
    xy = np.hypot(d[0], d[1])
    psi   = np.arctan2(d[1], d[0])
    theta = np.arctan2(-d[2], xy)   # + = nose down toward ground target
    return psi, theta


def bearing_errors_deg(drone_pos, phi, theta, psi):
    """Yaw and pitch bearing errors to TARGET_POS in degrees.

    Uses the bearing vector d_body = R^T * (target - drone) in body frame.
    Yaw error  : atan2(d_body_y, d_body_x)  — target left/right of boresight
    Pitch error: atan2(-d_body_z, d_body_x) — target above/below boresight
    """
    d = TARGET_POS - drone_pos
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
ax.set_title("Quadrotor Circle-Tracking MPC — 3-D View",
             color="#ccccff", fontsize=12, pad=12)

PAD = 1.2
ax.set_xlim(-CIRCLE_R - PAD, CIRCLE_R + PAD)
ax.set_ylim(-CIRCLE_R - PAD, CIRCLE_R + PAD)
ax.set_zlim(0.0, CIRCLE_Z + PAD)
ax.view_init(elev=25, azim=-60)

# ── Static elements ───────────────────────────────────────────────────────────
theta_c = np.linspace(0, 2 * np.pi, 300)
ax.plot(CIRCLE_R * np.cos(theta_c), CIRCLE_R * np.sin(theta_c),
        np.full(300, CIRCLE_Z),
        "--", color="#555577", linewidth=1.2, label="Reference circle")

ax.scatter([0], [0], [0], marker="*", s=220, color="#e05252", zorder=5)
ax.text(0.15, 0.0, 0.05, "Target", color="#e05252", fontsize=8)

# Vertical dashed column from ground to circle — helps read height
ax.plot([0, 0], [0, 0], [0, CIRCLE_Z], ":", color="#554444",
        linewidth=0.8, alpha=0.5)

# ── Animated artists ──────────────────────────────────────────────────────────
trail_line, = ax.plot([], [], [], color="#4488ff", linewidth=1.4, alpha=0.55,
                      label="Drone trail")
drone_dot,  = ax.plot([], [], [], "o", color="#66aaff", markersize=9,
                      markeredgecolor="#ffffff", markeredgewidth=0.7)
ref_dot,    = ax.plot([], [], [], "o", color="#55ee88", markersize=6, alpha=0.7,
                      label="Reference position")

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

# Screen-space text overlays
time_text   = ax.text2D(0.03, 0.97, "", transform=ax.transAxes,
                        fontsize=10, color="#ddddff",  va="top")
xy_err_text = ax.text2D(0.03, 0.91, "", transform=ax.transAxes,
                        fontsize=9,  color="#88ffaa",  va="top")
yaw_text    = ax.text2D(0.97, 0.97, "", transform=ax.transAxes,
                        fontsize=9,  color="#ffaa44",  va="top", ha="right")
pitch_text  = ax.text2D(0.97, 0.91, "", transform=ax.transAxes,
                        fontsize=9,  color="#ff88aa",  va="top", ha="right")

# ── Legend (proxy artists) ────────────────────────────────────────────────────
import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([0], [0], color="#555577", linestyle="--", label="Reference circle"),
    mlines.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e05252",
                  markersize=12, linestyle="None", label="Target (ground)"),
    mlines.Line2D([0], [0], color="#4488ff", label="Drone trail"),
    mlines.Line2D([0], [0], color="#ffaa00", linewidth=1.5,
                  label="Frustum — actual"),
    mlines.Line2D([0], [0], color="#55ee88", linewidth=1.2, linestyle="--",
                  label="Frustum — ideal (ψ+θ to target)"),
    mlines.Line2D([0], [0], color="#00ddff", linewidth=1.8,
                  label="Boresight ray (ground hit)"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#55ee88",
                  markersize=7, linestyle="None", label="Reference position"),
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
    ref_dot.set_data_3d([], [], [])
    boresight_line.set_data_3d([], [], [])
    boresight_dot.set_data_3d([], [], [])
    for ln in frustum_act + frustum_ref:
        ln.set_data_3d([], [], [])
    for txt in (time_text, xy_err_text, yaw_text, pitch_text):
        txt.set_text("")
    return ([trail_line, drone_dot, ref_dot, boresight_line, boresight_dot]
            + frustum_act + frustum_ref
            + [time_text, xy_err_text, yaw_text, pitch_text])


def update(frame):
    s = max(0, frame - trail_steps)
    trail_line.set_data_3d(x_arr[s:frame+1], y_arr[s:frame+1], z_arr[s:frame+1])

    pos = np.array([x_arr[frame], y_arr[frame], z_arr[frame]])
    drone_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])
    ref_dot.set_data_3d([xr_arr[frame]], [yr_arr[frame]], [zr_arr[frame]])

    # Actual frustum — from logged attitude
    R_act = rotation_matrix_zyx(phi_arr[frame], theta_arr[frame], psi_arr[frame])
    _apply_frustum(frustum_act, pos,
                   phi_arr[frame], theta_arr[frame], psi_arr[frame])

    # Ideal frustum — yaw + pitch to point directly at ground target, zero roll
    psi_r, theta_r = ref_attitude(pos)
    _apply_frustum(frustum_ref, pos, 0.0, theta_r, psi_r)

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
        pos, phi_arr[frame], theta_arr[frame], psi_arr[frame])
    xy_err = np.hypot(pos[0] - xr_arr[frame], pos[1] - yr_arr[frame])

    t = t_arr[frame]
    time_text.set_text(f"t = {t:.2f} s  ({t / CIRCLE_PERIOD:.1f} laps)")
    xy_err_text.set_text(f"XY error: {xy_err:.3f} m")
    yaw_text.set_text(f"ψ err to target: {yaw_err:+.0f}°")
    pitch_text.set_text(f"θ err to target: {pitch_err:+.0f}°")

    return ([trail_line, drone_dot, ref_dot, boresight_line, boresight_dot]
            + frustum_act + frustum_ref
            + [time_text, xy_err_text, yaw_text, pitch_text])


ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    interval=int(Ts * 1000),
    blit=False,
)

plt.tight_layout()
plt.show()
