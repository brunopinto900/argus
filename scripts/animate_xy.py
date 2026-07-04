"""
Animated XY top-down view of the quadrotor circle-tracking MPC.

Run from the repo root after executing the C++ binary:
    ./build/argus_mpc          # produces logs/trajectory.csv
    python3 scripts/animate_xy.py

What is shown:
  - Grey dashed circle  : reference trajectory
  - Red star            : target object at the centre
  - Blue trail          : recent drone history (last 4 s)
  - Blue filled circle  : current drone position
  - Amber wedge         : camera frustum (direction = yaw angle psi)
  - Dashed line         : drone → target (shows pointing gap)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

CSV_PATH = "logs/trajectory.csv"
CIRCLE_R = 2.0
TRAIL_SECONDS = 4.0       # how many seconds of history to show
FRUSTUM_LEN   = 0.70      # camera frustum length [m]
FRUSTUM_FOV   = np.deg2rad(40.0)  # half-angle

# ── Load data ──────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: {CSV_PATH} not found.\nRun ./build/argus_mpc first.")
    sys.exit(1)

t_arr   = df["t"].to_numpy()
x_arr   = df["x"].to_numpy()
y_arr   = df["y"].to_numpy()
psi_arr = df["psi"].to_numpy()
xr_arr  = df["x_ref"].to_numpy()
yr_arr  = df["y_ref"].to_numpy()

Ts      = float(t_arr[1] - t_arr[0])
trail_steps = int(TRAIL_SECONDS / Ts)
n_frames    = len(t_arr)

# ── Figure setup ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

PAD = 1.2
ax.set_xlim(-CIRCLE_R - PAD, CIRCLE_R + PAD)
ax.set_ylim(-CIRCLE_R - PAD, CIRCLE_R + PAD)
ax.set_aspect("equal")
ax.grid(color="#2a2a4a", linewidth=0.5, zorder=0)
ax.set_xlabel("x [m]", color="#aaaacc")
ax.set_ylabel("y [m]", color="#aaaacc")
ax.tick_params(colors="#aaaacc")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2a4a")

# ── Static elements ────────────────────────────────────────────────────────
theta_circle = np.linspace(0, 2 * np.pi, 300)
ax.plot(CIRCLE_R * np.cos(theta_circle),
        CIRCLE_R * np.sin(theta_circle),
        "--", color="#555577", linewidth=1.2, zorder=1, label="Reference circle")

ax.plot(0, 0, marker="*", markersize=18, color="#e05252",
        zorder=5, label="Target object")
ax.text(0.12, 0.10, "Target", color="#e05252", fontsize=9,
        transform=ax.transData, zorder=5)

# ── Animated artists ──────────────────────────────────────────────────────
trail_line, = ax.plot([], [], color="#4488ff", linewidth=1.4,
                       alpha=0.55, zorder=3)

drone_dot, = ax.plot([], [], "o", color="#66aaff", markersize=11,
                     markeredgecolor="#ffffff", markeredgewidth=0.8, zorder=6)

# line from drone to target (shows pointing gap)
aim_line, = ax.plot([], [], "--", color="#e05252", linewidth=0.8,
                    alpha=0.4, zorder=2)

# ref position dot
ref_dot, = ax.plot([], [], "o", color="#55ee88", markersize=6,
                   alpha=0.7, zorder=4, label="Reference position")

# camera frustum — replaced each frame
frustum_patch = [None]

# text overlays
time_text = ax.text(0.03, 0.96, "", transform=ax.transAxes,
                    fontsize=11, color="#ddddff",
                    verticalalignment="top", zorder=10)
err_text  = ax.text(0.97, 0.96, "", transform=ax.transAxes,
                    fontsize=10, color="#88ffaa",
                    verticalalignment="top", horizontalalignment="right",
                    zorder=10)
status_text = ax.text(0.03, 0.03, "", transform=ax.transAxes,
                      fontsize=9, color="#ffaa44",
                      verticalalignment="bottom", zorder=10)

# ── Legend ────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#555577", label="Reference circle"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e05252",
               markersize=12, label="Target object", linestyle="None"),
    plt.Line2D([0], [0], color="#4488ff", label="Drone trail"),
    mpatches.Patch(facecolor="#ffcc44", alpha=0.5, edgecolor="#ffaa00",
                   label="Camera frustum (yaw)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#55ee88",
               markersize=7, label="Reference position", linestyle="None"),
]
ax.legend(handles=legend_handles, loc="lower right",
          facecolor="#1a1a2e", edgecolor="#444466",
          labelcolor="#ccccee", fontsize=8.5)

ax.set_title("Quadrotor Circle-Tracking MPC — XY View",
             color="#ccccff", fontsize=13, pad=10)


def make_frustum_vertices(x, y, psi):
    """Triangle vertices for the camera frustum wedge."""
    la = psi + FRUSTUM_FOV
    ra = psi - FRUSTUM_FOV
    apex  = [x, y]
    left  = [x + FRUSTUM_LEN * np.cos(la), y + FRUSTUM_LEN * np.sin(la)]
    right = [x + FRUSTUM_LEN * np.cos(ra), y + FRUSTUM_LEN * np.sin(ra)]
    return [apex, left, right]


def init():
    trail_line.set_data([], [])
    drone_dot.set_data([], [])
    aim_line.set_data([], [])
    ref_dot.set_data([], [])
    time_text.set_text("")
    err_text.set_text("")
    status_text.set_text("")
    return trail_line, drone_dot, aim_line, ref_dot, time_text, err_text, status_text


def update(frame):
    # ── Trail ──────────────────────────────────────────────────────────
    start = max(0, frame - trail_steps)
    trail_line.set_data(x_arr[start:frame + 1], y_arr[start:frame + 1])

    # ── Drone position ─────────────────────────────────────────────────
    cx, cy, psi = x_arr[frame], y_arr[frame], psi_arr[frame]
    drone_dot.set_data([cx], [cy])

    # ── Reference dot ─────────────────────────────────────────────────
    ref_dot.set_data([xr_arr[frame]], [yr_arr[frame]])

    # ── Line from drone to target ──────────────────────────────────────
    aim_line.set_data([cx, 0.0], [cy, 0.0])

    # ── Camera frustum ─────────────────────────────────────────────────
    if frustum_patch[0] is not None:
        frustum_patch[0].remove()
    verts = make_frustum_vertices(cx, cy, psi)
    frustum_patch[0] = Polygon(verts, closed=True,
                               facecolor="#ffcc44", alpha=0.35,
                               edgecolor="#ffaa00", linewidth=1.5, zorder=5)
    ax.add_patch(frustum_patch[0])

    # ── Text overlays ──────────────────────────────────────────────────
    t = t_arr[frame]
    xy_err = np.hypot(cx - xr_arr[frame], cy - yr_arr[frame])
    time_text.set_text(f"t = {t:.2f} s  ({t/10.0:.1f} laps)")
    err_text.set_text(f"XY error: {xy_err:.3f} m")

    # show a note about yaw not yet pointing at target
    psi_target = np.arctan2(-cy, -cx)    # angle that would point at origin
    psi_err = abs(((psi - psi_target) + np.pi) % (2 * np.pi) - np.pi)
    status_text.set_text(f"Yaw pointing: OFF  |  ψ error to target: {np.rad2deg(psi_err):.0f}°")

    return (trail_line, drone_dot, aim_line, ref_dot,
            time_text, err_text, status_text, frustum_patch[0])


ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    interval=int(Ts * 1000),   # real-time playback
    blit=False,
)

plt.tight_layout()
plt.show()
