"""
Animated view of the quadrotor moving-target tracking MPC: a 3-D geometry
panel (drone/frustum/obstacles over a flat ground, no ESDF coloring) and,
for the online-mapping/forest scenario, a true 2-D own_map ESDF panel
side by side.

Run from the repo root after executing the C++ binary:
    ./build/argus_mpc          # produces logs/trajectory.csv
    python3 scripts/animate_xy.py

Headless export (e.g. for docs/README assets), no display required:
    python3 scripts/animate_xy.py --save-png docs/tracking.png
    python3 scripts/animate_xy.py --save-gif docs/tracking.gif \
        --start-frame 20 --num-frames 160 --stride 2 --fps 15

Online-mapping/forest scenario -- the snapshot log (ESDF + obstacles) is
auto-detected next to --csv-path, no extra flags needed, and adds the
right-hand 2-D ESDF panel (single geometry-only panel otherwise):
    ./build/online_mapping_demo
    python3 scripts/animate_xy.py --csv-path logs/online_mapping_trajectory.csv

Left panel (3-D geometry -- drone/target/frustum over a flat ground,
no ESDF coloring, same scene every scenario shares):
  - Grey dashed path    : complete logged target trajectory (ground plane)
  - Yellow star (+stem) : current ground target position (moves) -- raised on
                          a short vertical stem above its true z so it's
                          never flush with the ground plane
  - Blue trail          : recent drone history (last 4 s)
  - Blue dot            : current drone position
  - Green dot           : current MPC reference position (above target at z_ref)
  - Amber frustum       : camera frustum (actual yaw + pitch)
  - Dashed green frustum: ideal frustum (yaw + pitch required to see target)
  - Amber polygon       : camera footprint projected onto the ground plane (z=0)
  - Text                : XY tracking error, z error, yaw/pitch bearing errors
  - Grey extrusions      : every ground-truth-occupied cell (from the
                          auto-detected snapshot log), cosmetic height --
                          own_map is still a single z-layer, see
                          design_decisions.md's "Terrain / varying height"

Right panel (online-mapping scenario only -- own_map's actual 2-D ESDF,
same content/convention as argus_follower/scripts/plot_online_mapping.py,
reused here so both panels animate off one shared frame index):
  - Red/blue heatmap    : own_map's distance field at the current time
  - Grey overlay        : every ground-truth-occupied cell, regardless of
                          whether own_map has observed it yet
  - Leader/follower/MPC : the same three paths as the left panel, top-down
  - Dotted circle        : sensor max range, centered on the drone
  - Amber wedge          : camera FOV (yaw only, top-down)
"""

import argparse
import os
import struct
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
    p.add_argument("--csv-path", default="logs/trajectory.csv",
                   help="Trajectory CSV to animate (any producer using the same column layout "
                        "as src/mpc_controller.cpp's log, e.g. examples/trailer_mpc_demo.cpp)")
    p.add_argument("--xy-limit", type=float, default=None,
                   help="Axis half-extent [m], applied symmetrically to both x and y around 0 "
                        "(default: fit tightly to the logged trajectory's own bounding box + "
                        "margin, independent of any snapshot log's much larger grid extent)")
    p.add_argument("--snapshot-log", default=None,
                   help="online_mapping_demo's ESDF snapshot log -- draws own_map's distance "
                        "field as a faded textured plane at z=0, and every ground-truth "
                        "obstacle as a cosmetic-height extrusion (same grid, no separate scenario "
                        "file needed). Default: auto-detected next to --csv-path (matches "
                        "online_mapping_demo's *_trajectory.csv / *_esdf_snapshots.bin naming); "
                        "silently skipped if none is found (e.g. for argus_mpc/trailer_mpc_demo "
                        "logs, which have no such file).")
    p.add_argument("--esdf-stride", type=int, default=6,
                   help="Downsample factor for the left panel's obstacle extrusion (every Nth "
                        "voxel per axis) -- the full 300x300 grid is too many boxes to animate "
                        "smoothly in 3-D. Doesn't affect the right ESDF panel, which renders the "
                        "full-resolution grid as a flat 2-D image (cheap either way).")
    p.add_argument("--obstacle-height", type=float, default=1.2,
                   help="Cosmetic extrusion height [m] for ground-truth obstacles")
    return p.parse_args()


_args = _parse_args()

# Headless (Agg) backend for --save-png/--save-gif — must be set before
# importing pyplot. Default interactive use (no flags) is unaffected.
import matplotlib
if _args.save_png or _args.save_gif:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CSV_PATH      = _args.csv_path
DRONE_Z_REF   = 1.5    # drone's z reference [m] — must match config/quadrotor.yaml mpc.hover_altitude
TARGET_COLOR  = "#f5ff33"  # high-contrast yellow -- never appears in the ESDF's red/blue hue sweep
TARGET_STEM_H = 0.5    # purely a visual landmark height, not physically meaningful -- lifts the
                        # target marker off the z=0 plane so it doesn't visually sit inside the
                        # ESDF texture painted on that same plane
TRAIL_SECONDS = 4.0
FRUSTUM_LEN   = 0.70   # camera frustum length [m]
BORESIGHT_LEN = 1.50   # drawn boresight-direction arrow length [m] -- fixed, NOT a ground-hit
                        # distance (which would vary wildly with pitch, see update()'s comment)
FRUSTUM_H_FOV = np.deg2rad(40.0)  # horizontal half-angle -- must match config/quadrotor.yaml's camera.fov_half_angle_deg
FRUSTUM_V_FOV = np.deg2rad(40.0)  # vertical half-angle -- same as above: the cost's r_fov cone is
                                   # circular (cos(alpha) vs cos(gamma), isotropic), so a narrower
                                   # vertical angle here would draw a target as "missed" even when
                                   # it's genuinely inside the cone the controller optimizes for.
# X_MIN/X_MAX/Y_MIN/Y_MAX resolved further down, after the trajectory is
# loaded (fit to its own bounding box by default).

# ── Load data ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: {CSV_PATH} not found.\nRun ./build/argus_mpc (or pass --csv-path) first.")
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
        hit = np.array([np.clip(hit[0], X_MIN, X_MAX),
                        np.clip(hit[1], Y_MIN, Y_MAX),
                        0.0])
        hits.append(hit)
    return np.array(hits)


K_OCCUPIED = 2  # argus_esdf::VoxelState::kOccupied -- see VoxelGrid.hpp


def load_snapshot_log(path):
    """Same binary layout as argus_follower/scripts/plot_online_mapping.py's
    loader -- duplicated here (not imported) to keep each plot script
    standalone, same pattern as the rest of this repo's scripts. Unlike
    that loader, this one keeps ground_truth_state (for the obstacle
    extrusion below -- no separate scenario YAML needed, same grid)."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        voxel_size, origin_x, origin_y = struct.unpack("<ddd", f.read(24))
        (sensor_max_range,) = struct.unpack("<d", f.read(8))
        n_voxels = nx * ny
        ground_truth_state = np.frombuffer(f.read(n_voxels), dtype=np.uint8).reshape(ny, nx)
        (num_snapshots,) = struct.unpack("<i", f.read(4))
        times = np.empty(num_snapshots)
        distances = np.empty((num_snapshots, ny, nx), dtype=np.float32)
        voxel_dtype = np.dtype([("distance", "<f4"), ("state", "u1")])
        for i in range(num_snapshots):
            (times[i],) = struct.unpack("<d", f.read(8))
            voxels = np.frombuffer(f.read(n_voxels * voxel_dtype.itemsize), dtype=voxel_dtype)
            distances[i] = voxels["distance"].reshape(ny, nx)
    xs = origin_x + (np.arange(nx) + 0.5) * voxel_size
    ys = origin_y + (np.arange(ny) + 0.5) * voxel_size
    return times, distances, xs, ys, ground_truth_state, voxel_size, sensor_max_range


def find_snapshot_log(csv_path):
    """Auto-detect online_mapping_demo's snapshot log next to a trajectory
    CSV, by its fixed naming convention (see online_mapping_demo.cpp:
    logs/online_mapping_trajectory.csv -> logs/online_mapping_esdf_snapshots.bin).
    Returns None (silently) for any other log, e.g. argus_mpc/trailer_mpc_demo's,
    which have no such file.

    Checks the CSV's own basename, not just whether the snapshot file
    happens to exist in the same directory -- every demo logs into the
    same logs/ folder, so a stale snapshot file left over from an earlier
    online_mapping_demo run would otherwise get wrongly attached to an
    unrelated circle/trailer CSV sitting right next to it.
    """
    if not os.path.basename(csv_path).startswith("online_mapping"):
        return None
    guess = os.path.join(os.path.dirname(csv_path) or ".",
                         "online_mapping_esdf_snapshots.bin")
    return guess if os.path.exists(guess) else None


# Same red(close)->blue(far) hue sweep as
# argus_follower/scripts/plot_online_mapping.py's esdf_to_rgb() -- this is
# the 2-D panel's whole job (showing own_map's actual ESDF), unlike the
# left 3-D geometry panel, which no longer renders any ESDF coloring at
# all (removed: it fights every marker drawn on top of it once the view is
# zoomed to the flight path rather than the full grid -- see git history
# for the version that tried it).
def esdf_to_rgb(distance, max_distance):
    t = np.clip(distance / max_distance, 0.0, 1.0)
    hue = t * (240.0 / 360.0)
    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([hue, np.full_like(t, 0.55), np.ones_like(t)], axis=-1)
    return hsv_to_rgb(hsv)


def build_obstacle_extrusions(ground_truth_state, xs, ys, voxel_size, height, stride):
    """Every ground-truth-occupied cell (downsampled by `stride`, same as
    the ESDF plane) as a small extruded box -- side faces + top cap. No
    scenario YAML needed: same snapshot-log grid the ESDF plane already
    reads, so it can never drift out of sync with what a given run
    actually used. Height is cosmetic (see --obstacle-height's help), not
    from real grid data -- see design_decisions.md's "Terrain / varying
    height". Blocky/pixelated by construction (one box per downsampled
    cell, not a fitted shape) -- an honest reflection of the grid's own
    resolution, same spirit as the 2-D script's grey overlay."""
    occupied = ground_truth_state[::stride, ::stride] == K_OCCUPIED
    half = stride * voxel_size / 2.0
    polys = []
    for iy, ix in zip(*np.nonzero(occupied)):
        cx, cy = xs[::stride][ix], ys[::stride][iy]
        ring = np.array([[cx - half, cy - half], [cx + half, cy - half],
                          [cx + half, cy + half], [cx - half, cy + half]])
        for i in range(4):
            p0, p1 = ring[i], ring[(i + 1) % 4]
            polys.append([[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0],
                          [p1[0], p1[1], height], [p0[0], p0[1], height]])
        polys.append([[p[0], p[1], height] for p in ring])  # top cap
    return polys


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


# ── Online-mapping snapshot log (auto-detected unless --snapshot-log given) ──
snapshot_path = _args.snapshot_log or find_snapshot_log(CSV_PATH)
esdf_times = esdf_distances = esdf_xs = esdf_ys = obstacle_polys = None
sensor_max_range = None
if snapshot_path:
    esdf_times, esdf_distances, esdf_xs, esdf_ys, ground_truth_state, voxel_size, \
        sensor_max_range = load_snapshot_log(snapshot_path)
    obstacle_polys = build_obstacle_extrusions(
        ground_truth_state, esdf_xs, esdf_ys, voxel_size, _args.obstacle_height,
        _args.esdf_stride)
    esdf_extent = [esdf_xs[0], esdf_xs[-1], esdf_ys[0], esdf_ys[-1]]

# View bounds (left/geometry panel only): fit tightly to where the
# drone/reference/target actually go (not the snapshot log's much larger,
# mostly-empty grid extent) so the action fills the frame instead of being
# a speck in a wide, static view. Independent x/y ranges (not a single
# symmetric square) since flight corridors here are typically long and
# narrow, not square. The right/ESDF panel (if any) uses the full grid
# extent instead -- see esdf_extent above -- since showing the whole map
# is its actual job.
TRAJ_MARGIN = 1.0  # m
_all_x = np.concatenate([x_arr, xt_arr, fx_arr])
_all_y = np.concatenate([y_arr, yt_arr, fy_arr])
if _args.xy_limit is not None:
    X_MIN, X_MAX = -_args.xy_limit, _args.xy_limit
    Y_MIN, Y_MAX = -_args.xy_limit, _args.xy_limit
else:
    X_MIN, X_MAX = _all_x.min() - TRAJ_MARGIN, _all_x.max() + TRAJ_MARGIN
    Y_MIN, Y_MAX = _all_y.min() - TRAJ_MARGIN, _all_y.max() + TRAJ_MARGIN


def esdf_frame_index(t):
    return int(np.argmin(np.abs(esdf_times - t)))


# ── Figure setup ─────────────────────────────────────────────────────────────
# Two panels side by side when a snapshot log is available (left: 3-D
# geometry, right: own_map's actual 2-D ESDF) -- single geometry-only
# panel otherwise (circle/trailer scenarios have no ESDF to show).
fig = plt.figure(figsize=(16, 8) if snapshot_path else (10, 8))
fig.patch.set_facecolor("#1a1a2e")
ax = fig.add_subplot(121 if snapshot_path else 111, projection="3d")
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
ax.set_title("Geometry (3-D)", color="#ccccff", fontsize=12, pad=12)

PAD = 1.2
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_zlim(0.0, DRONE_Z_REF + PAD)
ax.view_init(elev=60, azim=0)  # mostly top-down (reads like the 2-D ESDF panel) while keeping
                               # enough elevation to still show height/frustum/pitch in 3-D

# Right panel: own_map's actual 2-D ESDF (online-mapping scenario only) --
# same content as argus_follower/scripts/plot_online_mapping.py, reused
# here so it animates off the same frame index as the left panel.
if snapshot_path:
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="#aaaacc", labelsize=8)
    ax2.xaxis.label.set_color("#aaaacc")
    ax2.yaxis.label.set_color("#aaaacc")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("own_map ESDF (2-D)", color="#ccccff", fontsize=12, pad=12)
    ax2.set_aspect("equal")
    esdf_margin = 0.5 * sensor_max_range
    ax2.set_xlim(esdf_extent[0] - esdf_margin, esdf_extent[1] + esdf_margin)
    ax2.set_ylim(esdf_extent[2] - esdf_margin, esdf_extent[3] + esdf_margin)

# ── Static elements ───────────────────────────────────────────────────────────
# Full target trajectory (logged ground path) — shown as a faint dashed line
ax.plot(xt_arr, yt_arr, zt_arr,
        "--", color="#555577", linewidth=1.2, label="Reference trajectory")

# Obstacles: every ground-truth-occupied cell from the auto-detected
# snapshot log, static extruded boxes, cosmetic height — see
# build_obstacle_extrusions()'s docstring and design_decisions.md
# ("Terrain / varying height" todo item -- own_map is still a single
# z-layer, there's no real height data to draw instead).
if obstacle_polys:
    ax.add_collection3d(Poly3DCollection(
        obstacle_polys, facecolor="grey", edgecolor="dimgray",
        linewidth=0.5, alpha=0.5))
    ax.set_zlim(0.0, max(DRONE_Z_REF + PAD, _args.obstacle_height + 0.5))

# Right panel's static elements -- ground-truth obstacle outline (every
# true obstacle cell, regardless of observed state) and the sensor-range
# circle, same convention as plot_online_mapping.py.
if snapshot_path:
    ax2.contour(ground_truth_state == K_OCCUPIED, levels=[0.5], extent=esdf_extent,
                origin="lower", colors="dimgray", linewidths=1.5, linestyles="--")
    ax2.contourf(ground_truth_state == K_OCCUPIED, levels=[0.5, 1.5], extent=esdf_extent,
                 origin="lower", colors="grey", alpha=0.3)
    esdf_obstacle_proxy = mpatches.Patch(facecolor="grey", alpha=0.3, edgecolor="dimgray",
                                          linestyle="--", label="Obstacle (unobserved)")

    esdf_im = ax2.imshow(esdf_to_rgb(esdf_distances[0], max_distance=2.0),
                          origin="lower", extent=esdf_extent, interpolation="bilinear",
                          alpha=0.75, zorder=0)

    sensor_circle = plt.Circle((0, 0), sensor_max_range, fill=False, color="deepskyblue",
                                linestyle=":", linewidth=1.2,
                                label=f"Sensor range ({sensor_max_range:.0f} m)")
    ax2.add_patch(sensor_circle)

# ── Animated artists ──────────────────────────────────────────────────────────
trail_line, = ax.plot([], [], [], color="#4488ff", linewidth=1.4, alpha=0.55,
                      label="Drone trail")
drone_dot,  = ax.plot([], [], [], "o", color="#66aaff", markersize=9,
                      markeredgecolor="#ffffff", markeredgewidth=0.7)
target_halo, = ax.plot([], [], [], "o", color=TARGET_COLOR, markersize=26, alpha=0.22,
                       markeredgewidth=0)
target_stem, = ax.plot([], [], [], color=TARGET_COLOR, linewidth=1.2, alpha=0.6)
target_dot,  = ax.plot([], [], [], "*", color=TARGET_COLOR, markersize=16,
                       markeredgecolor="#ffffff", markeredgewidth=0.8,
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

# Right panel's animated artists -- leader/follower/MPC paths (top-down,
# same fx/fy, xt/yt, x/y arrays the left panel already uses) and the
# camera's yaw-only FOV wedge, same convention as plot_online_mapping.py.
if snapshot_path:
    leader_trail2d,   = ax2.plot([], [], "-", color="red", linewidth=2, label="Leader (target)")
    follower_trail2d, = ax2.plot([], [], ":", color="deepskyblue", linewidth=2,
                                  label="Follower (reference)")
    mpc_trail2d,      = ax2.plot([], [], "--", color="lime", linewidth=2, label="MPC path (actual)")
    leader_dot2d,   = ax2.plot([], [], "o", color="red", markersize=8)
    follower_dot2d, = ax2.plot([], [], "o", color="deepskyblue", markersize=8)
    mpc_dot2d,      = ax2.plot([], [], "o", color="lime", markersize=8)

    fov_wedge2d = mpatches.Wedge((0, 0), FRUSTUM_LEN * 4, 0, 0, color="orange", alpha=0.3,
                                  label="Camera FOV (yaw only)")
    ax2.add_patch(fov_wedge2d)

    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=ax2_handles + [esdf_obstacle_proxy],
               labels=ax2_labels + [esdf_obstacle_proxy.get_label()],
               loc="upper right", facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="#ccccee", fontsize=8)

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
legend_handles = [
    mlines.Line2D([0], [0], color="#555577", linestyle="--", label="Target trajectory"),
    mlines.Line2D([0], [0], marker="*", color="w", markerfacecolor=TARGET_COLOR,
                  markeredgecolor="#ffffff", markersize=12, linestyle="None",
                  label="Ground target"),
    mlines.Line2D([0], [0], color="#4488ff", label="Drone trail"),
    mlines.Line2D([0], [0], color="#ffaa00", linewidth=1.5,
                  label="Frustum — actual"),
    mlines.Line2D([0], [0], color="#55ee88", linewidth=1.2, linestyle="--",
                  label="Frustum — ideal (ψ+θ to target)"),
    mlines.Line2D([0], [0], color="#00ddff", linewidth=1.8,
                  label="Boresight direction (fixed length)"),
    mpatches.Patch(facecolor="#ffaa00", edgecolor="#ffcc44", alpha=0.5,
                   label="Camera footprint (ground)"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#55ee88",
                  markersize=7, linestyle="None", label="MPC reference"),
]
if obstacle_polys:
    legend_handles.append(mpatches.Patch(facecolor="grey", edgecolor="dimgray", alpha=0.5,
                                          label="Obstacle (cosmetic height)"))
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
    target_stem.set_data_3d([], [], [])
    target_halo.set_data_3d([], [], [])
    ref_dot.set_data_3d([], [], [])
    boresight_line.set_data_3d([], [], [])
    boresight_dot.set_data_3d([], [], [])
    ground_poly.set_verts([_POLY_DUMMY])
    for ln in frustum_act + frustum_ref:
        ln.set_data_3d([], [], [])
    for txt in (time_text, xy_err_text, z_err_text, yaw_text, pitch_text):
        txt.set_text("")
    artists = ([trail_line, drone_dot, target_dot, target_stem, target_halo, ref_dot,
                boresight_line, boresight_dot, ground_poly]
               + frustum_act + frustum_ref
               + [time_text, xy_err_text, z_err_text, yaw_text, pitch_text])
    if snapshot_path:
        leader_trail2d.set_data([], [])
        follower_trail2d.set_data([], [])
        mpc_trail2d.set_data([], [])
        leader_dot2d.set_data([], [])
        follower_dot2d.set_data([], [])
        mpc_dot2d.set_data([], [])
        artists += [esdf_im, leader_trail2d, follower_trail2d, mpc_trail2d,
                    leader_dot2d, follower_dot2d, mpc_dot2d, sensor_circle, fov_wedge2d]
    return artists


def update(frame):
    s = max(0, frame - trail_steps)
    trail_line.set_data_3d(x_arr[s:frame+1], y_arr[s:frame+1], z_arr[s:frame+1])

    pos     = np.array([x_arr[frame],  y_arr[frame],  z_arr[frame]])
    fov_pos = np.array([fx_arr[frame], fy_arr[frame], fz_arr[frame]])  # FOV target
    ref_pos = np.array([xt_arr[frame], yt_arr[frame], zt_arr[frame]])  # drone ref

    drone_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])
    target_top = fov_pos[2] + TARGET_STEM_H
    target_dot.set_data_3d([fov_pos[0]], [fov_pos[1]], [target_top])
    target_halo.set_data_3d([fov_pos[0]], [fov_pos[1]], [target_top])
    target_stem.set_data_3d([fov_pos[0], fov_pos[0]], [fov_pos[1], fov_pos[1]],
                             [fov_pos[2], target_top])
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

    # Boresight direction: body +x in world frame, drawn as a FIXED-length
    # arrow (BORESIGHT_LEN), not cast out to its ground intersection. The
    # ground-hit distance depends on pitch in a way that has nothing to do
    # with how far off-target the camera actually is -- a real but modest
    # ~20-30 deg angular error, at near-level pitch, produces a ground hit
    # many meters away (tiny vertical component, large horizontal throw),
    # which drew as a long near-horizontal line that *looked* like the
    # camera was pointing off into nowhere even though the true angular
    # error was small. A constant-length arrow keeps its visual length
    # meaningful (how far off the target direction it's tilted) regardless
    # of pitch, instead of conflating direction with ground-throw distance.
    boresight = R_act[:, 0]           # body x-axis in world frame
    hit = pos + BORESIGHT_LEN * boresight
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

    artists = ([trail_line, drone_dot, target_dot, target_stem, target_halo, ref_dot,
                boresight_line, boresight_dot, ground_poly]
               + frustum_act + frustum_ref
               + [time_text, xy_err_text, z_err_text, yaw_text, pitch_text])

    if snapshot_path:
        esdf_frame = esdf_frame_index(t)
        esdf_im.set_data(esdf_to_rgb(esdf_distances[esdf_frame], max_distance=2.0))

        leader_trail2d.set_data(fx_arr[s:frame+1], fy_arr[s:frame+1])
        follower_trail2d.set_data(xt_arr[s:frame+1], yt_arr[s:frame+1])
        mpc_trail2d.set_data(x_arr[s:frame+1], y_arr[s:frame+1])
        leader_dot2d.set_data([fov_pos[0]], [fov_pos[1]])
        follower_dot2d.set_data([ref_pos[0]], [ref_pos[1]])
        mpc_dot2d.set_data([pos[0]], [pos[1]])

        sensor_circle.center = (pos[0], pos[1])

        yaw_deg = np.degrees(psi_arr[frame])
        fov_half_deg = np.degrees(FRUSTUM_H_FOV)
        fov_wedge2d.set_center((pos[0], pos[1]))
        fov_wedge2d.set_theta1(yaw_deg - fov_half_deg)
        fov_wedge2d.set_theta2(yaw_deg + fov_half_deg)

        artists += [esdf_im, leader_trail2d, follower_trail2d, mpc_trail2d,
                    leader_dot2d, follower_dot2d, mpc_dot2d, sensor_circle, fov_wedge2d]

    return artists


plt.tight_layout(rect=[0, 0, 1, 0.94])  # leave headroom so the axis titles aren't clipped

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
