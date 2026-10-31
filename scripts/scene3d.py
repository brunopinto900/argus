"""
Shared scene-building and pose-interpolation helpers for the offline 3-D
viewer (scripts/view_scene3d.py) and camera-footage renderer
(scripts/render_camera_footage.py). Import only -- no __main__ here.

Reads the same scenario YAML (obstacles: list, shape: circle|square) and
trajectory-CSV column layout as the rest of scripts/ (run_scenario_suite.py,
plot_scenario_contact_sheet.py, animate_xy.py, examples/online_mapping_demo.cpp's
header) -- no new file formats introduced.

Obstacles have no z-extent in the scenario files or the ESDF/collision
pipeline (VoxelGrid is a single z-layer, height-independent). The
extrusion height used here is a renderer-only cosmetic choice, not a
physical property -- see DEFAULT_OBSTACLE_HEIGHT_M.

The ZYX rotation + forward-FOV-cone convention below is copied from
scripts/animate_xy.py's rotation_matrix_zyx()/frustum_edge_segs() rather
than imported -- that module runs argparse and builds a matplotlib figure
at module scope (no __main__ guard), so importing it would execute all of
that. Keep these two in sync by hand if the convention ever changes.
"""

import os
import struct

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation, Slerp

DEFAULT_OBSTACLE_HEIGHT_M = 2.5
DEFAULT_BLUR_SUBSAMPLES = 8
GROUND_MARGIN_M = 2.0


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


def frustum_edge_segs(pos, R, length, h_fov, v_fov):
    """(8, 2, 3) array of (start, end) line-segment pairs for one camera
    FOV wireframe -- camera boresight along body +x, same convention the
    r_fov cost and animate_xy.py's frustum already use."""
    lh = np.tan(h_fov) * length
    lv = np.tan(v_fov) * length
    corners_b = np.array([
        [length,  lh,  lv],
        [length, -lh,  lv],
        [length, -lh, -lv],
        [length,  lh, -lv],
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


def load_scenario_obstacles(path):
    """obstacles: list from a scenario YAML, same schema
    plot_scenario_contact_sheet.py/run_scenario_suite.py read
    (shape: circle -> center+radius, shape: square -> center+half_size)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("obstacles", [])


def load_camera_config(path):
    """(camera, mapping_sensor) sub-dicts from config/quadrotor.yaml --
    fov_half_angle_deg and update_rate_hz, the two figures the camera-
    footage renderer needs (resolution has no home in this file; it's a
    renderer CLI flag, matching compute_metrics.py's own hardcoded
    assumption)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["camera"], cfg["mapping_sensor"]


def load_trajectory(csv_path):
    return pd.read_csv(csv_path)


def scene_bounds(df, obstacles, margin=GROUND_MARGIN_M):
    """(xmin, xmax, ymin, ymax) over the drone path, leader path, and
    every obstacle, with a margin -- used to size the ground plane."""
    xs = [df["x"].to_numpy(), df["fx"].to_numpy()]
    ys = [df["y"].to_numpy(), df["fy"].to_numpy()]
    for o in obstacles:
        cx, cy = o["center"]
        r = o["radius"] if o["shape"] == "circle" else o["half_size"] * np.sqrt(2)
        xs.append(np.array([cx - r, cx + r]))
        ys.append(np.array([cy - r, cy + r]))
    xmin = min(a.min() for a in xs) - margin
    xmax = max(a.max() for a in xs) + margin
    ymin = min(a.min() for a in ys) - margin
    ymax = max(a.max() for a in ys) + margin
    return xmin, xmax, ymin, ymax


# ── Open3D builders ──────────────────────────────────────────────────────

def _tree_params(obstacle, height):
    """Deterministic per-tree randomization, seeded off the obstacle's own
    center so re-running the same scenario always draws the same trees
    (no flicker across a re-render, no dependence on obstacle list order)."""
    cx, cy = obstacle["center"]
    r = obstacle["radius"]
    seed = int(abs(hash((round(cx, 4), round(cy, 4)))) % (2**32))
    rng = np.random.default_rng(seed)
    trunk_h = height * 0.45
    trunk_r = max(r * 0.18, 0.04)
    canopy_r = r * rng.uniform(1.3, 1.9)
    jitter = rng.uniform(-0.04, 0.04, size=3)
    trunk_color = np.clip(np.array([0.40, 0.26, 0.12]) + jitter, 0, 1)
    canopy_color = np.clip(np.array([0.16, 0.42, 0.14]) + jitter, 0, 1)
    return trunk_h, trunk_r, canopy_r, trunk_color, canopy_color


def build_obstacle_parts_o3d(obstacle, height=DEFAULT_OBSTACLE_HEIGHT_M):
    """List of (name, mesh, rgba) parts for one obstacle. Circle obstacles
    become a tree (trunk + squashed-sphere canopy) -- the scenario files
    already call them "tree"/"trunk" in their own comments (see
    config/forest.yaml) -- square obstacles stay a plain box (buildings/
    walls, not trees)."""
    import open3d as o3d
    cx, cy = obstacle["center"]
    if obstacle["shape"] == "circle":
        trunk_h, trunk_r, canopy_r, trunk_color, canopy_color = _tree_params(obstacle, height)

        trunk = o3d.geometry.TriangleMesh.create_cylinder(radius=trunk_r, height=trunk_h,
                                                            resolution=10)
        _assign_cylindrical_uv_o3d(trunk, trunk_h, v_tile=max(1.0, trunk_h / 0.6))
        trunk.translate((cx, cy, trunk_h / 2))
        trunk.compute_vertex_normals()

        # Three overlapping squashed-sphere lobes (jittered offsets/radii,
        # seeded off the same obstacle center as everything else here) --
        # one perfect ellipsoid read as a lollipop in a rendered frame;
        # a lumpy clump is the cheapest fix that doesn't need an external
        # mesh asset.
        canopy = None
        lobe_rng = np.random.default_rng(int(abs(hash(("canopy", round(cx, 4), round(cy, 4))))
                                              % (2**32)))
        base_z = trunk_h + canopy_r * 0.55
        for lobe_r_scale, offset_scale in ((1.0, 0.0), (0.68, 0.75), (0.6, 0.75)):
            lobe = o3d.geometry.TriangleMesh.create_sphere(
                radius=canopy_r * lobe_r_scale, resolution=12)
            verts = np.asarray(lobe.vertices)
            verts[:, 2] *= 0.78
            lobe.vertices = o3d.utility.Vector3dVector(verts)
            ang = lobe_rng.uniform(0, 2 * np.pi)
            off_r = canopy_r * offset_scale * lobe_rng.uniform(0.5, 1.0)
            lobe.translate((cx + np.cos(ang) * off_r, cy + np.sin(ang) * off_r,
                             base_z + lobe_rng.uniform(-0.1, 0.1) * canopy_r))
            canopy = lobe if canopy is None else canopy + lobe
        canopy.compute_vertex_normals()

        return [("trunk", trunk, [*trunk_color, 1.0]), ("canopy", canopy, [*canopy_color, 1.0])]

    h = obstacle["half_size"]
    box = o3d.geometry.TriangleMesh.create_box(width=2 * h, height=2 * h, depth=height)
    box.translate((cx - h, cy - h, 0))
    box.compute_vertex_normals()
    return [("box", box, [0.5, 0.5, 0.55, 1.0])]


def sky_gradient_image(width=64, height=64):
    """Vertical-gradient sky image (deep blue zenith -> pale horizon) for
    Open3D's set_background(color, image) -- cheap stand-in for a real
    skybox/IBL (Open3D's bundled .ktx IBL presets turned out to need a
    resource path this renderer couldn't resolve reliably across
    environments; a flat color read as flatly unconvincing, so this
    splits the difference without an external asset dependency)."""
    import open3d as o3d
    zenith = np.array([0.30, 0.50, 0.78])
    horizon = np.array([0.80, 0.86, 0.92])
    t = np.linspace(0.0, 1.0, height)[:, None]
    column = (zenith[None, :] * (1 - t) + horizon[None, :] * t)
    img = np.tile(column[:, None, :], (1, width, 1))
    return o3d.geometry.Image((img * 255).astype(np.uint8))


GROUND_TILE_M = 1.5  # texture repeat period [m] -- see procedural_grass_texture()


def procedural_grass_texture(size=256, seed=0):
    """Tileable-ish grass texture: layered value noise (no external asset
    -- avoids a download dependency and licensing questions for a
    generated demo asset). Not a real tiling texture (edges don't
    perfectly wrap), but at GROUND_TILE_M's repeat period the seam is
    not visually obvious once lit/shaded."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = np.array([0.20, 0.45, 0.16])
    img = np.zeros((size, size, 3))
    for octave, (scale, amp) in enumerate([(2, 0.22), (8, 0.12), (32, 0.06)]):
        noise = rng.normal(size=(size, size))
        noise = gaussian_filter(noise, sigma=size / (scale * 4))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
        img += (noise[..., None] - 0.5) * amp
    img = np.clip(base[None, None, :] + img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def procedural_bark_texture(width=64, height=256, seed=1):
    """Vertical-streak bark texture for tree trunks."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = np.array([0.36, 0.23, 0.12])
    noise = rng.normal(size=(height, width))
    noise = gaussian_filter(noise, sigma=(height / 6, width / 20))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    img = np.clip(base[None, None, :] + (noise[..., None] - 0.5) * 0.3, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _assign_cylindrical_uv_o3d(mesh, height, v_tile=1.0):
    """Per-vertex UVs for a cylinder centered at the origin (Open3D's
    create_cylinder convention: z in [-height/2, height/2]) -- u wraps
    around the trunk, v runs along its length, tiled v_tile times so
    procedural_bark_texture's portrait streaks don't stretch into one
    smear on a tall trunk. Must be called before any translate() (world-
    position offsets would otherwise leak into the angle)."""
    import open3d as o3d
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    u = (np.arctan2(verts[:, 1], verts[:, 0]) / (2 * np.pi)) + 0.5
    v = ((verts[:, 2] / height) + 0.5) * v_tile
    uv_per_vertex = np.stack([u, v], axis=1)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_per_vertex[tris.reshape(-1)])


def _assign_planar_uv_xy_o3d(mesh, x0, y0, tile_m):
    """Per-vertex UVs from world (x,y) position, repeated every tile_m
    meters (Filament/OpenGL default-wraps UVs > 1, so this tiles the
    texture across a large ground instead of stretching one image across
    it) -- only correct for a flat, axis-aligned mesh like the ground."""
    import open3d as o3d
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    uv_per_vertex = np.stack([(verts[:, 0] - x0) / tile_m, (verts[:, 1] - y0) / tile_m], axis=1)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_per_vertex[tris.reshape(-1)])


def build_ground_plane_o3d(bounds, textured=False):
    import open3d as o3d
    xmin, xmax, ymin, ymax = bounds
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=xmax - xmin, height=ymax - ymin, depth=0.01)
    mesh.translate((xmin, ymin, -0.01))
    mesh.compute_vertex_normals()
    if textured:
        _assign_planar_uv_xy_o3d(mesh, xmin, ymin, GROUND_TILE_M)
    return mesh


def grid_line_segments(bounds, spacing=1.0, z=0.01):
    """(N, 2, 3) array of (start, end) line segments for a reference grid
    over bounds, spaced every `spacing` meters -- RViz's signature
    ground-plane look."""
    xmin, xmax, ymin, ymax = bounds
    xs = np.arange(np.floor(xmin / spacing) * spacing, xmax + spacing, spacing)
    ys = np.arange(np.floor(ymin / spacing) * spacing, ymax + spacing, spacing)
    segs = [[[x, ymin, z], [x, ymax, z]] for x in xs]
    segs += [[[xmin, y, z], [xmax, y, z]] for y in ys]
    return np.array(segs)


def build_grid_lineset_o3d(bounds, spacing=1.0, z=0.01, color=(0.85, 0.85, 0.85)):
    import open3d as o3d
    segs = grid_line_segments(bounds, spacing, z)
    points = segs.reshape(-1, 3)
    lines = [[2 * i, 2 * i + 1] for i in range(len(segs))]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                               lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def build_grid_lines_pv(bounds, spacing=1.0, z=0.01):
    import pyvista as pv
    segs = grid_line_segments(bounds, spacing, z)
    return pv.MultiBlock([pv.Line(s[0], s[1]) for s in segs]).combine()


AXIS_COLORS = ((0.85, 0.1, 0.1), (0.15, 0.75, 0.15), (0.15, 0.35, 0.95))  # X, Y, Z -- RViz convention


def axis_gizmo_segments(length=0.4):
    """((3, 2, 3)) X/Y/Z axis line segments from the origin, in LOCAL
    frame -- a small TF-style coordinate frame, same red/green/blue
    convention RViz uses for its TF display."""
    return np.array([
        [[0, 0, 0], [length, 0, 0]],
        [[0, 0, 0], [0, length, 0]],
        [[0, 0, 0], [0, 0, length]],
    ])


def build_axis_gizmo_o3d(length=0.4):
    import open3d as o3d
    segs = axis_gizmo_segments(length)
    points = segs.reshape(-1, 3)
    lines = [[0, 1], [2, 3], [4, 5]]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                               lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(AXIS_COLORS))
    return ls


def build_axis_gizmo_pv(length=0.4):
    """Axis gizmo mesh with a per-line "colors" cell-data array (RGB
    uint8) -- pass scalars="colors", rgb=True to add_mesh for the X/Y/Z
    red/green/blue coloring."""
    import pyvista as pv
    segs = axis_gizmo_segments(length)
    lines = pv.MultiBlock([pv.Line(s[0], s[1]) for s in segs]).combine()
    lines.cell_data["colors"] = (np.array(AXIS_COLORS) * 255).astype(np.uint8)
    return lines


def build_person_parts_o3d(height=1.3):
    """List of (name, mesh, rgba) low-poly person parts (two legs, torso,
    two arms, head) in LOCAL FRAME (feet at z=0) -- Open3D counterpart of
    build_person_parts_pv(), used for the leader marker in rendered
    camera footage. height=1.3 (not a real human's ~1.75m) is a
    deliberate cosmetic shrink so the marker reads as smaller than the
    DEFAULT_OBSTACLE_HEIGHT_M=2.5 obstacles instead of nearly matching
    them.

    Separate legs/arms (not one fused leg-cylinder, the original shape --
    read as a chess pawn, not a person, in a rendered frame) is the
    cheapest silhouette improvement available without an external mesh
    asset: a human silhouette is mostly legible from limb separation
    alone, even at low-poly/flat-shaded."""
    import open3d as o3d
    leg_h = height * 0.45
    torso_h = height * 0.33
    head_r = height * 0.09
    leg_r = height * 0.045
    arm_r = height * 0.035
    arm_h = torso_h * 0.95
    leg_gap = height * 0.06
    shoulder_w = height * 0.14

    def cylinder(radius, h, center):
        m = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=10)
        m.translate(center)
        m.compute_vertex_normals()
        return m

    leg_color = [0.22, 0.22, 0.25, 1.0]
    arm_color = [0.85, 0.25, 0.1, 1.0]
    parts = [
        ("leg_l", cylinder(leg_r, leg_h, (-leg_gap, 0, leg_h / 2)), leg_color),
        ("leg_r", cylinder(leg_r, leg_h, (leg_gap, 0, leg_h / 2)), leg_color),
        ("torso", cylinder(height * 0.11, torso_h, (0, 0, leg_h + torso_h / 2)), arm_color),
        ("arm_l", cylinder(arm_r, arm_h, (-shoulder_w, 0, leg_h + torso_h - arm_h / 2)),
         arm_color),
        ("arm_r", cylinder(arm_r, arm_h, (shoulder_w, 0, leg_h + torso_h - arm_h / 2)),
         arm_color),
    ]

    head = o3d.geometry.TriangleMesh.create_sphere(radius=head_r, resolution=12)
    head.translate((0, 0, leg_h + torso_h + head_r))
    head.compute_vertex_normals()
    parts.append(("head", head, [0.92, 0.80, 0.68, 1.0]))

    return parts


# ── PyVista builders ─────────────────────────────────────────────────────

def build_obstacle_parts_pv(obstacle, height=DEFAULT_OBSTACLE_HEIGHT_M):
    """List of (mesh, rgb) parts -- PyVista counterpart of
    build_obstacle_parts_o3d(), same tree-for-circle/box-for-square split."""
    import pyvista as pv
    cx, cy = obstacle["center"]
    if obstacle["shape"] == "circle":
        trunk_h, trunk_r, canopy_r, trunk_color, canopy_color = _tree_params(obstacle, height)
        trunk = pv.Cylinder(center=(cx, cy, trunk_h / 2), direction=(0, 0, 1),
                             radius=trunk_r, height=trunk_h, resolution=10)
        canopy = pv.Sphere(radius=canopy_r, theta_resolution=14, phi_resolution=14)
        canopy.points[:, 2] *= 0.8
        canopy.translate((cx, cy, trunk_h + canopy_r * 0.6), inplace=True)
        return [(trunk, tuple(trunk_color)), (canopy, tuple(canopy_color))]

    h = obstacle["half_size"]
    box = pv.Cube(center=(cx, cy, height / 2), x_length=2 * h, y_length=2 * h, z_length=height)
    return [(box, (0.5, 0.5, 0.55))]


def build_drone_parts_pv(body_len):
    """List of (mesh, color, opacity) quadrotor parts in BODY FRAME
    (centered at the drone's own origin, nose along +x) -- caller poses
    every part with the same world transform each frame (one rigid body,
    same assumption a real quad's frame+props share)."""
    import pyvista as pv
    arm_len = body_len * 1.3
    arm_r = body_len * 0.06
    rotor_r = body_len * 0.55
    frame = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                         radius=body_len * 0.28, height=body_len * 0.25)
    rotors = []
    for ang_deg in (45, 135, 225, 315):
        ang = np.deg2rad(ang_deg)
        tip = np.array([np.cos(ang), np.sin(ang), 0.0]) * arm_len
        arm = pv.Cylinder(center=tip / 2, direction=tip / arm_len, radius=arm_r, height=arm_len)
        frame = frame.merge(arm)
        rotor = pv.Cylinder(center=tip + np.array([0, 0, body_len * 0.08]), direction=(0, 0, 1),
                             radius=rotor_r, height=body_len * 0.015)
        rotors.append(rotor)
    parts = [(frame, "royalblue", 1.0)]
    parts += [(rotor, "gainsboro", 0.4) for rotor in rotors]
    return parts


def build_person_parts_pv(height=1.3):
    """List of (mesh, color) low-poly person parts (legs/torso/head) in
    LOCAL FRAME (feet at z=0) -- the leader marker, translation-only
    (no attitude to track for a person on the ground). height=1.3, same
    cosmetic shrink as build_person_parts_o3d()."""
    import pyvista as pv
    leg_h = height * 0.45
    torso_h = height * 0.33
    head_r = height * 0.09
    legs = pv.Cylinder(center=(0, 0, leg_h / 2), direction=(0, 0, 1),
                        radius=height * 0.05, height=leg_h)
    torso = pv.Cylinder(center=(0, 0, leg_h + torso_h / 2), direction=(0, 0, 1),
                         radius=height * 0.11, height=torso_h)
    head = pv.Sphere(radius=head_r, center=(0, 0, leg_h + torso_h + head_r))
    return [(legs, "dimgray"), (torso, "orangered"), (head, "navajowhite")]


def build_ground_plane_pv(bounds, tile_m=None):
    """Flat ground plane. With tile_m given, its default [0,1]-spanning
    texture coordinates are rescaled to repeat the texture every tile_m
    meters instead of stretching one image across the whole plane (same
    idea as _assign_planar_uv_xy_o3d, PyVista's Plane already ships UVs
    so no per-vertex rebuild needed, just a scale)."""
    import pyvista as pv
    xmin, xmax, ymin, ymax = bounds
    plane = pv.Plane(center=((xmin + xmax) / 2, (ymin + ymax) / 2, 0), direction=(0, 0, 1),
                      i_size=xmax - xmin, j_size=ymax - ymin)
    if tile_m:
        plane.active_texture_coordinates *= [(xmax - xmin) / tile_m, (ymax - ymin) / tile_m]
    return plane


# ── Pose interpolation (for both playback and motion blur) ───────────────

def interpolate_pose(df, t):
    """Linear position + SLERP attitude between the two trajectory rows
    bracketing time t. Returns (pos: (3,) ndarray, quat: scipy Rotation).
    Clamped to the CSV's own time range."""
    t_arr = df["t"].to_numpy()
    t = min(max(t, t_arr[0]), t_arr[-1])
    k = np.searchsorted(t_arr, t, side="right") - 1
    k = min(max(k, 0), len(t_arr) - 2)
    t0, t1 = t_arr[k], t_arr[k + 1]
    frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

    row0, row1 = df.iloc[k], df.iloc[k + 1]
    pos0 = row0[["x", "y", "z"]].to_numpy(dtype=float)
    pos1 = row1[["x", "y", "z"]].to_numpy(dtype=float)
    pos = pos0 + frac * (pos1 - pos0)

    R0 = rotation_matrix_zyx(row0["phi"], row0["theta"], row0["psi"])
    R1 = rotation_matrix_zyx(row1["phi"], row1["theta"], row1["psi"])
    key_rots = Rotation.from_matrix(np.stack([R0, R1]))
    quat = Slerp([0.0, 1.0], key_rots)([frac])[0]
    return pos, quat


# ── own_map ESDF snapshot log (2-D panel) ─────────────────────────────────
#
# Same binary layout and auto-detection convention as animate_xy.py's
# load_snapshot_log()/find_snapshot_log() -- copied, not imported, same
# "keep each script standalone" pattern used throughout scripts/.

def find_snapshot_log(csv_path):
    """Auto-detect online_mapping_demo's snapshot log next to a trajectory
    CSV (logs/online_mapping_trajectory.csv -> logs/online_mapping_esdf_
    snapshots.bin). None for any other log (e.g. argus_mpc/trailer_mpc_demo's)."""
    if not os.path.basename(csv_path).startswith("online_mapping"):
        return None
    guess = os.path.join(os.path.dirname(csv_path) or ".",
                          "online_mapping_esdf_snapshots.bin")
    return guess if os.path.exists(guess) else None


def load_snapshot_log(path):
    """(times, distances[T,ny,nx], xs, ys, ground_truth_state[ny,nx],
    voxel_size, sensor_max_range, own_map_states[T,ny,nx]) -- own_map's
    ESDF AND per-voxel belief state (kUnknown=0/kFree=1/kOccupied=2,
    VoxelGrid.hpp) over time, plus the static ground-truth occupancy
    grid. own_map_states is what a live point-cloud display should plot
    (the sensor's own growing belief, not the ground truth it hasn't
    necessarily seen yet) -- animate_xy.py's loader discards this same
    per-tick state field; kept here since the point-cloud panel needs
    it."""
    with open(path, "rb") as f:
        nx, ny = struct.unpack("<ii", f.read(8))
        voxel_size, origin_x, origin_y = struct.unpack("<ddd", f.read(24))
        (sensor_max_range,) = struct.unpack("<d", f.read(8))
        n_voxels = nx * ny
        ground_truth_state = np.frombuffer(f.read(n_voxels), dtype=np.uint8).reshape(ny, nx)
        (num_snapshots,) = struct.unpack("<i", f.read(4))
        times = np.empty(num_snapshots)
        distances = np.empty((num_snapshots, ny, nx), dtype=np.float32)
        own_map_states = np.empty((num_snapshots, ny, nx), dtype=np.uint8)
        voxel_dtype = np.dtype([("distance", "<f4"), ("state", "u1")])
        for i in range(num_snapshots):
            (times[i],) = struct.unpack("<d", f.read(8))
            voxels = np.frombuffer(f.read(n_voxels * voxel_dtype.itemsize), dtype=voxel_dtype)
            distances[i] = voxels["distance"].reshape(ny, nx)
            own_map_states[i] = voxels["state"].reshape(ny, nx)
    xs = origin_x + (np.arange(nx) + 0.5) * voxel_size
    ys = origin_y + (np.arange(ny) + 0.5) * voxel_size
    return (times, distances, xs, ys, ground_truth_state, voxel_size, sensor_max_range,
            own_map_states)


K_OCCUPIED = 2  # argus_esdf::VoxelState::kOccupied, see include/argus_esdf/VoxelGrid.hpp


def occupied_points_at(own_map_states, idx, xs, ys, z=0.05, stride=2):
    """World (x, y, z) points for own_map's kOccupied voxels at snapshot
    idx -- a live, growing point cloud of what the sensor has actually
    mapped so far (not the ground truth), same idea as RViz's
    PointCloud2/occupancy-grid lethal-cell display. stride downsamples
    (every Nth voxel per axis) -- the full grid is too many points to
    render smoothly per frame."""
    state = own_map_states[idx, ::stride, ::stride]
    iy, ix = np.nonzero(state == K_OCCUPIED)
    if len(iy) == 0:
        return np.zeros((0, 3))
    pts_x = xs[::stride][ix]
    pts_y = ys[::stride][iy]
    return np.stack([pts_x, pts_y, np.full_like(pts_x, z)], axis=1)


def esdf_to_rgb(distance, max_distance):
    """Red(close)->blue(far) hue sweep, same as animate_xy.py/
    plot_online_mapping.py's esdf_to_rgb() -- returns a float [0,1] RGB
    array, caller converts to uint8."""
    from matplotlib.colors import hsv_to_rgb
    t = np.clip(distance / max_distance, 0.0, 1.0)
    hue = t * (240.0 / 360.0)
    hsv = np.stack([hue, np.full_like(t, 0.55), np.ones_like(t)], axis=-1)
    return hsv_to_rgb(hsv)


def esdf_panel_image(distance, ground_truth_state, sensor_max_range):
    """uint8 RGB image for the 2-D ESDF panel: esdf_to_rgb's heatmap with
    every ground-truth-occupied cell overlaid in grey (regardless of
    whether own_map has observed it yet), same composition as
    plot_online_mapping.py's 2-D panel. Row 0 = ys[0] (south); flipped by
    the caller if the texture needs north-up."""
    rgb = esdf_to_rgb(distance, sensor_max_range)
    occupied = ground_truth_state == K_OCCUPIED
    rgb[occupied] = [0.4, 0.4, 0.4]
    return (rgb * 255).astype(np.uint8)


def _o3d_cylinder_between(p0, p1, radius, resolution=10):
    """Open3D cylinder spanning two points -- create_cylinder() is always
    local-z-aligned, so this rotates it onto the p0->p1 direction
    (Rodrigues' formula) before translating to the midpoint."""
    import open3d as o3d
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    vec = p1 - p0
    h = np.linalg.norm(vec)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution)
    z, v = np.array([0.0, 0.0, 1.0]), vec / h
    axis, c = np.cross(z, v), np.dot(z, v)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        axis = axis / s
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + K * s + K @ K * (1 - c)
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((p0 + p1) / 2)
    cyl.compute_vertex_normals()
    return cyl


def build_drone_parts_o3d(body_len=0.35):
    """List of (name, mesh, rgba) quadrotor parts in BODY FRAME (nose
    along +x) -- Open3D counterpart of build_drone_parts_pv(), for a
    chase-cam / third-person render (the drone's own camera footage never
    shows this -- a camera can't see its own airframe)."""
    import open3d as o3d
    arm_len = body_len * 1.3
    frame = o3d.geometry.TriangleMesh.create_cylinder(radius=body_len * 0.28,
                                                        height=body_len * 0.25, resolution=16)
    frame.compute_vertex_normals()
    parts = [("frame", frame, [0.25, 0.41, 0.88, 1.0])]
    for i, ang_deg in enumerate((45, 135, 225, 315)):
        ang = np.deg2rad(ang_deg)
        tip = np.array([np.cos(ang), np.sin(ang), 0.0]) * arm_len
        arm = _o3d_cylinder_between((0, 0, 0), tip, body_len * 0.06)
        parts.append((f"arm{i}", arm, [0.25, 0.41, 0.88, 1.0]))
        rotor = o3d.geometry.TriangleMesh.create_cylinder(radius=body_len * 0.55,
                                                            height=body_len * 0.015, resolution=20)
        rotor.translate(tip + np.array([0, 0, body_len * 0.08]))
        rotor.compute_vertex_normals()
        parts.append((f"rotor{i}", rotor, [0.85, 0.85, 0.85, 0.55]))
    return parts


def apply_post_process(frame, vignette_strength=0.35, saturation=1.12, contrast=1.06,
                        vignette_tint=None):
    """Light cinematic grade -- a touch more saturation/contrast and a
    soft vignette -- used by both render_hero_video.py and (by default)
    render_camera_footage.py's standalone output, since that footage now
    doubles as a portfolio clip, not just a diagnostic one. Pass
    --no-post-process to either script for the raw, unstylized frames.

    vignette_tint: optional (r,g,b) in 0..1, e.g. (0.9,0.2,0.2) for an OFF/
    ON comparison pair -- the vignette falloff blends toward this color
    instead of just darkening, so the two panels read as "red side" /
    "green side" even at a glance, muted, scrolling past in a feed."""
    from PIL import Image, ImageEnhance
    img = Image.fromarray(frame)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    falloff = np.clip(r - 0.6, 0.0, 1.0) ** 2
    if vignette_tint is not None:
        tint = np.array(vignette_tint, dtype=np.float32) * 255.0
        blend = np.clip(vignette_strength * falloff, 0.0, 1.0)[..., None]
        arr = arr * (1.0 - blend) + tint[None, None, :] * blend
    else:
        mask = 1.0 - vignette_strength * falloff
        arr *= mask[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def interpolate_point(df, t, cols):
    """Linear interpolation of arbitrary CSV columns (e.g. ("fx","fy","fz"))
    between the two rows bracketing time t -- same bracketing logic as
    interpolate_pose(), for data that has no attitude to SLERP (the
    leader's position)."""
    t_arr = df["t"].to_numpy()
    t = min(max(t, t_arr[0]), t_arr[-1])
    k = np.searchsorted(t_arr, t, side="right") - 1
    k = min(max(k, 0), len(t_arr) - 2)
    t0, t1 = t_arr[k], t_arr[k + 1]
    frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    p0 = df.iloc[k][list(cols)].to_numpy(dtype=float)
    p1 = df.iloc[k + 1][list(cols)].to_numpy(dtype=float)
    return p0 + frac * (p1 - p0)


def camera_extrinsics(pos, quat):
    """(eye, center, up) triple for Open3D's setup_camera(), from drone
    pose. Camera boresight = body +x (matches the r_fov cost / FOV cone
    convention above); no rig offset -- eye is the drone's own position."""
    R = quat.as_matrix()
    forward = R @ np.array([1.0, 0.0, 0.0])
    up = R @ np.array([0.0, 0.0, 1.0])
    eye = pos
    center = pos + forward
    return eye, center, up
