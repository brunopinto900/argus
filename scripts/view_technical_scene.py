"""
Clean "scientific instrument" 3-D view of a logged trajectory: white
background, a thin grey reference grid, own_map's occupied voxels drawn as
small flat-colored cubes (the sensor's actual occupancy belief, not a
cinematic tree/grass render), the camera frustum wireframe, and both the
drone's and target's full paths as 3-D lines -- the EVA-Tracker-style
schematic look, not view_scene3d.py's game-engine one. Static figures and
an interactive/scrubbable viewer share this one script, same as
view_scene3d.py's --animate/--off-screen split.

Shares scene3d.py with the other two viewers (same trajectory CSV, same
scenario YAML, same pose-interpolation/frustum math) -- only the palette,
ground, and marker representation differ. No obstacle meshes are drawn
from the scenario file; the voxel cloud from own_map's own belief is the
obstacle representation here, on purpose (own_map is 2-D/single-z-layer,
so "voxels" are rendered as a thin cube layer at the ground, not a true
3-D occupancy -- an honest stylization, not fabricated height data).

Static figure (paper/portfolio):
    python3 scripts/view_technical_scene.py --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml --frame 160 --off-screen \\
        --save-png paper/figures/technical_view.png

Interactive, scrubbable (needs a display):
    python3 scripts/view_technical_scene.py --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml --animate

Headless animated export:
    python3 scripts/view_technical_scene.py --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml --save-gif logs/technical_view.gif
"""

import argparse

import numpy as np
import pyvista as pv

import contact_events
import scene3d

DRONE_BODY_LEN = 0.30
FRUSTUM_LEN = 1.2
TARGET_MARKER_R = 0.14

GROUND_COLOR = "#f5f5f3"
GRID_COLOR = "#9a9a9a"
VOXEL_COLOR = "#7d828a"
DRONE_COLOR = "#20315c"
DRONE_ROTOR_COLOR = "#4a5a85"
TARGET_COLOR = "#b23a2e"
DRONE_PATH_COLOR = "#2b4c8c"
TARGET_PATH_COLOR = "#c0564a"
FRUSTUM_COLOR = "#2c2c2c"
BG_COLOR = "white"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", default="logs/online_mapping_trajectory.csv")
    p.add_argument("--scenario-config", default="config/forest.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--voxel-stride", type=int, default=2,
                   help="Downsample own_map's occupied-voxel cloud (every Nth cell per axis)")
    p.add_argument("--snapshot-path", default=None,
                   help="Explicit ESDF snapshot log path, overriding auto-detection -- "
                        "needed when --csv-path doesn't follow online_mapping_demo's own "
                        "'online_mapping_trajectory.csv' / 'online_mapping_esdf_snapshots.bin' "
                        "naming (e.g. logs/camera_footage/<run>/{trajectory,esdf_snapshots}.csv)")
    p.add_argument("--frame", type=int, default=None,
                   help="Static frame index to show/save (default: midpoint, or --animate)")
    p.add_argument("--animate", action="store_true", help="Scrub slider + live playback")
    p.add_argument("--fps", type=float, default=None,
                   help="Playback speed for space-to-play (default: real-time, from the "
                        "CSV's own tick rate)")
    p.add_argument("--off-screen", action="store_true")
    p.add_argument("--save-png", default=None)
    p.add_argument("--save-gif", default=None,
                   help="Headless animated export over the whole run (implies off-screen)")
    p.add_argument("--gif-stride", type=int, default=2, help="Frame stride for --save-gif")
    p.add_argument("--gif-fps", type=float, default=20.0)
    p.add_argument("--sightline", action="store_true",
                   help="Draw the drone-to-target sightline each frame (the occ_p0(s) "
                        "segment from the occlusion cost, Section III-D of the paper)")
    p.add_argument("--zoom", type=float, default=1.6,
                   help="Camera zoom multiplier (default 1.6; use higher for a tight, "
                        "single-occluder close-up)")
    p.add_argument("--no-legend", action="store_true")
    p.add_argument("--label-events", action="store_true",
                   help="Mark target-lost events (out-of-FOV-cone vs. occluded, "
                        "scripts/contact_events.py) along the drone path")
    p.add_argument("--d-fov", type=float, default=1.68)
    p.add_argument("--lost-contact-occlusion-pct", type=float, default=90.0)
    return p.parse_args()


def _pose_matrix(pos, quat):
    m = np.eye(4)
    m[:3, :3] = quat.as_matrix()
    m[:3, 3] = pos
    return m


VOXEL_DISPLAY_SIZE_M = 0.22  # own_map's real voxel_size (5cm) renders as an
# invisible speck at normal camera distance; this is a deliberate display-only
# enlargement (documented here, not silently baked into the data) so a sparse
# surface-voxel cloud still reads as "occupied region" the way the EVA-Tracker-
# style figures this is modeled on do.


def build_voxel_cloud_pv(plotter, pts, voxel_size):
    """Occupied-voxel point cloud as small flat-colored cube glyphs (not
    spheres -- "voxel", not "point cloud", is the look being asked for)."""
    if len(pts) == 0:
        return None
    cloud = pv.PolyData(pts)
    size = max(voxel_size, VOXEL_DISPLAY_SIZE_M)
    cube = pv.Cube(x_length=size, y_length=size, z_length=size * 0.7)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
    return plotter.add_mesh(glyphs, color=VOXEL_COLOR, opacity=1.0,
                             show_edges=True, edge_color="white", line_width=0.75,
                             lighting=True, specular=0.0)


def build_trajectory_line_pv(plotter, points, color, width=4):
    poly = pv.lines_from_points(points)
    return plotter.add_mesh(poly, color=color, line_width=width)


def build_static_scene(plotter, bounds):
    ground = scene3d.build_ground_plane_pv(bounds)
    plotter.add_mesh(ground, color=GROUND_COLOR, lighting=False, specular=0.0,
                      ambient=1.0)
    grid = scene3d.build_grid_lines_pv(bounds)
    plotter.add_mesh(grid, color=GRID_COLOR, opacity=0.85, line_width=1.2)


def build_dynamic_actors(plotter):
    drone_actors = []
    for mesh, _color, opacity in scene3d.build_drone_parts_pv(DRONE_BODY_LEN):
        drone_actors.append(plotter.add_mesh(mesh, color=DRONE_COLOR, opacity=1.0))
    target_actor = plotter.add_mesh(pv.Sphere(radius=TARGET_MARKER_R), color=TARGET_COLOR)
    return drone_actors, target_actor


def init_frustum_actor(plotter):
    placeholder = np.zeros((8, 2, 3))
    placeholder[:, 1, 0] = 0.01
    lines = pv.MultiBlock([pv.Line(s[0], s[1]) for s in placeholder]).combine()
    return plotter.add_mesh(lines, color=FRUSTUM_COLOR, line_width=1.5, opacity=0.8)


def update_frustum(actor, pos, quat, h_fov, v_fov):
    R = quat.as_matrix()
    segs = scene3d.frustum_edge_segs(pos, R, FRUSTUM_LEN, h_fov, v_fov)
    actor.mapper.dataset.points = segs.reshape(-1, 3)


FOV_LOST_COLOR = "#8e44ad"
OCC_LOST_COLOR = "#16a085"
EVENT_MARKER_R = 0.16
EVENT_TARGET_MARKER_SIZE = 0.22  # cube edge length -- a visibly different
# glyph (cube vs. the drone-position sphere) so the two event-marker sets
# stay distinguishable without doubling the legend.


def add_event_markers_pv(plotter, df, fov_half_angle_deg, d_fov_m,
                          lost_contact_occlusion_pct=90.0):
    """Static markers wherever contact_events marks the target lost --
    out-of-FOV-cone (purple) vs. occluded (teal), scripts/contact_events.py's
    split of Table 1's single merged "Reacq." count. Colors are deliberately
    far from TARGET_COLOR/TARGET_PATH_COLOR (both red) so an event marker is
    never mistaken for the target itself. Drawn at both the drone's position
    (sphere) and the target's own position (cube) at the same timesteps, so
    the event is visible on both paths, not just the drone's -- drawn once
    over the whole run, like the trajectory lines, not per-frame, so an
    event stays visible throughout playback."""
    m = contact_events.compute_contact_masks(df, fov_half_angle_deg, d_fov_m,
                                               lost_contact_occlusion_pct)
    occ_only = m["occlusion_lost"] & ~m["out_of_fov"]
    drone_pos = df[["x", "y", "z"]].to_numpy()
    target_pos = df[["fx", "fy", "fz"]].to_numpy()
    actors = []
    for mask, color, label in ((m["out_of_fov"], FOV_LOST_COLOR, "Contact lost: out of FOV cone"),
                                (occ_only, OCC_LOST_COLOR, "Contact lost: occluded")):
        if not mask.any():
            continue
        drone_cloud = pv.PolyData(drone_pos[mask])
        drone_glyphs = drone_cloud.glyph(geom=pv.Sphere(radius=EVENT_MARKER_R),
                                          scale=False, orient=False)
        actor = plotter.add_mesh(drone_glyphs, color=color, opacity=0.9)
        target_cloud = pv.PolyData(target_pos[mask])
        target_cube = pv.Cube(x_length=EVENT_TARGET_MARKER_SIZE, y_length=EVENT_TARGET_MARKER_SIZE,
                               z_length=EVENT_TARGET_MARKER_SIZE)
        target_glyphs = target_cloud.glyph(geom=target_cube, scale=False, orient=False)
        plotter.add_mesh(target_glyphs, color=color, opacity=0.9,
                          show_edges=True, edge_color="white", line_width=0.75)
        actors.append((actor, label, color))
    return actors


SIGHTLINE_COLOR = "#d98832"  # matches occ_geom.png's d(s)/occ_p0(s) orange


def init_sightline_actor(plotter):
    """The drone-to-target segment occ_p0(s) is linearized along (Section
    III-D of the paper) -- degenerate placeholder, same fixed-topology
    pattern as init_frustum_actor()."""
    line = pv.Line((0, 0, 0), (0.01, 0, 0))
    return plotter.add_mesh(line, color=SIGHTLINE_COLOR, line_width=2.5)


def update_sightline(actor, pos, target_pos):
    actor.mapper.dataset.points = np.array([pos, target_pos])


def render_frame(df, step, drone_actors, target_actor, frustum_actor, h_fov, v_fov,
                  sightline_actor=None):
    t = df["t"].to_numpy()[step]
    pos, quat = scene3d.interpolate_pose(df, t)
    drone_matrix = _pose_matrix(pos, quat)
    for actor in drone_actors:
        actor.user_matrix = drone_matrix
    target_pos = df.iloc[step][["fx", "fy", "fz"]].to_numpy(dtype=float)
    target_matrix = np.eye(4)
    target_matrix[:3, 3] = target_pos
    target_actor.user_matrix = target_matrix
    update_frustum(frustum_actor, pos, quat, h_fov, v_fov)
    if sightline_actor is not None:
        update_sightline(sightline_actor, pos, target_pos)
    return pos, target_pos


def main():
    args = _parse_args()

    camera_cfg, _ = scene3d.load_camera_config(args.config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    h_fov = v_fov = np.deg2rad(fov_half_deg)

    df = scene3d.load_trajectory(args.csv_path)
    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    bounds = scene3d.scene_bounds(df, obstacles)
    n_frames = len(df)
    t_arr = df["t"].to_numpy()

    snapshot_path = args.snapshot_path or scene3d.find_snapshot_log(args.csv_path)
    esdf_data = scene3d.load_snapshot_log(snapshot_path) if snapshot_path else None

    off_screen = args.off_screen or bool(args.save_png) or bool(args.save_gif)
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1280, 800))
    plotter.set_background(BG_COLOR)

    build_static_scene(plotter, bounds)
    build_trajectory_line_pv(plotter, df[["x", "y", "z"]].to_numpy(), DRONE_PATH_COLOR)
    build_trajectory_line_pv(plotter, df[["fx", "fy", "fz"]].to_numpy(), TARGET_PATH_COLOR)

    if esdf_data is not None:
        (esdf_times, _esdf_distances, esdf_xs, esdf_ys, _ground_truth_state, voxel_size,
         _sensor_max_range, own_map_states) = esdf_data

        def _voxels_at(t):
            idx = int(np.argmin(np.abs(esdf_times - t)))
            return scene3d.occupied_points_at(own_map_states, idx, esdf_xs, esdf_ys,
                                               z=voxel_size * 0.3, stride=args.voxel_stride)
    else:
        voxel_size = 0.1

        def _voxels_at(_t):
            return np.zeros((0, 3))

    voxel_actor = [None]

    def _update_voxels(t):
        if voxel_actor[0] is not None:
            plotter.remove_actor(voxel_actor[0])
            voxel_actor[0] = None
        pts = _voxels_at(t)
        voxel_actor[0] = build_voxel_cloud_pv(plotter, pts, voxel_size)

    drone_actors, target_actor = build_dynamic_actors(plotter)
    frustum_actor = init_frustum_actor(plotter)
    sightline_actor = init_sightline_actor(plotter) if args.sightline else None

    event_actors = []
    if args.label_events:
        event_actors = add_event_markers_pv(plotter, df, fov_half_deg, args.d_fov,
                                              args.lost_contact_occlusion_pct)

    if not args.no_legend:
        legend_labels = [["Drone path", DRONE_PATH_COLOR], ["Target path", TARGET_PATH_COLOR],
                          ["Drone", DRONE_COLOR], ["Target", TARGET_COLOR],
                          ["Occupied voxel (own_map)", VOXEL_COLOR]]
        if args.sightline:
            legend_labels.append(["Sightline occ_p0(s)", SIGHTLINE_COLOR])
        for _actor, label, color in event_actors:
            legend_labels.append([label, color])
        plotter.add_legend(labels=legend_labels, bcolor="white", face="rectangle",
                            size=(0.26, 0.18), loc="lower left")

    plotter.camera.azimuth = 200
    plotter.camera.elevation = 25
    plotter.camera.zoom(args.zoom)

    def _update_all(step):
        t = t_arr[step]
        render_frame(df, step, drone_actors, target_actor, frustum_actor, h_fov, v_fov,
                      sightline_actor)
        _update_voxels(t)

    if args.save_gif:
        plotter.open_gif(args.save_gif, fps=args.gif_fps)
        steps = range(0, n_frames, max(1, args.gif_stride))
        _update_all(0)
        plotter.render()
        for step in steps:
            _update_all(step)
            plotter.write_frame()
        plotter.close()
        print(f"Saved {args.save_gif} ({len(list(steps))} frames)")
        return

    if args.animate and not off_screen:
        slider = plotter.add_slider_widget(lambda v: (_update_all(int(v)), plotter.render()),
                                            [0, n_frames - 1], value=0, title="step")
        status_suffix = ""

        def _status_text():
            return "space: play/pause   1/2/3: chase/top/side view" + status_suffix

        plotter.add_text(_status_text(), position="upper_right", font_size=10, name="status")
        _update_all(0)

        dt_ms = max(1000.0 / args.fps, 15.0) if args.fps else \
            max(float(t_arr[1] - t_arr[0]) * 1000.0, 15.0)
        state = {"playing": False, "step": 0}

        def _toggle_play():
            nonlocal status_suffix
            state["playing"] = not state["playing"]
            status_suffix = "  [PLAYING]" if state["playing"] else "  [PAUSED]"
            plotter.add_text(_status_text(), position="upper_right", font_size=10, name="status")

        def _on_tick(_step_count):
            if not state["playing"]:
                return
            state["step"] = (state["step"] + 1) % n_frames
            _update_all(state["step"])
            slider.GetRepresentation().SetValue(state["step"])
            plotter.render()

        def _set_view(mode):
            if mode == "top":
                plotter.view_xy()
            elif mode == "side":
                plotter.view_yz()
            else:
                plotter.camera.azimuth = 200
                plotter.camera.elevation = 25
                plotter.reset_camera()
            plotter.render()

        plotter.add_key_event("space", _toggle_play)
        plotter.add_key_event("1", lambda: _set_view("chase"))
        plotter.add_key_event("2", lambda: _set_view("top"))
        plotter.add_key_event("3", lambda: _set_view("side"))
        plotter.add_timer_event(max_steps=10_000_000, duration=int(dt_ms), callback=_on_tick)
        plotter.show()
        return

    step = args.frame if args.frame is not None else n_frames // 2
    _update_all(step)

    if args.save_png:
        plotter.show(screenshot=args.save_png, auto_close=False)
        print(f"Saved {args.save_png} (step {step}/{n_frames})")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
