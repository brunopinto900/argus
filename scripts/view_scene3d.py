"""
Interactive/scrubbable 3-D playback of a logged trajectory CSV against
its scenario's obstacle layout (PyVista, no Gazebo) -- ground plane,
extruded obstacles, a leader marker, and an orientation-legible drone
marker + its forward camera FOV cone, all driven off the same trajectory
CSV scripts/render_camera_footage.py reads.

Own_map's actual 2-D ESDF (same red-close/blue-far heatmap + grey
ground-truth overlay as animate_xy.py's right panel) colors the 3-D
ground plane itself -- not a separate flat top-down panel -- auto-
detected next to --csv-path, falls back to a flat grey ground if there's
no snapshot log. A second, optional panel stacked above it shows live
camera footage: the drone's own forward view, rendered each frame via
render_camera_footage.py's Open3D scene (single-sample, no motion blur
-- that's render_camera_footage.py's job for an offline, frame-by-frame
look; this panel trades blur for interactive frame rate). Disable with
--no-camera-panel, leaving just the 3-D view full-window.

Both panels share one slider; space toggles real-time auto-play.

Run (interactive window, needs a display):
    python3 scripts/view_scene3d.py --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml --animate

Headless, single frame to PNG (no display required):
    python3 scripts/view_scene3d.py --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml --frame 110 --off-screen \\
        --save-png logs/scene3d_check.png
"""

import argparse

import numpy as np
import open3d as o3d
import pyvista as pv

import render_camera_footage as camfeed
import scene3d

DRONE_BODY_LEN = 0.35
FRUSTUM_LEN = 1.0


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", default="logs/online_mapping_trajectory.csv")
    p.add_argument("--scenario-config", default="config/forest.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--obstacle-height", type=float, default=scene3d.DEFAULT_OBSTACLE_HEIGHT_M)
    p.add_argument("--horizon-csv", default="logs/online_mapping_horizon.csv",
                   help="Optional predicted-horizon overlay; silently skipped if missing")
    p.add_argument("--frame", type=int, default=None,
                   help="Static frame index to show/save (default: midpoint, or --animate)")
    p.add_argument("--animate", action="store_true", help="Scrub slider + live playback")
    p.add_argument("--fps", type=float, default=None,
                   help="Playback speed for space-to-play (default: real-time, from the "
                        "CSV's own tick rate)")
    p.add_argument("--no-camera-panel", action="store_true",
                   help="Skip the live camera-footage panel (faster interactive playback)")
    p.add_argument("--camera-width", type=int, default=320,
                   help="Half the OAK-D-Lite's native 640x400 stereo-depth resolution "
                        "(render_camera_footage.py's default) -- smaller here for "
                        "interactive frame rate")
    p.add_argument("--camera-height", type=int, default=200)
    p.add_argument("--off-screen", action="store_true")
    p.add_argument("--save-png", default=None)
    return p.parse_args()


def _pose_matrix(pos, quat):
    m = np.eye(4)
    m[:3, :3] = quat.as_matrix()
    m[:3, 3] = pos
    return m


def build_static_scene(plotter, obstacles, bounds, obstacle_height, esdf_bounds=None):
    """Ground + obstacles. When esdf_bounds is given (own_map's own grid
    extent, not the trajectory-fitted `bounds`), the ground is sized to
    match it and returned as a textured actor the caller re-textures with
    the live ESDF heatmap each frame -- this replaces the old separate
    flat top-down ESDF panel: the same coloring, baked onto the actual
    ground the obstacles stand on, viewed from the same oblique 3-D
    camera as everything else instead of a disconnected second view."""
    if esdf_bounds is not None:
        ground = scene3d.build_ground_plane_pv(esdf_bounds)
        blank = pv.Texture(np.full((2, 2, 3), 110, dtype=np.uint8))
        ground_actor = plotter.add_mesh(ground, texture=blank)
        grid_bounds = esdf_bounds
    else:
        ground = scene3d.build_ground_plane_pv(bounds, tile_m=scene3d.GROUND_TILE_M)
        grass = pv.Texture(scene3d.procedural_grass_texture())
        ground_actor = plotter.add_mesh(ground, texture=grass)
        grid_bounds = bounds
    grid = scene3d.build_grid_lines_pv(grid_bounds)
    plotter.add_mesh(grid, color="white", opacity=0.3, line_width=1)
    for obstacle in obstacles:
        for mesh, color in scene3d.build_obstacle_parts_pv(obstacle, obstacle_height):
            plotter.add_mesh(mesh, color=color)
    return ground_actor


def update_ground_esdf(ground_actor, esdf_times, esdf_distances, ground_truth_state,
                        sensor_max_range, t):
    idx = int(np.argmin(np.abs(esdf_times - t)))
    img = scene3d.esdf_panel_image(esdf_distances[idx], ground_truth_state, sensor_max_range)
    ground_actor.texture = pv.Texture(np.flipud(img))


def build_dynamic_actors(plotter):
    drone_actors = [plotter.add_mesh(mesh, color=color, opacity=opacity)
                     for mesh, color, opacity in scene3d.build_drone_parts_pv(DRONE_BODY_LEN)]
    leader_actors = [plotter.add_mesh(mesh, color=color)
                      for mesh, color in scene3d.build_person_parts_pv()]
    # TF-style coordinate-frame gizmos (RViz convention: red/green/blue = X/Y/Z),
    # posed by the same rigid-body transform as the rest of each actor group.
    drone_actors.append(plotter.add_mesh(scene3d.build_axis_gizmo_pv(0.5), scalars="colors",
                                          rgb=True, line_width=3))
    leader_actors.append(plotter.add_mesh(scene3d.build_axis_gizmo_pv(0.4), scalars="colors",
                                           rgb=True, line_width=3))
    return drone_actors, leader_actors


def init_frustum_actor(plotter):
    """Fixed-topology (8 lines, 16 points) placeholder -- update_frustum()
    only ever overwrites point coordinates in place, never removes/re-adds
    this actor, so there's no one-frame gap where it's absent (that gap
    was the FOV panel's flicker during playback)."""
    placeholder = np.zeros((8, 2, 3))
    placeholder[:, 1, 0] = 0.01  # degenerate but distinct start/end points
    lines = pv.MultiBlock([pv.Line(s[0], s[1]) for s in placeholder]).combine()
    return plotter.add_mesh(lines, color="gold", line_width=2)


def update_frustum(actor, pos, quat, h_fov, v_fov):
    R = quat.as_matrix()
    segs = scene3d.frustum_edge_segs(pos, R, FRUSTUM_LEN, h_fov, v_fov)  # (8, 2, 3)
    actor.mapper.dataset.points = segs.reshape(-1, 3)


def render_frame(df, step, drone_actors, leader_actors, frustum_actor, h_fov, v_fov):
    t = df["t"].to_numpy()[step]
    pos, quat = scene3d.interpolate_pose(df, t)
    drone_matrix = _pose_matrix(pos, quat)
    for actor in drone_actors:
        actor.user_matrix = drone_matrix
    leader_pos = df.iloc[step][["fx", "fy", "fz"]].to_numpy(dtype=float)
    leader_matrix = np.eye(4)
    leader_matrix[:3, 3] = leader_pos
    for actor in leader_actors:
        actor.user_matrix = leader_matrix
    update_frustum(frustum_actor, pos, quat, h_fov, v_fov)
    return pos, quat, leader_pos


def build_camera_panel(plotter, obstacles, bounds, obstacle_height, width, height):
    """Live forward-camera view: a persistent Open3D OffscreenRenderer
    (built once, reused every frame -- same reused-renderer pattern
    render_camera_footage.py uses for its batch export) whose rendered
    RGB frame is pushed into a flat textured plane viewed by its own
    orthographic PyVista subplot camera, same trick as the ESDF panel.
    Single-sample (no --blur-subsamples averaging) -- this is the
    interactive, real-time-paced preview; render_camera_footage.py is the
    tool for an offline, properly motion-blurred look."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    leader_name = camfeed.build_scene(renderer, obstacles, bounds, obstacle_height, width, height)
    renderer.scene.view.set_ambient_occlusion(True)

    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1.0, j_size=height / width)
    blank = pv.Texture(np.zeros((2, 2, 3), dtype=np.uint8))
    plane_actor = plotter.add_mesh(plane, texture=blank)
    plotter.camera_position = "xy"
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    return renderer, leader_name, plane_actor


def update_camera_panel(renderer, leader_name, plane_actor, fov_half_deg, pos, quat, leader_pos):
    camfeed.update_leader_position(renderer, leader_name, leader_pos)
    eye, center, up = scene3d.camera_extrinsics(pos, quat)
    renderer.setup_camera(2 * fov_half_deg, center, eye, up)
    img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    plane_actor.texture = pv.Texture(img)


def main():
    args = _parse_args()

    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    camera_cfg, _ = scene3d.load_camera_config(args.config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    h_fov = v_fov = np.deg2rad(fov_half_deg)

    df = scene3d.load_trajectory(args.csv_path)
    bounds = scene3d.scene_bounds(df, obstacles)
    n_frames = len(df)
    t_arr = df["t"].to_numpy()

    try:
        horizon_df = scene3d.load_trajectory(args.horizon_csv)
    except FileNotFoundError:
        horizon_df = None

    snapshot_path = scene3d.find_snapshot_log(args.csv_path)
    esdf_data = scene3d.load_snapshot_log(snapshot_path) if snapshot_path else None
    show_camera_panel = not args.no_camera_panel

    # Camera footage (top, wide) stacked over the 3-D view (bottom, wide)
    # -- a vertical 2x1 grid, not the old side-by-side 1x3 (3-D/ESDF/
    # camera) layout; the ESDF panel is gone as a separate view, folded
    # into the 3-D ground's own texture instead (see build_static_scene).
    off_screen = args.off_screen or bool(args.save_png)
    if show_camera_panel:
        plotter = pv.Plotter(shape=(2, 1), off_screen=off_screen)
        camera_subplot, three_d_subplot = (0, 0), (1, 0)
    else:
        plotter = pv.Plotter(off_screen=off_screen)
        camera_subplot, three_d_subplot = None, (0, 0)

    plotter.subplot(*three_d_subplot)
    plotter.set_background("lightsteelblue")
    esdf_bounds = None
    if esdf_data is not None:
        (esdf_times, esdf_distances, esdf_xs, esdf_ys, ground_truth_state, _, sensor_max_range,
         own_map_states) = esdf_data
        esdf_bounds = (esdf_xs[0], esdf_xs[-1], esdf_ys[0], esdf_ys[-1])
    ground_actor = build_static_scene(plotter, obstacles, bounds, args.obstacle_height,
                                       esdf_bounds)
    drone_actors, leader_actors = build_dynamic_actors(plotter)
    frustum_actor = init_frustum_actor(plotter)
    points_actor = [None]  # own_map's live occupied-voxel point cloud (RViz-style),
                            # rebuilt each frame since its point count grows over time

    if horizon_df is not None:
        plotter.add_mesh(pv.PolyData(horizon_df[["hx", "hy", "hz"]].to_numpy()),
                          color="lime", point_size=3, render_points_as_spheres=True,
                          opacity=0.25)

    plotter.camera.azimuth = 200
    plotter.camera.elevation = 20

    camera_panel = None
    if show_camera_panel:
        plotter.subplot(*camera_subplot)
        renderer, leader_name, cam_plane_actor = build_camera_panel(
            plotter, obstacles, bounds, args.obstacle_height, args.camera_width,
            args.camera_height)
        camera_panel = (renderer, leader_name, cam_plane_actor)

    def _update_all(step):
        plotter.subplot(*three_d_subplot)
        t = t_arr[step]
        pos, quat, leader_pos = render_frame(df, step, drone_actors, leader_actors,
                                              frustum_actor, h_fov, v_fov)
        if esdf_data is not None:
            update_ground_esdf(ground_actor, esdf_times, esdf_distances, ground_truth_state,
                                sensor_max_range, t)
            idx = int(np.argmin(np.abs(esdf_times - t)))
            pts = scene3d.occupied_points_at(own_map_states, idx, esdf_xs, esdf_ys)
            if points_actor[0] is not None:
                plotter.remove_actor(points_actor[0])
            if len(pts):
                points_actor[0] = plotter.add_mesh(
                    pv.PolyData(pts), color="orange", point_size=4,
                    render_points_as_spheres=True, opacity=0.85)
        if camera_panel is not None:
            renderer, leader_name, cam_plane_actor = camera_panel
            update_camera_panel(renderer, leader_name, cam_plane_actor, fov_half_deg,
                                 pos, quat, leader_pos)

    if args.animate and not off_screen:
        plotter.subplot(*three_d_subplot)
        slider = plotter.add_slider_widget(lambda v: _update_all(int(v)), [0, n_frames - 1],
                                            value=0, title="step")
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
            plotter.subplot(*three_d_subplot)
            if mode == "top":
                plotter.view_xy()
            elif mode == "side":
                plotter.view_yz()
            else:
                plotter.camera.azimuth = 200
                plotter.camera.elevation = 20
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
