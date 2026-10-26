"""
Offline camera-footage renderer: replays a logged trajectory CSV through
a real rendered 3-D scene (open3d.visualization.rendering.OffscreenRenderer,
headless -- no display, no Gazebo) from the drone's own forward-facing
camera, and writes an RGB frame per tick to --out-dir.

Motion blur is REAL, not the analytic body-rate proxy
scripts/compute_metrics.py uses for the suite's gate metrics (that proxy
is untouched and stays the fast, suite-wide estimate -- this is a
separate, slower, visual-inspection tool for one scenario at a time).
Blur comes from rendering --blur-subsamples sub-poses across one assumed
camera-exposure window (1/mapping_sensor.update_rate_hz, same assumption
compute_metrics.py already makes) and averaging the resulting images.
Depth frames (--depth) are always single-sample -- averaging a depth
buffer across sub-poses has no clean physical meaning, so they stay
sharp by design.

Camera model: --width/--height default to 640x400, the OAK-D-Lite's
native stereo-depth resolution (same sensor compute_metrics.py's blur
estimate and config/quadrotor.yaml's mapping_sensor assume). The FOV
stays the solver's own circular cone (config/quadrotor.yaml's
camera.fov_half_angle_deg) rather than a literal rectangular lens
frustum -- the r_fov cost is rotationally symmetric by construction (see
ros2/argus_px4_bridge/doc/README.md's fov_markers entry), so rendering a
true rectangular frustum here would show a FOV shape the controller
itself doesn't actually optimize for. Lens distortion is intentionally
not modeled either, for the same reason: nothing else in this project's
simulation (cost, ESDF queries, FOV check) assumes anything but an ideal
pinhole, so adding distortion only here would make the footage LESS
representative of what the controller sees, not more.

Run (after a scenario has been simulated, e.g. ./build/online_mapping_demo
config/forest.yaml -- which writes logs/online_mapping_trajectory.csv):
    pip install -r scripts/requirements-viz.txt
    python3 scripts/render_camera_footage.py \\
        --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml \\
        --out-dir logs/camera_footage/forest

Assemble straight to an mp4 instead of a PNG sequence (--video implies
--rgb, skips writing individual frame_%05d.png files, via imageio's
bundled ffmpeg -- no system ffmpeg install needed):
    python3 scripts/render_camera_footage.py \\
        --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml \\
        --video logs/camera_footage/forest.mp4

Narrow a range and bump blur for a visible comparison:
    python3 scripts/render_camera_footage.py --start-step 100 --end-step 130 \\
        --blur-subsamples 1 --out-dir logs/camera_footage/sharp
    python3 scripts/render_camera_footage.py --start-step 100 --end-step 130 \\
        --blur-subsamples 8 --out-dir logs/camera_footage/blurred
"""

import argparse
import os

import numpy as np
import open3d as o3d

import scene3d


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", default="logs/online_mapping_trajectory.csv")
    p.add_argument("--scenario-config", default="config/forest.yaml",
                   help="Scenario YAML the CSV was generated from (obstacles: list)")
    p.add_argument("--config-path", default="config/quadrotor.yaml",
                   help="Drone/camera config (camera.fov_half_angle_deg, "
                        "mapping_sensor.update_rate_hz)")
    p.add_argument("--obstacle-height", type=float, default=scene3d.DEFAULT_OBSTACLE_HEIGHT_M,
                   help="Cosmetic extrusion height [m] for every obstacle "
                        "(obstacles have no z-extent in the scenario file)")
    p.add_argument("--out-dir", default="logs/camera_footage/run")
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--end-step", type=int, default=None, help="Exclusive; default: whole CSV")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--blur-subsamples", type=int, default=scene3d.DEFAULT_BLUR_SUBSAMPLES,
                   help="Sub-poses rendered and averaged per output frame "
                        "(1 = no blur, sharp)")
    p.add_argument("--width", type=int, default=640,
                   help="Matches the OAK-D-Lite's native stereo-depth resolution "
                        "(640x400), the sensor compute_metrics.py's blur estimate "
                        "and config/quadrotor.yaml's mapping_sensor assume")
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--supersample", type=int, default=2,
                   help="MSAA sample count factor for anti-aliased edges (Open3D's "
                        "View.set_sample_count/set_antialiasing). 1 = off. NOT a "
                        "resolution change -- rendering at higher resolution and "
                        "downsampling turned out to trigger an exposure bug in this "
                        "Open3D/Filament/llvmpipe combination (ground washed out "
                        "near-white at 2x resolution, confirmed before any "
                        "downsampling ran) -- MSAA sidesteps it entirely.")
    p.add_argument("--rgb", action="store_true")
    p.add_argument("--depth", action="store_true")
    p.add_argument("--both", action="store_true", help="Shorthand for --rgb --depth")
    p.add_argument("--no-hud", action="store_true",
                   help="Disable the telemetry HUD overlay (dist_obstacle/slack/solve "
                        "time/occlusion, with a red status badge when close or heavily "
                        "occluded) -- ties the footage back to what the MPC/ESDF "
                        "pipeline actually saw, not just a pretty render")
    p.add_argument("--d-fov", type=float, default=4.0,
                   help="Occlusion falloff scale [m] for the HUD's occlusion%% readout "
                        "-- same default as compute_metrics.py's --d-fov")
    p.add_argument("--video", default=None, metavar="PATH",
                   help="Write an mp4 at PATH instead of a frame_%%05d.png sequence "
                        "(implies --rgb; depth output via --depth/--both is unaffected "
                        "and still written as .npy per frame)")
    p.add_argument("--fps", type=float, default=None,
                   help="Video frame rate (default: --output-fps if set, else derived "
                        "from the CSV's own tick rate and --stride)")
    p.add_argument("--output-fps", type=float, default=None,
                   help="Resample --start-step/--end-step's time window at this many "
                        "frames/sec (e.g. 30 or 60) instead of one frame per logged "
                        "CSV row -- decouples video smoothness from the 20Hz control "
                        "loop. Default: off (one frame per row, per --stride).")
    return p.parse_args()


def _material(color, roughness=0.85, reflectance=0.4):
    """defaultLit material -- roughness/reflectance tuned per surface
    (matte ground/trees/clothing vs. a slightly less rough drone body),
    not left at Open3D's flat-plastic default."""
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = color
    mat.base_roughness = roughness
    mat.base_reflectance = reflectance
    mat.base_metallic = 0.0
    return mat


def build_scene(renderer, obstacles, bounds, obstacle_height, width=640, height=480):
    """Adds ground + obstacles + a leader (person) marker to
    renderer.scene. Built once, reused across every output frame. Returns
    the leader's part names (its transform is updated per frame, as one
    rigid body -- a person doesn't rotate, only translates)."""
    renderer.scene.set_background([1, 1, 1, 1], scene3d.sky_gradient_image(width, height))
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        (0.35, -0.55, -1.0))
    renderer.scene.scene.set_sun_light((0.35, -0.55, -1.0), (1.0, 0.98, 0.92), 50000)

    ground = scene3d.build_ground_plane_o3d(bounds)
    renderer.scene.add_geometry("ground", ground, _material([0.42, 0.42, 0.40, 1.0],
                                                              roughness=0.95))

    for i, obstacle in enumerate(obstacles):
        for part_name, mesh, color in scene3d.build_obstacle_parts_o3d(obstacle, obstacle_height):
            roughness = 0.9 if part_name == "canopy" else 0.8
            renderer.scene.add_geometry(f"obstacle_{i}_{part_name}", mesh,
                                         _material(color, roughness=roughness))

    leader_names = []
    for part_name, mesh, color in scene3d.build_person_parts_o3d():
        name = f"leader_{part_name}"
        roughness = 0.5 if part_name == "head" else 0.85
        renderer.scene.add_geometry(name, mesh, _material(color, roughness=roughness))
        leader_names.append(name)
    return leader_names


def draw_hud(frame, row, d_fov):
    """Overlay the solver's own telemetry for this tick -- dist_obstacle
    (own_map belief), the slacked constraint's slack value, solve time,
    and an occlusion%% readout (same 100*clip(1-occlusion_clearance/d_fov,
    0,1) formula compute_metrics.py's dashboard uses) -- plus a small
    status badge, red when close to the collision margin or heavily
    occluded. This is what ties the footage back to the actual MPC/ESDF
    contribution instead of being pure visual polish."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    dist = float(row["dist_obstacle"])
    slack = float(row["obstacle_slack"]) if "obstacle_slack" in row else float("nan")
    solve_ms = float(row["solve_time_ms"])
    occ_clear = float(row["occlusion_clearance"]) if "occlusion_clearance" in row else None
    occ_pct = 100.0 * np.clip(1.0 - occ_clear / d_fov, 0.0, 1.0) if occ_clear is not None \
        else None

    danger = dist < 0.4 or (occ_pct is not None and occ_pct > 70.0)
    badge_color = (220, 50, 50, 230) if danger else (60, 180, 90, 230)

    lines = [f"dist_obstacle: {dist:.2f} m", f"slack: {slack:.3f} m",
             f"solve: {solve_ms:.2f} ms"]
    if occ_pct is not None:
        lines.append(f"occlusion: {occ_pct:.0f}%")

    pad, line_h, box_w = 5, 13, 155
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle([4, 4, 4 + box_w, 4 + box_h], fill=(0, 0, 0, 120))
    for i, line in enumerate(lines):
        draw.text((4 + pad, 4 + pad + i * line_h), line, fill=(255, 255, 255, 255))

    r = 7
    cx, cy = frame.shape[1] - 14, 14
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=badge_color)
    return np.asarray(img)


def update_leader_position(renderer, leader_names, pos):
    m = np.array([[1, 0, 0, pos[0]],
                   [0, 1, 0, pos[1]],
                   [0, 0, 1, pos[2]],
                   [0, 0, 0, 1]], dtype=np.float64)
    for name in leader_names:
        renderer.scene.set_geometry_transform(name, m)


def main():
    args = _parse_args()
    if not (args.rgb or args.depth or args.both or args.video):
        args.rgb = True  # default mode
    do_rgb = args.rgb or args.both or bool(args.video)
    do_depth = args.depth or args.both

    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    camera_cfg, mapping_cfg = scene3d.load_camera_config(args.config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    t_exposure_s = 1.0 / mapping_cfg["update_rate_hz"]

    df = scene3d.load_trajectory(args.csv_path)
    bounds = scene3d.scene_bounds(df, obstacles)

    t_arr = df["t"].to_numpy()
    end_step = args.end_step if args.end_step is not None else len(df)
    end_step = min(end_step, len(df))

    if args.output_fps:
        # Dense, time-based sampling -- decoupled from the 20Hz control
        # loop's own tick rate, for a smooth demo video instead of one
        # frame per (possibly coarse) logged step. --start-step/--end-step
        # still pick the time WINDOW; --output-fps picks how finely it's
        # resampled within that window. --stride is meaningless here.
        t_start, t_end = t_arr[args.start_step], t_arr[end_step - 1]
        n_out = max(1, int(round((t_end - t_start) * args.output_fps)) + 1)
        sample_times = np.linspace(t_start, t_end, n_out)
    else:
        sample_times = t_arr[args.start_step:end_step:args.stride]

    video_writer = None
    if args.video:
        import imageio
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        if args.fps is not None:
            fps = args.fps
        elif args.output_fps:
            fps = args.output_fps
        else:
            fps = 1.0 / ((df["t"][1] - df["t"][0]) * args.stride)
        video_writer = imageio.get_writer(args.video, fps=fps, codec="libx264",
                                           quality=8, macro_block_size=None)
    if video_writer is None or do_depth:
        os.makedirs(args.out_dir, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    leader_name = build_scene(renderer, obstacles, bounds, args.obstacle_height,
                               args.width, args.height)
    renderer.scene.view.set_antialiasing(args.supersample > 1, temporal=False)
    renderer.scene.view.set_sample_count(max(1, args.supersample) * 2)
    renderer.setup_camera(2 * fov_half_deg, [0, 0, 1], [0, 0, 0], [0, 0, 1])

    n_sub = max(1, args.blur_subsamples)

    for out_idx, t_center in enumerate(sample_times):
        leader_pos = scene3d.interpolate_point(df, t_center, ("fx", "fy", "fz"))
        update_leader_position(renderer, leader_name, leader_pos)

        if do_rgb:
            accum = None
            sub_times = (
                [t_center] if n_sub == 1 else
                np.linspace(t_center - t_exposure_s / 2, t_center + t_exposure_s / 2, n_sub)
            )
            for t_sub in sub_times:
                pos, quat = scene3d.interpolate_pose(df, t_sub)
                eye, center, up = scene3d.camera_extrinsics(pos, quat)
                renderer.setup_camera(2 * fov_half_deg, center, eye, up)
                img = np.asarray(renderer.render_to_image(), dtype=np.float32)
                accum = img if accum is None else accum + img
            frame = (accum / n_sub).astype(np.uint8)
            if not args.no_hud:
                nearest_row = df.iloc[int(np.argmin(np.abs(t_arr - t_center)))]
                frame = draw_hud(frame, nearest_row, args.d_fov)
            if video_writer is not None:
                video_writer.append_data(frame)
            else:
                o3d.io.write_image(os.path.join(args.out_dir, f"frame_{out_idx:05d}.png"),
                                    o3d.geometry.Image(frame))

        if do_depth:
            pos, quat = scene3d.interpolate_pose(df, t_center)
            eye, center, up = scene3d.camera_extrinsics(pos, quat)
            renderer.setup_camera(2 * fov_half_deg, center, eye, up)
            depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
            np.save(os.path.join(args.out_dir, f"depth_{out_idx:05d}.npy"), depth)

    dest = args.video if video_writer is not None else args.out_dir
    if video_writer is not None:
        video_writer.close()
    print(f"Wrote {len(sample_times)} frames ({sample_times[0]:.2f}s-{sample_times[-1]:.2f}s) "
          f"to {dest}")


if __name__ == "__main__":
    main()
