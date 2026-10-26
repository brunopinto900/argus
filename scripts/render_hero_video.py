"""
Composite "hero" demo video: an external chase-cam 3-D view (drone, the
leader, extruded obstacles/trees) with the drone's own camera footage
picture-in-picture inset -- the single deliverable meant for a CV/
portfolio page or a thesis demo, not a diagnostic tool (that's
view_scene3d.py's job).

One Open3D OffscreenRenderer, one scene, rendered twice per output frame
(chase pass with the drone mesh shown, POV pass with it hidden -- a
camera can't see its own airframe) -- not two separate renderers: Open3D/
Filament's cleanup path reliably crashes at process exit with two
OffscreenRenderer instances alive at once (confirmed directly; harmless
since it happens after the video is written and closed, but avoided here
since it's needless noise). The chase frame is rendered at the full
output resolution and the POV frame is rendered at that SAME resolution
then downscaled for the inset -- free anti-aliasing on the inset, and
--width/--height's 1.6:1 default matches the OAK-D-Lite's 640x400 aspect
exactly, so the downscale is undistorted.

Run:
    python3 scripts/render_hero_video.py \\
        --csv-path logs/online_mapping_trajectory.csv \\
        --scenario-config config/forest.yaml \\
        --video logs/camera_footage/forest_hero.mp4
"""

import argparse
import os

import numpy as np
import open3d as o3d
from PIL import Image

import render_camera_footage as camfeed
import scene3d


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", default="logs/online_mapping_trajectory.csv")
    p.add_argument("--scenario-config", default="config/forest.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--obstacle-height", type=float, default=scene3d.DEFAULT_OBSTACLE_HEIGHT_M)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--end-step", type=int, default=None)
    p.add_argument("--output-fps", type=float, default=30.0)
    p.add_argument("--blur-subsamples", type=int, default=scene3d.DEFAULT_BLUR_SUBSAMPLES)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--inset-width", type=int, default=320,
                   help="Picture-in-picture size for the camera-footage inset")
    p.add_argument("--chase-distance", type=float, default=3.0, help="Behind the drone [m]")
    p.add_argument("--chase-height", type=float, default=1.4, help="Above the drone [m]")
    p.add_argument("--chase-fov-deg", type=float, default=55.0)
    p.add_argument("--no-hud", action="store_true", help="Disable the inset's telemetry HUD")
    p.add_argument("--no-post-process", action="store_true",
                   help="Disable the cinematic grade (saturation/contrast/vignette)")
    p.add_argument("--d-fov", type=float, default=4.0)
    p.add_argument("--video", default="logs/camera_footage/hero.mp4")
    return p.parse_args()


def chase_camera_pose(pos, quat, distance, height):
    """Yaw-only follow (ignores roll/pitch) with a fixed world up vector
    -- a chase cam that banked and pitched with the drone would be
    nauseating to watch; real game/film chase cams stay yaw-only too."""
    psi = quat.as_euler("ZYX")[0]
    heading = np.array([np.cos(psi), np.sin(psi), 0.0])
    eye = pos - heading * distance + np.array([0.0, 0.0, height])
    center = pos + np.array([0.0, 0.0, 0.15])
    return eye, center, (0.0, 0.0, 1.0)


def _translation_matrix(pos):
    m = np.eye(4)
    m[:3, 3] = pos
    return m


def _pose_matrix(pos, quat):
    m = np.eye(4)
    m[:3, :3] = quat.as_matrix()
    m[:3, 3] = pos
    return m


def composite_pip(main_img, inset_img, margin=16, border=3):
    main = Image.fromarray(main_img).convert("RGB")
    inset = Image.fromarray(inset_img).convert("RGB")
    bordered = Image.new("RGB", (inset.width + 2 * border, inset.height + 2 * border),
                          (245, 245, 245))
    bordered.paste(inset, (border, border))
    main.paste(bordered, (main.width - bordered.width - margin,
                           main.height - bordered.height - margin))
    return np.asarray(main)


def main():
    args = _parse_args()

    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    camera_cfg, mapping_cfg = scene3d.load_camera_config(args.config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    t_exposure_s = 1.0 / mapping_cfg["update_rate_hz"]

    df = scene3d.load_trajectory(args.csv_path)
    bounds = scene3d.scene_bounds(df, obstacles)
    t_arr = df["t"].to_numpy()

    end_step = min(args.end_step if args.end_step is not None else len(df), len(df))
    t_start, t_end = t_arr[args.start_step], t_arr[end_step - 1]
    n_out = max(1, int(round((t_end - t_start) * args.output_fps)) + 1)
    sample_times = np.linspace(t_start, t_end, n_out)

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    leader_names = camfeed.build_scene(renderer, obstacles, bounds, args.obstacle_height,
                                        args.width, args.height)
    drone_names = []
    for part_name, mesh, color in scene3d.build_drone_parts_o3d():
        name = f"drone_{part_name}"
        renderer.scene.add_geometry(name, mesh, camfeed._material(color, roughness=0.35))
        drone_names.append(name)
    renderer.scene.view.set_antialiasing(True)
    renderer.scene.view.set_sample_count(4)

    inset_h = int(round(args.inset_width * args.height / args.width))

    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    import imageio
    writer = imageio.get_writer(args.video, fps=args.output_fps, codec="libx264",
                                 quality=8, macro_block_size=None)

    n_sub = max(1, args.blur_subsamples)

    for t_center in sample_times:
        pos, quat = scene3d.interpolate_pose(df, t_center)
        leader_pos = scene3d.interpolate_point(df, t_center, ("fx", "fy", "fz"))
        camfeed.update_leader_position(renderer, leader_names, leader_pos)
        drone_matrix = _pose_matrix(pos, quat)
        for name in drone_names:
            renderer.scene.set_geometry_transform(name, drone_matrix)

        for name in drone_names:
            renderer.scene.show_geometry(name, True)
        eye, center, up = chase_camera_pose(pos, quat, args.chase_distance, args.chase_height)
        renderer.setup_camera(args.chase_fov_deg, center, eye, up)
        chase_img = np.asarray(renderer.render_to_image())

        for name in drone_names:
            renderer.scene.show_geometry(name, False)
        accum = None
        sub_times = ([t_center] if n_sub == 1 else
                     np.linspace(t_center - t_exposure_s / 2, t_center + t_exposure_s / 2, n_sub))
        for t_sub in sub_times:
            p_sub, q_sub = scene3d.interpolate_pose(df, t_sub)
            eye2, center2, up2 = scene3d.camera_extrinsics(p_sub, q_sub)
            renderer.setup_camera(2 * fov_half_deg, center2, eye2, up2)
            img = np.asarray(renderer.render_to_image(), dtype=np.float32)
            accum = img if accum is None else accum + img
        pov_frame = (accum / n_sub).astype(np.uint8)

        pov_inset = np.asarray(Image.fromarray(pov_frame).resize(
            (args.inset_width, inset_h), Image.LANCZOS))
        if not args.no_hud:
            nearest_row = df.iloc[int(np.argmin(np.abs(t_arr - t_center)))]
            pov_inset = camfeed.draw_hud(pov_inset, nearest_row, args.d_fov)

        composite = composite_pip(chase_img, pov_inset)
        if not args.no_post_process:
            composite = scene3d.apply_post_process(composite)
        writer.append_data(composite)

    writer.close()
    print(f"Wrote {len(sample_times)} frames ({sample_times[0]:.2f}s-{sample_times[-1]:.2f}s) "
          f"to {args.video}")


if __name__ == "__main__":
    main()
