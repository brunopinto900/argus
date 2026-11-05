"""
Side-by-side OFF vs. ON animated comparison in view_technical_scene.py's
clean white/grey "scientific instrument" style -- the README hero visual.
Replaces the old low-poly chase-cam render (docs/occlusion_comparison.gif):
same off/on narrative, same underlying run, but the honest voxel/frustum/
trajectory view instead of a cinematic placeholder-mesh scene, and no
picture-in-picture inset.

Reuses view_technical_scene.py's scene-building functions directly (one
PyVista plotter per condition), takes an off-screen screenshot per frame,
and composites the two panels with OWN_MAP labels via PIL -- same pattern
as esdf_sightline.png's static off/on composite, but animated.

    python3 scripts/render_technical_comparison.py \\
        --csv-off logs/camera_footage/band_best_off/trajectory.csv \\
        --csv-on logs/camera_footage/band_best_on/trajectory.csv \\
        --scenario-config config/forest.yaml \\
        --caption "Occlusion cost: off vs. on" \\
        --gif docs/occlusion_comparison.gif
"""

import argparse

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

import scene3d
import view_technical_scene as vts

PANEL_SIZE = (720, 560)
OFF_LABEL_COLOR = (178, 58, 46)
ON_LABEL_COLOR = (40, 120, 60)
LEGEND_H = 46
BANNER_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-off", required=True)
    p.add_argument("--csv-on", required=True)
    p.add_argument("--snapshot-off", default=None,
                   help="Default: auto-detect next to --csv-off")
    p.add_argument("--snapshot-on", default=None)
    p.add_argument("--scenario-config", default="config/forest.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--voxel-stride", type=int, default=1)
    p.add_argument("--d-fov", type=float, default=1.68)
    p.add_argument("--stride", type=int, default=2, help="Frame stride over the run")
    p.add_argument("--zoom", type=float, default=1.6)
    p.add_argument("--gif", default="docs/occlusion_comparison.gif")
    p.add_argument("--gif-fps", type=float, default=12.0)
    p.add_argument("--caption", default=None)
    p.add_argument("--label-events", action="store_true",
                   help="Mark target-lost events (out-of-FOV-cone vs. occluded) "
                        "along each panel's drone path")
    p.add_argument("--lost-contact-occlusion-pct", type=float, default=90.0)
    return p.parse_args()


def _build_plotter(csv_path, snapshot_path, scenario_config, config_path, zoom, bounds,
                    label_events=False, d_fov=1.68, lost_contact_occlusion_pct=90.0):
    camera_cfg, _ = scene3d.load_camera_config(config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    h_fov = v_fov = np.deg2rad(fov_half_deg)

    df = scene3d.load_trajectory(csv_path)
    snapshot_path = snapshot_path or scene3d.find_snapshot_log(csv_path)
    esdf_data = scene3d.load_snapshot_log(snapshot_path) if snapshot_path else None

    plotter = pv.Plotter(off_screen=True, window_size=PANEL_SIZE)
    plotter.set_background(vts.BG_COLOR)
    vts.build_static_scene(plotter, bounds)
    vts.build_trajectory_line_pv(plotter, df[["x", "y", "z"]].to_numpy(), vts.DRONE_PATH_COLOR)
    vts.build_trajectory_line_pv(plotter, df[["fx", "fy", "fz"]].to_numpy(), vts.TARGET_PATH_COLOR)
    event_actors = []
    if label_events:
        event_actors = vts.add_event_markers_pv(plotter, df, fov_half_deg, d_fov,
                                                  lost_contact_occlusion_pct)
    event_legend = [(label, color) for _actor, label, color in event_actors]
    drone_actors, target_actor = vts.build_dynamic_actors(plotter)
    frustum_actor = vts.init_frustum_actor(plotter)
    plotter.camera.azimuth = 200
    plotter.camera.elevation = 25
    plotter.camera.zoom(zoom)

    if esdf_data is not None:
        (esdf_times, _d, esdf_xs, esdf_ys, _gt, voxel_size, _rng, own_map_states) = esdf_data

        def _voxels_at(t, stride):
            idx = int(np.argmin(np.abs(esdf_times - t)))
            return scene3d.occupied_points_at(own_map_states, idx, esdf_xs, esdf_ys,
                                               z=voxel_size * 0.3, stride=stride)
    else:
        voxel_size = 0.1

        def _voxels_at(_t, _stride):
            return np.zeros((0, 3))

    return dict(plotter=plotter, df=df, h_fov=h_fov, v_fov=v_fov,
                drone_actors=drone_actors, target_actor=target_actor,
                frustum_actor=frustum_actor, voxel_actor=[None],
                voxels_at=_voxels_at, voxel_size=voxel_size, event_legend=event_legend)


def _render_panel(scene, step, voxel_stride, font):
    t = scene["df"]["t"].to_numpy()[step]
    vts.render_frame(scene["df"], step, scene["drone_actors"], scene["target_actor"],
                      scene["frustum_actor"], scene["h_fov"], scene["v_fov"])
    plotter = scene["plotter"]
    if scene["voxel_actor"][0] is not None:
        plotter.remove_actor(scene["voxel_actor"][0])
        scene["voxel_actor"][0] = None
    pts = scene["voxels_at"](t, voxel_stride)
    scene["voxel_actor"][0] = vts.build_voxel_cloud_pv(plotter, pts, scene["voxel_size"])
    plotter.render()
    img = Image.fromarray(plotter.screenshot(return_img=True))
    draw = ImageDraw.Draw(img, "RGBA")
    return img, draw


def _draw_banner(draw, text, color, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    box = (12, 10, 12 + tw + 2 * pad, 10 + th + 2 * pad)
    draw.rectangle(box, fill=(255, 255, 255, 220))
    draw.rectangle(box, outline=(60, 60, 60, 255), width=1)
    draw.text((box[0] + pad, box[1] + pad - bbox[1]), text, font=font, fill=color)


def main():
    args = _parse_args()

    df_off = scene3d.load_trajectory(args.csv_off)
    df_on = scene3d.load_trajectory(args.csv_on)
    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    bounds_off = scene3d.scene_bounds(df_off, obstacles)
    bounds_on = scene3d.scene_bounds(df_on, obstacles)
    bounds = tuple(min(a, b) if i % 2 == 0 else max(a, b)
                   for i, (a, b) in enumerate(zip(bounds_off, bounds_on)))

    scene_off = _build_plotter(args.csv_off, args.snapshot_off, args.scenario_config,
                                args.config_path, args.zoom, bounds,
                                label_events=args.label_events, d_fov=args.d_fov,
                                lost_contact_occlusion_pct=args.lost_contact_occlusion_pct)
    scene_on = _build_plotter(args.csv_on, args.snapshot_on, args.scenario_config,
                               args.config_path, args.zoom, bounds,
                               label_events=args.label_events, d_fov=args.d_fov,
                               lost_contact_occlusion_pct=args.lost_contact_occlusion_pct)

    n_frames = min(len(scene_off["df"]), len(scene_on["df"]))
    steps = list(range(0, n_frames, max(1, args.stride)))

    banner_font = ImageFont.truetype(BANNER_FONT, 24)
    legend_font = ImageFont.truetype(CAPTION_FONT, 18)
    caption_font = ImageFont.truetype(CAPTION_FONT, 20)

    w = PANEL_SIZE[0] * 2 + 4
    h = PANEL_SIZE[1] + LEGEND_H + (34 if args.caption else 0)

    legend_items = [
        ("Drone path", vts.DRONE_PATH_COLOR), ("Target path", vts.TARGET_PATH_COLOR),
        ("Drone", vts.DRONE_COLOR), ("Target", vts.TARGET_COLOR),
        ("Occupied voxel (own_map)", vts.VOXEL_COLOR),
    ]
    seen = {label for label, _ in legend_items}
    for label, color in scene_off["event_legend"] + scene_on["event_legend"]:
        if label not in seen:
            legend_items.append((label, color))
            seen.add(label)

    frames = []
    for step in steps:
        off_img, off_draw = _render_panel(scene_off, step, args.voxel_stride, banner_font)
        on_img, on_draw = _render_panel(scene_on, step, args.voxel_stride, banner_font)
        _draw_banner(off_draw, "OCCLUSION OFF", OFF_LABEL_COLOR, banner_font)
        _draw_banner(on_draw, "OCCLUSION ON", ON_LABEL_COLOR, banner_font)

        combined = Image.new("RGB", (w, h), (255, 255, 255))
        combined.paste(off_img, (0, 0))
        combined.paste(on_img, (PANEL_SIZE[0] + 4, 0))
        draw = ImageDraw.Draw(combined)
        x, y = 16, PANEL_SIZE[1] + 10
        for text, color in legend_items:
            draw.rectangle((x, y + 3, x + 12, y + 15), fill=color)
            bbox = draw.textbbox((0, 0), text, font=legend_font)
            draw.text((x + 18, y), text, font=legend_font, fill=(30, 30, 30))
            x += 24 + (bbox[2] - bbox[0]) + 22
        if args.caption:
            cap_y = PANEL_SIZE[1] + LEGEND_H + 4
            draw.rectangle((0, cap_y, w, h), fill=(20, 20, 20))
            bbox = draw.textbbox((0, 0), args.caption, font=caption_font)
            cw = bbox[2] - bbox[0]
            draw.text(((w - cw) // 2, cap_y + 6), args.caption, font=caption_font,
                       fill=(240, 240, 240))
        frames.append(combined)

    scene_off["plotter"].close()
    scene_on["plotter"].close()

    frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                    duration=int(1000 / args.gif_fps), loop=0)
    print(f"Saved {args.gif} ({len(frames)} frames, {w}x{h})")


if __name__ == "__main__":
    main()
