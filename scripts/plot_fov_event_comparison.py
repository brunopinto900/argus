"""
Static "bad vs. good" technical-view comparison, same style as
view_technical_scene.py/esdf_sightline.png, but contrasting two
*configurations* at one shared scenario tick instead of occlusion off/on:
Band follower (mid FOV-cone exit) vs. Band follower + centering (same
tick, target still framed) -- one concrete frame for the FOV-exit events
Fig. 6 (scripts/plot_fov_results.py) shows disappearing (3 to 0) when the
centering term is added.

    python3 scripts/plot_fov_event_comparison.py \\
        --csv-bad logs/camera_footage/band_tracking_fc/trajectory.csv \\
        --snapshot-bad logs/camera_footage/band_tracking_fc/esdf_snapshots.bin \\
        --csv-good logs/camera_footage/band_centering_fc/trajectory.csv \\
        --snapshot-good logs/camera_footage/band_centering_fc/esdf_snapshots.bin \\
        --scenario-config config/forest.yaml --frame 75 \\
        --save-png paper/figures/fov_event_comparison.png
"""

import argparse

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

import scene3d
import view_technical_scene as vts

PANEL_SIZE = (720, 560)
BAD_LABEL_COLOR = (178, 58, 46)
GOOD_LABEL_COLOR = (40, 120, 60)
LEGEND_H = 46
BANNER_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-bad", required=True)
    p.add_argument("--csv-good", required=True)
    p.add_argument("--snapshot-bad", default=None)
    p.add_argument("--snapshot-good", default=None)
    p.add_argument("--scenario-config", default="config/forest.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--voxel-stride", type=int, default=1)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--zoom", type=float, default=2.6)
    p.add_argument("--d-fov", type=float, default=1.68)
    p.add_argument("--lost-contact-occlusion-pct", type=float, default=90.0)
    p.add_argument("--label-bad", default="BAND FOLLOWER")
    p.add_argument("--label-good", default="BAND FOLLOWER + CENTERING")
    p.add_argument("--save-png", default=None)
    return p.parse_args()


def _build_panel(csv_path, snapshot_path, scenario_config, config_path, zoom, bounds,
                  frame, voxel_stride, d_fov, lost_contact_occlusion_pct, banner_font):
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
    event_actors = vts.add_event_markers_pv(plotter, df, fov_half_deg, d_fov,
                                             lost_contact_occlusion_pct)
    event_legend = [(label, color) for _actor, label, color in event_actors]

    drone_actors, target_actor = vts.build_dynamic_actors(plotter)
    frustum_actor = vts.init_frustum_actor(plotter)
    plotter.camera.azimuth = 200
    plotter.camera.elevation = 25
    plotter.camera.zoom(zoom)

    t = df["t"].to_numpy()[frame]
    vts.render_frame(df, frame, drone_actors, target_actor, frustum_actor, h_fov, v_fov)

    if esdf_data is not None:
        (esdf_times, _d, esdf_xs, esdf_ys, _gt, voxel_size, _rng, own_map_states) = esdf_data
        idx = int(np.argmin(np.abs(esdf_times - t)))
        pts = scene3d.occupied_points_at(own_map_states, idx, esdf_xs, esdf_ys,
                                          z=voxel_size * 0.3, stride=voxel_stride)
        if len(pts):
            vts.build_voxel_cloud_pv(plotter, pts, voxel_size)

    plotter.render()
    img = Image.fromarray(plotter.screenshot(return_img=True))
    draw = ImageDraw.Draw(img, "RGBA")
    plotter.close()
    return img, draw, event_legend


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

    df_bad = scene3d.load_trajectory(args.csv_bad)
    df_good = scene3d.load_trajectory(args.csv_good)
    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    bounds_bad = scene3d.scene_bounds(df_bad, obstacles)
    bounds_good = scene3d.scene_bounds(df_good, obstacles)
    bounds = tuple(min(a, b) if i % 2 == 0 else max(a, b)
                   for i, (a, b) in enumerate(zip(bounds_bad, bounds_good)))

    banner_font = ImageFont.truetype(BANNER_FONT, 22)
    legend_font = ImageFont.truetype(CAPTION_FONT, 18)

    bad_img, bad_draw, bad_legend = _build_panel(
        args.csv_bad, args.snapshot_bad, args.scenario_config, args.config_path,
        args.zoom, bounds, args.frame, args.voxel_stride, args.d_fov,
        args.lost_contact_occlusion_pct, banner_font)
    good_img, good_draw, good_legend = _build_panel(
        args.csv_good, args.snapshot_good, args.scenario_config, args.config_path,
        args.zoom, bounds, args.frame, args.voxel_stride, args.d_fov,
        args.lost_contact_occlusion_pct, banner_font)

    _draw_banner(bad_draw, args.label_bad, BAD_LABEL_COLOR, banner_font)
    _draw_banner(good_draw, args.label_good, GOOD_LABEL_COLOR, banner_font)

    w = PANEL_SIZE[0] * 2 + 4
    h = PANEL_SIZE[1] + LEGEND_H
    combined = Image.new("RGB", (w, h), (255, 255, 255))
    combined.paste(bad_img, (0, 0))
    combined.paste(good_img, (PANEL_SIZE[0] + 4, 0))

    draw = ImageDraw.Draw(combined)
    legend_items = [
        ("Drone path", vts.DRONE_PATH_COLOR), ("Target path", vts.TARGET_PATH_COLOR),
        ("Drone", vts.DRONE_COLOR), ("Target", vts.TARGET_COLOR),
        ("Occupied voxel (own_map)", vts.VOXEL_COLOR),
    ]
    seen = {label for label, _ in legend_items}
    for label, color in bad_legend + good_legend:
        if label not in seen:
            legend_items.append((label, color))
            seen.add(label)
    x, y = 16, PANEL_SIZE[1] + 10
    for text, color in legend_items:
        draw.rectangle((x, y + 3, x + 12, y + 15), fill=color)
        bbox = draw.textbbox((0, 0), text, font=legend_font)
        draw.text((x + 18, y), text, font=legend_font, fill=(30, 30, 30))
        x += 24 + (bbox[2] - bbox[0]) + 22

    if args.save_png:
        combined.save(args.save_png)
        print(f"Saved {args.save_png} ({w}x{h}, frame {args.frame}, "
              f"t={df_bad['t'].to_numpy()[args.frame]:.2f}s)")
    else:
        combined.show()


if __name__ == "__main__":
    main()
