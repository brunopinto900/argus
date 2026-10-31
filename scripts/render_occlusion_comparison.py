"""
Side-by-side occlusion-cost comparison video: camera footage (top row) +
own_map's 2-D ESDF (bottom row), occlusion OFF (left column, red label) vs.
occlusion ON (right column, green label) -- the single deliverable meant to
show, visually, what the squared-hinge occlusion cost (design_decisions.md's
"Occlusion-cost shape" section) actually buys on occlusion_cut, rather than
just the metrics table.

Both conditions replay the SAME scenario (config/occlusion_cut.yaml, same
static obstacles) from two different logged runs -- one with
occlusion_avoidance.mode: disabled, one with mode: cost (w=50/25, the
sweep's chosen point) -- so the obstacle geometry in both camera views and
both ESDF panels is identical; only the drone's actual flown trajectory
(and therefore the ESDF snapshot sequence, since own_map is built from
wherever the drone actually looked) differs.

One shared OffscreenRenderer renders both camera views each output frame
(not two renderer instances -- see render_hero_video.py's docstring: two
live OffscreenRenderers crash Open3D/Filament's cleanup path at process
exit). The ESDF panels come straight from scene3d.esdf_panel_image() (no
matplotlib) upscaled with PIL, same per-run snapshot log each condition's
online_mapping_demo run already writes next to its trajectory CSV.

Run (after producing both trajectory.csv + esdf_snapshots.bin pairs, see
design_decisions.md's "Occlusion on vs. off, side-by-side video" entry for
how those two runs were generated):
    python3 scripts/render_occlusion_comparison.py \\
        --csv-off logs/camera_footage/occ_off/trajectory.csv \\
        --csv-on  logs/camera_footage/occ_on/trajectory.csv \\
        --scenario-config config/occlusion_cut.yaml \\
        --video logs/camera_footage/occlusion_comparison.mp4
"""

import argparse
import os

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

import render_camera_footage as camfeed
import scene3d

LABEL_OFF = "OCCLUSION OFF"
LABEL_ON = "OCCLUSION ON"
COLOR_OFF = (230, 60, 60, 255)   # red -- the unmitigated baseline
COLOR_ON = (70, 200, 100, 255)   # green -- the fix
DRONE_MARKER_COLOR = (0, 220, 255, 255)   # cyan, matches the camera-rig eye point
LEADER_MARKER_COLOR = (255, 221, 51, 255)  # same high-contrast yellow as animate_xy.py's
                                             # TARGET_COLOR -- never collides with the
                                             # ESDF panel's own red/blue hue sweep
CHART_BG = (12, 16, 20)
CHART_GRID = (60, 68, 76)
CHART_AXIS_TEXT = (170, 178, 186)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-off", required=True)
    p.add_argument("--csv-on", required=True)
    p.add_argument("--snapshot-off", default=None,
                   help="Default: auto-detect next to --csv-off (online_mapping naming)")
    p.add_argument("--snapshot-on", default=None)
    p.add_argument("--scenario-config", default="config/occlusion_cut.yaml")
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--obstacle-height", type=float, default=scene3d.DEFAULT_OBSTACLE_HEIGHT_M)
    p.add_argument("--d-fov", type=float, default=1.68,
                   help="Occlusion falloff scale [m] for the HUD readout -- matches the "
                        "derived default, see design_decisions.md's d_FOV entry")
    p.add_argument("--panel-width", type=int, default=640)
    p.add_argument("--panel-height", type=int, default=400)
    p.add_argument("--blur-subsamples", type=int, default=3,
                   help="Lower than render_camera_footage.py's default (8): this script "
                        "renders 2x the camera frames per output frame already")
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--end-step", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output-fps", type=float, default=20.0,
                   help="Sampling rate of the trajectory CSVs, also the video's playback "
                        "fps unless --playback-fps is given")
    p.add_argument("--playback-fps", type=float, default=None,
                   help="Video fps, decoupled from --output-fps's sampling rate -- e.g. "
                        "--output-fps 20 --playback-fps 10 plays back at half speed "
                        "(every sampled tick held for 2 video frames) so a subtle "
                        "per-tick divergence is easier to actually see")
    p.add_argument("--video", default="logs/camera_footage/occlusion_comparison.mp4")
    p.add_argument("--no-post-process", action="store_true")
    p.add_argument("--label-off", default=LABEL_OFF)
    p.add_argument("--label-on", default=LABEL_ON)
    p.add_argument("--chart-height", type=int, default=240,
                   help="Height of the top-down path panel appended below the "
                        "camera/ESDF rows")
    return p.parse_args()


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_LABEL_FONT = None


def draw_label(frame_rgb, text, color):
    """Big bold colored condition label, top-center, with a dark backing
    box for legibility over bright sky/ground."""
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = _font(max(18, frame_rgb.shape[1] // 22))
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    bbox = draw.textbbox((0, 0), text, font=_LABEL_FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = frame_rgb.shape[1] // 2
    pad = 8
    box = [cx - tw // 2 - pad, 6, cx + tw // 2 + pad, 6 + th + 2 * pad]
    draw.rectangle(box, fill=(0, 0, 0, 150))
    draw.text((cx - tw // 2, 6 + pad - bbox[1]), text, font=_LABEL_FONT, fill=color)
    return np.asarray(img)


def esdf_frame(distances, idx, ground_truth_state, sensor_max_range, xs, ys,
                drone_pos, leader_pos, label, label_color, panel_w, panel_h):
    """own_map's ESDF heatmap at snapshot idx, resized to panel_w x panel_h,
    north-up (row 0 = south, flipped for display -- same convention
    view_scene3d.py's update_ground_esdf() uses for its PyVista texture),
    with drone/leader markers and the condition label burned in."""
    img = scene3d.esdf_panel_image(distances[idx], ground_truth_state, sensor_max_range)
    img = np.flipud(img)
    pil = Image.fromarray(img).resize((panel_w, panel_h), Image.BILINEAR).convert("RGB")

    # Label first, markers drawn after and on top -- otherwise the label's
    # backing box can fully occlude a marker that happens to sit near the
    # panel's top edge (confirmed happening with the leader marker before
    # this ordering fix).
    frame = draw_label(np.asarray(pil), label, label_color)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil, "RGBA")

    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]

    def to_px(pos):
        u = (pos[0] - x0) / (x1 - x0)
        v = 1.0 - (pos[1] - y0) / (y1 - y0)  # flipped same as the image above
        return u * panel_w, v * panel_h

    r = max(4, panel_w // 90)
    px, py = to_px(drone_pos)
    draw.ellipse([px - r, py - r, px + r, py + r], fill=DRONE_MARKER_COLOR,
                 outline=(0, 0, 0, 255))
    lx, ly = to_px(leader_pos)
    draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=LEADER_MARKER_COLOR,
                 outline=(0, 0, 0, 255))

    return np.asarray(pil)


def build_topdown_panel(obstacles, leader_xy, drone_xy_off, drone_xy_on, width, height):
    """Bird's-eye static map: obstacle footprints, the (shared, scripted --
    identical for both runs) leader path, and both conditions' FULL flown
    drone paths overlaid in one shared frame. The physical divergence
    between the two runs is usually much easier to read here than in
    either camera feed or ESDF panel -- those show one drone's-eye view
    at a time, this shows both actual routes at once. Per-frame moving
    position dots are a cheap PIL overlay on a copy of this (see
    topdown_frame_with_markers), same pattern as the chart strip.

    World y plotted along the screen's HORIZONTAL axis (not x) -- on
    occlusion_cut the leader travels mostly along y (~12.5 m) across a
    narrow x corridor (~3 m), so a literal x-horizontal/y-vertical map
    would be a tall sliver nearly invisible in a wide, short panel.
    Swapping which world axis maps to which screen axis keeps
    ax.set_aspect('equal') (so the obstacle footprint stays a true
    circle, not stretched) while actually filling a landscape panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array(CHART_BG) / 255)
    ax.set_facecolor(np.array(CHART_BG) / 255)

    for obstacle in obstacles:
        cx, cy = obstacle["center"]
        if obstacle["shape"] == "circle":
            patch = plt.Circle((cy, cx), obstacle["radius"], color=(0.45, 0.45, 0.45),
                                zorder=1)
        else:
            h = obstacle["half_size"]
            patch = plt.Rectangle((cy - h, cx - h), 2 * h, 2 * h, color=(0.45, 0.45, 0.45),
                                   zorder=1)
        ax.add_patch(patch)

    ax.plot(leader_xy[:, 1], leader_xy[:, 0], color=np.array(LEADER_MARKER_COLOR[:3]) / 255,
            linewidth=1.6, linestyle="--", zorder=2, label="leader path")
    ax.plot(drone_xy_off[:, 1], drone_xy_off[:, 0], color=np.array(COLOR_OFF[:3]) / 255,
            linewidth=2.2, zorder=3, label="drone, OFF")
    ax.plot(drone_xy_on[:, 1], drone_xy_on[:, 0], color=np.array(COLOR_ON[:3]) / 255,
            linewidth=2.2, zorder=3, label="drone, ON")

    # Framed on the two DRONE paths only (not the leader, which runs
    # further ahead) -- the leader line still draws, just clipped past
    # the frame edge if it extends beyond it. Framing on the leader too
    # would stretch the axis and shrink the one divergence this panel
    # exists to show: drone-vs-drone, not drone-vs-leader.
    all_x = np.concatenate([drone_xy_off[:, 0], drone_xy_on[:, 0]])
    all_y = np.concatenate([drone_xy_off[:, 1], drone_xy_on[:, 1]])
    margin = 1.0
    ax.set_xlim(all_y.min() - margin, all_y.max() + margin)
    ax.set_ylim(all_x.min() - margin, all_x.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]", color=np.array(CHART_AXIS_TEXT) / 255, fontsize=10)
    ax.set_ylabel("x [m]", color=np.array(CHART_AXIS_TEXT) / 255, fontsize=10)
    ax.tick_params(colors=np.array(CHART_AXIS_TEXT) / 255, labelsize=9)
    ax.grid(color=np.array(CHART_GRID) / 255, linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(np.array(CHART_GRID) / 255)
    ax.legend(loc="upper right", fontsize=9, facecolor=tuple(np.array(CHART_BG) / 255),
              edgecolor=tuple(np.array(CHART_GRID) / 255), labelcolor="white")
    fig.tight_layout(pad=1.2)

    # The data-limit <-> pixel mapping used for the per-frame markers must
    # go through the AXES' actual pixel bounding box, not the full figure
    # canvas: tick labels, axis labels and the legend all eat into the
    # canvas, and ax.set_aspect('equal') shrinks the axes further to
    # preserve the circle/square obstacle shapes, so the plotted square is
    # noticeably smaller than (width, height). Mapping data coords
    # linearly across the full canvas (the old approach) put markers
    # outside the actual plotted square -- exactly the "goes off screen"
    # symptom. ax.get_window_extent() gives the true post-layout pixel box.
    fig.canvas.draw()
    renderer_ = fig.canvas.get_renderer()
    bbox_px = ax.get_window_extent(renderer_)
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    # bbox_px is in Matplotlib's bottom-left-origin display coords; flip to
    # the image array's top-left-origin row/col convention.
    ax_x0, ax_x1 = bbox_px.x0, bbox_px.x1
    ax_top, ax_bottom = fig_h_px - bbox_px.y1, fig_h_px - bbox_px.y0

    if buf.shape[1] != width or buf.shape[0] != height:
        sx, sy = width / buf.shape[1], height / buf.shape[0]
        ax_x0, ax_x1 = ax_x0 * sx, ax_x1 * sx
        ax_top, ax_bottom = ax_top * sy, ax_bottom * sy
        buf = np.asarray(Image.fromarray(buf).resize((width, height), Image.BILINEAR))

    return buf, (all_y.min() - margin, all_y.max() + margin, all_x.min() - margin,
                 all_x.max() + margin, (ax_x0, ax_x1, ax_top, ax_bottom))


def topdown_frame_with_markers(static_topdown, extent, pos_off, pos_on, leader_pos):
    """Per-frame moving-position dots on a COPY of the cached static map.
    extent is (y0, y1, x0, x1, px_bbox) -- see build_topdown_panel; px_bbox
    is the axes' own pixel box within the full panel image, used for the
    data->pixel mapping instead of the full panel's width/height."""
    pil = Image.fromarray(static_topdown.copy())
    draw = ImageDraw.Draw(pil, "RGBA")

    y0, y1, x0, x1, (ax_x0, ax_x1, ax_top, ax_bottom) = extent

    def to_px(x, y):
        u = (y - y0) / (y1 - y0)
        v = 1.0 - (x - x0) / (x1 - x0)
        return ax_x0 + u * (ax_x1 - ax_x0), ax_top + v * (ax_bottom - ax_top)

    r = max(5, int((ax_x1 - ax_x0) // 110) + 5)
    for pos, color in ((pos_off, COLOR_OFF), (pos_on, COLOR_ON), (leader_pos, LEADER_MARKER_COLOR)):
        px, py = to_px(pos[0], pos[1])
        # Leader often runs ahead of the drone-framed view -- clamp to the
        # plotted square's own edge (not the full panel) rather than draw
        # off the map or on top of the axis labels/legend around it.
        px = np.clip(px, ax_x0 + r + 2, ax_x1 - r - 2)
        py = np.clip(py, ax_top + r + 2, ax_bottom - r - 2)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0, 255))

    return np.asarray(pil)


def render_camera_frame(renderer, leader_names, df, t_center, fov_half_deg,
                         t_exposure_s, n_sub, d_fov, no_post_process, label, label_color):
    leader_pos = scene3d.interpolate_point(df, t_center, ("fx", "fy", "fz"))
    camfeed.update_leader_position(renderer, leader_names, leader_pos)

    accum = None
    sub_times = ([t_center] if n_sub == 1 else
                 np.linspace(t_center - t_exposure_s / 2, t_center + t_exposure_s / 2, n_sub))
    for t_sub in sub_times:
        pos, quat = scene3d.interpolate_pose(df, t_sub)
        eye, center, up = scene3d.camera_extrinsics(pos, quat)
        renderer.setup_camera(2 * fov_half_deg, center, eye, up)
        img = np.asarray(renderer.render_to_image(), dtype=np.float32)
        accum = img if accum is None else accum + img
    frame = (accum / n_sub).astype(np.uint8)

    t_arr = df["t"].to_numpy()
    nearest_row = df.iloc[int(np.argmin(np.abs(t_arr - t_center)))]
    frame = camfeed.draw_hud(frame, nearest_row, d_fov)
    if not no_post_process:
        frame = scene3d.apply_post_process(frame)
    return draw_label(frame, label, label_color), pos


def main():
    args = _parse_args()

    snapshot_off = args.snapshot_off or scene3d.find_snapshot_log(args.csv_off)
    snapshot_on = args.snapshot_on or scene3d.find_snapshot_log(args.csv_on)
    if not snapshot_off or not snapshot_on:
        raise SystemExit("Need both runs' own_map ESDF snapshot logs -- "
                          "pass --snapshot-off/--snapshot-on or place them next to "
                          "the trajectory CSVs using online_mapping_demo's own naming.")

    obstacles = scene3d.load_scenario_obstacles(args.scenario_config)
    camera_cfg, mapping_cfg = scene3d.load_camera_config(args.config_path)
    fov_half_deg = camera_cfg["fov_half_angle_deg"]
    t_exposure_s = 1.0 / mapping_cfg["update_rate_hz"]

    df_off = scene3d.load_trajectory(args.csv_off)
    df_on = scene3d.load_trajectory(args.csv_on)

    (esdf_t_off, esdf_d_off, esdf_xs_off, esdf_ys_off, gt_off, _vs_off, smax_off,
     _om_off) = scene3d.load_snapshot_log(snapshot_off)
    (esdf_t_on, esdf_d_on, esdf_xs_on, esdf_ys_on, gt_on, _vs_on, smax_on,
     _om_on) = scene3d.load_snapshot_log(snapshot_on)

    bounds = scene3d.scene_bounds(df_off, obstacles)
    t_arr = df_off["t"].to_numpy()
    end_step = args.end_step if args.end_step is not None else min(len(df_off), len(df_on))
    end_step = min(end_step, len(df_off), len(df_on))

    if args.output_fps:
        t_start, t_end = t_arr[args.start_step], t_arr[end_step - 1]
        n_out = max(1, int(round((t_end - t_start) * args.output_fps)) + 1)
        sample_times = np.linspace(t_start, t_end, n_out)
    else:
        sample_times = t_arr[args.start_step:end_step:args.stride]

    # Top-down path context restricted to the rendered window (start_step:
    # end_step) -- the pre-divergence lead-in is identical for both and
    # only adds clutter to the bird's-eye map.
    sl = slice(args.start_step, end_step)
    static_topdown, topdown_extent = build_topdown_panel(
        obstacles,
        df_off[["fx", "fy"]].to_numpy()[sl],
        df_off[["x", "y"]].to_numpy()[sl],
        df_on[["x", "y"]].to_numpy()[sl],
        2 * args.panel_width, args.chart_height)

    panel_w, panel_h = args.panel_width, args.panel_height
    renderer = o3d.visualization.rendering.OffscreenRenderer(panel_w, panel_h)
    leader_names = camfeed.build_scene(renderer, obstacles, bounds, args.obstacle_height,
                                        panel_w, panel_h)
    renderer.scene.view.set_antialiasing(True, temporal=False)
    renderer.scene.view.set_sample_count(4)
    renderer.scene.view.set_ambient_occlusion(True)

    import imageio
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    # quality=10 (imageio's 1-10 scale) was confirmed to pick x264's "High
    # 4:4:4 Predictive" profile despite yuv420p chroma -- decodes fine in
    # ffmpeg itself but most standard players (browsers, QuickTime, many
    # OS defaults) can't open that profile at all, so the file silently
    # "won't start" even though it isn't corrupt. Forcing an explicit
    # High/yuv420p profile + CRF instead of the quality scale avoids this.
    playback_fps = args.playback_fps or args.output_fps or 20.0
    hold = max(1, int(round(playback_fps / (args.output_fps or 20.0)))) \
        if args.playback_fps and args.playback_fps < (args.output_fps or 20.0) else 1
    # --playback-fps below --output-fps holds each sampled tick for
    # `hold` extra video frames (slow motion); above it would require
    # actual interpolation between ticks, not attempted here -- use
    # --output-fps to resample more finely instead.
    writer = imageio.get_writer(
        args.video, fps=playback_fps, codec="libx264",
        macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "high",
                        "-crf", "18", "-preset", "medium", "-movflags", "+faststart"])

    n_sub = max(1, args.blur_subsamples)
    try:
        for t_center in sample_times:
            cam_off, drone_pos_off = render_camera_frame(
                renderer, leader_names, df_off, t_center, fov_half_deg, t_exposure_s,
                n_sub, args.d_fov, args.no_post_process, args.label_off, COLOR_OFF)
            cam_on, drone_pos_on = render_camera_frame(
                renderer, leader_names, df_on, t_center, fov_half_deg, t_exposure_s,
                n_sub, args.d_fov, args.no_post_process, args.label_on, COLOR_ON)

            idx_off = int(np.argmin(np.abs(esdf_t_off - t_center)))
            idx_on = int(np.argmin(np.abs(esdf_t_on - t_center)))
            leader_off = scene3d.interpolate_point(df_off, t_center, ("fx", "fy", "fz"))
            leader_on = scene3d.interpolate_point(df_on, t_center, ("fx", "fy", "fz"))
            esdf_off_img = esdf_frame(esdf_d_off, idx_off, gt_off, smax_off, esdf_xs_off,
                                       esdf_ys_off, drone_pos_off, leader_off, args.label_off,
                                       COLOR_OFF, panel_w, panel_h)
            esdf_on_img = esdf_frame(esdf_d_on, idx_on, gt_on, smax_on, esdf_xs_on,
                                      esdf_ys_on, drone_pos_on, leader_on, args.label_on,
                                      COLOR_ON, panel_w, panel_h)

            leader_now = scene3d.interpolate_point(df_off, t_center, ("fx", "fy"))
            topdown = topdown_frame_with_markers(
                static_topdown, topdown_extent, drone_pos_off[:2], drone_pos_on[:2], leader_now)

            top = np.concatenate([cam_off, cam_on], axis=1)
            bottom = np.concatenate([esdf_off_img, esdf_on_img], axis=1)
            panel_w_total = top.shape[1]
            if topdown.shape[1] != panel_w_total:
                topdown = np.asarray(Image.fromarray(topdown).resize(
                    (panel_w_total, topdown.shape[0]), Image.BILINEAR))
            composite = np.concatenate([top, bottom, topdown], axis=0)
            for _ in range(hold):
                writer.append_data(composite)
    finally:
        writer.close()

    print(f"Wrote {len(sample_times)} frames ({sample_times[0]:.2f}s-{sample_times[-1]:.2f}s) "
          f"to {args.video}")


if __name__ == "__main__":
    main()
