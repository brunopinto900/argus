# Tooling Reference

Deep-dive usage reference for the visualization, metrics, and comparison
scripts. See [README.md](README.md) for the project overview, quickstart
(`./run.sh`), and the three core executables (`argus_mpc`,
`trailer_mpc_demo`, `online_mapping_demo`) these tools operate on.

## Occlusion-aware cost (implementation)

Implementation reference — the result, the table, and the gradient
derivation are in [README.md](README.md#paper-summary)'s Paper summary and
in the [paper](paper/paper.pdf) itself; this section covers only what's
specific to the code. `online_mapping_demo` adds an **opt-in occlusion-aware
cost** (`occlusion_avoidance.mode: cost`, default `disabled`) that reuses
the onboard-built ESDF (`own_map`), marching it along the line from drone to
target each stage to find `s*` — how far along that line the worst point
sits — and linearizing the distance field there instead of at the drone's
own position:

```
occ_p0 = (1-s*)·drone_p0 + s*·p_target
r_occ  = max(0, τ_occ − occ_lin(x))      # squared hinge, NLS auto-squares it
```

```yaml
occlusion_avoidance:
  mode: cost         # 'disabled' (default) or 'cost'
  tau_occ_m: 1.68     # activation margin -- derived from
                       # trailer.distance_to_target * tan(camera.fov_half_angle_deg)
```

Measured with `scripts/run_variant.py --scenario forest_clutter`; "reacq" =
lost-contact events, out of the FOV cone *or* >90% occluded (see
`scripts/compute_metrics.py`'s `--lost-contact-occlusion-pct`). `r_center`
is `camera.center_mode: cost`, an exploratory, untuned-weight residual (see
`generate_mpc.py`'s `CENTER_MODE`). See `design_decisions.md` (local,
gitignored) for the full derivation.

## Telemetry dashboard

One figure, all state over time — works on any of the three logs above,
auto-adding an obstacle-distance panel when the log has one:

```bash
python3 scripts/plot_telemetry_dashboard.py --csv-path logs/trailer_mpc_trajectory.csv
python3 scripts/plot_telemetry_dashboard.py --csv-path logs/online_mapping_trajectory.csv --real-target-m 0.27
python3 scripts/plot_telemetry_dashboard.py --csv-path logs/trajectory.csv   # argus_mpc, the default --csv-path
```

Panels: XY path (with every obstacle outline, when present — pass
`--scenario-config` if plotting a non-default scenario file), yaw error
(MPC vs. bearing to the FOV target), velocity, attitude, body rates, XY
tracking error, distance to the FOV target, and (only when the log has a
`dist_obstacle` column) distance to the obstacle with the `d_safe` config
knob, the actual physical safety target (`--real-target-m`), and the run's
closest approach marked. Add `--save-png PATH` for a headless export
instead of an interactive window; `--keepaway-m` draws a constant
reference line on the distance-to-target panel (e.g.
`trailer.distance_to_target`).

## Animations

3-D trajectory view (`argus_mpc`/`trailer_mpc_demo`, already covered by
`--save-gif`/`--save-png` in README's How to run):

```bash
python3 scripts/animate_xy.py --csv-path logs/trailer_mpc_trajectory.csv
```

Same 3-D view also works on `online_mapping_demo`'s log — no extra flags
needed, the ESDF snapshot log is auto-detected next to `--csv-path` (by
`online_mapping_demo`'s fixed `*_trajectory.csv`/`*_esdf_snapshots.bin`
naming) and draws two extra layers: every ground-truth obstacle as an
extruded shape (cosmetic height only — `own_map` is still a single
z-layer, no real height data to draw instead), and `own_map`'s ESDF as a
faded textured plane at z=0:

```bash
python3 scripts/animate_xy.py --csv-path logs/online_mapping_trajectory.csv
```

2-D ESDF map animation (`online_mapping_demo` only) — own_map's distance
field sweeping over time, every ground-truth obstacle shown in grey
(regardless of observed state), sensor-range circle, camera FOV wedge, and
all three paths (leader/follower/MPC actual) overlaid:

```bash
python3 argus_follower/scripts/plot_online_mapping.py
python3 argus_follower/scripts/plot_online_mapping.py --save-gif docs/online_mapping.gif --fps 10
```

## Offline 3-D viewer and camera footage (no Gazebo)

Real rendered 3-D playback and camera footage from a logged trajectory CSV
— ground (trees for circle obstacles, boxes for square ones; a low-poly
person for the leader; a real quadrotor mesh for the drone), and (for the
footage renderer) the drone's own forward-facing camera view at the
OAK-D-Lite's native 640x400 resolution, with real motion blur from
sub-frame temporal sampling, not an analytic estimate. Needs its own
dependencies (`scripts/requirements-viz.txt`, separate from
`requirements.txt` — only needed for these five scripts):

```bash
pip install -r scripts/requirements-viz.txt
```

Interactive/scrubbable 3-D scene (PyVista; `--off-screen --save-png` for a
headless single-frame export, no display needed). Two extra panels
auto-enable alongside the main 3-D view: own_map's live 2-D ESDF heatmap
(auto-detected next to `--csv-path`, silently skipped if there's no
snapshot log — e.g. `online_mapping_demo`'s other, non-mapping logs), and
the drone's own live camera footage (`--no-camera-panel` to drop it for a
faster interactive frame rate). Space toggles real-time play/pause once
the window is open; the slider scrubs manually at any time:

```bash
python3 scripts/view_scene3d.py --csv-path logs/online_mapping_trajectory.csv \
    --scenario-config config/forest.yaml --animate
```

Camera-footage frames (Open3D `OffscreenRenderer`, always headless) —
`--blur-subsamples 1` for a sharp baseline, higher for visible motion
blur (default 8), `--video PATH` to assemble an mp4 instead of a PNG
sequence, `--output-fps 30` to resample smoothly instead of one frame
per 20Hz control tick. Includes a telemetry HUD (`dist_obstacle`/slack/
solve time/occlusion%, red badge when close or heavily occluded) by
default — `--no-hud` for a clean shot:

```bash
python3 scripts/render_camera_footage.py --csv-path logs/online_mapping_trajectory.csv \
    --scenario-config config/forest.yaml --video logs/camera_footage/forest.mp4
```

This is a separate, visual-inspection tool — it doesn't feed back into
`run_scenario_suite.py`'s gates or replace `compute_metrics.py`'s
analytic blur-from-body-rate estimate, which stays the fast, suite-wide
metric.

Composite "hero" demo video — chase-cam 3-D view (drone mesh + leader +
trees) with the camera footage picture-in-picture, a light cinematic
grade applied (saturation/contrast/vignette) — the actual portfolio/CV
deliverable, not a diagnostic tool:

```bash
python3 scripts/render_hero_video.py --csv-path logs/online_mapping_trajectory.csv \
    --scenario-config config/forest.yaml --video logs/camera_footage/forest_hero.mp4
```

Occlusion OFF-vs-ON comparison video (`scripts/render_occlusion_comparison.py`)
— the tool behind the [Occlusion-aware cost](#occlusion-aware-cost-implementation)
section and the top-of-README GIF: two logged runs (same scenario, occlusion cost off
and on) side by side, either as full technical composites (3-D view + camera
footage + topdown panel for each) or, with `--chase-cam`, as a chase-cam view
with the onboard camera picture-in-picture, plus `--social` (big plain-
language stat readout instead of the technical HUD) and `--caption "..."`
(lower-third caption bar) for presentation-ready output:

```bash
python3 scripts/render_occlusion_comparison.py \
    --csv-off logs/band_final_off/trajectory.csv --csv-on logs/band_final_on/trajectory.csv \
    --chase-cam --social --caption "Occlusion cost: off vs. on" \
    --video logs/camera_footage/occlusion_comparison_chase.mp4
```

## Run metrics

`online_mapping_demo` only (needs its `clearance_true`/`occlusion_clearance`/
`solve_time_ms` log columns). Prints a summary table — collision safety
(measured against `--real-target-m`, the actual physical requirement, not
the inflated `d_safe` cost knob), tracking error, FOV bearing
(`pitch_bearing_deviation_from_design_deg` is the real pitch "error," since
0 is not the design target — see `config/quadrotor.yaml`'s
`trailer.target_bearing_deg`), occlusion, solver health, and smoothness —
plus a 7-panel figure:

```bash
./build/online_mapping_demo
python3 scripts/compute_metrics.py
python3 scripts/compute_metrics.py --save-png docs/metrics.png   # headless export
```

## Config variants (A/B comparisons)

Named config overlays in **[`config/variants.yaml`](config/variants.yaml)** (merged onto `config/quadrotor.yaml` — any key, including mode switches like `obstacle_avoidance.mode: slack|cost`) let you A/B formulations without hand-editing weights or checking out old commits. `scripts/run_variant.py` applies one or more variants, rebuilds only if the resolved config changed, runs the scenario suite, restores the committed config afterward, and prints a pooled gates/clearance/occlusion/reacquisition/solve-time table. `--scenario NAME` (repeatable) restricts every variant to just that scenario instead of the full 16-scenario suite — cached separately per `(variant, build_hash, scenario-set)` under `logs/variants/<name>/<hash>/<scenario-set>/`, so a quick single-scenario comparison never costs a full-suite run, and re-running an unchanged request is instant.

```bash
python3 scripts/run_variant.py --list
python3 scripts/run_variant.py --variant band_tracking --variant band_azimuth0_center --scenario forest_clutter
python3 scripts/run_variant.py --variant trailer_on_slack_on --fresh   # ignore caches, fresh timing
```

`scripts/run_scenario_suite.py` takes the same `--scenario NAME` directly, for running the suite (or one scenario of it) against whatever's currently in `config/quadrotor.yaml` without going through a named variant.
