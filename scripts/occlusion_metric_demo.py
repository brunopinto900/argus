"""
Standalone validation demo for the occlusion metric: how much an obstacle
degrades the quality of the view of the target, computed from pure
synthetic geometry -- no VoxelGrid, no MPC, nothing from the real
pipeline. Not integrated anywhere yet; this is only for checking the math
behaves correctly on a handful of deliberately chosen edge cases before
deciding whether/how to wire it into the real pipeline.

Final metric (after several wrong turns during design discussion -- see
design_decisions.md for the full reasoning trail):

    d   = lateral (perpendicular) clearance from an obstacle's NEAR EDGE
          to the drone->target bearing vector, measured only over the
          segment between the drone and the target (depth-gated: an
          obstacle whose projection onto the bearing direction falls
          at or beyond the target's own range cannot degrade the view,
          no matter how well aligned or how large it is -- a wall behind
          the target you're looking at does not blur the shot).

    occ = 100 * max(0, 1 - d / d_FOV)

          d_FOV is the lateral-clearance scale beyond which an obstacle
          is considered too far off-axis to matter: d=0 (obstacle's body
          actually crosses the bearing vector) -> 100%; d>=d_FOV -> 0%;
          linear in between. This is the original formula the whole
          exercise started from (theta_occ = theta_FOV * max(0, 1 -
          d/d_FOV)), just with `d` finally pinned down precisely as
          "edge-to-bearing-vector lateral distance", not LOS range and
          not angular distance to the target's own footprint (both were
          tried and rejected during design discussion).

    Multiple obstacles combine via MAX occlusion (equivalently: MIN
    remaining clearance), not sum/average/union -- the single worst
    occluder determines shot quality; a second, less-bad occluder
    elsewhere in frame doesn't make a clear shot any worse.

Why earlier framings were wrong (kept here as a record, not a pitch):
  - A single ray/point test along the bearing vector only checks one
    infinitesimal direction; a big/close obstacle whose edge still
    grazes the line reads as "clear" even though the view is nearly
    ruined -- this is why the metric is edge-distance, not a hit test.
  - Measuring against the TARGET's own angular footprint (how much of
    the target's silhouette is covered) answers a different question
    ("is the target visible") than the one asked here ("is the shot
    degraded") -- an obstacle sitting next to, not on, the target still
    clutters the frame despite leaving the target's own silhouette
    untouched.
  - Measuring against the whole FOV wedge (how much of the camera's
    total view is cluttered) over-counts obstacles that are nowhere
    near the sightline to the target and under-weights the one thing
    that actually matters for tracking this particular target.
  - A depth gate keyed on the TARGET's range (not a fixed max range)
    is what correctly lets a wall far behind the target stay harmless
    while a wall at the same alignment but nearer than the target still
    degrades the shot.

Two implementations are computed for every case:
  - exact_occlusion_pct(): closed-form perpendicular distance from each
    gated obstacle's center to the infinite bearing line, minus its
    radius, minimized over obstacles. Only possible here because this
    demo has clean analytic obstacle shapes (center/radius) to work with.
  - sampled_occlusion_pct(): march n_samples points along the actual
    drone->target segment and, at each one, take the nearest gated
    obstacle's surface distance -- the running minimum across the march
    stands in for d. This is the method that will actually generalize to
    the real pipeline, where the only thing available at runtime is
    own_map's distance queries (the same ESDF already used for the MPC's
    obstacle cost), not a clean list of analytic obstacle shapes.
    Plotted here purely to confirm it converges to the exact answer
    before ever depending on it for real data.

Each row is ONE deliberately chosen, big/legible scenario -- not a dense
sweep -- with an explicit, derived (not guessed) expected value, so a
mismatch is visible at a glance rather than something you have to trust
a curve to notice.

Run:
    python3 scripts/occlusion_metric_demo.py
    python3 scripts/occlusion_metric_demo.py --save-png logs/occlusion_metric_demo.png
"""

import argparse

import numpy as np
import yaml


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-png", metavar="PATH",
                    help="Render headlessly to PATH instead of showing a window")
    p.add_argument("--n-samples", type=int, default=65,
                    help="March steps for sampled_occlusion_pct() (default: %(default)s)")
    p.add_argument("--d-fov", type=float, default=1.68,
                    help="Lateral-clearance falloff scale [m] (default: %(default)s, "
                         "derived as trailer.distance_to_target * tan(fov_half_angle_deg) "
                         "-- see design_decisions.md's 'd_FOV/tau_occ changed to 1.68m, "
                         "derived -- was 4.0m, picked'; matches scripts/compute_metrics.py's "
                         "default)")
    p.add_argument("--config-path", default="config/quadrotor.yaml",
                    help="Source of camera.fov_half_angle_deg, drawn on every scenario panel "
                         "purely for visual context -- it is not part of the occlusion math "
                         "(default: %(default)s)")
    return p.parse_args()


_args = _parse_args()

import matplotlib
if _args.save_png:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    _cfg = yaml.safe_load(open(_args.config_path))
    FOV_HALF_ANGLE_DEG = (_cfg.get("camera") or {}).get("fov_half_angle_deg", 40.0)
except FileNotFoundError:
    FOV_HALF_ANGLE_DEG = 40.0

D_FOV = _args.d_fov


# ── Core occlusion math ──────────────────────────────────────────────────

def bearing_unit(drone, target):
    """Unit vector from drone to target."""
    v = np.array(target, dtype=float) - np.array(drone, dtype=float)
    return v / np.linalg.norm(v)


def obstacle_projection(drone, u, obs):
    """(s, edge_dist): `s` is how far along the bearing direction `u` the
    obstacle's center projects (negative = behind the drone); `edge_dist`
    is the lateral clearance from the obstacle's NEAR EDGE to the bearing
    line (0 if the line actually passes through the obstacle's body)."""
    w = np.array(obs["center"], dtype=float) - np.array(drone, dtype=float)
    s = float(np.dot(w, u))
    perp = w - s * u
    center_dist = float(np.linalg.norm(perp))
    edge_dist = max(0.0, center_dist - obs["size"])
    return s, edge_dist


def depth_gated_edge_distances(drone, target, obstacles):
    """Edge-distances of every obstacle whose projection along the bearing
    direction falls strictly between the drone and the target -- anything
    at or beyond the target's own range is excluded here, once, so both
    metrics below share identical gating instead of re-deriving it."""
    u = bearing_unit(drone, target)
    R_t = float(np.linalg.norm(np.array(target) - np.array(drone)))
    out = []
    for obs in obstacles:
        s, edge_dist = obstacle_projection(drone, u, obs)
        if 0.0 < s < R_t:
            out.append(edge_dist)
    return out


def exact_occlusion_pct(drone, target, obstacles, d_fov=D_FOV):
    """Closed-form: worst (smallest-clearance) depth-gated obstacle sets
    the result -- max occlusion across obstacles, not a sum or union."""
    edge_dists = depth_gated_edge_distances(drone, target, obstacles)
    if not edge_dists:
        return 0.0
    worst_clearance = min(edge_dists)
    return 100.0 * max(0.0, 1.0 - worst_clearance / d_fov)


def sampled_occlusion_pct(drone, target, obstacles, n_samples, d_fov=D_FOV):
    """Approximate: march n_samples points along the actual drone->target
    segment and track the minimum nearest-obstacle surface distance seen
    along the way. Stands in for what the real pipeline will do (a bounded
    march/raycast against own_map's distance field), where clean analytic
    circles won't be available the way they are in this synthetic demo."""
    gated = [obs for obs in obstacles
             if 0.0 < obstacle_projection(drone, bearing_unit(drone, target), obs)[0]
             < float(np.linalg.norm(np.array(target) - np.array(drone)))]
    if not gated:
        return 0.0

    u = bearing_unit(drone, target)
    R_t = float(np.linalg.norm(np.array(target) - np.array(drone)))
    drone_arr = np.array(drone, dtype=float)

    min_clearance = np.inf
    for t in np.linspace(0.0, R_t, n_samples):
        point = drone_arr + t * u
        for obs in gated:
            d = float(np.hypot(point[0] - obs["center"][0], point[1] - obs["center"][1])) - obs["size"]
            min_clearance = min(min_clearance, max(0.0, d))
    return 100.0 * max(0.0, 1.0 - min_clearance / d_fov)


# ── Shared scenario constants -- deliberately large, so geometry is
# legible at a glance instead of needing to infer it from a number ──────

DRONE = (0.0, 0.0)
TARGET = (10.0, 0.0)
R_TARGET = 1.0  # drawn for visual context only -- not part of the occlusion math
R_T = float(np.hypot(TARGET[0] - DRONE[0], TARGET[1] - DRONE[1]))

OBS_RADIUS = 1.5

XLIM, YLIM = (-2, 14), (-7, 7)


def obstacle_at(lateral_offset, s=5.0, radius=OBS_RADIUS):
    """One circular obstacle `s` meters along the bearing direction
    (which runs along +x here, drone->target), offset `lateral_offset`
    meters perpendicular to it. Valid only for this demo's axis-aligned
    drone/target pair, where the bearing direction is exactly +x."""
    return {"shape": "circle", "center": (s, lateral_offset), "size": radius}


# ── Drawing helpers ───────────────────────────────────────────────────────

def draw_base_scene(ax, title):
    boresight_deg = np.degrees(np.arctan2(TARGET[1] - DRONE[1], TARGET[0] - DRONE[0]))
    fov_radius = 1.2 * max(abs(XLIM[0]), abs(XLIM[1]), abs(YLIM[0]), abs(YLIM[1]))
    ax.add_patch(mpatches.Wedge(DRONE, fov_radius,
                                 boresight_deg - FOV_HALF_ANGLE_DEG,
                                 boresight_deg + FOV_HALF_ANGLE_DEG,
                                 facecolor="gold", edgecolor="none", alpha=0.12, zorder=0,
                                 label=f"camera FOV (±{FOV_HALF_ANGLE_DEG:.0f}°, context only)"))

    u = bearing_unit(DRONE, TARGET)
    perp = np.array([-u[1], u[0]])
    p0, p1 = np.array(DRONE), np.array(TARGET)
    band = [p0 + perp * D_FOV, p1 + perp * D_FOV, p1 - perp * D_FOV, p0 - perp * D_FOV]
    ax.add_patch(mpatches.Polygon(band, closed=True, facecolor="tab:orange", alpha=0.12,
                                   edgecolor="none", zorder=0,
                                   label=f"d_FOV falloff band (±{D_FOV:.1f} m)"))

    ax.plot(*DRONE, marker="^", color="tab:blue", markersize=12, zorder=5)
    ax.annotate("drone", DRONE, textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.add_patch(mpatches.Circle(TARGET, R_TARGET, facecolor="tab:green",
                                  edgecolor="tab:green", alpha=0.5, zorder=4))
    ax.plot(*TARGET, marker="*", color="tab:green", markersize=12, zorder=5)
    ax.annotate("target", TARGET, textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.plot([DRONE[0], TARGET[0]], [DRONE[1], TARGET[1]], "--",
             color="gray", linewidth=1.2, zorder=1, label="bearing vector")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def draw_obstacles(ax, obstacles, color="tab:red", alpha=0.4):
    for obs in obstacles:
        ax.add_patch(mpatches.Circle(obs["center"], obs["size"], facecolor=color,
                                      edgecolor=color, alpha=alpha, zorder=3))


def annotate_metric(ax, exact_val, sampled_val, expected_val=None, expected_label=""):
    """Prints the computed numbers directly on the scenario panel,
    color-coded green/red against the precomputed expected value where
    one exists, so a mismatch is visible without a second axes."""
    lines = [f"exact:    {exact_val:5.1f}%", f"sampled:  {sampled_val:5.1f}%"]
    color = "black"
    if expected_val is not None:
        lines.append(f"expected: {expected_val:5.1f}% {expected_label}")
        color = "tab:green" if abs(exact_val - expected_val) < 1.0 else "tab:red"
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=10,
             family="monospace", va="bottom", ha="left", color=color, zorder=10,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor=color, boxstyle="round"))


# ── Cases ─────────────────────────────────────────────────────────────────

def case_clear(ax):
    """Obstacle's edge well past d_FOV from the bearing vector -- the
    baseline "nothing should trigger" case."""
    off = OBS_RADIUS + D_FOV + 1.0  # edge clearance = D_FOV + 1.0
    obs = [obstacle_at(off)]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)

    draw_base_scene(ax, "Case 1: clearly clear\n"
                         f"(edge clearance = d_FOV+1.0m = {off - OBS_RADIUS:.1f} m)")
    draw_obstacles(ax, obs)
    annotate_metric(ax, exact, sampled, 0.0, "(well outside the falloff band)")


def case_partial(ax):
    """Obstacle's edge sits at exactly half of d_FOV -- validates the
    linear falloff isn't just correct at the two endpoints."""
    off = OBS_RADIUS + 0.5 * D_FOV
    obs = [obstacle_at(off)]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)

    draw_base_scene(ax, "Case 2: halfway into the falloff band\n"
                         f"(edge clearance = d_FOV/2 = {0.5 * D_FOV:.1f} m)")
    draw_obstacles(ax, obs)
    annotate_metric(ax, exact, sampled, 50.0, "(linear falloff midpoint)")


def case_boundary(ax):
    """Obstacle's edge sits exactly at d_FOV -- the boundary where
    occlusion first becomes zero, validates no off-by-one either way."""
    off = OBS_RADIUS + D_FOV
    obs = [obstacle_at(off)]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)

    draw_base_scene(ax, "Case 3: exactly at the d_FOV boundary\n"
                         f"(edge clearance = d_FOV = {D_FOV:.1f} m)")
    draw_obstacles(ax, obs)
    annotate_metric(ax, exact, sampled, 0.0, "(falloff just reaches zero)")


def case_dead_ahead(ax):
    """Obstacle's body actually crosses the bearing vector, nearer than
    the target -- the "obviously should fully degrade the shot" case."""
    obs = [obstacle_at(0.0)]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)

    draw_base_scene(ax, "Case 4: dead ahead, crossing the bearing vector\n"
                         "(edge clearance = 0 m, nearer than the target)")
    draw_obstacles(ax, obs)
    annotate_metric(ax, exact, sampled, 100.0, "(directly in the way)")


def case_behind_target(ax):
    """Identical obstacle and alignment as Case 4, just moved past the
    target's own range -- must read 0% regardless of size or alignment,
    proving the depth gate dominates (a wall behind the target you're
    looking at does not blur the shot)."""
    obs = [{"shape": "circle", "center": (R_T + 5.0, 0.0), "size": OBS_RADIUS}]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)

    draw_base_scene(ax, "Case 5: same obstacle as Case 4, moved past the target\n"
                         "(perfectly aligned, but farther than R_t)")
    draw_obstacles(ax, obs, color="tab:brown")
    annotate_metric(ax, exact, sampled, 0.0, "(behind the target cannot degrade the shot)")


def case_max_combination(ax):
    """Two obstacles of differing severity -- validates that the combined
    result is the WORST one (max occlusion / min clearance), not an
    average or a sum: a second, milder occluder elsewhere in frame must
    not make an already-clear shot any clearer, nor should it make a
    badly-occluded shot score as anything other than badly-occluded."""
    severe = obstacle_at(OBS_RADIUS + 0.2 * D_FOV, s=5.0)   # edge clearance 0.2*d_FOV -> 80%
    mild   = obstacle_at(-(OBS_RADIUS + 0.8 * D_FOV), s=3.0)  # edge clearance 0.8*d_FOV -> 20%
    obs = [severe, mild]
    exact = exact_occlusion_pct(DRONE, TARGET, obs)
    sampled = sampled_occlusion_pct(DRONE, TARGET, obs, _args.n_samples)
    exact_severe = exact_occlusion_pct(DRONE, TARGET, [severe])
    exact_mild = exact_occlusion_pct(DRONE, TARGET, [mild])

    draw_base_scene(ax, "Case 6: two obstacles, differing severity\n"
                         f"(alone: {exact_severe:.0f}% / {exact_mild:.0f}% -- combined must be the max)")
    draw_obstacles(ax, [severe], color="tab:red")
    draw_obstacles(ax, [mild], color="tab:brown")
    annotate_metric(ax, exact, sampled, max(exact_severe, exact_mild), "(worst occluder wins)")


def main():
    cases = [case_clear, case_partial, case_boundary,
             case_dead_ahead, case_behind_target, case_max_combination]
    n_cols = 3
    n_rows = (len(cases) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 7.5 * n_rows))
    fig.suptitle("Occlusion metric -- edge-case validation "
                 "(d = lateral clearance to bearing vector, synthetic geometry, not from the real pipeline)")
    for case_fn, ax in zip(cases, axes.flat):
        case_fn(ax)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if _args.save_png:
        fig.savefig(_args.save_png, dpi=150)
        print(f"Saved {_args.save_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
