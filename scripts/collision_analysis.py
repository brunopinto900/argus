#!/usr/bin/env python3
"""General collision-clearance diagnostic for any scenario in config/.

Answers (without assuming the conclusion): is a logged near-obstacle
closest-approach a transient or a steady-state equilibrium; which cost
term (tracking vs. avoidance) dominates at that instant; did the
solver's own open-loop plan already violate clearance, or only the
realized closed-loop trajectory; and how does realized clearance respond
to the avoidance cost weight.

Usage (run from repo root, after building online_mapping_demo):
    ./build/online_mapping_demo config/standoff_conflict.yaml
    python3 scripts/collision_analysis.py standoff_conflict

    # weight sweep (slow -- regenerates + rebuilds the solver per point):
    python3 scripts/collision_analysis.py standoff_conflict --sweep-robs-weight

Reads:
    logs/online_mapping_trajectory.csv  (closed-loop realized states)
    logs/online_mapping_horizon.csv     (open-loop predicted horizon per tick)
    config/<scenario>.yaml              (obstacle geometry)
    config/quadrotor.yaml               (cost weights, d_safe_m)

Writes PNGs under logs/collision_analysis/<scenario>/.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
LOGS_DIR = REPO_ROOT / "logs"
REAL_TARGET_M = 0.27

# generate_mpc.py's AcadosOcpSolver dlopen()s libacados/libqpOASES_e at
# import time -- needs ACADOS_SOURCE_DIR/lib on LD_LIBRARY_PATH, which is
# normally only set by sourcing a shell profile interactively, not
# inherited by a bare subprocess. Set explicitly so --sweep-robs-weight
# doesn't depend on how this script itself was launched.
ACADOS_SOURCE_DIR = pathlib.Path.home() / "acados"
SUBPROCESS_ENV = dict(os.environ)
SUBPROCESS_ENV["ACADOS_SOURCE_DIR"] = str(ACADOS_SOURCE_DIR)
SUBPROCESS_ENV["LD_LIBRARY_PATH"] = str(ACADOS_SOURCE_DIR / "lib") + ":" + SUBPROCESS_ENV.get("LD_LIBRARY_PATH", "")
VENV_PYTHON = pathlib.Path.home() / ".venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def load_quadrotor_cfg():
    with open(CONFIG_DIR / "quadrotor.yaml") as f:
        return yaml.safe_load(f)


def load_obstacles(scenario_name):
    with open(CONFIG_DIR / f"{scenario_name}.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("obstacles", [])


def obstacle_clearance_xy(x, y, obstacles):
    """Ground-truth clearance from a point to the nearest obstacle surface."""
    worst = np.inf
    for obs in obstacles:
        cx, cy = obs["center"]
        if obs["shape"] == "circle":
            d = np.hypot(x - cx, y - cy) - obs["radius"]
        else:
            hs = obs["half_size"]
            dx = max(abs(x - cx) - hs, 0.0)
            dy = max(abs(y - cy) - hs, 0.0)
            d = np.hypot(dx, dy) if (abs(x - cx) > hs or abs(y - cy) > hs) \
                else -max(hs - abs(x - cx), hs - abs(y - cy))
        worst = min(worst, d)
    return worst


def load_traj():
    path = LOGS_DIR / "online_mapping_trajectory.csv"
    if not path.exists():
        sys.exit(f"missing {path} -- run ./build/online_mapping_demo config/<scenario>.yaml first")
    return pd.read_csv(path)


def load_horizon():
    path = LOGS_DIR / "online_mapping_horizon.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── 2. clearance vs time ─────────────────────────────────────────────────

def plot_clearance_vs_time(df, scenario_name, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["t"], df["clearance_true"], color="tab:blue", lw=1.3, label="clearance_true (drone-to-obstacle, ground truth)")
    ax.axhline(REAL_TARGET_M, color="tab:red", linestyle="--", lw=1.5,
               label=f"hard gate = {REAL_TARGET_M} m")
    worst_i = df["clearance_true"].idxmin()
    ax.plot(df["t"][worst_i], df["clearance_true"][worst_i], "rv", ms=10, zorder=5,
            label=f"min = {df['clearance_true'][worst_i]:.3f} m @ t={df['t'][worst_i]:.2f}s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("clearance [m]")
    ax.set_title(f"{scenario_name}: drone-obstacle clearance vs time")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "clearance_vs_time.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, worst_i


# ── 3a. stage cost breakdown + gradient magnitudes at the critical tick ──

def cost_breakdown_at_tick(df, tick_idx, qcfg):
    W_stage = qcfg["cost"]["W_stage"]
    W_x, W_y, W_z = W_stage[0], W_stage[1], W_stage[2]
    W_robs = W_stage[9]
    D_SAFE = qcfg["obstacle_avoidance"]["d_safe_m"]

    row = df.iloc[tick_idx]
    ex, ey, ez = row["x"] - row["xt"], row["y"] - row["yt"], row["z"] - row["zt"]
    tracking_cost = W_x * ex**2 + W_y * ey**2 + W_z * ez**2
    # |d(cost)/d(x,y,z)| via the position components (dominant terms)
    tracking_grad_mag = 2.0 * np.hypot(W_x * ex, W_y * ey)

    d_lin = row["dist_obstacle"]  # own_map's belief -- what the solver itself saw
    r_obs = max(0.0, D_SAFE - d_lin)
    avoidance_cost = W_robs * r_obs**2
    # scalar hinge-residual gradient magnitude (direction needs the ESDF
    # gradient at that stage, not currently logged -- see script docstring
    # caveat printed below)
    avoidance_grad_mag = 2.0 * W_robs * r_obs

    return {
        "t": row["t"], "d_lin_own_map": d_lin, "clearance_true": row["clearance_true"],
        "tracking_cost": tracking_cost, "tracking_grad_mag": tracking_grad_mag,
        "avoidance_cost": avoidance_cost, "avoidance_grad_mag": avoidance_grad_mag,
        "D_SAFE": D_SAFE, "r_obs_residual": r_obs,
    }


def plot_cost_breakdown(breakdown, scenario_name, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["tracking", "avoidance (r_obs)"]

    costs = [breakdown["tracking_cost"], breakdown["avoidance_cost"]]
    axes[0].bar(labels, costs, color=["tab:blue", "tab:red"])
    axes[0].set_ylabel("stage cost contribution")
    axes[0].set_title("cost magnitude")
    for i, v in enumerate(costs):
        axes[0].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)

    grads = [breakdown["tracking_grad_mag"], breakdown["avoidance_grad_mag"]]
    axes[1].bar(labels, grads, color=["tab:blue", "tab:red"])
    axes[1].set_ylabel("|d(cost)/d(state)| (position terms; avoidance is\nscalar hinge magnitude, direction not logged)")
    axes[1].set_title("gradient magnitude -- which term is \"pulling harder\"")
    for i, v in enumerate(grads):
        axes[1].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{scenario_name}: cost breakdown @ t={breakdown['t']:.2f}s "
                 f"(own_map dist={breakdown['d_lin_own_map']:.3f}m, "
                 f"ground-truth clearance={breakdown['clearance_true']:.3f}m, "
                 f"D_SAFE={breakdown['D_SAFE']:.3f}m)")
    fig.tight_layout()
    out = out_dir / "cost_breakdown.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── 3b. open-loop (solver's plan) vs closed-loop (realized) trajectory ──

def plot_open_vs_closed_loop(traj_df, horizon_df, tick_idx, scenario_name, obstacles, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if horizon_df is None:
        print("  (no logs/online_mapping_horizon.csv -- rebuild online_mapping_demo "
              "with the lastPredictedHorizon() accessor and rerun the scenario first)")
        return None

    step = traj_df.iloc[tick_idx]["step"]
    plan = horizon_df[horizon_df["step"] == step].sort_values("stage")
    if plan.empty:
        print(f"  no horizon rows logged for step={step}")
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    for obs in obstacles:
        cx, cy = obs["center"]
        if obs["shape"] == "circle":
            ax.add_patch(patches.Circle((cx, cy), obs["radius"], color="tab:red", alpha=0.3))
        else:
            hs = obs["half_size"]
            ax.add_patch(patches.Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, color="tab:red", alpha=0.3))

    # closed-loop realized trajectory (a window around the tick)
    lo = max(0, tick_idx - 40)
    hi = min(len(traj_df), tick_idx + 10)
    window = traj_df.iloc[lo:hi]
    ax.plot(window["xt"], window["yt"], color="tab:green", lw=1.5, linestyle=":",
            label="reference (follower) path")
    ax.plot(window["x"], window["y"], color="tab:blue", lw=2, label="closed-loop realized (recent history)")
    ax.plot(traj_df.iloc[tick_idx]["x"], traj_df.iloc[tick_idx]["y"], "bo", ms=10, zorder=5, label="realized state @ this tick")

    # open-loop predicted horizon from this exact tick
    ax.plot(plan["hx"], plan["hy"], color="tab:orange", lw=2, linestyle="--", marker="o", ms=4,
            label="open-loop plan (solver's horizon @ this tick)")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{scenario_name}: open-loop plan vs closed-loop realized @ t={traj_df.iloc[tick_idx]['t']:.2f}s\n"
                 "(if the plan itself clips the obstacle: a planning/cost problem; "
                 "if only realized does: a tracking/execution problem)")
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "open_vs_closed_loop.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── 3c. open-loop vs closed-loop across several ticks around the worst one ──

def plot_open_vs_closed_loop_grid(traj_df, horizon_df, center_idx, offsets,
                                   scenario_name, obstacles, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if horizon_df is None:
        print("  (no logs/online_mapping_horizon.csv -- skipping multi-tick panel)")
        return None

    idxs = sorted(set(i for i in (center_idx + o for o in offsets) if 0 <= i < len(traj_df)))
    n = len(idxs)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), squeeze=False)

    for ax_i, tick_idx in enumerate(idxs):
        ax = axes[ax_i // ncols][ax_i % ncols]
        for obs in obstacles:
            cx, cy = obs["center"]
            if obs["shape"] == "circle":
                ax.add_patch(patches.Circle((cx, cy), obs["radius"], color="tab:red", alpha=0.3))
            else:
                hs = obs["half_size"]
                ax.add_patch(patches.Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, color="tab:red", alpha=0.3))

        step = traj_df.iloc[tick_idx]["step"]
        plan = horizon_df[horizon_df["step"] == step].sort_values("stage")

        lo = max(0, tick_idx - 40)
        hi = min(len(traj_df), tick_idx + 10)
        window = traj_df.iloc[lo:hi]
        ax.plot(window["xt"], window["yt"], color="tab:green", lw=1.2, linestyle=":",
                label="reference (follower) path")
        ax.plot(window["x"], window["y"], color="tab:blue", lw=1.6, label="closed-loop (recent history)")
        ax.plot(traj_df.iloc[tick_idx]["x"], traj_df.iloc[tick_idx]["y"], "bo", ms=7, zorder=5)
        if not plan.empty:
            ax.plot(plan["hx"], plan["hy"], color="tab:orange", lw=1.6, linestyle="--",
                    marker="o", ms=3, label="open-loop plan")

        offset = tick_idx - center_idx
        tag = "WORST" if offset == 0 else (f"t{offset:+d}")
        ax.set_aspect("equal")
        ax.set_title(f"{tag}  t={traj_df.iloc[tick_idx]['t']:.2f}s  "
                     f"clr={traj_df.iloc[tick_idx]['clearance_true']:.3f}m", fontsize=9)
        ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(loc="best", fontsize=7)

    for ax_i in range(n, nrows * ncols):
        axes[ax_i // ncols][ax_i % ncols].axis("off")

    fig.suptitle(f"{scenario_name}: open-loop plan vs closed-loop realized, ticks around worst clearance\n"
                 "(first panel, scanning backward, where the plan already curves away from the "
                 "obstacle is where the close pass was committed)")
    fig.tight_layout()
    out = out_dir / "open_vs_closed_loop_grid.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── 6. avoidance-weight sweep (slow: regenerates + rebuilds per point) ──

def _run(cmd, **kwargs):
    """subprocess.run with the acados env wired in, and stderr/stdout
    actually printed on failure instead of silently swallowed."""
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=SUBPROCESS_ENV,
                             capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  !! command failed: {' '.join(str(c) for c in cmd)}")
        print("  --- stdout ---")
        print(result.stdout[-3000:])
        print("  --- stderr ---")
        print(result.stderr[-3000:])
        result.check_returncode()
    return result


def _regen_and_rebuild():
    _run([PYTHON, "scripts/generate_mpc.py"])
    _run(["cmake", "--build", "build", "--target", "online_mapping_demo", "-j4"])


def sweep_robs_weight(scenario_name, weight_values, out_dir):
    import matplotlib.pyplot as plt

    qpath = CONFIG_DIR / "quadrotor.yaml"
    backup = qpath.read_text()
    results = []
    try:
        for w in weight_values:
            cfg = yaml.safe_load(qpath.read_text())
            cfg["cost"]["W_stage"][9] = float(w)
            cfg["cost"]["W_terminal"][9] = float(w)
            qpath.write_text(yaml.dump(cfg, default_flow_style=None, sort_keys=False))

            print(f"  [sweep] W_robs={w}: regenerating + rebuilding solver...")
            _regen_and_rebuild()

            print(f"  [sweep] W_robs={w}: running {scenario_name}...")
            _run(["./build/online_mapping_demo", f"config/{scenario_name}.yaml"])
            df = pd.read_csv(LOGS_DIR / "online_mapping_trajectory.csv")
            min_clear = float(df["clearance_true"].min())
            any_infeasible = bool((df["solve_status"] != 0).any())
            results.append((w, min_clear, any_infeasible))
            print(f"  [sweep] W_robs={w}: min_clearance={min_clear:.3f}m infeasible_any={any_infeasible}")
    finally:
        qpath.write_text(backup)
        print("  [sweep] restoring original quadrotor.yaml, regenerating + rebuilding...")
        _regen_and_rebuild()

    fig, ax = plt.subplots(figsize=(8, 5))
    ws = [r[0] for r in results]
    clears = [r[1] for r in results]
    infeas = [r[2] for r in results]
    colors = ["tab:red" if f else "tab:blue" for f in infeas]
    ax.plot(ws, clears, color="gray", lw=1, zorder=1)
    ax.scatter(ws, clears, c=colors, s=60, zorder=2,
               label="blue = feasible every tick, red = solve_status!=0 somewhere")
    ax.axhline(REAL_TARGET_M, color="black", linestyle="--", lw=1.2, label=f"hard gate = {REAL_TARGET_M}m")
    ax.set_xlabel("W_robs (avoidance cost weight)")
    ax.set_ylabel("resulting min clearance_true [m]")
    ax.set_title(f"{scenario_name}: realized clearance vs avoidance weight")
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "robs_weight_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")
    return out, results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", help="scenario name, e.g. standoff_conflict (no .yaml)")
    p.add_argument("--sweep-robs-weight", action="store_true",
                   help="also run the avoidance-weight sweep (slow: regen+rebuild per point)")
    p.add_argument("--sweep-values", default="20,50,100,200,400,800",
                   help="comma-separated W_robs values to sweep")
    args = p.parse_args()

    out_dir = LOGS_DIR / "collision_analysis" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    obstacles = load_obstacles(args.scenario)
    qcfg = load_quadrotor_cfg()
    df = load_traj()
    horizon_df = load_horizon()

    print(f"--- {args.scenario} ---")
    out2, worst_i = plot_clearance_vs_time(df, args.scenario, out_dir)
    print(f"[2] saved {out2}")
    print(f"    min clearance = {df['clearance_true'][worst_i]:.3f} m @ t={df['t'][worst_i]:.2f}s "
          f"(step {int(df['step'][worst_i])})")

    breakdown = cost_breakdown_at_tick(df, worst_i, qcfg)
    out3a = plot_cost_breakdown(breakdown, args.scenario, out_dir)
    print(f"[3a] saved {out3a}")
    print(f"     tracking_cost={breakdown['tracking_cost']:.3f} avoidance_cost={breakdown['avoidance_cost']:.3f}")
    print(f"     tracking_grad={breakdown['tracking_grad_mag']:.3f} avoidance_grad={breakdown['avoidance_grad_mag']:.3f}")

    out3b = plot_open_vs_closed_loop(df, horizon_df, worst_i, args.scenario, obstacles, out_dir)
    if out3b:
        print(f"[3b] saved {out3b}")

    offsets = [-80, -60, -40, -20, -10, 0]
    out3c = plot_open_vs_closed_loop_grid(df, horizon_df, worst_i, offsets, args.scenario, obstacles, out_dir)
    if out3c:
        print(f"[3c] saved {out3c}")

    if args.sweep_robs_weight:
        values = [float(v) for v in args.sweep_values.split(",")]
        print(f"[6] sweeping W_robs over {values} (this takes a while -- regen+rebuild per point)")
        out6, results = sweep_robs_weight(args.scenario, values, out_dir)
        print(f"[6] saved {out6}")


if __name__ == "__main__":
    main()
