"""
Sweep the realized body-rate (p,q,r) cost weight across the FULL
16-scenario suite, aggregating every metric run_scenario_suite.py already
computes per scenario -- collision, FOV exceedance, tracking error,
blur/smoothness (realized rate magnitude), rate saturation -- instead of
eyeballing one scenario, one weight value, at a time.

This formalizes what's been done ad hoc through this session (edit
config/quadrotor.yaml, regenerate, rebuild, rerun the suite, read the
printed table, repeat) into one repeatable, scriptable sweep. Same pattern
as collision_analysis.py's --sweep-robs-weight (regenerate + rebuild per
point, restore the original config in a finally block), generalized two
ways: across the full suite instead of one scenario, and across an
arbitrary weight list instead of a single before/after comparison.

Reuses run_scenario_suite.py's SCENARIOS list and compute_gate_and_stats()
directly (same reuse pattern as plot_scenario_contact_sheet.py) rather
than reimplementing metric extraction -- one definition of each metric,
used everywhere. Does NOT reuse run_scenario() itself -- that also
renders+saves a full dashboard PNG per scenario via compute_metrics.py,
which a sweep never looks at but would pay for on every weight value (96
renders for a 6-value x 16-scenario sweep). See _run_scenario_lean()'s
docstring.

Reports WORST-CASE per metric across the 16 scenarios for each weight
value, not just the mean: a weight that looks good on average while
quietly breaking one scenario is exactly the failure mode this project
keeps running into (see design_decisions.md's azimuth-weight-100 and
r_elev-weight-100 regressions, both found only by checking individual
scenarios, not suite-wide averages).

Run (slow -- regenerates + rebuilds + runs 16 scenarios PER weight value,
budget ~1-2 min per value):
    python3 scripts/sweep_body_rate_weight.py --values 0,1,5,10,20,50
    python3 scripts/sweep_body_rate_weight.py --values 0,1,5,10,20,50 \\
        --save-csv logs/suite/body_rate_weight_sweep.csv \\
        --save-png logs/suite/body_rate_weight_sweep.png

Always restores config/quadrotor.yaml to what it was before the sweep
started (even on failure/Ctrl-C, via try/finally) and leaves the solver
regenerated/rebuilt against that restored config -- the repo is never left
mid-sweep.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import run_scenario_suite as suite  # noqa: E402 -- reuse SCENARIOS/run_scenario/compute_gate_and_stats

REPO_ROOT = suite.REPO_ROOT
VENV_PYTHON = suite.VENV_PYTHON
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "quadrotor.yaml")

# p,q,r indices in W_stage/W_terminal -- see config/quadrotor.yaml's cost:
# layout comment. TERMINAL_RATIO=2.0 matches every other state-cost pair's
# stage:terminal convention already in that file (x/y/z 100->200, vx/vy/vz
# 10->20, etc.) -- not swept independently here.
STAGE_IDX = [8, 9, 10]
TERMINAL_IDX = [8, 9, 10]
TERMINAL_RATIO = 2.0


def _run(cmd):
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=suite._ENV,
                             capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  !! command failed: {' '.join(str(c) for c in cmd)}")
        print("  --- stdout ---"); print(result.stdout[-3000:])
        print("  --- stderr ---"); print(result.stderr[-3000:])
        result.check_returncode()


def _regen_and_rebuild():
    _run([VENV_PYTHON, "scripts/generate_mpc.py"])
    _run(["cmake", "--build", "build", "--target", "online_mapping_demo", "-j4"])


def _set_weight(w):
    cfg = yaml.safe_load(open(CONFIG_PATH))
    for i in STAGE_IDX:
        cfg["cost"]["W_stage"][i] = float(w)
    for i in TERMINAL_IDX:
        cfg["cost"]["W_terminal"][i] = float(w) * TERMINAL_RATIO
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)


def _run_scenario_lean(scenario_file):
    """Deliberately NOT suite.run_scenario(): that also shells out to
    compute_metrics.py to render+save a full 9-panel matplotlib dashboard
    PNG per scenario, which a sweep never looks at (only the aggregated
    metrics dict matters here) but pays for on every single weight value --
    96 dashboard renders for a 6-value x 16-scenario sweep, the dominant
    cost by far, not the sub-millisecond MPC solve. This just runs the
    binary and reads the trajectory CSV straight from logs/, skipping the
    dashboard/summary-CSV generation and the logs/suite/ archival copy
    entirely. Use suite.run_scenario() instead (or rerun
    run_scenario_suite.py directly) when you actually want the dashboards
    for a specific weight value worth drilling into."""
    subprocess.run(["./build/online_mapping_demo", f"config/{scenario_file}"],
                    cwd=REPO_ROOT, env=suite._ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pd.read_csv(os.path.join(REPO_ROOT, "logs", "online_mapping_trajectory.csv"))


def run_suite_once():
    """Run every scenario in suite.SCENARIOS against the CURRENTLY BUILT
    binary, return one metrics dict per scenario (suite.compute_gate_and_
    stats' own shape, same fields the printed suite table and
    logs/suite/baseline.json already use)."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    rows = []
    for name, scenario_file, _ in suite.SCENARIOS:
        df = _run_scenario_lean(scenario_file)
        m = suite.compute_gate_and_stats(df, cfg)
        m["scenario"] = name
        rows.append(m)
    return rows


def aggregate(rows):
    """One row per weight value. Worst-case (not mean) for every
    gate-relevant metric -- see module docstring for why. Mean is also
    reported for a couple of metrics where "typical" behavior matters too
    (FOV exceedance, tracking error), but worst-case is what should decide
    whether a weight value is usable."""
    df = pd.DataFrame(rows)
    return {
        "n_gates_pass":                   int(df["gates_pass"].sum()),
        "n_scenarios":                    len(df),
        "collision_any":                  bool((~df["collision_free"]).any()),
        "min_clearance_true_m":           float(df["min_clearance_true_m"].min()),
        "rate_saturation_pct_max":        float(df["rate_saturation_pct"].max()),
        "fov_exceedance_pct_mean":        float(df["fov_exceedance_pct"].mean()),
        "fov_exceedance_pct_max":         float(df["fov_exceedance_pct"].max()),
        "xy_tracking_error_mean_m":       float(df["xy_tracking_error_mean_m"].mean()),
        "xy_tracking_error_max_m":        float(df["xy_tracking_error_max_m"].max()),
        "body_rate_magnitude_mean_deg_s": float(df["body_rate_magnitude_mean_deg_s"].mean()),
        "body_rate_magnitude_max_deg_s":  float(df["body_rate_magnitude_max_deg_s"].max()),
        "occluded_pct_mean":              float(df["occluded_pct"].mean()),
        "occluded_pct_max":               float(df["occluded_pct"].max()),
    }


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--values", default="0,1,5,10,20,50",
                    help="comma-separated p,q,r STAGE weights to sweep "
                         "(terminal = %.1fx each, see TERMINAL_RATIO)" % TERMINAL_RATIO)
    p.add_argument("--save-csv", default=os.path.join(suite.SUITE_DIR, "body_rate_weight_sweep.csv"))
    p.add_argument("--save-png", default=os.path.join(suite.SUITE_DIR, "body_rate_weight_sweep.png"))
    return p.parse_args()


def main():
    args = _parse_args()
    values = [float(v) for v in args.values.split(",")]
    backup = open(CONFIG_PATH).read()

    results = []
    try:
        for w in values:
            print(f"[sweep] p,q,r weight={w} (terminal={w * TERMINAL_RATIO}): "
                  f"regenerating + rebuilding...")
            _set_weight(w)
            _regen_and_rebuild()
            print(f"[sweep] p,q,r weight={w}: running full {len(suite.SCENARIOS)}-scenario suite...")
            rows = run_suite_once()
            agg = aggregate(rows)
            agg["weight"] = w
            results.append(agg)
            print(f"[sweep] weight={w}: gates_pass={agg['n_gates_pass']}/{agg['n_scenarios']}  "
                  f"min_clearance={agg['min_clearance_true_m']:.3f}m  "
                  f"fov_exc_max={agg['fov_exceedance_pct_max']:.1f}%  "
                  f"blur_max={agg['body_rate_magnitude_max_deg_s']:.0f}deg/s  "
                  f"rate_sat_max={agg['rate_saturation_pct_max']:.1f}%")
    finally:
        print("[sweep] restoring original quadrotor.yaml, regenerating + rebuilding...")
        with open(CONFIG_PATH, "w") as f:
            f.write(backup)
        _regen_and_rebuild()

    out = pd.DataFrame(results).set_index("weight")
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    out.to_csv(args.save_csv)
    print(f"\nSaved {args.save_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("n_gates_pass", "gates passing (of %d)" % len(suite.SCENARIOS)),
        ("min_clearance_true_m", "worst-case clearance [m]"),
        ("fov_exceedance_pct_max", "worst-case FOV exceedance [%]"),
        ("xy_tracking_error_max_m", "worst-case xy tracking error [m]"),
        ("body_rate_magnitude_max_deg_s", "worst-case realized rate / blur [deg/s]"),
        ("rate_saturation_pct_max", "worst-case rate SATURATION [%]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(out.index, out[col], marker="o")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("p,q,r stage weight")
        ax.grid(alpha=0.3)
    fig.suptitle("Realized body-rate (p,q,r) weight sweep -- "
                  "worst-case per metric across the full suite")
    fig.tight_layout()
    fig.savefig(args.save_png, dpi=130)
    print(f"Saved {args.save_png}")


if __name__ == "__main__":
    main()
