"""Apply one or more named overlays from config/variants.yaml, (re)build only if
the resolved config/source actually changed, run the scenario suite
(loading cached per-scenario results when available -- see
run_scenario_suite.py's --no-cache), and print a pooled solve/onboard-
time comparison table across all requested variants.

Each variant's full result (resolved config + per-scenario and pooled
stats) is stored under logs/variants/<name>/<build_hash>/ -- same idea as
an ML experiment tracker: config and metrics live together, keyed by a
hash of everything that could have changed the result, so re-running an
unchanged variant is instant (cache hit) and changing a weight or the
solver source automatically invalidates only what's actually stale.

Examples:
    python3 scripts/run_variant.py --variant trailer_on_slack_on
    python3 scripts/run_variant.py --variant trailer_off_slack_off \\
        --variant trailer_off_slack_on --variant trailer_on_slack_off \\
        --variant trailer_on_slack_on
    python3 scripts/run_variant.py --variant trailer_on_slack_on --fresh   # ignore all caches
    python3 scripts/run_variant.py --list                                  # show available variants
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

import config_variants as cv

REPO_ROOT = cv.REPO_ROOT
VENV_PYTHON = os.path.expanduser("~/.venv/bin/python3")
ACADOS_DIR = os.path.expanduser("~/acados")
VARIANTS_LOG_DIR = os.path.join(REPO_ROOT, "logs", "variants")
BUILD_MARKER = os.path.join(REPO_ROOT, "c_generated_code", ".argus_build_hash")

_ENV = dict(os.environ)
_ENV["ACADOS_SOURCE_DIR"] = ACADOS_DIR
_ENV["LD_LIBRARY_PATH"] = os.path.join(ACADOS_DIR, "lib") + ":" + _ENV.get("LD_LIBRARY_PATH", "")


def _git_dirty(path):
    out = subprocess.run(["git", "status", "--porcelain", "--", path],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return bool(out.stdout.strip())


def _git_restore(path):
    subprocess.run(["git", "checkout", "--", path], cwd=REPO_ROOT, check=True)


def _current_build_marker():
    if not os.path.exists(BUILD_MARKER):
        return None
    with open(BUILD_MARKER) as f:
        return f.read().strip()


def _regenerate_and_build(build_hash):
    """Assumes config/quadrotor.yaml on disk ALREADY has the content for
    build_hash -- just runs codegen + the C++ build and stamps the marker.
    Kept separate from writing the config so the final restore-to-
    committed step (which uses `git checkout` to get the real file back,
    comments included) never has to go through yaml.safe_dump -- that
    strips every comment, which git checkout's version doesn't have to."""
    print(f"  [{build_hash}] generating + building...")
    subprocess.run([VENV_PYTHON, "scripts/generate_mpc.py"], cwd=REPO_ROOT, env=_ENV,
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(["cmake", "--build", "build", "--target", "online_mapping_demo",
                     "-j", str(os.cpu_count())],
                    cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(BUILD_MARKER, "w") as f:
        f.write(build_hash)


def _rebuild_for(resolved_cfg, build_hash):
    """Regenerate + rebuild only if the generated code doesn't already
    match build_hash -- codegen+build is ~10-20s, worth skipping when two
    consecutive variants (or re-running the same one) resolve identically.
    Writes resolved_cfg to config/quadrotor.yaml via yaml.safe_dump, which
    drops every comment -- fine for a variant overlay (transient, restored
    by git checkout before this process exits), NOT fine for writing back
    the committed config itself; see _regenerate_and_build()'s docstring,
    used for that case instead."""
    if _current_build_marker() == build_hash:
        print(f"  [{build_hash}] generated code already matches, skipping rebuild")
        return
    quadrotor_path = os.path.join(REPO_ROOT, "config", "quadrotor.yaml")
    with open(quadrotor_path, "w") as f:
        yaml.safe_dump(resolved_cfg, f, default_flow_style=None, sort_keys=False)
    _regenerate_and_build(build_hash)


def run_one_variant(name, fresh=False, render_plots=False):
    resolved = cv.resolve_variant(name)
    build_hash = cv.build_hash(resolved)
    variant_dir = os.path.join(VARIANTS_LOG_DIR, name, build_hash)
    results_path = os.path.join(variant_dir, "results.json")

    if not fresh and os.path.exists(results_path):
        print(f"[{name}] cached result at {results_path}")
        with open(results_path) as f:
            return json.load(f)

    print(f"[{name}] build_hash={build_hash}")
    _rebuild_for(resolved, build_hash)

    suite_args = [VENV_PYTHON, "scripts/run_scenario_suite.py"]
    if not render_plots:
        suite_args.append("--no-plots")
    if fresh:
        suite_args.append("--no-cache")
    subprocess.run(suite_args, cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    cache_root = os.path.join(REPO_ROOT, "logs", "suite_cache", build_hash)
    per_scenario = {}
    all_solve, all_pot = [], []
    for scen_name in sorted(os.listdir(cache_root)):
        traj_path = os.path.join(cache_root, scen_name, "trajectory.csv")
        if not os.path.exists(traj_path):
            continue
        df = pd.read_csv(traj_path)
        st, pot = df["solve_time_ms"].values, df["pipeline_onboard_time_ms"].values
        per_scenario[scen_name] = {
            "solve_mean_ms": float(np.mean(st)), "solve_max_ms": float(np.max(st)),
            "onboard_mean_ms": float(np.mean(pot)), "onboard_max_ms": float(np.max(pot)),
        }
        all_solve.append(st)
        all_pot.append(pot)

    all_solve, all_pot = np.concatenate(all_solve), np.concatenate(all_pot)
    pooled = {
        "solve_mean_ms": float(np.mean(all_solve)), "solve_max_ms": float(np.max(all_solve)),
        "onboard_mean_ms": float(np.mean(all_pot)), "onboard_max_ms": float(np.max(all_pot)),
    }
    result = {"variant": name, "build_hash": build_hash, "resolved_config": resolved,
              "per_scenario": per_scenario, "pooled": pooled}

    os.makedirs(variant_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(variant_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(resolved, f, default_flow_style=None, sort_keys=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", default=[],
                    help="Name of an overlay entry in config/variants.yaml (repeatable). "
                         "Omit entirely to run just the committed baseline config.")
    p.add_argument("--fresh", action="store_true",
                    help="Ignore every cache (variant results AND the scenario suite's "
                         "own cache) -- re-simulate everything for a fresh wall-clock "
                         "timing measurement")
    p.add_argument("--plots", action="store_true",
                    help="Render each scenario's dashboard PNG too (off by default -- "
                         "this tool is for quick solve/onboard-time comparisons)")
    p.add_argument("--list", action="store_true", help="List available variants and exit")
    args = p.parse_args()

    if args.list:
        names = sorted(cv.load_all_variants())
        print("Available variants:" if names else "No variants defined yet.")
        for n in names:
            print(f"  {n}")
        return

    variants = args.variant or [None]  # None == baseline, no overlay
    quadrotor_path = os.path.join(REPO_ROOT, "config", "quadrotor.yaml")
    if _git_dirty(quadrotor_path):
        print(f"ERROR: {quadrotor_path} has uncommitted changes -- refusing to overwrite them.\n"
              f"Commit/stash them first, or `git checkout -- {quadrotor_path}` if they're "
              f"leftover from an interrupted run_variant.py.", file=sys.stderr)
        sys.exit(1)

    original_marker = _current_build_marker()
    results = []
    try:
        for name in variants:
            results.append(run_one_variant(name or "baseline", fresh=args.fresh,
                                            render_plots=args.plots))
    finally:
        print("\nRestoring committed config/quadrotor.yaml and rebuilding...")
        _git_restore(quadrotor_path)  # real file back, comments included
        committed_hash = cv.build_hash(cv.load_base_config())
        if _current_build_marker() != committed_hash:
            _regenerate_and_build(committed_hash)
        else:
            print(f"  [{committed_hash}] generated code already matches, skipping rebuild")

    header = f"{'variant':<24}{'solve mean':<14}{'solve max':<14}{'onboard mean':<16}{'onboard max':<14}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        pl = r["pooled"]
        print(f"{r['variant']:<24}{pl['solve_mean_ms']:<14.3f}{pl['solve_max_ms']:<14.3f}"
              f"{pl['onboard_mean_ms']:<16.3f}{pl['onboard_max_ms']:<14.3f}")


if __name__ == "__main__":
    main()
