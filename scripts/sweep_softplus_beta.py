"""
Single-scenario weight sweep for camera.softplus_beta (config/quadrotor.yaml),
same spirit as design_decisions.md's "r_obs weight" sweep: does this knob
change behavior in one deliberately chosen scenario, without wrecking
something else -- not a Monte Carlo robustness sweep.

Scenario (mpc_controller.cpp's "fov_edge" mode, see its comments): a static
target directly ahead at the drone's own altitude (no pitch component),
drone held at its initial position (no position-tracking confound), initial
yaw deliberately offset so gamma_0 = fov_half_angle_deg + 5 deg -- just
outside the FOV cone. Isolates the pure yaw-correction transient as the MPC
turns the drone to bring the target back inside the cone, which is exactly
the behavior softplus_beta's knee sharpness is expected to affect (see
design_decisions.md's "Softplus sharpness" / interview.md's matching entry).

softplus_beta is baked into the symbolic cost at code-generation time (not a
runtime parameter), so each value in the sweep requires: edit the config,
regenerate_mpc.py, rebuild argus_mpc, run the scenario, analyze the log --
then the script puts the config back to its original value and regenerates/
rebuilds once more so the repo is left exactly as it started.

Metrics computed per beta, from the same bearing-angle formula compute_
metrics.py's bearing_angles_deg() uses:
  - fov_exceedance_duration_s: how long gamma stays above fov_half_angle_deg
    (same question the original "weak discrimination at beta=5" complaint
    was about)
  - peak_yaw_jerk_rad_s2: max |diff(r_cmd)| / Ts during the run -- proxy for
    "how violent is the correction" (what large beta was predicted to make
    worse)
  - yaw_rate_sign_changes: zero-crossing count of r_cmd -- direct chattering
    counter, independent of how large or fast the turn itself is

Run (needs the acados venv + env vars; this script exports them itself):
    python3 scripts/sweep_softplus_beta.py
"""

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "quadrotor.yaml")
VENV_PYTHON = os.path.expanduser("~/.venv/bin/python3")
ACADOS_DIR = os.path.expanduser("~/acados")

BETAS = [5, 20, 50, 100, 500]
TS = 0.05  # must match config/quadrotor.yaml mpc.Ts
FOV_HALF_ANGLE_DEG = 40.0  # must match config/quadrotor.yaml camera.fov_half_angle_deg

_ENV = dict(os.environ)
_ENV["ACADOS_SOURCE_DIR"] = ACADOS_DIR
_ENV["LD_LIBRARY_PATH"] = os.path.join(ACADOS_DIR, "lib") + ":" + _ENV.get("LD_LIBRARY_PATH", "")

_BETA_LINE_RE = re.compile(r"(softplus_beta:\s*)([0-9.eE+-]+)(.*)")


def read_current_beta():
    with open(CONFIG_PATH) as f:
        for line in f:
            m = _BETA_LINE_RE.search(line)
            if m:
                return float(m.group(2))
    raise RuntimeError("softplus_beta line not found in config/quadrotor.yaml")


def set_beta(beta):
    """Regex substitution on just the one line -- preserves the rest of the
    YAML file's formatting/comments exactly, unlike a yaml.safe_load +
    safe_dump round-trip."""
    with open(CONFIG_PATH) as f:
        lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        m = _BETA_LINE_RE.search(line)
        if m:
            lines[i] = _BETA_LINE_RE.sub(rf"\g<1>{beta}\g<3>", line)
            found = True
            break
    if not found:
        raise RuntimeError("softplus_beta line not found in config/quadrotor.yaml")
    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)


def regenerate_and_build():
    subprocess.run([VENV_PYTHON, "scripts/generate_mpc.py"],
                    cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(["cmake", "--build", "build", "-j4", "--target", "argus_mpc"],
                    cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_scenario():
    subprocess.run(["./build/argus_mpc", "fov_edge"],
                    cwd=REPO_ROOT, env=_ENV, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pd.read_csv(os.path.join(REPO_ROOT, "logs", "trajectory.csv"))


def bearing_gamma_deg(df):
    """Same ZYX-rotation formula as compute_metrics.py's bearing_angles_deg(),
    reduced to just the combined angle gamma this scenario needs."""
    cp, sp = np.cos(df["phi"]), np.sin(df["phi"])
    ct, st = np.cos(df["theta"]), np.sin(df["theta"])
    cy, sy = np.cos(df["psi"]), np.sin(df["psi"])
    dx = df["fx"] - df["x"]
    dy = df["fy"] - df["y"]
    dz = df["fz"] - df["z"]
    db_x = cy * ct * dx + sy * ct * dy - st * dz
    db_y = (cy * st * sp - sy * cp) * dx + (sy * st * sp + cy * cp) * dy + ct * sp * dz
    db_z = (cy * st * cp + sy * sp) * dx + (sy * st * cp - sy * cp) * dy + ct * cp * dz
    rng = np.sqrt(db_x**2 + db_y**2 + db_z**2)
    return np.degrees(np.arccos(np.clip(db_x / rng, -1.0, 1.0)))


def analyze(df):
    gamma = bearing_gamma_deg(df).values
    r_cmd = df["r_cmd"].values

    exceedance_steps = int(np.sum(gamma > FOV_HALF_ANGLE_DEG))
    fov_exceedance_duration_s = exceedance_steps * TS

    jerk = np.diff(r_cmd) / TS
    peak_yaw_jerk_rad_s2 = float(np.max(np.abs(jerk)))

    signs = np.sign(r_cmd)
    signs = signs[signs != 0]
    yaw_rate_sign_changes = int(np.sum(np.diff(signs) != 0))

    return {
        "fov_exceedance_duration_s": fov_exceedance_duration_s,
        "peak_yaw_jerk_rad_s2": peak_yaw_jerk_rad_s2,
        "yaw_rate_sign_changes": yaw_rate_sign_changes,
        "gamma_settle_time_s": float(np.argmax(gamma <= FOV_HALF_ANGLE_DEG) * TS),
    }


def main():
    original_beta = read_current_beta()
    print(f"Original softplus_beta = {original_beta}")
    results = []
    try:
        for beta in BETAS:
            print(f"\n--- beta = {beta} ---")
            set_beta(beta)
            regenerate_and_build()
            df = run_scenario()
            m = analyze(df)
            m["beta"] = beta
            results.append(m)
            print(m)
    finally:
        print(f"\nRestoring softplus_beta = {original_beta} and rebuilding ...")
        set_beta(original_beta)
        regenerate_and_build()

    print("\n" + "=" * 70)
    print(f"{'beta':>6} {'fov_exceed_s':>14} {'peak_jerk':>12} {'sign_changes':>14} {'settle_s':>10}")
    for m in results:
        print(f"{m['beta']:>6} {m['fov_exceedance_duration_s']:>14.3f} "
              f"{m['peak_yaw_jerk_rad_s2']:>12.2f} {m['yaw_rate_sign_changes']:>14} "
              f"{m['gamma_settle_time_s']:>10.3f}")


if __name__ == "__main__":
    main()
