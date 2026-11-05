"""
FOV-cone results figure for the paper: combined bearing angle gamma(t)
against the FOV half-angle threshold, for each of Table 1's four
configurations, with shaded spans showing *why* contact was lost at each
point -- out of the FOV cone vs. occluded past the lost-contact threshold
(scripts/contact_events.py) -- instead of Table 1's single merged
"Reacq." count. This is the FOV-side counterpart to
compute_metrics.py's occlusion_panel/occlusion_comparison.gif, which
already get a dedicated plot and render; FOV only had the geometry figure
and a tuning anecdote before this.

    python3 scripts/plot_fov_results.py \\
        --config-path config/quadrotor.yaml \\
        --csv logs/suite_cache/9003ae057c11dd27/forest_clutter/trajectory.csv "Baseline, occlusion off" \\
        --csv logs/suite_cache/231817e249ef4cb4/forest_clutter/trajectory.csv "Baseline, occlusion on" \\
        --csv logs/suite_cache/b03864ca5b0bc369/forest_clutter/trajectory.csv "Band follower, occlusion on" \\
        --csv logs/suite_cache/3700de04c93c3197/forest_clutter/trajectory.csv "Band follower + centering" \\
        --save-png paper/figures/fov_results.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import yaml

import contact_events


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-path", default="config/quadrotor.yaml")
    p.add_argument("--csv", nargs=2, action="append", metavar=("PATH", "LABEL"),
                    required=True, help="Repeatable: trajectory CSV + panel label")
    p.add_argument("--d-fov", type=float, default=1.68)
    p.add_argument("--lost-contact-occlusion-pct", type=float, default=90.0)
    p.add_argument("--save-png", default=None)
    return p.parse_args()


def contiguous_true_runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def main():
    args = _parse_args()
    cfg = yaml.safe_load(open(args.config_path))
    fov_half_deg = cfg["camera"]["fov_half_angle_deg"]

    n = len(args.csv)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]

    labeled = {"fov": False, "occ": False}
    for ax, (path, label) in zip(axes, args.csv):
        df = pd.read_csv(path)
        m = contact_events.compute_contact_masks(df, fov_half_deg, args.d_fov,
                                                   args.lost_contact_occlusion_pct)
        t = df["t"].to_numpy()
        ax.plot(t, m["gamma"], color="black", linewidth=1.1, label="bearing $\\gamma$" if ax is axes[0] else None)
        ax.axhline(fov_half_deg, color="tab:red", linestyle="--", linewidth=1,
                    label=f"FOV edge ({fov_half_deg:.0f}°)" if ax is axes[0] else None)

        occ_only = m["occlusion_lost"] & ~m["out_of_fov"]
        for b, e in contiguous_true_runs(m["out_of_fov"]):
            ax.axvspan(t[b], t[e], color="tab:red", alpha=0.2,
                        label="target lost: out of FOV cone" if not labeled["fov"] else None)
            labeled["fov"] = True
        for b, e in contiguous_true_runs(occ_only):
            ax.axvspan(t[b], t[e], color="tab:orange", alpha=0.25,
                        label="target lost: occluded" if not labeled["occ"] else None)
            labeled["occ"] = True

        n_fov_events = len(contiguous_true_runs(m["out_of_fov"]))
        n_occ_events = len(contiguous_true_runs(occ_only))
        ax.set_ylabel("$\\gamma$ [deg]", fontsize=8)
        ax.set_title(f"{label}  —  {n_fov_events} FOV-exit event(s), {n_occ_events} occlusion-only event(s)",
                      fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("t [s]")
    fig.legend(loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FOV-cone bearing and target-lost events, forest_clutter scenario", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    if args.save_png:
        fig.savefig(args.save_png, dpi=200)
        print(f"Saved {args.save_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
