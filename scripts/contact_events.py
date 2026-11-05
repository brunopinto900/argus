"""
Shared "is the target in contact" computation: the combined bearing angle
gamma, the FOV-cone exceedance mask, and the occlusion-based lost-contact
mask, split apart instead of pre-merged -- the same OR that
compute_metrics.py's compute_summary() builds into its single
lost_contact/reacquisition_events numbers, but exposed here as separate
boolean arrays so a plot or a 3-D render can show *why* contact was lost,
not just that it was. compute_metrics.py's own copy (bearing_angles_deg(),
the lost_contact OR in compute_summary()) stays the source of truth for the
formulas; this module mirrors them for reuse by scripts/plot_fov_results.py
and scripts/render_technical_comparison.py -- keep the two in sync by hand.

No argparse/plotting here, just arrays, so it has no side effects to import.
"""

import numpy as np


def bearing_angles_deg(df):
    """Combined bearing angle gamma (boresight to true target direction) and
    its yaw/pitch decomposition. Same ZYX convention as
    compute_metrics.py's bearing_angles_deg()."""
    cp, sp = np.cos(df["phi"]), np.sin(df["phi"])
    ct, st = np.cos(df["theta"]), np.sin(df["theta"])
    cy, sy = np.cos(df["psi"]), np.sin(df["psi"])
    dx = df["fx"] - df["x"]
    dy = df["fy"] - df["y"]
    dz = df["fz"] - df["z"]
    db_x = cy * ct * dx + sy * ct * dy - st * dz
    db_y = (cy * st * sp - sy * cp) * dx + (sy * st * sp + cy * cp) * dy + ct * sp * dz
    db_z = (cy * st * cp + sy * sp) * dx + (sy * st * cp - cy * sp) * dy + ct * cp * dz
    yaw_offset = np.asarray(np.degrees(np.arctan2(db_y, db_x)))
    pitch_offset = np.asarray(np.degrees(np.arctan2(-db_z, db_x)))
    rng = np.sqrt(db_x**2 + db_y**2 + db_z**2)
    gamma = np.asarray(np.degrees(np.arccos(np.clip(db_x / rng, -1.0, 1.0))))
    return yaw_offset, pitch_offset, gamma


def compute_contact_masks(df, fov_half_angle_deg, d_fov_m, lost_contact_occlusion_pct=90.0):
    """Returns a dict of boolean arrays (len == len(df)):
    out_of_fov        -- gamma exceeds the FOV half-angle
    occlusion_lost     -- occlusion so severe contact is lost even though
                           still inside the cone (the other half of
                           compute_metrics.py's lost_contact OR)
    lost_contact        -- out_of_fov | occlusion_lost (what
                           reacquisition_events actually counts)
    plus 'gamma' (deg) and, if occlusion columns are present, 'occ_pct'.
    """
    _, _, gamma = bearing_angles_deg(df)
    out_of_fov = gamma > fov_half_angle_deg

    occlusion_lost = np.zeros(len(df), dtype=bool)
    occ_pct = None
    if "occlusion_clearance" in df.columns:
        occ_pct = 100.0 * np.clip(1.0 - df["occlusion_clearance"].to_numpy() / d_fov_m, 0.0, 1.0)
        occlusion_lost = occ_pct > lost_contact_occlusion_pct

    lost_contact = out_of_fov | occlusion_lost
    out = dict(gamma=gamma, out_of_fov=out_of_fov, occlusion_lost=occlusion_lost,
               lost_contact=lost_contact)
    if occ_pct is not None:
        out["occ_pct"] = occ_pct
    return out
