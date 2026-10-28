"""
Generate the ACADOS OCP solver for the quadrotor circle-tracking MPC.

Run once from the repo root:
    python3 scripts/generate_mpc.py

Output: c_generated_code/   (C solver + CMakeLists, ready to link from C++)
        quadrotor_ocp.json   (OCP spec, used by ACADOS internals)

State vector  (nx = 12):  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Input vector  (nu =  4):  [T, p_cmd, q_cmd, r_cmd]
Runtime param (np = 10):  [p_target_x, p_target_y, p_target_z,
                           obs_p0_x, obs_p0_y, obs_d0, obs_grad_x, obs_grad_y,
                           azimuth_ux, azimuth_uy]
  — injected before every solve. The obs_* block is a per-stage local
  affine reconstruction of own_map's ESDF around a linearization point
  p0 (own_map.query() can't enter the OCP symbolically -- it's runtime
  voxel data, not a closed-form expression CasADi can differentiate at
  code-gen time; see whiteboard.md). Unlike p_target (same value every
  stage today), obs_* is expected to differ per stage -- the caller
  queries the ESDF at each stage's predicted position separately.
  azimuth_u{x,y} is the desired horizontal unit DIRECTION from the
  leader to the drone (e.g. "behind" the leader's heading) -- feeds
  r_azimuth below; direction only, not a position, see that residual's
  comment for why.

Cost output, obstacle_avoidance.mode='slack' (default, ny=19): [x, y, z, vx, vy,
vz, phi, theta, p, q, r, r_fov, r_dist, r_elev, r_azimuth, T, p_cmd, q_cmd, r_cmd]
Cost output, obstacle_avoidance.mode='cost' (legacy, ny=20): same with an r_obs
entry inserted right after r_fov -- see OBS_MODE above D_SAFE.
  - r_fov = softplus(cos(alpha_fov) - cos(gamma)) — zero inside FOV cone, grows outside
  - mode='slack': collision avoidance is NOT a cost term -- see "Collision-avoidance
    constraint" below: d_lin = obs_d0 + obs_grad . (p - obs_p0) is a slacked
    nonlinear constraint (con_h_expr), not an NLS residual. See
    design_decisions.md's "slack-variable collision constraint" entry for why
    (mode='cost''s squared-hinge residual never guaranteed d_safe for any finite
    weight -- its gradient vanishes at the boundary; a slacked constraint's
    linear slack penalty does not). mode='cost' is kept only for A/B comparison
    (scripts/run_variant.py), not recommended for actual use.
  - r_dist = ||p_drone_xy - p_leader_xy|| - D_DESIRED — HORIZONTAL-only standoff-
    distance cost (ground-plane distance, not 3D range -- see r_elev below for why),
    added alongside (not replacing) x/y/z tracking of the trailer reference point;
    see design_decisions.md's collision-clearance analysis for why
  - r_elev = (z_drone - z_leader) - horiz_dist * tan(target_bearing) — dedicated
    elevation/pitch-angle residual, decoupled from r_dist so standoff distance and
    look-down angle each have their own target instead of competing for one shared
    3D-distance budget (the "landing zone" dive this replaces -- see
    design_decisions.md). horiz_dist reuses the same expression as r_dist.
  - r_azimuth = 1 - cos(phi_rel - azimuth_target), expressed without atan2 as
    1 - (dot of the drone's and the desired unit direction from the leader) --
    dedicated azimuth (which side of the leader, e.g. "behind") residual,
    decoupled from r_dist/r_elev for the same reason: a single trailer-point
    position target couples distance+elevation+azimuth together, and any one
    of them being forced off (e.g. by avoidance) used to drag the others with
    it. See design_decisions.md.
  - phi, theta penalised to keep the drone roughly level
  - p, q, r (REALIZED body rates, not commands) penalised directly -- distinct
    from the rate-command penalty below: the plant is a second-order model
    (intentional lag vs. the MPC's prediction model), so a smooth commanded
    rate sequence can still produce an oscillatory realized rate. Only
    phi/theta were in the cost among the states before this; p/q/r were not,
    see design_decisions.md's roll-oscillation note.
  - rate commands penalised to limit aggressiveness

The yref for each horizon step is updated at runtime in C++ via:
    acados_ocp_solver_set(solver, i, "yref", yref_ptr)
"""

import os
import sys
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver

# Make ACADOS shared libs visible before the solver is loaded
_acados_lib = os.path.join(os.path.expanduser('~'), 'acados', 'lib')
os.environ['LD_LIBRARY_PATH'] = _acados_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# allow 'from quadrotor_model / config import ...' when running from repo root
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_model import export_quadrotor_ode_model
from config import load_config, load_target_config


# ── Load all parameters from config/quadrotor.yaml ────────────────────────────
_cfg        = load_config()
_model_cfg  = _cfg['model']
_mpc_cfg    = _cfg['mpc']
_con_cfg    = _cfg['constraints']
_cost_cfg   = _cfg['cost']
_plant_cfg  = _cfg['plant']
_cam_cfg    = _cfg['camera']
_obs_cfg    = _cfg['obstacle_avoidance']

# ── Target trajectory (config/target.yaml) — only the circle radius/period/
#    height are needed at codegen time; waypoints are loaded at runtime.
_circ_cfg   = load_target_config()['circle']

N              = _mpc_cfg['N']
Ts             = _mpc_cfg['Ts']
HOVER_ALTITUDE = _mpc_cfg['hover_altitude']

MASS    = _model_cfg['mass']
G       = _model_cfg['gravity']
T_HOVER = MASS * G

T_MIN        = _con_cfg['T_min']
T_MAX        = _con_cfg['T_max_factor'] * T_HOVER
RATE_MAX     = _con_cfg['rate_max']
YAW_RATE_MAX = _con_cfg['yaw_rate_max']
ANGLE_MAX    = np.deg2rad(_con_cfg['angle_max_deg'])
Z_MIN        = float(_con_cfg['z_min'])

W_STAGE    = np.diag(_cost_cfg['W_stage'])
W_TERMINAL = np.diag(_cost_cfg['W_terminal'])

FOV_ALPHA = float(np.deg2rad(_cam_cfg['fov_half_angle_deg']))
FOV_BETA  = float(_cam_cfg['softplus_beta'])
COS_ALPHA = float(np.cos(FOV_ALPHA))

# 'slack' (default): d_lin >= d_safe as a slacked nonlinear constraint.
# 'cost': the old squared-hinge r_obs = max(0, d_safe - d_lin) NLS residual
# instead -- kept only so the two formulations can be A/B compared from one
# codebase (scripts/run_variant.py) without git worktree surgery. Changes
# ny/ny_e (adds/removes the r_obs slot), so W_stage/W_terminal in whatever
# config is loaded must already be shaped for the selected mode -- see the
# shape assertions in generate_mpc() below.
OBS_MODE = str(_obs_cfg.get('mode', 'slack')).lower()
assert OBS_MODE in ('slack', 'cost'), \
    f"obstacle_avoidance.mode must be 'slack' or 'cost', got {OBS_MODE!r}"

D_SAFE = float(_obs_cfg['d_safe_m'])
# Slack-penalty weights for the collision-avoidance constraint below --
# linear term is the one that matters (an exact penalty: once it exceeds
# the constraint's Lagrange multiplier, d_safe is enforced exactly, not
# asymptotically, unlike the old squared-hinge cost term it replaces).
# Quadratic term only smooths the QP near the boundary, not what provides
# the guarantee. See design_decisions.md.
SLACK_LINEAR    = float(_obs_cfg.get('slack_linear_weight', 1.0e4))
SLACK_QUADRATIC = float(_obs_cfg.get('slack_quadratic_weight', 100.0))
# Desired horizontal drone-to-leader distance for r_dist -- reuses the
# same standoff distance the trailer reference point is built from
# (config/quadrotor.yaml's trailer.distance_to_target).
D_DESIRED = float(_cfg['trailer']['distance_to_target'])
# Desired look-down (elevation) angle for r_elev -- same design angle the
# old kFovZRef altitude pin used (config/quadrotor.yaml's
# trailer.target_bearing_deg), now targeted directly as its own residual
# instead of only being realized indirectly via a fixed-z reference.
TARGET_BEARING_RAD = float(np.deg2rad(_cfg['trailer']['target_bearing_deg']))




def generate_mpc():
    ocp   = AcadosOcp()
    model = export_quadrotor_ode_model()
    # Runtime parameters: FOV target position (3) + per-stage obstacle
    # ESDF linearization (5: p0_x, p0_y, d0, grad_x, grad_y) + desired
    # azimuth direction (2: ux, uy) -- see the module docstring's
    # "Runtime param" section for what each block means.
    model.p = ca.SX.sym('p_runtime', 10)
    p_target   = model.p[0:3]
    obs_p0     = model.p[3:5]
    obs_d0     = model.p[5]
    obs_grad   = model.p[6:8]
    azimuth_u  = model.p[8:10]
    ocp.model = model
    ocp.parameter_values = np.zeros(10)   # nominal; overwritten before every solve

    nx = model.x.rows()   # 12
    nu = model.u.rows()   #  4

    # ── Horizon ───────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * Ts

    # ── Cost function ─────────────────────────────────────────────────────────
    # State indices: x(0) y(1) z(2) vx(3) vy(4) vz(5) phi(6) theta(7) psi(8) p(9) q(10) r(11)
    x_sym = model.x
    u_sym = model.u

    # ZYX body-to-world rotation matrix: v_world = R_bw @ v_body.
    # Column 0 is the body +x axis (camera boresight) in world frame.
    phi, theta, psi = x_sym[6], x_sym[7], x_sym[8]
    cp, sp = ca.cos(phi),   ca.sin(phi)
    ct, st = ca.cos(theta), ca.sin(theta)
    cy, sy = ca.cos(psi),   ca.sin(psi)
    R_col0 = ca.vertcat(cy*ct,             sy*ct,             -st   )
    R_col1 = ca.vertcat(cy*st*sp - sy*cp, sy*st*sp + cy*cp,  ct*sp )
    R_col2 = ca.vertcat(cy*st*cp + sy*sp, sy*st*cp - cy*sp,  ct*cp )
    R_bw   = ca.horzcat(R_col0, R_col1, R_col2)

    # Bearing vector to target in body frame.
    d_world = p_target - x_sym[:3]
    range_  = ca.sqrt(ca.dot(d_world, d_world) + 1e-12)  # smooth norm: C∞ at δ=0
    d_body  = ca.mtimes(R_bw.T, d_world)

    # FOV cone slack s: positive = target outside the cone.
    #   cos(gamma) = forward component / range  (gamma = angle from boresight)
    #   s = cos(alpha_FOV) - cos(gamma)
    s_fov = COS_ALPHA - d_body[0] / range_

    # Softplus residual: r_fov^2 = (1/beta)*log(1 + exp(beta*s)).
    # Vanishes inside the cone, grows monotonically outside -- smooth everywhere.
    # Stable form: log(1+exp(u)) = max(u,0) + log(1+exp(-|u|))  avoids overflow for large β·s.
    _u    = FOV_BETA * s_fov
    soft  = (1.0 / FOV_BETA) * (ca.fmax(_u, 0) + ca.log(1.0 + ca.exp(-ca.fabs(_u))))
    r_fov = soft

    # ── Collision-avoidance constraint: d_lin >= d_safe, slacked ─────────────
    # d_lin is the local affine reconstruction of own_map's ESDF around the
    # per-stage linearization point obs_p0, built from (obs_d0, obs_grad)
    # supplied as runtime parameters by the caller (own_map.query(), done
    # in C++ -- the ESDF itself can't be expressed symbolically here, see
    # the module docstring and whiteboard.md).
    #
    # Previously an NLS cost residual r_obs = max(0, d_safe - d_lin) (a
    # squared hinge, since NLS squares every residual). That never
    # guaranteed d_safe for any finite weight: the squared penalty's
    # gradient vanishes at the boundary, so nothing pushes back until
    # you're already meaningfully inside it (see design_decisions.md's
    # d_safe-inflation note -- the 2.7x fudge factor that was compensating
    # for exactly this). Replaced with a slacked nonlinear constraint
    # (con_h_expr below): d_lin >= d_safe - s, s >= 0, with a slack
    # penalty that includes a LINEAR term (SLACK_LINEAR) -- an exact
    # penalty, not an asymptotic one. This is what lets trailer x/y/z
    # tracking be re-enabled (below) without the old cost-vs-cost fight:
    # the constraint acts on the drone's actual predicted state at every
    # stage, independent of what the tracking cost wants.
    d_lin = obs_d0 + ca.dot(obs_grad, x_sym[:2] - obs_p0)

    # ── Standoff-distance + elevation-angle residuals, decoupled ────────────
    # Originally one 3D-distance residual (r_dist = ||p_drone - p_leader||
    # - D_DESIRED, see design_decisions.md's collision-clearance analysis
    # for why it replaced the trailer x/y/z tracking term). That single
    # residual let the solver trade horizontal offset for vertical offset
    # freely to hit the same 2m total -- under horizontal pressure (e.g. a
    # sharp turn) it would dive toward the leader's own altitude to claw
    # distance back, since nothing penalised *which* split it picked (the
    # "landing zone" behaviour found on pedestrian_walk). Split into two
    # residuals with independent targets, same decoupling real gimbal-
    # follow systems use (control ground-plane distance and look-down
    # angle/gimbal pitch separately, not one slant-range number):
    #   r_dist  -- horizontal-only distance to the leader, target D_DESIRED
    #   r_elev  -- vertical offset vs. the horizontal-distance-scaled value
    #              that produces the designed TARGET_BEARING_RAD look-down
    #              angle (same relation the old kFovZRef used, now a cost
    #              target instead of a fixed yref pin)
    horiz_dist = ca.sqrt(d_world[0]**2 + d_world[1]**2 + 1e-12)
    vert_offset = x_sym[2] - p_target[2]  # drone z - leader z (+ve: drone above)
    r_dist = horiz_dist - D_DESIRED
    r_elev = vert_offset - horiz_dist * ca.tan(TARGET_BEARING_RAD)

    # ── Azimuth residual: which side of the leader (e.g. "behind") ──────────
    # r_azimuth = 1 - cos(phi_rel - azimuth_target), where phi_rel is the
    # drone's actual bearing from the leader. Written as a dot product of
    # unit vectors instead of atan2(...) to stay smooth everywhere (same
    # trick r_fov already uses for its own angle, avoiding the wraparound/
    # branch-cut problems atan2 would introduce): 0 when the drone sits
    # exactly on the desired azimuth, grows smoothly as it swings away,
    # independent of distance or elevation. azimuth_u is caller-supplied
    # (unit vector from leader to the desired azimuth point, see
    # QuadrotorMpc.hpp's Reference struct) -- derived, in this project's
    # caller, directly from the already-heading-filtered trailer reference
    # point (TrailerFollower's PDx/PDy), reusing its smoothing instead of
    # recomputing a heading filter here. See design_decisions.md.
    dx = x_sym[0] - p_target[0]
    dy = x_sym[1] - p_target[1]
    r_azimuth = 1.0 - (dx * azimuth_u[0] + dy * azimuth_u[1]) / horiz_dist

    body_rates = x_sym[9:12]  # p, q, r -- realized rates, distinct from the u_sym rate commands below

    if OBS_MODE == 'cost':
        # Legacy squared-hinge collision cost -- see the OBS_MODE comment
        # above D_SAFE. Never guarantees d_safe for any finite weight (the
        # NLS-squared penalty's gradient vanishes at the boundary).
        r_obs = ca.fmax(0.0, D_SAFE - d_lin)
        cost_y_expr   = ca.vertcat(x_sym[:8], body_rates, r_fov, r_obs, r_dist, r_elev, r_azimuth, u_sym)  # ny   = 20
        cost_y_expr_e = ca.vertcat(x_sym[:8], body_rates, r_fov, r_obs, r_dist, r_elev, r_azimuth)          # ny_e = 16
    else:
        cost_y_expr   = ca.vertcat(x_sym[:8], body_rates, r_fov, r_dist, r_elev, r_azimuth, u_sym)  # ny   = 19
        cost_y_expr_e = ca.vertcat(x_sym[:8], body_rates, r_fov, r_dist, r_elev, r_azimuth)          # ny_e = 15

    expected_ny  = 20 if OBS_MODE == 'cost' else 19
    expected_nye = 16 if OBS_MODE == 'cost' else 15
    assert W_STAGE.shape[0] == expected_ny, (
        f"obstacle_avoidance.mode={OBS_MODE!r} needs cost.W_stage with {expected_ny} "
        f"entries (got {W_STAGE.shape[0]}) -- see config/variants.yaml for ready-made "
        f"W_stage/W_terminal pairs per mode")
    assert W_TERMINAL.shape[0] == expected_nye, (
        f"obstacle_avoidance.mode={OBS_MODE!r} needs cost.W_terminal with {expected_nye} "
        f"entries (got {W_TERMINAL.shape[0]})")

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr   = cost_y_expr
    ocp.model.cost_y_expr_e = cost_y_expr_e

    ocp.cost.W   = W_STAGE
    ocp.cost.W_e = W_TERMINAL

    # Nominal yref: hover at origin, level, target inside FOV (r_fov
    # reference = 0). C++ overwrites this every step with the actual
    # reference. No r_obs entry any more -- collision avoidance is a
    # constraint now, not a cost residual, see above.
    yref_hover_list = [0.0, 0.0, HOVER_ALTITUDE,  # x, y, z
                        0.0, 0.0, 0.0,            # vx, vy, vz
                        0.0, 0.0,                 # phi, theta
                        0.0, 0.0, 0.0,            # p, q, r (realized body rates: no rotation)
                        0.0]                      # r_fov  (target inside cone)
    if OBS_MODE == 'cost':
        yref_hover_list.append(0.0)  # r_obs (clear of any obstacle)
    yref_hover_list += [0.0,         # r_dist (exactly D_DESIRED away, horizontal)
                         0.0,        # r_elev (exactly at the designed look-down angle)
                         0.0,        # r_azimuth (exactly at the desired azimuth direction)
                         T_HOVER, 0.0, 0.0, 0.0]  # T, rates
    yref_hover   = np.array(yref_hover_list)
    yref_hover_e = yref_hover[:expected_nye]

    ocp.cost.yref   = yref_hover
    ocp.cost.yref_e = yref_hover_e

    # ── Constraints ───────────────────────────────────────────────────────
    # Input box constraints
    ocp.constraints.lbu    = np.array([T_MIN, -RATE_MAX, -RATE_MAX, -YAW_RATE_MAX])
    ocp.constraints.ubu    = np.array([T_MAX,  RATE_MAX,  RATE_MAX,  YAW_RATE_MAX])
    ocp.constraints.idxbu  = np.array([0, 1, 2, 3])

    # State box constraints: z (idx 2, hard ground-clearance floor -- fixed
    # world altitude, not relative to the leader; see config/quadrotor.yaml's
    # constraints.z_min comment and design_decisions.md's collision-clearance
    # analysis for why), phi (idx 6), theta (idx 7) — running stages. No
    # physical upper altitude limit, so z's upper bound is just a large
    # permissive number, not a real constraint.
    Z_MAX = 1.0e3
    ocp.constraints.lbx    = np.array([Z_MIN, -ANGLE_MAX, -ANGLE_MAX])
    ocp.constraints.ubx    = np.array([Z_MAX,  ANGLE_MAX,  ANGLE_MAX])
    ocp.constraints.idxbx  = np.array([2, 6, 7])

    # Same limits at the terminal stage
    ocp.constraints.lbx_e   = ocp.constraints.lbx
    ocp.constraints.ubx_e   = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    # Collision-avoidance: slacked nonlinear constraint d_lin >= D_SAFE
    # (see the r_obs -> con_h_expr comment above for why this replaced the
    # squared-hinge cost term). uh is a large permissive number -- only the
    # lower bound is ever meaningfully active, d_lin has no real upper
    # limit. idxsh marks it soft: acados adds a slack s>=0 to lh with
    # penalty zl*s + 0.5*Zl*s^2 (zu/Zu are for the upper-bound slack, which
    # never activates here, so left at 0). SLACK_LINEAR is the exact-
    # penalty term that actually provides the guarantee; SLACK_QUADRATIC
    # just smooths the QP near the boundary.
    # Only added in 'slack' mode -- in 'cost' mode, collision avoidance is
    # entirely the r_obs NLS residual above, no constraint/slack at all.
    if OBS_MODE == 'slack':
        H_UB = 1.0e6
        ocp.model.con_h_expr   = ca.vertcat(d_lin)
        ocp.model.con_h_expr_e = ca.vertcat(d_lin)
        ocp.constraints.lh     = np.array([D_SAFE])
        ocp.constraints.uh     = np.array([H_UB])
        ocp.constraints.lh_e   = np.array([D_SAFE])
        ocp.constraints.uh_e   = np.array([H_UB])
        ocp.constraints.idxsh  = np.array([0])
        ocp.constraints.idxsh_e = np.array([0])
        ocp.cost.zl   = np.array([SLACK_LINEAR])
        ocp.cost.zu   = np.array([0.0])
        ocp.cost.Zl   = np.array([SLACK_QUADRATIC])
        ocp.cost.Zu   = np.array([0.0])
        ocp.cost.zl_e = np.array([SLACK_LINEAR])
        ocp.cost.zu_e = np.array([0.0])
        ocp.cost.Zl_e = np.array([SLACK_QUADRATIC])
        ocp.cost.Zu_e = np.array([0.0])

    # Initial state (placeholder; set at runtime in C++ before every solve)
    ocp.constraints.x0 = np.array([0.0, 0.0, HOVER_ALTITUDE,  # position
                                    0.0, 0.0, 0.0,             # velocity
                                    0.0, 0.0, 0.0,             # euler angles
                                    0.0, 0.0, 0.0])            # body rates

    # ── Solver options ────────────────────────────────────────────────────
    ocp.solver_options.qp_solver            = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx       = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type      = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4    # RK4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type      = 'SQP_RTI'   # one RTI step per MPC cycle
    ocp.solver_options.qp_solver_cond_N     = N // 2
    ocp.solver_options.tol                  = 1e-3

    # ── Code export directory ─────────────────────────────────────────────
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ocp.code_gen_options.code_export_directory = os.path.join(repo_root, 'c_generated_code')

    # ── Generate ──────────────────────────────────────────────────────────
    solver = AcadosOcpSolver(ocp, json_file='quadrotor_ocp.json')
    print(f"\nOCP solver generated in: {ocp.code_gen_options.code_export_directory}")
    print(f"nx={nx}, nu={nu}, N={N}, Ts={Ts}s, Tf={N*Ts}s")
    return solver


def generate_sim():
    """Generate a standalone integrator (plant simulator) from the same model.

    Produces acados_sim_solver_quadrotor.c/.h in c_generated_code/.
    The C++ application uses this instead of its own RK4, so the ODE
    only ever lives in quadrotor_model.py.
    """
    sim = AcadosSim()
    model = export_quadrotor_ode_model()
    sim.model = model

    sim.solver_options.T               = Ts   # integrate exactly one sample period
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages      = 4    # RK4
    sim.solver_options.num_steps       = 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sim.code_gen_options.code_export_directory = os.path.join(repo_root, 'c_generated_code')

    AcadosSimSolver(sim, json_file='quadrotor_sim.json')
    print(f"Sim solver generated in: {sim.code_gen_options.code_export_directory}")


def generate_cpp_params_header(out_dir: str):
    """Write argus_params.h — auto-generated C++ constants from config/*.yaml.

    Included by mpc_controller.cpp so C++ never hardcodes values that live
    in the YAML config.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'argus_params.h')
    with open(path, 'w') as f:
        f.write("// Auto-generated by scripts/generate_mpc.py — do not edit.\n")
        f.write("// Source of truth: config/quadrotor.yaml, config/target.yaml\n")
        f.write("#pragma once\n\n")
        f.write(f"static constexpr double ARGUS_MASS     = {MASS};\n")
        f.write(f"static constexpr double ARGUS_G        = {G};\n")
        f.write(f"static constexpr double ARGUS_T_HOVER  = {T_HOVER};\n\n")
        f.write(f"static constexpr double ARGUS_MPC_TS   = {Ts};\n")
        f.write(f"static constexpr double ARGUS_HOVER_ALTITUDE = {HOVER_ALTITUDE};\n\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_R      = {_circ_cfg['radius']};\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_Z      = {_circ_cfg['z']};\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_PERIOD = {_circ_cfg['period']};\n\n")
        f.write(f"static constexpr double ARGUS_PLANT_WN_RP    = {_plant_cfg['wn_rp']};\n")
        f.write(f"static constexpr double ARGUS_PLANT_ZETA_RP  = {_plant_cfg['zeta_rp']};\n")
        f.write(f"static constexpr double ARGUS_PLANT_WN_YAW   = {_plant_cfg['wn_yaw']};\n")
        f.write(f"static constexpr double ARGUS_PLANT_ZETA_YAW = {_plant_cfg['zeta_yaw']};\n\n")
        f.write(f"static constexpr double ARGUS_FOV_HALF_ANGLE_DEG = {_cam_cfg['fov_half_angle_deg']};\n")
        f.write(f"static constexpr double ARGUS_SOFTPLUS_BETA      = {FOV_BETA};\n\n")
        f.write("// 1 = collision avoidance is the slacked d_lin >= d_safe constraint;\n")
        f.write("// 0 = legacy squared-hinge r_obs cost residual. See generate_mpc.py's\n")
        f.write("// OBS_MODE / config/quadrotor.yaml's obstacle_avoidance.mode.\n")
        f.write(f"#define ARGUS_OBS_SLACK_MODE {1 if OBS_MODE == 'slack' else 0}\n")
    print(f"C++ params header written to: {path}")


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generated_dir = os.path.join(repo_root, 'c_generated_code')
    generate_mpc()
    generate_sim()
    generate_cpp_params_header(generated_dir)
