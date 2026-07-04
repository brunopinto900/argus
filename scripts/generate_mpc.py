"""
Generate the ACADOS OCP solver for the quadrotor circle-tracking MPC.

Run once from the repo root:
    python3 scripts/generate_mpc.py

Output: c_generated_code/   (C solver + CMakeLists, ready to link from C++)
        quadrotor_ocp.json   (OCP spec, used by ACADOS internals)

State vector  (nx = 12):  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Input vector  (nu =  4):  [T, tau_phi, tau_theta, tau_psi]

Cost output   (ny = 12):  [x, y, z, vx, vy, vz, phi, theta, T, tau_phi, tau_theta, tau_psi]
  - psi is intentionally excluded from the cost for now (yaw pointing added later)
  - phi, theta penalised to keep the drone roughly level
  - inputs penalised to limit aggressiveness

The yref for each horizon step is updated at runtime in C++ via:
    acados_ocp_solver_set(solver, i, "yref", yref_ptr)
"""

import os
import sys
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

# Make ACADOS shared libs visible before the solver is loaded
_acados_lib = os.path.join(os.path.expanduser('~'), 'acados', 'lib')
os.environ['LD_LIBRARY_PATH'] = _acados_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# allow 'from quadrotor_model import ...' when running from repo root
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_model import export_quadrotor_ode_model


# ── MPC tuning parameters ─────────────────────────────────────────────────────
N  = 20      # prediction horizon steps
Ts = 0.05    # sampling time [s]   → Tf = 1.0 s look-ahead

# ── Quadrotor physical params (must match quadrotor_model.py) ─────────────────
MASS = 1.0
G    = 9.81
T_HOVER = MASS * G          # ~9.81 N

# ── Input limits ──────────────────────────────────────────────────────────────
T_MIN   = 0.0
T_MAX   = 2.0 * T_HOVER     # 2× hover thrust
TAU_MAX = 0.5               # ±0.5 Nm for roll/pitch/yaw torques

# ── State limits (attitude only; avoids Euler singularity) ────────────────────
ANGLE_MAX = np.deg2rad(60.0)   # ±60° for roll and pitch

# ── Cost weights ──────────────────────────────────────────────────────────────
# Stage cost diagonal: [x, y, z, vx, vy, vz, phi, theta, T, tau_phi, tau_theta, tau_psi]
W_STAGE = np.diag([
    100.0, 100.0, 100.0,   # position tracking (primary objective)
     10.0,  10.0,  10.0,   # velocity tracking
      5.0,   5.0,          # roll, pitch (keep level)
      0.1,                 # thrust regularisation
      0.1,   0.1,   0.1,  # torque regularisation
])

# Terminal cost diagonal: [x, y, z, vx, vy, vz, phi, theta]
W_TERMINAL = np.diag([
    200.0, 200.0, 200.0,
     20.0,  20.0,  20.0,
     10.0,  10.0,
])


def generate_mpc():
    ocp   = AcadosOcp()
    model = export_quadrotor_ode_model()
    ocp.model = model

    nx = model.x.rows()   # 12
    nu = model.u.rows()   #  4

    # ── Horizon ───────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * Ts

    # ── Cost function ─────────────────────────────────────────────────────
    # cost_y_expr selects which states/inputs enter the cost.
    # Indices in state: x(0) y(1) z(2) vx(3) vy(4) vz(5) phi(6) theta(7) psi(8) p(9) q(10) r(11)
    # psi (index 8) and body rates (9-11) are intentionally left out for now.
    x_sym = model.x
    u_sym = model.u

    cost_y_expr   = ca.vertcat(x_sym[:8], u_sym)   # [pos, vel, phi, theta, inputs]  ny=12
    cost_y_expr_e = x_sym[:8]                       # terminal: [pos, vel, phi, theta] ny_e=8

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr   = cost_y_expr
    ocp.model.cost_y_expr_e = cost_y_expr_e

    ocp.cost.W   = W_STAGE
    ocp.cost.W_e = W_TERMINAL

    # Nominal yref: hovering at origin, zero velocity, level attitude.
    # In C++ the MPC loop overwrites this every step with the circle reference.
    yref_hover   = np.array([0.0, 0.0, 1.5,        # x, y, z
                              0.0, 0.0, 0.0,        # vx, vy, vz
                              0.0, 0.0,             # phi, theta
                              T_HOVER, 0.0, 0.0, 0.0])  # T, tau_*
    yref_hover_e = yref_hover[:8]

    ocp.cost.yref   = yref_hover
    ocp.cost.yref_e = yref_hover_e

    # ── Constraints ───────────────────────────────────────────────────────
    # Input box constraints
    ocp.constraints.lbu    = np.array([T_MIN,    -TAU_MAX, -TAU_MAX, -TAU_MAX])
    ocp.constraints.ubu    = np.array([T_MAX,     TAU_MAX,  TAU_MAX,  TAU_MAX])
    ocp.constraints.idxbu  = np.array([0, 1, 2, 3])

    # State box constraints on phi (idx 6) and theta (idx 7) — running stages
    ocp.constraints.lbx    = np.array([-ANGLE_MAX, -ANGLE_MAX])
    ocp.constraints.ubx    = np.array([ ANGLE_MAX,  ANGLE_MAX])
    ocp.constraints.idxbx  = np.array([6, 7])

    # Same attitude limits at the terminal stage
    ocp.constraints.lbx_e   = ocp.constraints.lbx
    ocp.constraints.ubx_e   = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    # Initial state (placeholder; set at runtime in C++ before every solve)
    ocp.constraints.x0 = np.array([0.0, 0.0, 1.5,   # position
                                    0.0, 0.0, 0.0,   # velocity
                                    0.0, 0.0, 0.0,   # euler angles
                                    0.0, 0.0, 0.0])  # body rates

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
    print(f"\nCode generated in: {ocp.code_export_directory}")
    print(f"nx={nx}, nu={nu}, N={N}, Ts={Ts}s, Tf={N*Ts}s")
    return solver


if __name__ == '__main__':
    generate_mpc()
