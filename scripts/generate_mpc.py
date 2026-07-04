"""
Generate the ACADOS OCP solver for the quadrotor circle-tracking MPC.

Run once from the repo root:
    python3 scripts/generate_mpc.py

Output: c_generated_code/   (C solver + CMakeLists, ready to link from C++)
        quadrotor_ocp.json   (OCP spec, used by ACADOS internals)

State vector  (nx = 12):  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Input vector  (nu =  4):  [T, p_cmd, q_cmd, r_cmd]

Cost output   (ny = 12):  [x, y, z, vx, vy, vz, phi, theta, T, p_cmd, q_cmd, r_cmd]
  - psi is intentionally excluded from the cost for now (yaw pointing added later)
  - phi, theta penalised to keep the drone roughly level
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
from config import load_config


# ── Load all parameters from config/quadrotor.yaml ────────────────────────────
_cfg        = load_config()
_model_cfg  = _cfg['model']
_mpc_cfg    = _cfg['mpc']
_con_cfg    = _cfg['constraints']
_cost_cfg   = _cfg['cost']
_circ_cfg   = _cfg['circle']

N       = _mpc_cfg['N']
Ts      = _mpc_cfg['Ts']

MASS    = _model_cfg['mass']
G       = _model_cfg['gravity']
T_HOVER = MASS * G

T_MIN        = _con_cfg['T_min']
T_MAX        = _con_cfg['T_max_factor'] * T_HOVER
RATE_MAX     = _con_cfg['rate_max']
YAW_RATE_MAX = _con_cfg['yaw_rate_max']
ANGLE_MAX    = np.deg2rad(_con_cfg['angle_max_deg'])

W_STAGE    = np.diag(_cost_cfg['W_stage'])
W_TERMINAL = np.diag(_cost_cfg['W_terminal'])


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
    ocp.constraints.lbu    = np.array([T_MIN, -RATE_MAX, -RATE_MAX, -YAW_RATE_MAX])
    ocp.constraints.ubu    = np.array([T_MAX,  RATE_MAX,  RATE_MAX,  YAW_RATE_MAX])
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
    """Write argus_params.h — auto-generated C++ constants from quadrotor.yaml.

    Included by mpc_controller.cpp so C++ never hardcodes values that live
    in the YAML config.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'argus_params.h')
    with open(path, 'w') as f:
        f.write("// Auto-generated by scripts/generate_mpc.py — do not edit.\n")
        f.write("// Source of truth: config/quadrotor.yaml\n")
        f.write("#pragma once\n\n")
        f.write(f"static constexpr double ARGUS_MASS     = {MASS};\n")
        f.write(f"static constexpr double ARGUS_G        = {G};\n")
        f.write(f"static constexpr double ARGUS_T_HOVER  = {T_HOVER};\n\n")
        f.write(f"static constexpr double ARGUS_MPC_TS   = {Ts};\n\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_R      = {_circ_cfg['radius']};\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_Z_REF  = {_circ_cfg['z_ref']};\n")
        f.write(f"static constexpr double ARGUS_CIRCLE_PERIOD = {_circ_cfg['period']};\n")
    print(f"C++ params header written to: {path}")


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generated_dir = os.path.join(repo_root, 'c_generated_code')
    generate_mpc()
    generate_sim()
    generate_cpp_params_header(generated_dir)
