import casadi as ca
from acados_template import AcadosModel


def export_quadrotor_ode_model() -> AcadosModel:
    """
    12-state quadrotor model with Euler angles (ZYX convention).

    States : [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    Inputs : [T, tau_phi, tau_theta, tau_psi]

    Euler angle kinematics have a singularity at theta = ±90 deg.
    Keep attitude well away from that limit (constraints in OCP handle this).
    """

    # ── Physical parameters ────────────────────────────────────────────────
    m  = 1.0     # total mass            [kg]
    g  = 9.81    # gravitational accel   [m/s²]
    Ix = 0.01    # roll  inertia         [kg·m²]
    Iy = 0.01    # pitch inertia         [kg·m²]
    Iz = 0.02    # yaw   inertia         [kg·m²]

    # ── State symbols ──────────────────────────────────────────────────────
    x     = ca.SX.sym('x')
    y     = ca.SX.sym('y')
    z     = ca.SX.sym('z')
    vx    = ca.SX.sym('vx')
    vy    = ca.SX.sym('vy')
    vz    = ca.SX.sym('vz')
    phi   = ca.SX.sym('phi')    # roll
    theta = ca.SX.sym('theta')  # pitch
    psi   = ca.SX.sym('psi')    # yaw
    p     = ca.SX.sym('p')      # roll  rate  [rad/s]
    q     = ca.SX.sym('q')      # pitch rate  [rad/s]
    r     = ca.SX.sym('r')      # yaw   rate  [rad/s]

    state = ca.vertcat(x, y, z, vx, vy, vz, phi, theta, psi, p, q, r)

    # ── Input symbols ──────────────────────────────────────────────────────
    T         = ca.SX.sym('T')          # collective thrust  [N]
    tau_phi   = ca.SX.sym('tau_phi')    # roll  torque       [Nm]
    tau_theta = ca.SX.sym('tau_theta')  # pitch torque       [Nm]
    tau_psi   = ca.SX.sym('tau_psi')    # yaw   torque       [Nm]

    u = ca.vertcat(T, tau_phi, tau_theta, tau_psi)

    # ── xdot symbols (required for implicit form f_impl = xdot - f_expl) ──
    x_dot     = ca.SX.sym('x_dot')
    y_dot     = ca.SX.sym('y_dot')
    z_dot     = ca.SX.sym('z_dot')
    vx_dot    = ca.SX.sym('vx_dot')
    vy_dot    = ca.SX.sym('vy_dot')
    vz_dot    = ca.SX.sym('vz_dot')
    phi_dot   = ca.SX.sym('phi_dot')
    theta_dot = ca.SX.sym('theta_dot')
    psi_dot   = ca.SX.sym('psi_dot')
    p_dot     = ca.SX.sym('p_dot')
    q_dot     = ca.SX.sym('q_dot')
    r_dot     = ca.SX.sym('r_dot')

    xdot = ca.vertcat(x_dot, y_dot, z_dot,
                      vx_dot, vy_dot, vz_dot,
                      phi_dot, theta_dot, psi_dot,
                      p_dot, q_dot, r_dot)

    # ── Thrust in world frame: R_ZYX * [0, 0, T] ──────────────────────────
    #   R = Rz(psi) · Ry(theta) · Rx(phi)
    Tw_x = T * (ca.cos(psi)*ca.sin(theta)*ca.cos(phi) + ca.sin(psi)*ca.sin(phi))
    Tw_y = T * (ca.sin(psi)*ca.sin(theta)*ca.cos(phi) - ca.cos(psi)*ca.sin(phi))
    Tw_z = T *  ca.cos(theta)*ca.cos(phi)

    # ── Explicit ODE ───────────────────────────────────────────────────────
    f_expl = ca.vertcat(
        # position kinematics
        vx,
        vy,
        vz,
        # translational dynamics
        Tw_x / m,
        Tw_y / m,
        Tw_z / m - g,
        # Euler angle kinematics (ZYX, body rates → Euler rates)
        p + ca.sin(phi)*ca.tan(theta)*q + ca.cos(phi)*ca.tan(theta)*r,
        ca.cos(phi)*q - ca.sin(phi)*r,
        (ca.sin(phi)*q + ca.cos(phi)*r) / ca.cos(theta),
        # rotational dynamics – Euler's equations, diagonal inertia
        (tau_phi   - (Iz - Iy)*q*r) / Ix,
        (tau_theta - (Ix - Iz)*p*r) / Iy,
        (tau_psi   - (Iy - Ix)*p*q) / Iz,
    )

    # ── AcadosModel ───────────────────────────────────────────────────────
    model = AcadosModel()
    model.name        = 'quadrotor'
    model.x           = state
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    model.x_labels = ['x [m]', 'y [m]', 'z [m]',
                      'vx [m/s]', 'vy [m/s]', 'vz [m/s]',
                      'phi [rad]', 'theta [rad]', 'psi [rad]',
                      'p [rad/s]', 'q [rad/s]', 'r [rad/s]']
    model.u_labels = ['T [N]', 'tau_phi [Nm]', 'tau_theta [Nm]', 'tau_psi [Nm]']
    model.t_label  = 't [s]'

    return model
