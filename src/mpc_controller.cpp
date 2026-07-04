#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_quadrotor.h"
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Model constants — must match scripts/quadrotor_model.py ───────────────
namespace model {
    constexpr double m  = 1.0;
    constexpr double g  = 9.81;
    constexpr double Ix = 0.01;
    constexpr double Iy = 0.01;
    constexpr double Iz = 0.02;
}

// ── MPC constants — must match scripts/generate_mpc.py ────────────────────
namespace mpc {
    constexpr int    N       = QUADROTOR_N;    // 20
    constexpr int    NX      = QUADROTOR_NX;   // 12
    constexpr int    NU      = QUADROTOR_NU;   //  4
    constexpr int    NY      = QUADROTOR_NY;   // 12
    constexpr int    NYN     = QUADROTOR_NYN;  //  8
    constexpr double Ts      = 0.05;
    constexpr double T_hover = model::m * model::g;
}

// ── Circle trajectory parameters ──────────────────────────────────────────
namespace circle {
    constexpr double R      = 2.0;
    constexpr double z_ref  = 1.5;
    constexpr double period = 10.0;
    constexpr double omega  = 2.0 * M_PI / period;
}

using State = std::array<double, QUADROTOR_NX>;
using Input = std::array<double, QUADROTOR_NU>;

// ── Quadrotor ODE — identical to scripts/quadrotor_model.py ───────────────
static void quadrotor_ode(const State& x, const Input& u, State& xd)
{
    const double phi = x[6], theta = x[7], psi = x[8];
    const double p   = x[9], q     = x[10], r  = x[11];
    const double T   = u[0], t_phi = u[1], t_th = u[2], t_psi = u[3];

    const double Twx = T * (std::cos(psi)*std::sin(theta)*std::cos(phi) + std::sin(psi)*std::sin(phi));
    const double Twy = T * (std::sin(psi)*std::sin(theta)*std::cos(phi) - std::cos(psi)*std::sin(phi));
    const double Twz = T *  std::cos(theta)*std::cos(phi);

    xd[0]  = x[3];
    xd[1]  = x[4];
    xd[2]  = x[5];
    xd[3]  = Twx / model::m;
    xd[4]  = Twy / model::m;
    xd[5]  = Twz / model::m - model::g;
    xd[6]  = p + std::sin(phi)*std::tan(theta)*q + std::cos(phi)*std::tan(theta)*r;
    xd[7]  = std::cos(phi)*q - std::sin(phi)*r;
    xd[8]  = (std::sin(phi)*q + std::cos(phi)*r) / std::cos(theta);
    xd[9]  = (t_phi - (model::Iz - model::Iy)*q*r) / model::Ix;
    xd[10] = (t_th  - (model::Ix - model::Iz)*p*r) / model::Iy;
    xd[11] = (t_psi - (model::Iy - model::Ix)*p*q) / model::Iz;
}

// ── RK4 integrator ─────────────────────────────────────────────────────────
static State rk4_step(const State& x, const Input& u, double dt)
{
    State k1{}, k2{}, k3{}, k4{}, tmp{}, xnext{};

    quadrotor_ode(x, u, k1);
    for (int i = 0; i < mpc::NX; ++i) tmp[i] = x[i] + 0.5*dt*k1[i];

    quadrotor_ode(tmp, u, k2);
    for (int i = 0; i < mpc::NX; ++i) tmp[i] = x[i] + 0.5*dt*k2[i];

    quadrotor_ode(tmp, u, k3);
    for (int i = 0; i < mpc::NX; ++i) tmp[i] = x[i] + dt*k3[i];

    quadrotor_ode(tmp, u, k4);
    for (int i = 0; i < mpc::NX; ++i)
        xnext[i] = x[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

    return xnext;
}

// ── Circle references ──────────────────────────────────────────────────────
// yref layout:  [x,y,z, vx,vy,vz, phi,theta, T, tau_phi,tau_theta,tau_psi]
static void circle_yref(double t, double yref[QUADROTOR_NY])
{
    const double ct = std::cos(circle::omega * t);
    const double st = std::sin(circle::omega * t);
    yref[0]  = circle::R * ct;
    yref[1]  = circle::R * st;
    yref[2]  = circle::z_ref;
    yref[3]  = -circle::R * circle::omega * st;
    yref[4]  =  circle::R * circle::omega * ct;
    yref[5]  =  0.0;
    yref[6]  =  0.0;           // phi: level
    yref[7]  =  0.0;           // theta: level
    yref[8]  =  mpc::T_hover;
    yref[9]  =  0.0;
    yref[10] =  0.0;
    yref[11] =  0.0;
}

// yref_e layout: [x,y,z, vx,vy,vz, phi,theta]
static void circle_yref_terminal(double t, double yref_e[QUADROTOR_NYN])
{
    double tmp[QUADROTOR_NY];
    circle_yref(t, tmp);
    for (int i = 0; i < QUADROTOR_NYN; ++i) yref_e[i] = tmp[i];
}

// ─────────────────────────────────────────────────────────────────────────
int main()
{
    namespace fs = std::filesystem;

    // ── Create solver ─────────────────────────────────────────────────────
    quadrotor_solver_capsule* capsule = quadrotor_acados_create_capsule();
    int status = quadrotor_acados_create(capsule);
    if (status) {
        std::cerr << "quadrotor_acados_create() returned " << status << "\n";
        return 1;
    }

    ocp_nlp_config* config  = quadrotor_acados_get_nlp_config(capsule);
    ocp_nlp_dims*   dims    = quadrotor_acados_get_nlp_dims(capsule);
    ocp_nlp_in*     nlp_in  = quadrotor_acados_get_nlp_in(capsule);
    ocp_nlp_out*    nlp_out = quadrotor_acados_get_nlp_out(capsule);

    // ── Initial state: on the circle at t=0 with tangential velocity ──────
    State x0 = {
        circle::R, 0.0,        circle::z_ref,    // position
        0.0,       circle::R * circle::omega, 0.0, // velocity (tangential)
        0.0,       0.0,        0.0,               // euler angles (level)
        0.0,       0.0,        0.0                // body rates
    };

    // Warm-start all stages with x0 and hover thrust
    Input u_init = {mpc::T_hover, 0.0, 0.0, 0.0};
    for (int i = 0; i <= mpc::N; ++i)
        ocp_nlp_out_set(config, dims, nlp_out, nlp_in, i, "x", x0.data());
    for (int i = 0; i < mpc::N; ++i)
        ocp_nlp_out_set(config, dims, nlp_out, nlp_in, i, "u", u_init.data());

    // ── Open CSV log ──────────────────────────────────────────────────────
    fs::create_directories("logs");
    std::ofstream csv("logs/trajectory.csv");
    csv << std::fixed << std::setprecision(6);
    csv << "step,t,"
           "x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,"
           "T,tau_phi,tau_theta,tau_psi,"
           "x_ref,y_ref,z_ref,solve_status\n";

    // ── Simulation loop ───────────────────────────────────────────────────
    constexpr double T_sim = 20.0;    // 2 full circles
    const int Nsim = static_cast<int>(T_sim / mpc::Ts);

    State x = x0;
    Input u = u_init;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Running " << Nsim << " MPC steps (" << T_sim << " s) ...\n\n";

    for (int step = 0; step < Nsim; ++step)
    {
        const double t = step * mpc::Ts;

        // 1. Pin initial state
        ocp_nlp_constraints_model_set(config, dims, nlp_in, nlp_out, 0, "lbx", x.data());
        ocp_nlp_constraints_model_set(config, dims, nlp_in, nlp_out, 0, "ubx", x.data());

        // 2. Set reference for every horizon stage
        double yref[QUADROTOR_NY];
        double yref_e[QUADROTOR_NYN];
        for (int j = 0; j < mpc::N; ++j) {
            circle_yref(t + j * mpc::Ts, yref);
            ocp_nlp_cost_model_set(config, dims, nlp_in, j, "yref", yref);
        }
        circle_yref_terminal(t + mpc::N * mpc::Ts, yref_e);
        ocp_nlp_cost_model_set(config, dims, nlp_in, mpc::N, "yref", yref_e);

        // 3. Solve (SQP-RTI: one iteration)
        status = quadrotor_acados_solve(capsule);

        // 4. Read first optimal control
        ocp_nlp_out_get(config, dims, nlp_out, 0, "u", u.data());

        // 5. Log
        const double xr = circle::R * std::cos(circle::omega * t);
        const double yr = circle::R * std::sin(circle::omega * t);
        csv << step << "," << t;
        for (double v : x) csv << "," << v;
        for (double v : u) csv << "," << v;
        csv << "," << xr << "," << yr << "," << circle::z_ref
            << "," << status << "\n";

        if (step % 40 == 0) {
            const double err = std::hypot(x[0] - xr, x[1] - yr);
            std::cout << "t=" << std::setw(6) << t << " s  "
                      << "pos=(" << std::setw(7) << x[0] << ", "
                                 << std::setw(7) << x[1] << ")  "
                      << "xy_err=" << std::setw(6) << err << " m  "
                      << (status == 0 ? "OK" : "WARN") << "\n";
        }

        // 6. Simulate plant one step with RK4
        x = rk4_step(x, u, mpc::Ts);

        // 7. Shift warm-start: move stage i+1 → stage i
        State x_s{};
        Input u_s{};
        for (int i = 0; i < mpc::N - 1; ++i) {
            ocp_nlp_out_get(config, dims, nlp_out, i + 1, "x", x_s.data());
            ocp_nlp_out_get(config, dims, nlp_out, i + 1, "u", u_s.data());
            ocp_nlp_out_set(config, dims, nlp_out, nlp_in, i, "x", x_s.data());
            ocp_nlp_out_set(config, dims, nlp_out, nlp_in, i, "u", u_s.data());
        }
        ocp_nlp_out_get(config, dims, nlp_out, mpc::N, "x", x_s.data());
        ocp_nlp_out_set(config, dims, nlp_out, nlp_in, mpc::N, "x", x_s.data());
    }

    csv.close();
    std::cout << "\nDone. Trajectory saved to logs/trajectory.csv\n";
    std::cout << "Visualise with: python3 scripts/animate_xy.py\n";

    // ── Cleanup ───────────────────────────────────────────────────────────
    quadrotor_acados_free(capsule);
    quadrotor_acados_free_capsule(capsule);
    return 0;
}
