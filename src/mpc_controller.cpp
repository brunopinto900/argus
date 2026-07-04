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
// Auto-generated from config/quadrotor.yaml — re-run generate_mpc.py to update
#include "argus_params.h"
#include "plant_dynamics.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── MPC constants ─────────────────────────────────────────────────────────
namespace mpc {
    constexpr int    N       = QUADROTOR_N;        // from generated solver header
    constexpr int    NX      = QUADROTOR_NX;
    constexpr int    NU      = QUADROTOR_NU;
    constexpr int    NY      = QUADROTOR_NY;
    constexpr int    NYN     = QUADROTOR_NYN;
    constexpr double Ts      = ARGUS_MPC_TS;               // from argus_params.h
    constexpr double T_hover = ARGUS_T_HOVER;      // from argus_params.h
}

// ── Circle trajectory parameters (from argus_params.h) ────────────────────
namespace circle {
    constexpr double R      = ARGUS_CIRCLE_R;
    constexpr double z_ref  = ARGUS_CIRCLE_Z_REF;
    constexpr double period = ARGUS_CIRCLE_PERIOD;
    constexpr double omega  = 2.0 * M_PI / period;
}

using State = std::array<double, QUADROTOR_NX>;
using Input = std::array<double, QUADROTOR_NU>;

// ── Circle references ──────────────────────────────────────────────────────
// yref layout:  [x,y,z, vx,vy,vz, phi,theta, cos(psi),sin(psi), T, p_cmd,q_cmd,r_cmd]
static void circle_yref(double t, double yref[QUADROTOR_NY])
{
    const double ct  = std::cos(circle::omega * t);
    const double st  = std::sin(circle::omega * t);
    const double xr  = circle::R * ct;
    const double yr  = circle::R * st;
    // yaw that points from the reference position toward the origin
    const double psi_ref = std::atan2(-yr, -xr);

    yref[0]  = xr;
    yref[1]  = yr;
    yref[2]  = circle::z_ref;
    yref[3]  = -circle::R * circle::omega * st;
    yref[4]  =  circle::R * circle::omega * ct;
    yref[5]  =  0.0;
    yref[6]  =  0.0;                        // phi: level
    yref[7]  =  0.0;                        // theta: level
    yref[8]  =  std::cos(psi_ref);          // yaw unit vector
    yref[9]  =  std::sin(psi_ref);
    yref[10] =  mpc::T_hover;
    yref[11] =  0.0;
    yref[12] =  0.0;
    yref[13] =  0.0;
}

// yref_e layout: [x,y,z, vx,vy,vz, phi,theta, cos(psi),sin(psi)]
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
        circle::R, 0.0,        circle::z_ref,      // position
        0.0,       circle::R * circle::omega, 0.0,  // velocity (tangential)
        0.0,       0.0,        0.0,                 // euler angles (level)
        0.0,       0.0,        0.0                  // body rates
    };

    // ── Plant simulator: second-order dynamics, different from MPC model ──
    PlantDynamics plant(x0, mpc::T_hover, ARGUS_MASS,
                        ARGUS_PLANT_WN_RP,  ARGUS_PLANT_ZETA_RP,
                        ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);

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
           "T,p_cmd,q_cmd,r_cmd,"
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

        // 6. Simulate plant one step (second-order dynamics)
        plant.update(u, mpc::Ts);
        x = plant.state();

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
