#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <argus_mpc/QuadrotorMpc.hpp>
// Auto-generated from config/quadrotor.yaml — re-run generate_mpc.py to update
#include "argus_params.h"
#include "plant_dynamics.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── MPC / circle-trajectory constants (from argus_params.h) ──────────────
namespace mpc {
    constexpr double Ts      = ARGUS_MPC_TS;
    constexpr double T_hover = ARGUS_T_HOVER;
}

namespace circle {
    constexpr double R      = ARGUS_CIRCLE_R;
    constexpr double z_ref  = ARGUS_CIRCLE_Z_REF;
    constexpr double period = ARGUS_CIRCLE_PERIOD;
    constexpr double omega  = 2.0 * M_PI / period;
}

using PlantState = std::array<double, 12>;
using PlantInput = std::array<double, 4>;

// Circle reference at time t — fed into QuadrotorMpc per horizon stage, so
// the solver sees a real look-ahead of where the circle is going, not a
// single held point.
static argus_mpc::Reference circle_reference(double t)
{
    const double ct = std::cos(circle::omega * t);
    const double st = std::sin(circle::omega * t);

    argus_mpc::Reference ref;
    ref.x  = circle::R * ct;
    ref.y  = circle::R * st;
    ref.z  = circle::z_ref;
    ref.vx = -circle::R * circle::omega * st;
    ref.vy =  circle::R * circle::omega * ct;
    ref.vz = 0.0;
    return ref;
}

// ─────────────────────────────────────────────────────────────────────────
int main()
{
    namespace fs = std::filesystem;

    argus_mpc::QuadrotorMpc mpc;
    const int N = mpc.horizonLength();

    // ── Initial state: on the circle at t=0 with tangential velocity ──────
    PlantState x0 = {
        circle::R, 0.0,        circle::z_ref,       // position
        0.0,       circle::R * circle::omega, 0.0,  // velocity (tangential)
        0.0,       0.0,        0.0,                 // euler angles (level)
        0.0,       0.0,        0.0                  // body rates
    };

    // ── Plant simulator: second-order dynamics, different from MPC model ──
    PlantDynamics plant(x0, mpc::T_hover, ARGUS_MASS,
                        ARGUS_PLANT_WN_RP,  ARGUS_PLANT_ZETA_RP,
                        ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);

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

    PlantState x = x0;
    PlantInput u = {mpc::T_hover, 0.0, 0.0, 0.0};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Running " << Nsim << " MPC steps (" << T_sim << " s) ...\n\n";

    for (int step = 0; step < Nsim; ++step)
    {
        const double t = step * mpc::Ts;

        const argus_mpc::State x0_mpc{
            x[0], x[1], x[2],
            x[3], x[4], x[5],
            x[6], x[7], x[8],
            x[9], x[10], x[11],
        };

        std::vector<argus_mpc::Reference> horizon(N);
        for (int j = 0; j < N; ++j)
            horizon[j] = circle_reference(t + j * mpc::Ts);

        const argus_mpc::Command cmd = mpc.step(x0_mpc, horizon);
        u = {cmd.thrust * mpc::T_hover, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate};

        // ── Log ───────────────────────────────────────────────────────────
        const double xr = circle::R * std::cos(circle::omega * t);
        const double yr = circle::R * std::sin(circle::omega * t);
        csv << step << "," << t;
        for (double v : x) csv << "," << v;
        for (double v : u) csv << "," << v;
        csv << "," << xr << "," << yr << "," << circle::z_ref
            << "," << cmd.solve_status << "\n";

        if (step % 40 == 0) {
            const double err = std::hypot(x[0] - xr, x[1] - yr);
            std::cout << "t=" << std::setw(6) << t << " s  "
                      << "pos=(" << std::setw(7) << x[0] << ", "
                                 << std::setw(7) << x[1] << ")  "
                      << "xy_err=" << std::setw(6) << err << " m  "
                      << (cmd.solve_status == 0 ? "OK" : "WARN") << "\n";
        }

        // ── Simulate plant one step (second-order dynamics) ────────────────
        plant.update(u, mpc::Ts);
        x = plant.state();
    }

    csv.close();
    std::cout << "\nDone. Trajectory saved to logs/trajectory.csv\n";
    std::cout << "Visualise with: python3 scripts/animate_xy.py\n";

    return 0;
}
