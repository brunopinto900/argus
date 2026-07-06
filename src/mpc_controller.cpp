#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <argus_mpc/QuadrotorMpc.hpp>
// Auto-generated from config/quadrotor.yaml — re-run generate_mpc.py to update
#include "argus_params.h"
#include "plant_dynamics.hpp"
#include "TargetSimulator.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mpc {
    constexpr double Ts      = ARGUS_MPC_TS;
    constexpr double T_hover = ARGUS_T_HOVER;
    constexpr double z_ref   = ARGUS_CIRCLE_Z_REF;  // drone altitude [m]
}

using PlantState = std::array<double, 12>;
using PlantInput = std::array<double, 4>;

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    // ── Parse mode: which TargetSource drives the tracked target ──────────────
    bool is_circle = true;
    if (argc > 1) {
        if (std::strcmp(argv[1], "moving_target") == 0)
            is_circle = false;
        else if (std::strcmp(argv[1], "circle") != 0) {
            std::cerr << "Usage: argus_mpc [circle|moving_target]\n";
            return 1;
        }
    }
    std::cout << "Mode: " << (is_circle ? "circle tracking" : "moving target") << "\n";

    std::unique_ptr<TargetSource> target;
    if (is_circle) {
        target = std::make_unique<CircleTargetSimulator>(
            ARGUS_CIRCLE_R, mpc::z_ref, ARGUS_CIRCLE_PERIOD);
    } else {
        const TargetConfig cfg = load_target_config(ARGUS_CONFIG_DIR "/quadrotor.yaml");
        target = std::make_unique<TargetSimulator>(cfg);
    }

    argus_mpc::QuadrotorMpc mpc;
    const int N = mpc.horizonLength();

    // ── Initial drone state: hovering above the target's start position ───────
    const argus_mpc::TargetState t0 = target->state();
    PlantState x0 = {t0.x,  t0.y,  mpc::z_ref,
                      0.0,   0.0,   0.0,
                      0.0,   0.0,   0.0,
                      0.0,   0.0,   0.0};

    PlantDynamics plant(x0, mpc::T_hover, ARGUS_MASS,
                        ARGUS_PLANT_WN_RP,  ARGUS_PLANT_ZETA_RP,
                        ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);

    // ── Open CSV log ──────────────────────────────────────────────────────────
    fs::create_directories("logs");
    std::ofstream csv("logs/trajectory.csv");
    csv << std::fixed << std::setprecision(6);
    csv << "step,t,"
           "x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,"
           "T,p_cmd,q_cmd,r_cmd,"
           "xt,yt,zt,"       // drone position reference (hover point above target)
           "fx,fy,fz,"       // FOV target position (the tracked target itself)
           "solve_status\n";

    // ── Simulation loop ───────────────────────────────────────────────────────
    constexpr double T_sim = 30.0;
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

        // Drone position reference: hover above the target's look-ahead
        // position; FOV target (model.p) is the target's actual position.
        const argus_mpc::TargetState tgt = target->step(mpc::Ts);
        std::vector<argus_mpc::Reference> horizon(N);
        for (int j = 0; j < N; ++j) {
            horizon[j] = {tgt.x + (j * mpc::Ts) * tgt.vx,
                           tgt.y + (j * mpc::Ts) * tgt.vy,
                           mpc::z_ref};
        }
        const double xt_log = tgt.x, yt_log = tgt.y, zt_log = mpc::z_ref;  // drone position reference
        const double fx_log = tgt.x, fy_log = tgt.y, fz_log = tgt.z;       // FOV target

        const argus_mpc::Command cmd = mpc.step(x0_mpc, horizon, tgt);
        u = {cmd.thrust * mpc::T_hover, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate};

        // ── Log ───────────────────────────────────────────────────────────────
        csv << step << "," << t;
        for (double v : x) csv << "," << v;
        for (double v : u) csv << "," << v;
        csv << "," << xt_log << "," << yt_log << "," << zt_log
            << "," << fx_log << "," << fy_log << "," << fz_log
            << "," << cmd.solve_status << "\n";

        if (step % 40 == 0) {
            const double err = std::hypot(x[0] - xt_log, x[1] - yt_log);
            std::cout << "t=" << std::setw(6) << t << " s  "
                      << "drone=(" << std::setw(7) << x[0] << ", "
                                   << std::setw(7) << x[1] << ")  "
                      << "ref=("   << std::setw(6) << xt_log << ", "
                                   << std::setw(6) << yt_log << ")  "
                      << "xy_err=" << std::setw(6) << err << " m  "
                      << (cmd.solve_status == 0 ? "OK" : "WARN") << "\n";
        }

        // ── Simulate plant one step ───────────────────────────────────────────
        plant.update(u, mpc::Ts);
        x = plant.state();
    }

    csv.close();
    std::cout << "\nDone. Trajectory saved to logs/trajectory.csv\n";
    std::cout << "Visualise with: python3 scripts/animate_xy.py\n";

    return 0;
}
