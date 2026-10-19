// Wires argus_mpc::QuadrotorMpc to argus_follower::TrailerFollower: the
// trailer-kinematics goal-point horizon (position + velocity, per stage)
// becomes the MPC's own drone-position reference, and the tracked target's
// actual position/velocity drives the FOV cost -- same roles
// src/mpc_controller.cpp's DroneReference/TargetState split already has,
// just with a different reference source (hitch-lag kinematics instead of
// "hover above" / "orbit around"). No obstacle awareness yet -- see
// design_decisions.md's "Explicitly out of scope" note; this only closes
// the MPC<->follower loop.
//
// Usage: ./build/trailer_mpc_demo [line|circle|zigzag] [t_end_seconds]
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>
#include <argus_mpc/QuadrotorMpc.hpp>

// Auto-generated from config/quadrotor.yaml -- re-run generate_mpc.py to update.
#include "argus_params.h"
#include "plant_dynamics.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using argus_follower::SplineXY;
using argus_follower::TrailerFollower;
using PlantState = std::array<double, 12>;
using PlantInput = std::array<double, 4>;

namespace {

constexpr double kTs             = ARGUS_MPC_TS;
constexpr double kTHover         = ARGUS_T_HOVER;
// Static FOV-aware z reference, NOT the circle demo's ARGUS_HOVER_ALTITUDE
// -- that 1.5m comes from a different derivation tied to circular motion's
// own banking requirement (tan(theta) = r*omega^2/g), which doesn't apply
// to this scenario's arbitrary leader path. Here z_ref is instead derived
// from tan(theta_FOV) = z_ref / distance_to_target, solved for z_ref at the
// configured trailer.target_bearing_deg -- see design_decisions.md's
// "Static FOV-aware z_ref" for the full derivation and why this is computed
// once from the configured standoff rather than recomputed online every
// control cycle (oscillation risk).
// The tracked leader's own altitude for the FOV cost -- a ground-level
// target (person/vehicle), distinct from the drone's z reference above
// it. TrailerFollower/TargetMotionSimulator are 2-D (x, y only), so this
// is fixed here, not sourced from either of those.
constexpr double kTargetZ = 0.0;

// Same loader as argus_follower/examples/follower_demo.cpp.
struct TrailerConfig
{
    double distance_to_target = 1.0;
    double heading_gain       = 1.0;
    int    N                  = 20;
    double Ts                 = 0.05;
    double target_bearing_deg = 30.0;
};

TrailerConfig loadTrailerConfig(const std::string& yaml_path)
{
    const YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["trailer"]) throw std::runtime_error(yaml_path + ": missing 'trailer' section");
    const YAML::Node n = root["trailer"];

    TrailerConfig cfg;
    if (n["distance_to_target"]) cfg.distance_to_target = n["distance_to_target"].as<double>();
    if (n["heading_gain"])       cfg.heading_gain        = n["heading_gain"].as<double>();
    if (n["N"])                  cfg.N                   = n["N"].as<int>();
    if (n["Ts"])                 cfg.Ts                  = n["Ts"].as<double>();
    if (n["target_bearing_deg"]) cfg.target_bearing_deg  = n["target_bearing_deg"].as<double>();
    return cfg;
}

SplineXY buildTarget(const std::string& motion, double t_end, double sample_dt)
{
    if (motion == "line") {
        return argus_follower::straightLineTarget(0.0, 0.0, 1.0, t_end, sample_dt);
    }
    if (motion == "circle") {
        return argus_follower::circleTarget(0.0, 0.0, 5.0, 0.5, t_end, sample_dt);
    }
    if (motion == "zigzag") {
        return argus_follower::zigzagTarget(0.0, 0.0, 1.0, 1.0, 5.0, t_end, sample_dt);
    }
    throw std::invalid_argument("unknown motion '" + motion + "' (expected line|circle|zigzag)");
}

}  // namespace

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    const std::string motion = argc > 1 ? argv[1] : "circle";
    const double t_end = argc > 2 ? std::stod(argv[2]) : 20.0;

    const TrailerConfig trailer_cfg = loadTrailerConfig(ARGUS_CONFIG_DIR "/quadrotor.yaml");
    // z_ref = distance_to_target * tan(target_bearing_deg) -- see the
    // comment above kTHover.
    const double kFovZRef = trailer_cfg.distance_to_target *
                             std::tan(trailer_cfg.target_bearing_deg * M_PI / 180.0);

    argus_mpc::QuadrotorMpc mpc;
    const int N = mpc.horizonLength();
    if (trailer_cfg.N != N) {
        throw std::runtime_error(
            "config/quadrotor.yaml: trailer.N (" + std::to_string(trailer_cfg.N) +
            ") must equal mpc.N (" + std::to_string(N) + ") -- the follower's horizon "
            "becomes the MPC's own reference horizon, stage for stage");
    }
    if (std::abs(trailer_cfg.Ts - kTs) > 1e-9) {
        throw std::runtime_error("config/quadrotor.yaml: trailer.Ts must equal mpc.Ts");
    }

    SplineXY target = buildTarget(motion, t_end, trailer_cfg.Ts);
    TrailerFollower follower(trailer_cfg.distance_to_target, trailer_cfg.heading_gain);

    const int n_steps = static_cast<int>(std::round(t_end / trailer_cfg.Ts));

    // First horizon call fixes the drone's initial position (and advances
    // the follower's heading state by one step, same as every subsequent
    // call) -- not recomputed again for step 0 below, see
    // online_mapping_demo.cpp's identical pattern for why calling it twice
    // would double-advance that state.
    const auto horizon0 = follower.computeGoalPoints(target.x, target.y, N, 0.0, trailer_cfg.Ts);
    PlantState x = {horizon0.PDx[0], horizon0.PDy[0], kFovZRef,
                     0.0,            0.0,             0.0,
                     0.0,            0.0,             0.0,
                     0.0,            0.0,             0.0};
    PlantDynamics plant(x, kTHover, ARGUS_MASS, ARGUS_PLANT_WN_RP, ARGUS_PLANT_ZETA_RP,
                         ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);

    fs::create_directories("logs");
    const std::string out_path = "logs/trailer_mpc_trajectory.csv";
    std::ofstream csv(out_path);
    csv << std::fixed << std::setprecision(6);
    csv << "step,t,"
           "x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,"
           "T,p_cmd,q_cmd,r_cmd,"
           "xt,yt,zt,"   // drone position reference (follower's goal point, stage 0)
           "fx,fy,fz,"   // FOV target (the tracked leader itself)
           "solve_status\n";

    PlantInput u = {kTHover, 0.0, 0.0, 0.0};

    std::cout << "Mode: trailer-follower MPC tracking (" << motion << ")\n"
              << "Running " << n_steps << " MPC steps (" << t_end << " s) ...\n\n"
              << std::fixed << std::setprecision(4);

    for (int step = 0; step < n_steps; ++step) {
        const double t = step * trailer_cfg.Ts;

        const auto horizon = (step == 0)
            ? horizon0
            : follower.computeGoalPoints(target.x, target.y, N, t, trailer_cfg.Ts);

        std::vector<argus_mpc::Reference> mpc_horizon(N);
        for (int j = 0; j < N; ++j) {
            mpc_horizon[j] = {horizon.PDx[j], horizon.PDy[j], kFovZRef};
        }

        const argus_mpc::TargetState tgt{
            target.x(t), target.y(t), kTargetZ,
            target.x.derivative(t), target.y.derivative(t), 0.0,
        };

        const argus_mpc::State x0_mpc{
            x[0], x[1], x[2], x[3],  x[4],  x[5],
            x[6], x[7], x[8], x[9], x[10], x[11],
        };
        const argus_mpc::Command cmd = mpc.step(x0_mpc, mpc_horizon, tgt);
        u = {cmd.thrust * kTHover, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate};

        csv << step << ',' << t;
        for (double v : x) csv << ',' << v;
        for (double v : u) csv << ',' << v;
        csv << ',' << mpc_horizon[0].x << ',' << mpc_horizon[0].y << ',' << mpc_horizon[0].z
            << ',' << tgt.x << ',' << tgt.y << ',' << tgt.z
            << ',' << cmd.solve_status << '\n';

        if (step % 40 == 0) {
            const double err = std::hypot(x[0] - mpc_horizon[0].x, x[1] - mpc_horizon[0].y);
            std::cout << "t=" << std::setw(6) << t << " s  "
                      << "drone=(" << std::setw(7) << x[0] << ", " << std::setw(7) << x[1] << ")  "
                      << "ref=("   << std::setw(6) << mpc_horizon[0].x << ", "
                                   << std::setw(6) << mpc_horizon[0].y << ")  "
                      << "xy_err=" << std::setw(6) << err << " m  "
                      << (cmd.solve_status == 0 ? "OK" : "WARN") << "\n";
        }

        plant.update(u, trailer_cfg.Ts);
        x = plant.state();
    }

    const argus_mpc::SolverTiming timing = mpc.solverTiming();
    std::cout << "\nSolver timing (acados_solve only, " << timing.count << " calls):\n"
              << "  min  = " << timing.min_ms  << " ms\n"
              << "  mean = " << timing.mean_ms << " ms\n"
              << "  max  = " << timing.max_ms  << " ms\n";

    std::cout << "\nLogged " << n_steps << " steps to " << out_path
              << "\nVisualise with: python3 scripts/animate_xy.py --csv-path " << out_path << "\n";
    return 0;
}
