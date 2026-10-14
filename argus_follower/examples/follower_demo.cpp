// Standalone (no ROS2, no Gazebo) run of TrailerFollower against a
// synthetic target trajectory, logged to CSV for scripts/plot_follower.py
// to turn into the XY-path and velocity-overlap plots -- same "C++ logs a
// CSV, Python plots it" split as src/mpc_controller.cpp + animate_xy.py.
//
// Usage: ./build/follower_demo [line|circle|zigzag] [t_end_seconds]
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>

using argus_follower::SplineXY;
using argus_follower::TrailerFollower;

namespace {

// argus_follower itself has no yaml-cpp dependency (kept pure, like
// quadrotor_mpc/argus_esdf) -- this loader is private to this demo, same
// role as src/TargetSimulator.hpp's load_target_config() for argus_mpc.
struct TrailerConfig
{
    double distance_to_target = 1.0;
    double heading_gain       = 1.0;
    int    N                  = 20;
    double Ts                 = 0.05;
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
    const std::string motion = argc > 1 ? argv[1] : "circle";
    const double t_end = argc > 2 ? std::stod(argv[2]) : 20.0;

    const TrailerConfig cfg = loadTrailerConfig(ARGUS_CONFIG_DIR "/quadrotor.yaml");

    SplineXY target = buildTarget(motion, t_end, cfg.Ts);
    TrailerFollower follower(cfg.distance_to_target, cfg.heading_gain);

    const int n_steps = static_cast<int>(std::round(t_end / cfg.Ts));

    namespace fs = std::filesystem;
    fs::create_directories("logs");
    const std::string out_path = "logs/follower_trajectory.csv";
    std::ofstream csv(out_path);
    csv << "t,leader_x,leader_y,leader_vx,leader_vy,"
           "follower_x,follower_y,follower_vx,follower_vy\n";

    for (int step = 0; step < n_steps; ++step) {
        const double t = step * cfg.Ts;

        // Leader's own (position, velocity) at the current instant -- not
        // the horizon's predicted future taus, which computeGoalPoints()
        // below also evaluates internally.
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const double leader_vx = target.x.derivative(t);
        const double leader_vy = target.y.derivative(t);

        // Index 0 of the N-step horizon is the follower's goal for this
        // same instant -- the one real step this call actually commits to.
        const auto horizon = follower.computeGoalPoints(target.x, target.y, cfg.N, t, cfg.Ts);

        csv << t << ',' << leader_x << ',' << leader_y << ',' << leader_vx << ',' << leader_vy
            << ',' << horizon.PDx[0] << ',' << horizon.PDy[0] << ',' << horizon.VFx[0] << ','
            << horizon.VFy[0] << '\n';
    }

    std::cout << "Logged " << n_steps << " steps (motion=" << motion << ") to " << out_path
              << "\nPlot with: python3 argus_follower/scripts/plot_follower.py\n";
    return 0;
}
