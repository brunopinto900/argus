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

#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>

using argus_follower::SplineXY;
using argus_follower::TrailerFollower;

namespace {

constexpr int kHorizonN = 20;
constexpr double kDt = 0.05;
constexpr double kDistanceToTarget = 1.0;

SplineXY buildTarget(const std::string& motion, double t_end)
{
    if (motion == "line") {
        return argus_follower::straightLineTarget(0.0, 0.0, 1.0, t_end, kDt);
    }
    if (motion == "circle") {
        return argus_follower::circleTarget(0.0, 0.0, 5.0, 0.5, t_end, kDt);
    }
    if (motion == "zigzag") {
        return argus_follower::zigzagTarget(0.0, 0.0, 1.0, 1.0, 5.0, t_end, kDt);
    }
    throw std::invalid_argument("unknown motion '" + motion + "' (expected line|circle|zigzag)");
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string motion = argc > 1 ? argv[1] : "circle";
    const double t_end = argc > 2 ? std::stod(argv[2]) : 20.0;

    SplineXY target = buildTarget(motion, t_end);
    TrailerFollower follower(kDistanceToTarget);

    const int n_steps = static_cast<int>(std::round(t_end / kDt));

    namespace fs = std::filesystem;
    fs::create_directories("logs");
    const std::string out_path = "logs/follower_trajectory.csv";
    std::ofstream csv(out_path);
    csv << "t,leader_x,leader_y,leader_vx,leader_vy,"
           "follower_x,follower_y,follower_vx,follower_vy\n";

    for (int step = 0; step < n_steps; ++step) {
        const double t = step * kDt;

        // Leader's own (position, velocity) at the current instant -- not
        // the horizon's predicted future taus, which computeGoalPoints()
        // below also evaluates internally.
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const double leader_vx = target.x.derivative(t);
        const double leader_vy = target.y.derivative(t);

        // Index 0 of the N-step horizon is the follower's goal for this
        // same instant -- the one real step this call actually commits to.
        const auto horizon = follower.computeGoalPoints(target.x, target.y, kHorizonN, t, kDt);

        csv << t << ',' << leader_x << ',' << leader_y << ',' << leader_vx << ',' << leader_vy
            << ',' << horizon.PDx[0] << ',' << horizon.PDy[0] << ',' << horizon.VFx[0] << ','
            << horizon.VFy[0] << '\n';
    }

    std::cout << "Logged " << n_steps << " steps (motion=" << motion << ") to " << out_path
              << "\nPlot with: python3 argus_follower/scripts/plot_follower.py\n";
    return 0;
}
