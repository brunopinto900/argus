// Standalone (no ROS2, no Gazebo) demo: a static square obstacle and its
// ESDF (argus_esdf::VoxelGrid), a target following
// straightLineWithBypassTarget, and the (currently obstacle-unaware)
// argus_follower::TrailerFollower reference tracking that target -- logged
// for argus_follower/scripts/plot_follower_obstacle.py to animate.
//
// There is no obstacle avoidance here yet (see the top-level todo file) --
// this is a before/after baseline for once avoidance is added to the OCP.
// See the printed "closest approach" report at the end of a run.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include <argus_esdf/VoxelGrid.hpp>
#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>

using argus_esdf::VoxelGrid;
using argus_follower::SplineXY;
using argus_follower::TrailerFollower;

namespace {

constexpr double kVoxelSize = 0.05;                        // m
constexpr int kGridN = 240;                                // 240*0.05 = 12m x 12m footprint
constexpr double kHalfExtent = kGridN * kVoxelSize / 2.0;
constexpr double kObstacleHalfSize = 0.8;                  // 1.6m x 1.6m square, centered at origin

// Target path: straightLineWithBypassTarget along x=0, bypass centered on
// the obstacle (y=0). amplitude > kObstacleHalfSize so the target itself
// clears the obstacle.
constexpr double kTargetSpeed = 1.0;         // m/s
constexpr double kYStart = -6.0;             // m
constexpr double kBypassAmplitude = 1.5;     // m
constexpr double kBypassHalfWidth = 2.5;     // m -- half-width of the bypass window
constexpr double kTEnd = 14.0;               // s

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

// Same square-obstacle construction as examples/esdf_2d_demo.cpp, centered
// at the origin -- the target's straight approach runs directly at it.
VoxelGrid buildObstacleGrid()
{
    VoxelGrid grid(Eigen::Vector3d(-kHalfExtent, -kHalfExtent, 0.0),
                   Eigen::Vector3i(kGridN, kGridN, 1), kVoxelSize);
    for (int x = 0; x < kGridN; ++x) {
        for (int y = 0; y < kGridN; ++y) {
            const Eigen::Vector3d center = grid.voxelCenter({x, y, 0});
            if (std::abs(center.x()) <= kObstacleHalfSize &&
                std::abs(center.y()) <= kObstacleHalfSize) {
                const Eigen::Vector3d sensor = center + Eigen::Vector3d(0.0, 0.0, 10.0 * kVoxelSize);
                grid.insertPointCloud(sensor, {center});
            }
        }
    }
    grid.computeEsdf();
    return grid;
}

// Flat CSV: a one-line header (nx, ny, voxel_size, origin_x, origin_y),
// then ny rows of nx comma-separated distances -- y ascending (row 0 is
// the grid's own y=0 row), so the plotting script can imshow() it directly
// with origin="lower" and no further reindexing.
void writeEsdfCsv(const VoxelGrid& grid, const std::string& path)
{
    std::ofstream out(path);
    out << grid.dims().x() << ',' << grid.dims().y() << ',' << grid.voxelSize() << ','
        << grid.origin().x() << ',' << grid.origin().y() << '\n';
    for (int y = 0; y < grid.dims().y(); ++y) {
        for (int x = 0; x < grid.dims().x(); ++x) {
            if (x > 0) out << ',';
            out << grid.distanceAt({x, y, 0});
        }
        out << '\n';
    }
}

// Nearest-voxel (no interpolation -- the grid is dims.z()==1, so query()'s
// trilinear interpolation is always invalid on it, see VoxelGrid.hpp)
// distance at a world-frame (x, y) point.
double distanceAtPoint(const VoxelGrid& grid, double x, double y)
{
    const Eigen::Vector3i idx(
        static_cast<int>(std::floor((x - grid.origin().x()) / grid.voxelSize())),
        static_cast<int>(std::floor((y - grid.origin().y()) / grid.voxelSize())), 0);
    if (!grid.inBounds(idx)) return std::numeric_limits<double>::infinity();
    return grid.distanceAt(idx);
}

}  // namespace

int main()
{
    namespace fs = std::filesystem;
    fs::create_directories("logs");

    const VoxelGrid grid = buildObstacleGrid();
    const std::string esdf_path = "logs/follower_obstacle_esdf.csv";
    writeEsdfCsv(grid, esdf_path);
    std::cout << "Wrote " << esdf_path << " (" << grid.dims().x() << "x" << grid.dims().y()
              << ")\n";

    const TrailerConfig cfg = loadTrailerConfig(ARGUS_CONFIG_DIR "/quadrotor.yaml");

    const SplineXY target = argus_follower::straightLineWithBypassTarget(
        /*x0=*/0.0, kYStart, kTargetSpeed, kBypassAmplitude, /*bypass_y_center=*/0.0,
        kBypassHalfWidth, kTEnd, cfg.Ts);
    TrailerFollower follower(cfg.distance_to_target, cfg.heading_gain);

    const int n_steps = static_cast<int>(std::round(kTEnd / cfg.Ts));
    const std::string traj_path = "logs/follower_obstacle_trajectory.csv";
    std::ofstream traj(traj_path);
    traj << "t,leader_x,leader_y,follower_x,follower_y\n";

    double min_follower_dist = std::numeric_limits<double>::infinity();
    double min_leader_dist   = std::numeric_limits<double>::infinity();

    for (int step = 0; step < n_steps; ++step) {
        const double t = step * cfg.Ts;
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const auto horizon = follower.computeGoalPoints(target.x, target.y, cfg.N, t, cfg.Ts);
        const double follower_x = horizon.PDx[0];
        const double follower_y = horizon.PDy[0];

        traj << t << ',' << leader_x << ',' << leader_y << ',' << follower_x << ','
             << follower_y << '\n';

        min_leader_dist   = std::min(min_leader_dist, distanceAtPoint(grid, leader_x, leader_y));
        min_follower_dist = std::min(min_follower_dist, distanceAtPoint(grid, follower_x, follower_y));
    }

    std::cout << "Logged " << n_steps << " steps to " << traj_path << "\n\n"
              << "Closest approach to the obstacle:\n"
              << "  leader (target): " << min_leader_dist << " m"
              << (min_leader_dist <= kVoxelSize ? "  <-- COLLISION" : "") << "\n"
              << "  follower (ref):  " << min_follower_dist << " m"
              << (min_follower_dist <= kVoxelSize ? "  <-- COLLISION" : "")
              << "\n\nAnimate with: python3 argus_follower/scripts/plot_follower_obstacle.py\n";
    return 0;
}
