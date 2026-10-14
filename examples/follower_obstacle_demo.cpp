// Standalone (no ROS2, no Gazebo) obstacle-avoidance *test fixture*: a
// static square obstacle and its ESDF (argus_esdf::VoxelGrid), a target
// orbiting around it, and the (currently obstacle-unaware)
// argus_follower::TrailerFollower reference tracking that target -- logged
// for argus_follower/scripts/plot_follower_obstacle.py to animate.
//
// There is deliberately no obstacle avoidance here yet (see the top-level
// todo file) -- the point of this demo is to visualize whether/when the
// trailer reference's own corner-cutting behaviour alone would carry the
// follower's path through the obstacle, as a before/after baseline for
// once avoidance is added to the OCP.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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

constexpr double kOrbitRadius = 4.0;     // m -- target's orbit around the obstacle
constexpr double kAngularSpeed = 0.4;    // rad/s

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
// at the origin so the target's circular orbit is concentric with it.
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

    const double t_end = 2.0 * (2.0 * M_PI / kAngularSpeed);  // two full orbits
    const SplineXY target =
        argus_follower::circleTarget(0.0, 0.0, kOrbitRadius, kAngularSpeed, t_end, cfg.Ts);
    TrailerFollower follower(cfg.distance_to_target, cfg.heading_gain);

    const int n_steps = static_cast<int>(std::round(t_end / cfg.Ts));
    const std::string traj_path = "logs/follower_obstacle_trajectory.csv";
    std::ofstream traj(traj_path);
    traj << "t,leader_x,leader_y,follower_x,follower_y\n";

    for (int step = 0; step < n_steps; ++step) {
        const double t = step * cfg.Ts;
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const auto horizon = follower.computeGoalPoints(target.x, target.y, cfg.N, t, cfg.Ts);
        traj << t << ',' << leader_x << ',' << leader_y << ',' << horizon.PDx[0] << ','
             << horizon.PDy[0] << '\n';
    }

    std::cout << "Logged " << n_steps << " steps to " << traj_path
              << "\nAnimate with: python3 argus_follower/scripts/plot_follower_obstacle.py\n";
    return 0;
}
