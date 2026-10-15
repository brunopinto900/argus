// Standalone (no ROS2, no Gazebo) demo of the online/incremental mapping
// loop: two independent clocks -- mapping_sensor.update_rate_hz (60Hz) for
// the synthetic 360-degree sensor sweep + VoxelGrid insertion/ESDF
// recompute, and trailer.Ts (20Hz) for the follower's own control update --
// against the same obstacle-bypass scenario as follower_obstacle_demo.cpp.
//
// Unlike follower_obstacle_demo.cpp (instant, full access to the ground
// truth ESDF), the follower here only ever has its own incrementally-built
// `own_map`, starting all-unknown and resolved solely through simulated
// sensor sweeps -- see design_decisions.md for why the two clocks are
// decoupled and why unknown space is seeded as not-free.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <argus_esdf/VoxelGrid.hpp>
#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>
#include <argus_perception/ObservableSlice.hpp>
#include <argus_perception/SyntheticLidar.hpp>

using argus_esdf::VoxelGrid;
using argus_follower::SplineXY;
using argus_follower::TrailerFollower;
using argus_perception::computeObservableSlice;
using argus_perception::integrateLidarScan;
using argus_perception::verifyAgainstGroundTruth;

namespace {

constexpr double kVoxelSize = 0.05;                        // m
// 300*0.05 = 15m x 15m footprint -- wider than the flight path (12m long)
// + 8m sensor range would ever fully sweep, so unexplored (kUnknown)
// space persists off to the sides for the whole run instead of the
// cumulative swept area converging on ~the entire grid partway through
// (what a tighter, e.g. 12x12m, footprint does here).
constexpr int kGridN = 300;
constexpr double kHalfExtent = kGridN * kVoxelSize / 2.0;
constexpr double kObstacleHalfSize = 0.8;                  // 1.6m x 1.6m square, centered at origin

// Sensor pose z: the grid is a single z-layer (dims.z()==1) covering
// world z in [0, kVoxelSize) -- any pose/point outside that lands out of
// bounds. Mid-layer keeps raycast()/traceRay() comfortably inside it.
constexpr double kSensorZ = kVoxelSize / 2.0;

// Same scenario as follower_obstacle_demo.cpp: a straight approach with a
// bypass large enough for the leader to clear the obstacle but tight
// enough that the (obstacle-unaware) follower's hitch-lag does not.
constexpr double kTargetSpeed = 1.0;         // m/s
constexpr double kYStart = -6.0;             // m
constexpr double kBypassAmplitude = 1.5;     // m
constexpr double kBypassHalfWidth = 2.5;     // m
constexpr double kTEnd = 14.0;               // s

// Snapshot logging rate for the own_map animation -- decoupled from both
// the 60Hz sensor/map update and the 20Hz control loop (see
// design_decisions.md's "Logging / visualization": logging every sensor
// tick over a multi-second run is hundreds of MB, 10Hz is already smooth
// to animate).
constexpr double kSnapshotRateHz = 10.0;

struct TrailerConfig
{
    double distance_to_target = 1.0;
    double heading_gain       = 1.0;
    int    N                  = 20;
    double Ts                 = 0.05;
};

TrailerConfig loadTrailerConfig(const YAML::Node& root, const std::string& yaml_path)
{
    if (!root["trailer"]) throw std::runtime_error(yaml_path + ": missing 'trailer' section");
    const YAML::Node n = root["trailer"];

    TrailerConfig cfg;
    if (n["distance_to_target"]) cfg.distance_to_target = n["distance_to_target"].as<double>();
    if (n["heading_gain"])       cfg.heading_gain        = n["heading_gain"].as<double>();
    if (n["N"])                  cfg.N                   = n["N"].as<int>();
    if (n["Ts"])                 cfg.Ts                  = n["Ts"].as<double>();
    return cfg;
}

struct MappingSensorConfig
{
    double max_range_m    = 8.0;
    int    num_rays       = 1006;
    double update_rate_hz = 60.0;
};

MappingSensorConfig loadMappingSensorConfig(const YAML::Node& root, const std::string& yaml_path)
{
    if (!root["mapping_sensor"]) {
        throw std::runtime_error(yaml_path + ": missing 'mapping_sensor' section");
    }
    const YAML::Node n = root["mapping_sensor"];

    MappingSensorConfig cfg;
    if (n["max_range_m"])    cfg.max_range_m    = n["max_range_m"].as<double>();
    if (n["num_rays"])       cfg.num_rays       = n["num_rays"].as<int>();
    if (n["update_rate_hz"]) cfg.update_rate_hz = n["update_rate_hz"].as<double>();
    return cfg;
}

// Same square-obstacle construction as follower_obstacle_demo.cpp -- this
// is the privileged ground truth; the follower itself never queries it
// directly, only through simulateLidarScan().
VoxelGrid buildGroundTruthGrid()
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
    return grid;
}

double distanceAtPoint(const VoxelGrid& grid, double x, double y)
{
    const Eigen::Vector3i idx(
        static_cast<int>(std::floor((x - grid.origin().x()) / grid.voxelSize())),
        static_cast<int>(std::floor((y - grid.origin().y()) / grid.voxelSize())), 0);
    if (!grid.inBounds(idx)) return std::numeric_limits<double>::infinity();
    return grid.distanceAt(idx);
}

// Fraction of voxels own_map has actually resolved (kFree or kOccupied) --
// a quick sanity signal that the sensor loop is making progress, not the
// full observable-slice coverage report (that belongs to step 9's console
// report, not this one).
double resolvedFraction(const VoxelGrid& grid)
{
    int resolved = 0;
    const int total = grid.dims().x() * grid.dims().y() * grid.dims().z();
    for (int x = 0; x < grid.dims().x(); ++x) {
        for (int y = 0; y < grid.dims().y(); ++y) {
            if (grid.stateAt({x, y, 0}) != argus_esdf::VoxelState::kUnknown) ++resolved;
        }
    }
    return static_cast<double>(resolved) / total;
}

// One frame of own_map's (distance, state) per voxel, row-major with y
// outer / x inner (matches the existing CSV convention elsewhere in this
// repo) -- captured by value since own_map keeps mutating after this.
// float32 (not double) for distance and a 1-byte state are enough for
// visualization and keep the log an order of magnitude smaller than a
// per-tick CSV dump would be.
struct OwnMapSnapshot
{
    double t = 0.0;
    std::vector<float> distance;
    std::vector<argus_esdf::VoxelState> state;
};

OwnMapSnapshot captureSnapshot(const VoxelGrid& grid, double t)
{
    OwnMapSnapshot snap;
    snap.t = t;
    const int nx = grid.dims().x();
    const int ny = grid.dims().y();
    snap.distance.reserve(static_cast<std::size_t>(nx) * ny);
    snap.state.reserve(static_cast<std::size_t>(nx) * ny);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const Eigen::Vector3i idx(x, y, 0);
            snap.distance.push_back(static_cast<float>(grid.distanceAt(idx)));
            snap.state.push_back(grid.stateAt(idx));
        }
    }
    return snap;
}

// Binary layout (little-endian, this machine's native order -- no cross-
// platform exchange needed, this is a local demo log):
//   int32 nx, int32 ny, float64 voxel_size, float64 origin_x, float64 origin_y,
//   float64 sensor_max_range
//   nx*ny x uint8 ground_truth_state   (y outer, x inner -- written once,
//     ground truth is static; purely for visualization/comparison, the
//     own_map pipeline itself never reads it back)
//   int32 num_snapshots
//   then num_snapshots blocks of:
//     float64 t
//     nx*ny x { float32 distance, uint8 state }   (y outer, x inner)
void writeSnapshotLog(const std::string& path, const VoxelGrid& grid,
                       const VoxelGrid& ground_truth, double sensor_max_range,
                       const std::vector<OwnMapSnapshot>& snapshots)
{
    std::ofstream out(path, std::ios::binary);
    const std::int32_t nx = grid.dims().x();
    const std::int32_t ny = grid.dims().y();
    const double voxel_size = grid.voxelSize();
    const double origin_x = grid.origin().x();
    const double origin_y = grid.origin().y();
    const std::int32_t num_snapshots = static_cast<std::int32_t>(snapshots.size());

    out.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
    out.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
    out.write(reinterpret_cast<const char*>(&voxel_size), sizeof(voxel_size));
    out.write(reinterpret_cast<const char*>(&origin_x), sizeof(origin_x));
    out.write(reinterpret_cast<const char*>(&origin_y), sizeof(origin_y));
    out.write(reinterpret_cast<const char*>(&sensor_max_range), sizeof(sensor_max_range));

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const auto s = static_cast<std::uint8_t>(ground_truth.stateAt({x, y, 0}));
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
    }

    out.write(reinterpret_cast<const char*>(&num_snapshots), sizeof(num_snapshots));

    for (const auto& snap : snapshots) {
        out.write(reinterpret_cast<const char*>(&snap.t), sizeof(snap.t));
        for (std::size_t i = 0; i < snap.distance.size(); ++i) {
            const auto s = static_cast<std::uint8_t>(snap.state[i]);
            out.write(reinterpret_cast<const char*>(&snap.distance[i]), sizeof(float));
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
    }
}

}  // namespace

int main()
{
    namespace fs = std::filesystem;
    fs::create_directories("logs");

    VoxelGrid ground_truth = buildGroundTruthGrid();
    ground_truth.computeEsdf();  // for the closest-approach report below only -- own_map
                                  // never queries ground_truth directly, see design_decisions.md
    VoxelGrid own_map(ground_truth.origin(), ground_truth.dims(), ground_truth.voxelSize());

    const YAML::Node config_root = YAML::LoadFile(ARGUS_CONFIG_DIR "/quadrotor.yaml");
    const TrailerConfig trailer_cfg = loadTrailerConfig(config_root, ARGUS_CONFIG_DIR "/quadrotor.yaml");
    const MappingSensorConfig sensor_cfg =
        loadMappingSensorConfig(config_root, ARGUS_CONFIG_DIR "/quadrotor.yaml");

    const double sensor_dt = 1.0 / sensor_cfg.update_rate_hz;
    const double control_dt = trailer_cfg.Ts;
    const int steps_per_control = static_cast<int>(std::round(control_dt / sensor_dt));
    if (steps_per_control < 1 ||
        std::abs(control_dt - steps_per_control * sensor_dt) > 1e-9) {
        throw std::runtime_error(
            "mapping_sensor.update_rate_hz and trailer.Ts must divide evenly "
            "(control_dt must be an integer multiple of sensor_dt)");
    }

    const SplineXY target = argus_follower::straightLineWithBypassTarget(
        /*x0=*/0.0, kYStart, kTargetSpeed, kBypassAmplitude, /*bypass_y_center=*/0.0,
        kBypassHalfWidth, kTEnd, control_dt);
    TrailerFollower follower(trailer_cfg.distance_to_target, trailer_cfg.heading_gain);

    const int n_control_steps = static_cast<int>(std::round(kTEnd / control_dt));
    const std::string traj_path = "logs/online_mapping_trajectory.csv";
    std::ofstream traj(traj_path);
    traj << "t,leader_x,leader_y,follower_x,follower_y\n";

    double min_follower_dist = std::numeric_limits<double>::infinity();
    double min_leader_dist   = std::numeric_limits<double>::infinity();

    // Drone pose for the mapping sensor: held at the follower's previous
    // control-step position and linearly interpolated across the
    // steps_per_control sensor sub-ticks up to its new position -- the
    // sensor clock is faster than the control clock, so it samples the
    // platform's motion between control updates rather than re-observing
    // a static point steps_per_control times over. No previous position
    // exists yet on the first control step, so it holds at that step's own
    // position instead of interpolating.
    double prev_x = 0.0;
    double prev_y = 0.0;

    const int snapshot_decimation =
        static_cast<int>(std::round(sensor_cfg.update_rate_hz / kSnapshotRateHz));
    int sensor_tick = 0;
    std::vector<OwnMapSnapshot> snapshots;
    std::vector<Eigen::Vector3d> sensor_poses;  // every pose actually swept, for the
                                                 // observable-slice report below

    for (int step = 0; step < n_control_steps; ++step) {
        const double t = step * control_dt;
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const auto horizon = follower.computeGoalPoints(target.x, target.y, trailer_cfg.N, t, control_dt);
        const double cur_x = horizon.PDx[0];
        const double cur_y = horizon.PDy[0];

        if (step == 0) {
            prev_x = cur_x;
            prev_y = cur_y;
        }

        for (int sub = 0; sub < steps_per_control; ++sub) {
            const double frac = static_cast<double>(sub) / steps_per_control;
            const double pose_x = prev_x + frac * (cur_x - prev_x);
            const double pose_y = prev_y + frac * (cur_y - prev_y);
            const Eigen::Vector3d pose(pose_x, pose_y, kSensorZ);
            sensor_poses.push_back(pose);

            integrateLidarScan(ground_truth, own_map, pose, sensor_cfg.num_rays,
                                sensor_cfg.max_range_m);
            own_map.computeEsdf(/*treat_unknown_as_occupied=*/true);

            if (sensor_tick % snapshot_decimation == 0) {
                snapshots.push_back(captureSnapshot(own_map, sensor_tick * sensor_dt));
            }
            ++sensor_tick;
        }

        traj << t << ',' << leader_x << ',' << leader_y << ',' << cur_x << ',' << cur_y << '\n';

        min_leader_dist   = std::min(min_leader_dist, distanceAtPoint(ground_truth, leader_x, leader_y));
        min_follower_dist = std::min(min_follower_dist, distanceAtPoint(ground_truth, cur_x, cur_y));

        prev_x = cur_x;
        prev_y = cur_y;
    }

    const std::string snapshot_path = "logs/online_mapping_esdf_snapshots.bin";
    writeSnapshotLog(snapshot_path, own_map, ground_truth, sensor_cfg.max_range_m, snapshots);

    // Independent re-raycast against ground truth from every pose actually
    // swept, deliberately not reusing the per-tick insertPointCloud() calls
    // above -- catches real pipeline bugs (pose/frame mismatches, missed
    // ticks), not just re-testing raycast()/traceRay() themselves. See
    // design_decisions.md's "Testing / verification strategy".
    const auto observable_slice = computeObservableSlice(ground_truth, sensor_poses,
                                                           sensor_cfg.num_rays, sensor_cfg.max_range_m);
    const auto verification = verifyAgainstGroundTruth(own_map, ground_truth, observable_slice);
    const int total_voxels = own_map.dims().x() * own_map.dims().y() * own_map.dims().z();

    std::cout << "Ran " << n_control_steps << " control steps (" << control_dt << "s) x "
              << steps_per_control << " sensor sub-ticks (" << sensor_dt << "s) each.\n"
              << "Logged trajectory to " << traj_path << "\n"
              << "Logged " << snapshots.size() << " own_map snapshots (" << kSnapshotRateHz
              << "Hz) to " << snapshot_path << "\n\n"
              << "own_map resolved fraction: " << resolvedFraction(own_map) * 100.0 << "%\n\n"
              << "Observable-slice verification (own_map vs. ground truth, voxels the\n"
              << "sensor actually had a chance to resolve):\n"
              << "  observable voxels: " << verification.observable_count << " / " << total_voxels
              << " (" << (100.0 * verification.observable_count / total_voxels) << "% of the grid)\n"
              << "  matched:            " << verification.matched_count << "\n"
              << "  mismatched:         " << verification.mismatched_count
              << (verification.allMatch() ? "" : "  <-- MISMATCH") << "\n\n"
              << "Closest approach to the obstacle (ground truth):\n"
              << "  leader (target): " << min_leader_dist << " m"
              << (min_leader_dist <= kVoxelSize ? "  <-- COLLISION" : "") << "\n"
              << "  follower (ref):  " << min_follower_dist << " m"
              << (min_follower_dist <= kVoxelSize ? "  <-- COLLISION" : "")
              << "\n\nAnimate with: python3 argus_follower/scripts/plot_online_mapping.py\n";
    return 0;
}
