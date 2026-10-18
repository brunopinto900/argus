// Standalone (no ROS2, no Gazebo) demo of the online/incremental mapping
// loop: two independent clocks -- mapping_sensor.update_rate_hz (60Hz) for
// the synthetic 360-degree sensor sweep + VoxelGrid insertion/ESDF
// recompute, and trailer.Ts (20Hz) for the follower's own MPC control
// update -- against the same obstacle-bypass scenario as
// follower_obstacle_demo.cpp.
//
// Unlike follower_obstacle_demo.cpp (instant, full access to the ground
// truth ESDF), the follower here only ever has its own incrementally-built
// `own_map`, starting all-unknown and resolved solely through simulated
// sensor sweeps -- see design_decisions.md for why the two clocks are
// decoupled and why unknown space is seeded as not-free.
//
// The mapping sensor rides the actual closed-loop drone (QuadrotorMpc +
// PlantDynamics, same wiring as examples/trailer_mpc_demo.cpp), not the
// follower's open-loop reference directly -- the follower's goal-point
// horizon is only ever the MPC's position reference, same as that demo.
// The MPC's collision-avoidance hinge cost is now live too: own_map is
// queried once per control tick at the drone's current position and that
// single (p0, d0, grad) linearization is reused for every horizon stage
// -- see design_decisions.md's "ESDF collision cost" and "d_safe" notes
// for why, and its caveats (d_safe is still an unvalidated placeholder).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <argus_esdf/VoxelGrid.hpp>
#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>
#include <argus_mpc/QuadrotorMpc.hpp>
#include <argus_perception/ObservableSlice.hpp>
#include <argus_perception/SyntheticLidar.hpp>

// Auto-generated from config/quadrotor.yaml -- re-run generate_mpc.py to update.
#include "argus_params.h"
#include "plant_dynamics.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using argus_esdf::VoxelGrid;
using argus_follower::SplineXY;
using argus_follower::TrailerFollower;
using argus_perception::computeObservableSlice;
using argus_perception::integrateLidarScan;
using argus_perception::verifyAgainstGroundTruth;
using PlantState = std::array<double, 12>;
using PlantInput = std::array<double, 4>;

namespace {

constexpr double kVoxelSize = 0.05;                        // m
// 300*0.05 = 15m x 15m footprint -- wider than the flight path (12m long)
// + 8m sensor range would ever fully sweep, so unexplored (kUnknown)
// space persists off to the sides for the whole run instead of the
// cumulative swept area converging on ~the entire grid partway through
// (what a tighter, e.g. 12x12m, footprint does here).
constexpr int kGridN = 300;
constexpr double kHalfExtent = kGridN * kVoxelSize / 2.0;

// Sensor pose z: the grid is a single z-layer (dims.z()==1) covering
// world z in [0, kVoxelSize) -- any pose/point outside that lands out of
// bounds. Mid-layer keeps raycast()/traceRay() comfortably inside it.
constexpr double kSensorZ = kVoxelSize / 2.0;

constexpr double kTs            = ARGUS_MPC_TS;
constexpr double kTHover        = ARGUS_T_HOVER;
// Static FOV-aware z reference, NOT the circle demo's ARGUS_HOVER_ALTITUDE
// -- that 1.5m comes from a different derivation tied to circular motion's
// own banking requirement (tan(theta) = r*omega^2/g), which doesn't apply
// to this scenario's threaded leader path. Here z_ref is instead derived
// from tan(theta_FOV) = z_ref / distance_to_target, solved for z_ref at a
// chosen target bearing angle -- see design_decisions.md's "Static
// FOV-aware z_ref" for the full derivation and why this is computed once
// from the configured standoff rather than recomputed online every
// control cycle (oscillation risk).
constexpr double kTargetBearingDeg = 30.0;  // headroom under the 40 deg FOV half-angle
// Ground-level tracked leader, distinct from the drone's z reference
// above it -- same reasoning as examples/trailer_mpc_demo.cpp.
constexpr double kTargetZ = 0.0;

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

// One obstacle from config/forest.yaml -- either a circle (center + radius)
// or a square (center + half_size). Matches the project's existing
// shape-query conventions (queryPlanar()'s own bilinear math, circle/square
// distance checks) rather than inventing a third representation.
struct ObstacleSpec
{
    enum class Shape { kCircle, kSquare } shape = Shape::kCircle;
    double center_x = 0.0, center_y = 0.0;
    double size = 0.2;  // radius (circle) or half_size (square)
};

// Full scenario: obstacles + the leader's path/pacing through them, both
// from one YAML file -- see config/forest.yaml / config/scenario_single_
// obstacle.yaml. Pass a different file as the demo's first CLI arg to
// switch scenarios without rebuilding.
struct ScenarioConfig
{
    std::vector<ObstacleSpec> obstacles;
    std::vector<std::pair<double, double>> leader_waypoints;
    argus_follower::SpeedKeyframes leader_speed_profile;
    double t_end = 14.0;
};

ScenarioConfig loadScenarioConfig(const std::string& yaml_path)
{
    const YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["obstacles"]) throw std::runtime_error(yaml_path + ": missing 'obstacles' section");
    if (!root["leader"])    throw std::runtime_error(yaml_path + ": missing 'leader' section");

    ScenarioConfig cfg;
    for (const YAML::Node& n : root["obstacles"]) {
        ObstacleSpec spec;
        const std::string shape = n["shape"].as<std::string>();
        spec.center_x = n["center"][0].as<double>();
        spec.center_y = n["center"][1].as<double>();
        if (shape == "circle") {
            spec.shape = ObstacleSpec::Shape::kCircle;
            spec.size  = n["radius"].as<double>();
        } else if (shape == "square") {
            spec.shape = ObstacleSpec::Shape::kSquare;
            spec.size  = n["half_size"].as<double>();
        } else {
            throw std::runtime_error(yaml_path + ": unknown obstacle shape '" + shape + "'");
        }
        cfg.obstacles.push_back(spec);
    }

    const YAML::Node leader = root["leader"];
    if (!leader["waypoints"])     throw std::runtime_error(yaml_path + ": leader: missing 'waypoints'");
    if (!leader["speed_profile"]) throw std::runtime_error(yaml_path + ": leader: missing 'speed_profile'");
    cfg.t_end = leader["t_end"].as<double>();
    for (const YAML::Node& wp : leader["waypoints"]) {
        cfg.leader_waypoints.emplace_back(wp[0].as<double>(), wp[1].as<double>());
    }
    for (const YAML::Node& kf : leader["speed_profile"]) {
        cfg.leader_speed_profile.emplace_back(kf[0].as<double>(), kf[1].as<double>());
    }
    return cfg;
}

bool insideObstacle(const ObstacleSpec& o, double x, double y)
{
    if (o.shape == ObstacleSpec::Shape::kCircle) {
        return std::hypot(x - o.center_x, y - o.center_y) <= o.size;
    }
    return std::abs(x - o.center_x) <= o.size && std::abs(y - o.center_y) <= o.size;
}

// Forest-like multi-obstacle ground truth, built from config/forest.yaml --
// this is the privileged ground truth; the follower itself never queries it
// directly, only through integrateLidarScan().
VoxelGrid buildGroundTruthGrid(const std::vector<ObstacleSpec>& obstacles)
{
    VoxelGrid grid(Eigen::Vector3d(-kHalfExtent, -kHalfExtent, 0.0),
                   Eigen::Vector3i(kGridN, kGridN, 1), kVoxelSize);
    for (int x = 0; x < kGridN; ++x) {
        for (int y = 0; y < kGridN; ++y) {
            const Eigen::Vector3d center = grid.voxelCenter({x, y, 0});
            const bool occupied = std::any_of(
                obstacles.begin(), obstacles.end(),
                [&](const ObstacleSpec& o) { return insideObstacle(o, center.x(), center.y()); });
            if (occupied) {
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

// Planar counterpart of VoxelGrid::query() -- same bilinear interpolant and
// the same analytic-derivative math, restricted to the z=0 layer. Needed
// because query() itself requires both z and z+1 neighbors to be in bounds
// for its trilinear interpolation, which is never true on this demo's
// single-layer (dims.z()==1) grids -- every query() call here would report
// valid=false. Scoped to this file rather than changing VoxelGrid's shared
// 3D query(), which other demos/tests rely on as-is.
struct PlanarQuery
{
    double distance = 0.0;
    double grad_x = 0.0, grad_y = 0.0;
    bool valid = false;
};

PlanarQuery queryPlanar(const VoxelGrid& grid, double x, double y)
{
    PlanarQuery result;
    const double rx = (x - grid.origin().x()) / grid.voxelSize() - 0.5;
    const double ry = (y - grid.origin().y()) / grid.voxelSize() - 0.5;
    const int ix0 = static_cast<int>(std::floor(rx));
    const int iy0 = static_cast<int>(std::floor(ry));
    const Eigen::Vector3i i00(ix0, iy0, 0), i11(ix0 + 1, iy0 + 1, 0);
    if (!grid.inBounds(i00) || !grid.inBounds(i11)) return result;  // valid=false

    const double wx = rx - ix0, wy = ry - iy0;
    const double c00 = grid.distanceAt({ix0, iy0, 0});
    const double c10 = grid.distanceAt({ix0 + 1, iy0, 0});
    const double c01 = grid.distanceAt({ix0, iy0 + 1, 0});
    const double c11 = grid.distanceAt({ix0 + 1, iy0 + 1, 0});

    const auto lerp = [](double a, double b, double t) { return a + (b - a) * t; };
    result.distance = lerp(lerp(c00, c10, wx), lerp(c01, c11, wx), wy);
    result.grad_x = lerp(c10 - c00, c11 - c01, wy) / grid.voxelSize();
    result.grad_y = lerp(c01 - c00, c11 - c10, wx) / grid.voxelSize();
    result.valid = true;
    return result;
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

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;
    fs::create_directories("logs");

    const std::string scenario_path = argc > 1 ? argv[1] : ARGUS_CONFIG_DIR "/forest.yaml";
    const ScenarioConfig scenario = loadScenarioConfig(scenario_path);
    std::cout << "Scenario: " << scenario_path << " (" << scenario.obstacles.size()
              << " obstacles)\n";
    VoxelGrid ground_truth = buildGroundTruthGrid(scenario.obstacles);
    ground_truth.computeEsdf();  // for the closest-approach report below only -- own_map
                                  // never queries ground_truth directly, see design_decisions.md
    VoxelGrid own_map(ground_truth.origin(), ground_truth.dims(), ground_truth.voxelSize());

    const YAML::Node config_root = YAML::LoadFile(ARGUS_CONFIG_DIR "/quadrotor.yaml");
    const TrailerConfig trailer_cfg = loadTrailerConfig(config_root, ARGUS_CONFIG_DIR "/quadrotor.yaml");
    // z_ref = distance_to_target * tan(kTargetBearingDeg) -- see the
    // kTargetBearingDeg comment above.
    const double kFovZRef = trailer_cfg.distance_to_target *
                             std::tan(kTargetBearingDeg * M_PI / 180.0);
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

    const SplineXY target = argus_follower::waypointTarget(
        scenario.leader_waypoints, scenario.leader_speed_profile, scenario.t_end, control_dt);
    TrailerFollower follower(trailer_cfg.distance_to_target, trailer_cfg.heading_gain);

    argus_mpc::QuadrotorMpc mpc;
    const int N = mpc.horizonLength();
    if (trailer_cfg.N != N) {
        throw std::runtime_error(
            "config/quadrotor.yaml: trailer.N (" + std::to_string(trailer_cfg.N) +
            ") must equal mpc.N (" + std::to_string(N) + ")");
    }
    if (std::abs(control_dt - kTs) > 1e-9) {
        throw std::runtime_error("config/quadrotor.yaml: trailer.Ts must equal mpc.Ts");
    }

    const int n_control_steps = static_cast<int>(std::round(scenario.t_end / control_dt));
    const std::string traj_path = "logs/online_mapping_trajectory.csv";
    std::ofstream traj(traj_path);
    traj << std::fixed << std::setprecision(6);
    // Same canonical layout as src/mpc_controller.cpp / trailer_mpc_demo.cpp
    // (step,t,full 12-state,u,xt/yt/zt=drone position reference,
    // fx/fy/fz=FOV target,solve_status) so scripts/plot_telemetry_dashboard.py
    // works unchanged on any of the three logs -- plus dist_obstacle, which
    // only this demo has (own_map's belief, not ground truth; see
    // design_decisions.md's "Ground truth vs. the drone's own map").
    // xt/yt/zt here = the follower's goal point; fx/fy/fz = the leader.
    traj << "step,t,"
            "x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,"
            "T,p_cmd,q_cmd,r_cmd,"
            "xt,yt,zt,"
            "fx,fy,fz,"
            "solve_status,"
            "dist_obstacle\n";

    // Staleness measurement -- a separate concern from the hinge cost
    // actually wired into step() above: this measures, per control tick,
    // how wrong a frozen tangent-plane approximation built at the step's
    // starting position (p0) would have been by the time the drone
    // actually reached its end-of-step position (p1). See
    // design_decisions.md's "Staleness error -- formula and which sign is
    // dangerous".
    const std::string staleness_path = "logs/staleness_log.csv";
    std::ofstream staleness(staleness_path);
    staleness << "t,p0_x,p0_y,p1_x,p1_y,step_dist,"
                 "d0,grad0_x,grad0_y,"
                 "true_dist_p1,grad1_x,grad1_y,"
                 "d_lin_p1,error\n";

    double min_follower_dist = std::numeric_limits<double>::infinity();
    double min_leader_dist   = std::numeric_limits<double>::infinity();
    double min_mpc_dist      = std::numeric_limits<double>::infinity();

    // First horizon call fixes the drone's initial position (and advances
    // the follower's heading state by one step, same as every subsequent
    // call) -- not recomputed again for step 0 below, see
    // examples/trailer_mpc_demo.cpp's identical pattern.
    const auto horizon0 = follower.computeGoalPoints(target.x, target.y, N, 0.0, control_dt);
    PlantState x = {horizon0.PDx[0], horizon0.PDy[0], kFovZRef,
                     0.0,            0.0,             0.0,
                     0.0,            0.0,             0.0,
                     0.0,            0.0,             0.0};
    PlantDynamics plant(x, kTHover, ARGUS_MASS, ARGUS_PLANT_WN_RP, ARGUS_PLANT_ZETA_RP,
                         ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);
    PlantInput u = {kTHover, 0.0, 0.0, 0.0};

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

        const auto horizon = (step == 0)
            ? horizon0
            : follower.computeGoalPoints(target.x, target.y, N, t, control_dt);
        const double ref_x = horizon.PDx[0];
        const double ref_y = horizon.PDy[0];

        std::vector<argus_mpc::Reference> mpc_horizon(N);
        for (int j = 0; j < N; ++j) {
            mpc_horizon[j] = {horizon.PDx[j], horizon.PDy[j], kFovZRef};
        }
        const argus_mpc::TargetState tgt{
            leader_x, leader_y, kTargetZ,
            target.x.derivative(t), target.y.derivative(t), 0.0,
        };
        const argus_mpc::State x0_mpc{
            x[0], x[1], x[2], x[3],  x[4],  x[5],
            x[6], x[7], x[8], x[9], x[10], x[11],
        };

        // p0: drone's position as of this solve, queried against own_map as
        // it stands right now -- the one ESDF evaluation this whole solve
        // gets (see design_decisions.md's "ESDF collision cost" section).
        // Simplest-possible per-stage choice: reuse the same (p0, d0, grad)
        // linearization for every horizon stage rather than querying at
        // each stage's predicted position -- QuadrotorMpc doesn't expose
        // the previous solve's predicted trajectory to build a better p0
        // from. An invalid query (own_map not yet resolved there) falls
        // back to the no-op default so an unresolved point never fabricates
        // a spurious obstacle cost.
        const Eigen::Vector2d p0(x[0], x[1]);
        const auto q0 = queryPlanar(own_map, p0.x(), p0.y());
        argus_mpc::ObstacleLinearization obs_stage;
        if (q0.valid) {
            obs_stage = {p0.x(), p0.y(), q0.distance, q0.grad_x, q0.grad_y};
        }
        const std::vector<argus_mpc::ObstacleLinearization> obstacle(N, obs_stage);

        const argus_mpc::Command cmd = mpc.step(x0_mpc, mpc_horizon, tgt, obstacle);
        u = {cmd.thrust * kTHover, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate};

        // Plant (and therefore the mapping sensor riding it) integrates at
        // the faster sensor rate, holding this control step's command
        // constant across all steps_per_control sub-ticks -- MPC solves at
        // control_dt, the inner loop/plant responds continuously, same
        // zero-order-hold architecture a real flight controller uses.
        for (int sub = 0; sub < steps_per_control; ++sub) {
            plant.update(u, sensor_dt);
            x = plant.state();

            const Eigen::Vector3d pose(x[0], x[1], kSensorZ);
            sensor_poses.push_back(pose);

            integrateLidarScan(ground_truth, own_map, pose, sensor_cfg.num_rays,
                                sensor_cfg.max_range_m);
            own_map.computeEsdf(/*treat_unknown_as_occupied=*/true);

            min_mpc_dist = std::min(min_mpc_dist, distanceAtPoint(ground_truth, x[0], x[1]));

            if (sensor_tick % snapshot_decimation == 0) {
                snapshots.push_back(captureSnapshot(own_map, sensor_tick * sensor_dt));
            }
            ++sensor_tick;
        }

        // p1: drone's actual position one control tick later, after own_map
        // has kept being updated underneath it (same map object, further
        // along). Comparing d_lin_p1 (built from the stale p0/d0/grad0) to
        // a fresh query at p1 is the staleness error itself; q1.distance
        // (own_map's own belief, not ground truth) also doubles as the
        // dist_obstacle column below, for the dashboard's closest-approach
        // panel.
        const Eigen::Vector2d p1(x[0], x[1]);
        const auto q1 = queryPlanar(own_map, p1.x(), p1.y());

        traj << step << ',' << t;
        for (double v : x) traj << ',' << v;
        for (double v : u) traj << ',' << v;
        traj << ',' << ref_x << ',' << ref_y << ',' << kFovZRef  // xt,yt,zt: follower goal point
             << ',' << leader_x << ',' << leader_y << ',' << kTargetZ  // fx,fy,fz: the leader
             << ',' << cmd.solve_status << ',';
        if (q1.valid) traj << q1.distance; else traj << "nan";
        traj << '\n';

        if (q0.valid && q1.valid) {
            const double step_dist = (p1 - p0).norm();
            const double d_lin_p1 = q0.distance + q0.grad_x * (p1.x() - p0.x()) +
                                     q0.grad_y * (p1.y() - p0.y());
            const double error = d_lin_p1 - q1.distance;
            staleness << t << ',' << p0.x() << ',' << p0.y() << ',' << p1.x() << ',' << p1.y() << ','
                      << step_dist << ',' << q0.distance << ',' << q0.grad_x << ','
                      << q0.grad_y << ',' << q1.distance << ',' << q1.grad_x << ','
                      << q1.grad_y << ',' << d_lin_p1 << ',' << error << '\n';
        }

        min_leader_dist   = std::min(min_leader_dist, distanceAtPoint(ground_truth, leader_x, leader_y));
        min_follower_dist = std::min(min_follower_dist, distanceAtPoint(ground_truth, ref_x, ref_y));
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
              << "Logged per-tick staleness measurements to " << staleness_path << "\n"
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
              << "  leader (target):     " << min_leader_dist << " m"
              << (min_leader_dist <= kVoxelSize ? "  <-- COLLISION" : "") << "\n"
              << "  follower (ref):      " << min_follower_dist << " m"
              << (min_follower_dist <= kVoxelSize ? "  <-- COLLISION" : "") << "\n"
              << "  drone (MPC actual):  " << min_mpc_dist << " m"
              << (min_mpc_dist <= kVoxelSize ? "  <-- COLLISION" : "")
              << "\n\nAnimate with: python3 argus_follower/scripts/plot_online_mapping.py\n";
    return 0;
}
