// Standalone (no ROS2, no Gazebo) validation of the full depth-camera ->
// point-cloud -> ESDF pipeline used by argus_mapping_node, against a
// synthetic depth image of the exact same box used in the SITL world
// (argus_box_world.sdf's obstacle_box_0: pose (0.3, 0.8, 0.75), size
// 0.5 x 0.5 x 1.5). Companion to examples/esdf_2d_demo.cpp, which only
// exercises VoxelGrid's own distance transform; this one additionally
// exercises the reprojection formula and optical->body->world chain from
// argus_mapping_node.cpp's cloudCallback() (duplicated here, not shared —
// same reasoning as esdf_2d_demo.cpp's distanceToColor(): small enough
// that a shared header isn't worth it for a demo/test's benefit). That
// chain is exactly the code that was wrong before the depth-reprojection
// fix (see the top-level todo file's "Depth camera point cloud is
// fundamentally broken" entry) — validating it used to mean a full
// SITL launch (Gazebo + PX4 + ROS2, 30s-1min+ per check just to inspect
// one topic); this gets the same answer (does the pipeline place the box
// where it actually is?) in well under a second, with no simulator.
//
// How the synthetic input is generated: rather than trusting a real
// (possibly buggy) point per pixel, ground-truth depth is computed via
// analytic ray-AABB intersection against the box, for every pixel of a
// simulated camera at a known world pose. To make this an actual
// regression test for the bug that was found — not just "does
// insertPointCloud() work" — that ground-truth depth is written into the
// synthetic message with the same sign flip the real OakD-Lite sensor has
// (z negative instead of positive-forward). The pipeline under test then
// has to recover the correct point the same way cloudCallback() does:
// sign-correct z, and rebuild X/Y from depth + pixel row/col + intrinsics,
// never trusting a precomputed lateral field.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include <argus_esdf/VoxelGrid.hpp>

using argus_esdf::VoxelGrid;

namespace {

// ── The box: exactly argus_box_world.sdf's obstacle_box_0 ──────────────────
const Eigen::Vector3d kBoxCenter(0.3, 0.8, 0.75);
const Eigen::Vector3d kBoxHalfSize(0.25, 0.25, 0.75);  // size 0.5 x 0.5 x 1.5

// ── Camera: a plausible hover-and-look-at-the-box pose, not a replay of any
// specific SITL flight. Facing -X (yaw=180deg, zero pitch/roll) puts the
// box's near (+X) face directly ahead and centered in frame, which keeps
// the geometry easy to hand-verify against the numbers checked at the end.
const Eigen::Vector3d kSensorOriginWorld(2.0, 0.8, 0.75);

Eigen::Matrix3d bodyToWorldRotation()  // yaw = 180 deg, zero pitch/roll
{
    Eigen::Matrix3d r;
    r << -1.0, 0.0, 0.0,
          0.0, -1.0, 0.0,
          0.0, 0.0, 1.0;
    return r;
}
const Eigen::Matrix3d kBodyToWorld = bodyToWorldRotation();

// Same optical (X-right, Y-down, Z-forward) -> body FLU rotation as
// argus_mapping_node.cpp's kOpticalToBody: body_x = optical_z,
// body_y = -optical_x, body_z = -optical_y.
Eigen::Matrix3d opticalToBodyRotation()
{
    Eigen::Matrix3d r;
    r << 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
         0.0, -1.0, 0.0;
    return r;
}
const Eigen::Matrix3d kOpticalToWorld = kBodyToWorld * opticalToBodyRotation();

// ── Camera intrinsics: OakD-Lite's real depth_camera sensor (PX4-Autopilot
// Tools/simulation/gz/models/OakD-Lite/model.sdf: horizontal_fov=1.274,
// 640x480). fy=fx assumes square pixels (one focal length, FOV differs
// between axes only because width != height) — the same assumption
// argus_mapping_node gets for free from /camera_info at runtime; there's
// no CameraInfo topic here, so it's derived directly from the same SDF
// values instead.
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr double kHorizontalFov = 1.274;
const double kFx = kWidth / (2.0 * std::tan(kHorizontalFov / 2.0));
const double kFy = kFx;
const double kCx = kWidth / 2.0;
const double kCy = kHeight / 2.0;

constexpr double kVoxelSize = 0.1;
const Eigen::Vector3d kGridOrigin(-3.0, -3.0, 0.0);
const Eigen::Vector3i kGridDims(60, 60, 25);
constexpr double kColorMaxDistance = 1.5;  // matches esdf_viz_max_distance's default
constexpr int kPixelsPerVoxel = 4;

struct Rgb { uint8_t r, g, b; };

// Same colormap as argus_mapping_node's distanceToColor(), minus alpha —
// this demo writes flat (opaque) PPMs.
Rgb distanceToColor(double distance, double max_distance)
{
    const double t = std::clamp(distance / max_distance, 0.0, 1.0);
    const double h = t * 240.0;
    const double x = 1.0 - std::abs(std::fmod(h / 60.0, 2.0) - 1.0);
    double r = 0.0, g = 0.0, b = 0.0;
    if (h < 60.0)       { r = 1.0; g = x;   b = 0.0; }
    else if (h < 120.0) { r = x;   g = 1.0; b = 0.0; }
    else if (h < 180.0) { r = 0.0; g = 1.0; b = x;   }
    else                { r = 0.0; g = x;   b = 1.0; }
    return {static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(b * 255)};
}

// Ray-AABB slab intersection: smallest t >= 0 such that origin + t*dir
// enters [box_min, box_max]. dir need not be normalized — every caller
// here passes a dir whose optical-frame Z component is exactly 1, so the
// returned t is directly the depth D the pinhole formula expects
// (point_optical = D * dir_optical), not a normalized ray length.
std::optional<double> rayBoxIntersection(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir,
                                          const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max)
{
    double t_min = 0.0;
    double t_max = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(dir[axis]) < 1e-12) {
            if (origin[axis] < box_min[axis] || origin[axis] > box_max[axis]) return std::nullopt;
            continue;
        }
        double t1 = (box_min[axis] - origin[axis]) / dir[axis];
        double t2 = (box_max[axis] - origin[axis]) / dir[axis];
        if (t1 > t2) std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max) return std::nullopt;
    }
    return t_min;
}

// Builds one synthetic depth frame (one z value per pixel, no-return
// pixels skipped) via analytic ray-box intersection, then feeds it through
// the same reprojection cloudCallback() uses post-fix: sign-correct z,
// rebuild X/Y from depth + pixel row/col + intrinsics — never touching a
// precomputed lateral field.
std::vector<Eigen::Vector3d> synthesizeAndReprojectPoints()
{
    const Eigen::Vector3d box_min = kBoxCenter - kBoxHalfSize;
    const Eigen::Vector3d box_max = kBoxCenter + kBoxHalfSize;

    std::vector<Eigen::Vector3d> points_world;
    points_world.reserve(kWidth * kHeight);

    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const Eigen::Vector3d dir_optical((col + 0.5 - kCx) / kFx, (row + 0.5 - kCy) / kFy, 1.0);
            const Eigen::Vector3d dir_world = kOpticalToWorld * dir_optical;
            const auto hit = rayBoxIntersection(kSensorOriginWorld, dir_world, box_min, box_max);
            if (!hit) continue;  // no-return pixel, same as a real sensor's inf/NaN

            const double ground_truth_depth = *hit;
            // The bug being regression-tested: the real sensor writes
            // -depth into z, not +depth.
            const double buggy_z_field = -ground_truth_depth;

            // --- from here down: verbatim logic from cloudCallback()'s fix ---
            const double depth = -buggy_z_field;
            const double optical_x = depth * (col - kCx) / kFx;
            const double optical_y = depth * (row - kCy) / kFy;
            points_world.push_back(kOpticalToWorld * Eigen::Vector3d(optical_x, optical_y, depth) +
                                    kSensorOriginWorld);
        }
    }
    return points_world;
}

void printAsciiXY(const VoxelGrid& grid, int z_idx)
{
    static const std::string kRamp = "@%#*+=-:. ";
    for (int y = grid.dims().y() - 1; y >= 0; --y) {
        std::string row;
        row.reserve(grid.dims().x());
        for (int x = 0; x < grid.dims().x(); ++x) {
            const double d = grid.distanceAt({x, y, z_idx});
            const double t = std::clamp(d / kColorMaxDistance, 0.0, 1.0);
            row += kRamp[static_cast<std::size_t>(t * (kRamp.size() - 1))];
        }
        std::cout << row << "\n";
    }
}

void writePpmXY(const VoxelGrid& grid, int z_idx, const std::string& path)
{
    const int w = grid.dims().x() * kPixelsPerVoxel;
    const int h = grid.dims().y() * kPixelsPerVoxel;
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (int py = 0; py < h; ++py) {
        const int y = grid.dims().y() - 1 - py / kPixelsPerVoxel;
        for (int px = 0; px < w; ++px) {
            const int x = px / kPixelsPerVoxel;
            const Rgb c = distanceToColor(grid.distanceAt({x, y, z_idx}), kColorMaxDistance);
            out.put(static_cast<char>(c.r));
            out.put(static_cast<char>(c.g));
            out.put(static_cast<char>(c.b));
        }
    }
}

void printAsciiXZ(const VoxelGrid& grid, int y_idx)
{
    static const std::string kRamp = "@%#*+=-:. ";
    for (int z = grid.dims().z() - 1; z >= 0; --z) {
        std::string row;
        row.reserve(grid.dims().x());
        for (int x = 0; x < grid.dims().x(); ++x) {
            const double d = grid.distanceAt({x, y_idx, z});
            const double t = std::clamp(d / kColorMaxDistance, 0.0, 1.0);
            row += kRamp[static_cast<std::size_t>(t * (kRamp.size() - 1))];
        }
        std::cout << row << "\n";
    }
}

void writePpmXZ(const VoxelGrid& grid, int y_idx, const std::string& path)
{
    const int w = grid.dims().x() * kPixelsPerVoxel;
    const int h = grid.dims().z() * kPixelsPerVoxel;
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (int py = 0; py < h; ++py) {
        const int z = grid.dims().z() - 1 - py / kPixelsPerVoxel;
        for (int px = 0; px < w; ++px) {
            const int x = px / kPixelsPerVoxel;
            const Rgb c = distanceToColor(grid.distanceAt({x, y_idx, z}), kColorMaxDistance);
            out.put(static_cast<char>(c.r));
            out.put(static_cast<char>(c.g));
            out.put(static_cast<char>(c.b));
        }
    }
}

}  // namespace

int main()
{
    std::cout << "argus_esdf depth-camera pipeline validation (SITL box replica)\n"
              << "box: center=(" << kBoxCenter.transpose() << ") half_size=(" << kBoxHalfSize.transpose() << ")\n"
              << "sensor: origin=(" << kSensorOriginWorld.transpose() << "), yaw=180deg (facing -X)\n\n";

    const std::vector<Eigen::Vector3d> points_world = synthesizeAndReprojectPoints();
    std::cout << "synthesized " << points_world.size() << " reprojected points from a "
              << kWidth << "x" << kHeight << " depth frame\n";

    VoxelGrid grid(kGridOrigin, kGridDims, kVoxelSize);
    grid.insertPointCloud(kSensorOriginWorld, points_world);
    grid.computeEsdf();

    namespace fs = std::filesystem;
    fs::create_directories("logs");

    const int z_idx = static_cast<int>(std::floor((kBoxCenter.z() - kGridOrigin.z()) / kVoxelSize));
    const int y_idx = static_cast<int>(std::floor((kBoxCenter.y() - kGridOrigin.y()) / kVoxelSize));

    std::cout << "\n--- XY slice at z=" << (kGridOrigin.z() + (z_idx + 0.5) * kVoxelSize)
              << "m (box mid-height, top-down view) ---\n";
    printAsciiXY(grid, z_idx);
    writePpmXY(grid, z_idx, "logs/esdf_depth_pipeline_xy.ppm");

    std::cout << "\n--- XZ slice at y=" << (kGridOrigin.y() + (y_idx + 0.5) * kVoxelSize)
              << "m (box center, side view) ---\n";
    printAsciiXZ(grid, y_idx);
    writePpmXZ(grid, y_idx, "logs/esdf_depth_pipeline_xz.ppm");

    std::cout << "\nWrote logs/esdf_depth_pipeline_xy.ppm and logs/esdf_depth_pipeline_xz.ppm\n";

    // ── Sanity checks — same spirit as esdf_2d_demo.cpp: distinct from the
    // gtest suite, exits nonzero (not a crash) if the pipeline stops
    // placing the box where it actually is. These specifically
    // regression-test the depth-reprojection bug (see the top-level todo
    // file): before the fix, the occupied z-range extended to ~2.45m (well
    // above the box's real 1.5m height) and the occupied centroid was
    // multiple metres from the box — both checked below. ─────────────────
    bool ok = true;

    const Eigen::Vector3d near_face_point(kBoxCenter.x() + kBoxHalfSize.x(), kBoxCenter.y(), kBoxCenter.z());
    const Eigen::Vector3i near_face_idx(
        static_cast<int>(std::floor((near_face_point.x() - kGridOrigin.x()) / kVoxelSize)),
        static_cast<int>(std::floor((near_face_point.y() - kGridOrigin.y()) / kVoxelSize)),
        static_cast<int>(std::floor((near_face_point.z() - kGridOrigin.z()) / kVoxelSize)));
    const double near_face_distance = grid.distanceAt(near_face_idx);
    if (near_face_distance > kVoxelSize) {
        std::cerr << "FAIL: box's near face should read ~0m, got " << near_face_distance << "m\n";
        ok = false;
    }

    Eigen::Vector3d occupied_centroid = Eigen::Vector3d::Zero();
    int occupied_count = 0;
    double occupied_z_min = std::numeric_limits<double>::infinity();
    double occupied_z_max = -std::numeric_limits<double>::infinity();
    for (int z = 0; z < kGridDims.z(); ++z) {
        for (int y = 0; y < kGridDims.y(); ++y) {
            for (int x = 0; x < kGridDims.x(); ++x) {
                const Eigen::Vector3i idx(x, y, z);
                if (grid.distanceAt(idx) > 1e-9) continue;
                const Eigen::Vector3d c = grid.voxelCenter(idx);
                occupied_centroid += c;
                ++occupied_count;
                occupied_z_min = std::min(occupied_z_min, c.z());
                occupied_z_max = std::max(occupied_z_max, c.z());
            }
        }
    }

    if (occupied_count == 0) {
        std::cerr << "FAIL: no occupied voxels at all -- box was never detected\n";
        ok = false;
    } else {
        occupied_centroid /= occupied_count;
        const double centroid_error = (occupied_centroid - kBoxCenter).norm();
        std::cout << "\noccupied voxels: " << occupied_count
                  << ", centroid=(" << occupied_centroid.transpose() << ")"
                  << ", centroid error from box center=" << centroid_error << "m"
                  << ", z range=[" << occupied_z_min << ", " << occupied_z_max << "]m\n";
        if (centroid_error > 1.0) {
            std::cerr << "FAIL: occupied centroid is " << centroid_error
                      << "m from the box's real position -- wrong location\n";
            ok = false;
        }
        const double margin = 2 * kVoxelSize;
        if (occupied_z_min < kBoxCenter.z() - kBoxHalfSize.z() - margin ||
            occupied_z_max > kBoxCenter.z() + kBoxHalfSize.z() + margin) {
            std::cerr << "FAIL: occupied z range [" << occupied_z_min << ", " << occupied_z_max
                      << "] exceeds the box's real height ["
                      << (kBoxCenter.z() - kBoxHalfSize.z()) << ", " << (kBoxCenter.z() + kBoxHalfSize.z())
                      << "] -- this is exactly the symptom of the depth-reprojection bug\n";
            ok = false;
        }
    }

    const Eigen::Vector3i far_idx(2, 2, 2);  // grid corner, far from the box
    const double far_distance = grid.distanceAt(far_idx);
    if (far_distance < 2.0) {
        std::cerr << "FAIL: far grid corner should read a large distance, got " << far_distance << "m\n";
        ok = false;
    }

    if (ok) std::cout << "\nOK: box detected in the right place, right size, nothing bogus far away\n";
    return ok ? 0 : 1;
}
