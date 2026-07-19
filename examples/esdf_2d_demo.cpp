// Standalone (no ROS2) sanity check for argus_esdf::VoxelGrid: builds a
// small square obstacle in a single-voxel-thick ("2-D-ish") grid, runs the
// real computeEsdf() pipeline, and renders the resulting distance field two
// ways — ASCII art to stdout for an instant terminal check, and a PPM image
// (logs/esdf_2d_demo.ppm) for a proper look. A correct field should show a
// solid square with concentric rings growing outward from it; a broken
// transform tends to look wrong immediately (asymmetric, streaky along one
// axis, or discontinuous) rather than subtly off, which is exactly why this
// is a useful companion to the exact-value assertions in
// tests/test_voxel_grid.cpp — some classes of bug are far more obvious to
// the eye than to a handful of point checks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <argus_esdf/VoxelGrid.hpp>

using argus_esdf::VoxelGrid;

namespace {

constexpr double kVoxelSize = 0.05;   // m
constexpr int kGridN = 80;            // 80x80x1 -> 4m x 4m footprint
constexpr double kHalfExtent = kGridN * kVoxelSize / 2.0;
constexpr double kSquareHalfSize = 0.4;  // m -- 0.8m x 0.8m obstacle, centered
constexpr double kColorMaxDistance = 1.0;  // m -- clamps the colormap/ASCII ramp
constexpr int kPixelsPerVoxel = 4;    // PPM upscale factor, purely cosmetic

struct Rgb { uint8_t r, g, b; };

// Same red(close)->blue(far) HSV hue sweep as argus_mapping_node's
// distanceToColor() in ros2/argus_mapping — duplicated here in plain C++
// (no std_msgs::ColorRGBA) since this file deliberately has zero ROS2
// dependency. Small enough that sharing it isn't worth a new argus_esdf
// public API just for a demo's benefit.
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

// Builds the grid and marks a filled square obstacle occupied. Deliberately
// goes through the real public API (insertPointCloud()) rather than poking
// any internal state, so this exercises exactly what argus_mapping_node
// itself calls — one short straight-down ray per obstacle voxel is enough
// to mark it occupied without disturbing its neighbors.
VoxelGrid buildSquareObstacleGrid()
{
    VoxelGrid grid(Eigen::Vector3d(-kHalfExtent, -kHalfExtent, 0.0),
                   Eigen::Vector3i(kGridN, kGridN, 1), kVoxelSize);

    for (int x = 0; x < kGridN; ++x) {
        for (int y = 0; y < kGridN; ++y) {
            const Eigen::Vector3d center = grid.voxelCenter({x, y, 0});
            if (std::abs(center.x()) <= kSquareHalfSize && std::abs(center.y()) <= kSquareHalfSize) {
                const Eigen::Vector3d sensor = center + Eigen::Vector3d(0.0, 0.0, 10.0 * kVoxelSize);
                grid.insertPointCloud(sensor, {center});
            }
        }
    }
    grid.computeEsdf();
    return grid;
}

void printAscii(const VoxelGrid& grid)
{
    // Dense near the obstacle, sparse far away -- printed top row (max y)
    // first so it reads right-side-up in a terminal.
    static const std::string kRamp = "@%#*+=-:. ";
    for (int y = grid.dims().y() - 1; y >= 0; --y) {
        std::string row;
        row.reserve(grid.dims().x());
        for (int x = 0; x < grid.dims().x(); ++x) {
            const double d = grid.distanceAt({x, y, 0});
            const double t = std::clamp(d / kColorMaxDistance, 0.0, 1.0);
            const std::size_t idx = static_cast<std::size_t>(t * (kRamp.size() - 1));
            row += kRamp[idx];
        }
        std::cout << row << "\n";
    }
}

void writePpm(const VoxelGrid& grid, const std::string& path)
{
    const int w = grid.dims().x() * kPixelsPerVoxel;
    const int h = grid.dims().y() * kPixelsPerVoxel;

    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";

    // Row 0 of the PPM is the top of the image -- same max-y-first order as
    // printAscii() above, so the two views match.
    for (int py = 0; py < h; ++py) {
        const int y = grid.dims().y() - 1 - py / kPixelsPerVoxel;
        for (int px = 0; px < w; ++px) {
            const int x = px / kPixelsPerVoxel;
            const double d = grid.distanceAt({x, y, 0});
            const Rgb c = distanceToColor(d, kColorMaxDistance);
            out.put(static_cast<char>(c.r));
            out.put(static_cast<char>(c.g));
            out.put(static_cast<char>(c.b));
        }
    }
}

}  // namespace

int main()
{
    const VoxelGrid grid = buildSquareObstacleGrid();

    std::cout << "argus_esdf 2-D square-obstacle sanity check\n"
              << "grid: " << grid.dims().x() << "x" << grid.dims().y()
              << ", voxel_size=" << grid.voxelSize() << "m, "
              << "square=" << (2.0 * kSquareHalfSize) << "m x " << (2.0 * kSquareHalfSize) << "m\n"
              << "ramp: '@' = at/inside obstacle -> ' ' = " << kColorMaxDistance << "m+ clear\n\n";
    printAscii(grid);

    namespace fs = std::filesystem;
    fs::create_directories("logs");
    const std::string ppm_path = "logs/esdf_2d_demo.ppm";
    writePpm(grid, ppm_path);
    std::cout << "\nWrote " << ppm_path
              << " (view with e.g. `feh`, `eog`, or `convert` to PNG)\n";

    // Cheap sanity assertions distinct from the gtest suite -- this binary
    // exits nonzero (not a crash) if the field doesn't look like a distance
    // field at all, e.g. someone breaks computeEsdf() and this stops being
    // rebuilt/run as part of `colcon build`/CI the way the gtest target is.
    //
    // distanceAt() (grid indices), not query() (world points + trilinear
    // interpolation): on a dims.z()==1 grid, EVERY query() call is
    // correctly invalid -- trilinear interpolation needs a neighbor cell on
    // both sides in every axis, and a single z-layer never has one above
    // it. Worth knowing before reaching for query() on a thin/2-D grid
    // elsewhere; distanceAt() has no such restriction since it doesn't
    // interpolate.
    const double center = grid.distanceAt({kGridN / 2, kGridN / 2, 0});
    const double far_corner = grid.distanceAt({5, 5, 0});
    bool ok = true;
    if (center > 1e-6) { std::cerr << "FAIL: center of obstacle should be ~0m, got " << center << "\n"; ok = false; }
    if (far_corner < kSquareHalfSize) {
        std::cerr << "FAIL: far corner should be far from the obstacle, got " << far_corner << "\n";
        ok = false;
    }
    if (ok) std::cout << "OK: center~0m (" << center << "), far corner far (" << far_corner << "m)\n";
    return ok ? 0 : 1;
}
