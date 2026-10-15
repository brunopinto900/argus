#include <gtest/gtest.h>

#include <argus_esdf/VoxelGrid.hpp>
#include <argus_perception/ObservableSlice.hpp>
#include <argus_perception/SyntheticLidar.hpp>

using argus_esdf::VoxelGrid;
using argus_esdf::VoxelState;
using argus_perception::computeObservableSlice;
using argus_perception::verifyAgainstGroundTruth;

namespace {
// Same fixture shape as test_voxel_grid.cpp's squareObstacleGrid() /
// test_synthetic_lidar.cpp's buildSquareGroundTruth().
VoxelGrid buildSquareGroundTruth()
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    for (int x = 9; x <= 11; ++x) {
        for (int y = 9; y <= 11; ++y) {
            const Eigen::Vector3d center = grid.voxelCenter({x, y, 0});
            const Eigen::Vector3d sensor = center + Eigen::Vector3d(0.0, 0.0, 1.0);
            grid.insertPointCloud(sensor, {center});
        }
    }
    return grid;
}
}  // namespace

// Runs the real pipeline (simulateLidarScan against ground truth,
// insertPointCloud into own_map) from a small scripted set of poses, then
// checks it with computeObservableSlice()/verifyAgainstGroundTruth() using
// those same poses -- the "happy path": a correctly-built own_map must
// match ground truth everywhere its sensor actually looked.
TEST(ObservableSlice, CorrectlyBuiltOwnMapMatchesEverywhereObserved)
{
    const VoxelGrid ground_truth = buildSquareGroundTruth();
    VoxelGrid own_map(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);

    const std::vector<Eigen::Vector3d> poses = {
        Eigen::Vector3d(10.5, 2.5, 0.5),
        Eigen::Vector3d(2.5, 10.5, 0.5),
    };
    constexpr int kNumRays = 360;
    constexpr double kMaxRange = 15.0;

    for (const auto& pose : poses) {
        own_map.insertPointCloud(pose, argus_perception::simulateLidarScan(ground_truth, pose,
                                                                             kNumRays, kMaxRange));
    }

    const auto slice = computeObservableSlice(ground_truth, poses, kNumRays, kMaxRange);
    const auto report = verifyAgainstGroundTruth(own_map, ground_truth, slice);

    EXPECT_GT(report.observable_count, 0);
    EXPECT_TRUE(report.allMatch()) << "mismatched=" << report.mismatched_count
                                    << " of " << report.observable_count;
    EXPECT_EQ(report.mismatched_count, 0);
}

// Voxels never swept by any pose must stay kUnknown in own_map -- not part
// of the observable slice, so not checked, but worth confirming the slice
// itself is a strict subset of the full grid (otherwise the "happy path"
// test above would be vacuously true over the whole grid).
TEST(ObservableSlice, SliceExcludesUnobservedRegionsOfTheGrid)
{
    const VoxelGrid ground_truth = buildSquareGroundTruth();
    const std::vector<Eigen::Vector3d> poses = {Eigen::Vector3d(10.5, 2.5, 0.5)};
    constexpr int kNumRays = 360;
    constexpr double kMaxRange = 10.0;  // obstacle's near face is ~6.5 away -- within range

    const auto slice = computeObservableSlice(ground_truth, poses, kNumRays, kMaxRange);
    constexpr int kTotalVoxels = 21 * 21;
    EXPECT_GT(slice.size(), 0u);
    EXPECT_LT(static_cast<int>(slice.size()), kTotalVoxels);
}

// A deliberately wrong own_map (never actually updated from any sensor
// sweep) must be caught -- confirms the check can actually fail, not just
// trivially pass.
TEST(ObservableSlice, DetectsMismatchInAnIncorrectOwnMap)
{
    const VoxelGrid ground_truth = buildSquareGroundTruth();
    VoxelGrid stale_own_map(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    // stale_own_map is never updated -- still all-kUnknown everywhere,
    // including where the obstacle actually is.

    const std::vector<Eigen::Vector3d> poses = {Eigen::Vector3d(10.5, 2.5, 0.5)};
    constexpr int kNumRays = 360;
    constexpr double kMaxRange = 15.0;

    const auto slice = computeObservableSlice(ground_truth, poses, kNumRays, kMaxRange);
    const auto report = verifyAgainstGroundTruth(stale_own_map, ground_truth, slice);

    EXPECT_GT(report.observable_count, 0);
    EXPECT_FALSE(report.allMatch());
    EXPECT_GT(report.mismatched_count, 0);
}
