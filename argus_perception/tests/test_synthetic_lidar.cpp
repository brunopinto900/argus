#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

#include <argus_esdf/VoxelGrid.hpp>
#include <argus_perception/SyntheticLidar.hpp>

using argus_esdf::VoxelGrid;
using argus_esdf::VoxelState;
using argus_perception::integrateLidarScan;
using argus_perception::simulateLidarScan;

namespace {
// 21x21x1 grid, unit voxels, with a filled 3x3 square obstacle at indices
// (9..11, 9..11) -- same fixture shape as test_voxel_grid.cpp's own
// squareObstacleGrid(), duplicated locally since that one isn't exported.
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

TEST(SyntheticLidar, RejectsNonPositiveNumRays)
{
    VoxelGrid empty(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 1), 1.0);
    EXPECT_THROW(simulateLidarScan(empty, Eigen::Vector3d(2.5, 2.5, 0.5), 0, 10.0),
                 std::invalid_argument);
    EXPECT_THROW(simulateLidarScan(empty, Eigen::Vector3d(2.5, 2.5, 0.5), -5, 10.0),
                 std::invalid_argument);
}

TEST(SyntheticLidar, EmptyGroundTruthProducesNoHits)
{
    VoxelGrid empty(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    const auto points = simulateLidarScan(empty, Eigen::Vector3d(10.5, 10.5, 0.5), 360, 10.0);
    EXPECT_TRUE(points.empty());
}

// Every returned point must equal what calling VoxelGrid::raycast()
// directly for that same sampled angle would return -- simulateLidarScan()
// should be a thin, exact sweep over raycast(), not approximate it.
TEST(SyntheticLidar, MatchesIndividualRaycastCallsForEveryAngle)
{
    const VoxelGrid ground_truth = buildSquareGroundTruth();
    const Eigen::Vector3d sensor_origin(10.5, 2.5, 0.5);  // south of the square
    constexpr int kNumRays = 360;
    constexpr double kMaxRange = 10.0;

    const auto points = simulateLidarScan(ground_truth, sensor_origin, kNumRays, kMaxRange);

    std::vector<Eigen::Vector3d> expected;
    for (int i = 0; i < kNumRays; ++i) {
        const double angle = (2.0 * M_PI * i) / kNumRays;
        const Eigen::Vector3d dir(std::cos(angle), std::sin(angle), 0.0);
        if (const auto hit = ground_truth.raycast(sensor_origin, dir, kMaxRange)) {
            expected.push_back(*hit);
        }
    }

    ASSERT_EQ(points.size(), expected.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        EXPECT_TRUE(points[i].isApprox(expected[i], 1e-9)) << "ray " << i;
    }
    EXPECT_GT(points.size(), 0u);  // sanity: the square was actually seen, not a vacuous all-miss pass
}

// A nearer obstacle must block the sensor's view of a farther one behind
// it -- independent of MatchesIndividualRaycastCallsForEveryAngle above
// (that test only checks equivalence to repeated raycast() calls, not
// occlusion correctness as a standalone behaviour).
TEST(SyntheticLidar, NearerObstacleOccludesFartherOneBehindIt)
{
    VoxelGrid ground_truth(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    const auto fill = [&](int x0, int x1, int y0, int y1) {
        for (int x = x0; x <= x1; ++x) {
            for (int y = y0; y <= y1; ++y) {
                const Eigen::Vector3d center = ground_truth.voxelCenter({x, y, 0});
                const Eigen::Vector3d sensor = center + Eigen::Vector3d(0.0, 0.0, 1.0);
                ground_truth.insertPointCloud(sensor, {center});
            }
        }
    };
    fill(9, 11, 5, 7);    // near square, south
    fill(9, 11, 13, 15);  // far square, north -- directly behind the near one

    const auto points =
        simulateLidarScan(ground_truth, Eigen::Vector3d(10.5, 0.5, 0.5), 360, 20.0);

    ASSERT_GT(points.size(), 0u);
    for (const auto& p : points) {
        EXPECT_LT(p.y(), 9.0) << "a point from the occluded far square leaked through";
    }
}

// The gap simulateLidarScan() alone has: a ray reaching max_range without
// hitting anything contributes nothing, so open space stays kUnknown even
// after being "seen" as clear. integrateLidarScan() must resolve it --
// every voxel within max_range of an all-empty ground truth should end up
// kFree, not kUnknown, once integrated.
TEST(SyntheticLidar, IntegrateLidarScanMarksEmptySpaceFreeNotUnknown)
{
    const VoxelGrid ground_truth(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    VoxelGrid own_map(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    const Eigen::Vector3d origin(10.5, 10.5, 0.5);
    constexpr double kMaxRange = 5.0;

    integrateLidarScan(ground_truth, own_map, origin, /*num_rays=*/360, kMaxRange);

    // A small disk around origin, well within kMaxRange on every axis, must
    // be resolved free -- not just the four cardinal directions a sparser
    // check might miss gaps on.
    for (int x = 8; x <= 13; ++x) {
        for (int y = 8; y <= 13; ++y) {
            EXPECT_EQ(own_map.stateAt({x, y, 0}), VoxelState::kFree) << "x=" << x << " y=" << y;
        }
    }
}

// A ray that does hit something must still integrate exactly like
// insertPointCloud() would -- integrateLidarScan() isn't just the miss-ray
// fix, it must preserve hit behaviour too (occupied hit, free path to it).
TEST(SyntheticLidar, IntegrateLidarScanStillMarksHitsOccupied)
{
    const VoxelGrid ground_truth = buildSquareGroundTruth();
    VoxelGrid own_map(Eigen::Vector3d::Zero(), Eigen::Vector3i(21, 21, 1), 1.0);
    const Eigen::Vector3d origin(10.5, 2.5, 0.5);  // south of the square

    integrateLidarScan(ground_truth, own_map, origin, /*num_rays=*/360, /*max_range=*/10.0);

    EXPECT_EQ(own_map.stateAt({10, 9, 0}), VoxelState::kOccupied);  // square's near face
    EXPECT_EQ(own_map.stateAt({10, 5, 0}), VoxelState::kFree);      // between origin and the square
}

TEST(SyntheticLidar, IntegrateLidarScanRejectsNonPositiveNumRays)
{
    const VoxelGrid ground_truth(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 1), 1.0);
    VoxelGrid own_map(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 1), 1.0);
    EXPECT_THROW(integrateLidarScan(ground_truth, own_map, Eigen::Vector3d(2.5, 2.5, 0.5), 0, 10.0),
                 std::invalid_argument);
}
