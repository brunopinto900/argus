#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

#include <argus_esdf/VoxelGrid.hpp>

using argus_esdf::EsdfQuery;
using argus_esdf::VoxelGrid;
using argus_esdf::VoxelState;

TEST(VoxelGrid, RejectsInvalidConstruction)
{
    EXPECT_THROW(VoxelGrid(Eigen::Vector3d::Zero(), Eigen::Vector3i(0, 5, 5), 1.0),
                 std::invalid_argument);
    EXPECT_THROW(VoxelGrid(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 5), 0.0),
                 std::invalid_argument);
    EXPECT_THROW(VoxelGrid(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 5), -1.0),
                 std::invalid_argument);
}

TEST(VoxelGrid, FreshGridIsAllUnknown)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 5), 1.0);
    EXPECT_EQ(grid.stateAt({2, 2, 2}), VoxelState::kUnknown);
    EXPECT_EQ(grid.stateAt({-1, 0, 0}), VoxelState::kUnknown);  // out of bounds
    EXPECT_EQ(grid.stateAt({10, 0, 0}), VoxelState::kUnknown);  // out of bounds
}

// dims=(10,1,1) collapses this to an effectively 1-D line of voxels along x,
// simplest case to hand-verify the DDA raycast against.
TEST(VoxelGrid, InsertPointCloudMarksRayFreeAndHitOccupied)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(10, 1, 1), 1.0);
    const Eigen::Vector3d sensor(0.5, 0.5, 0.5);   // center of voxel (0,0,0)
    const Eigen::Vector3d point(7.5, 0.5, 0.5);    // center of voxel (7,0,0)
    grid.insertPointCloud(sensor, {point});

    for (int x = 0; x <= 6; ++x) {
        EXPECT_EQ(grid.stateAt({x, 0, 0}), VoxelState::kFree) << "x=" << x;
    }
    EXPECT_EQ(grid.stateAt({7, 0, 0}), VoxelState::kOccupied);
    EXPECT_EQ(grid.stateAt({8, 0, 0}), VoxelState::kUnknown);
    EXPECT_EQ(grid.stateAt({9, 0, 0}), VoxelState::kUnknown);
}

// A later point's ray happening to pass straight through an earlier point's
// hit voxel (within the same insertPointCloud() call) must not erase it —
// order within a batch shouldn't matter for already-observed obstacles.
TEST(VoxelGrid, OccupiedVoxelNotDowngradedByLaterRayThroughIt)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(10, 1, 1), 1.0);
    const Eigen::Vector3d sensor(0.5, 0.5, 0.5);
    grid.insertPointCloud(sensor, {Eigen::Vector3d(3.5, 0.5, 0.5),
                                    Eigen::Vector3d(7.5, 0.5, 0.5)});
    EXPECT_EQ(grid.stateAt({3, 0, 0}), VoxelState::kOccupied);
    EXPECT_EQ(grid.stateAt({7, 0, 0}), VoxelState::kOccupied);
}

namespace {
// Shared fixture: an 11^3 grid, unit voxels, with a single isolated occupied
// voxel at index (5,5,5) — hand-verifiable EDT ground truth.
VoxelGrid singleObstacleGrid()
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(11, 11, 11), 1.0);
    const Eigen::Vector3d sensor(0.5, 0.5, 0.5);
    const Eigen::Vector3d obstacle_center(5.5, 5.5, 5.5);  // voxel (5,5,5)
    grid.insertPointCloud(sensor, {obstacle_center});
    grid.computeEsdf();
    return grid;
}
}  // namespace

TEST(VoxelGrid, ExactDistanceTransformOnSingleObstacle)
{
    VoxelGrid grid = singleObstacleGrid();

    EXPECT_NEAR(grid.distanceAt({5, 5, 5}), 0.0, 1e-9);
    EXPECT_NEAR(grid.distanceAt({6, 5, 5}), 1.0, 1e-9);          // face-adjacent
    EXPECT_NEAR(grid.distanceAt({5, 6, 5}), 1.0, 1e-9);
    EXPECT_NEAR(grid.distanceAt({6, 6, 5}), std::sqrt(2.0), 1e-9);  // edge-adjacent
    EXPECT_NEAR(grid.distanceAt({6, 6, 6}), std::sqrt(3.0), 1e-9);  // corner-adjacent
    EXPECT_NEAR(grid.distanceAt({7, 5, 5}), 2.0, 1e-9);
}

TEST(VoxelGrid, QueryAtVoxelCenterMatchesDistanceAt)
{
    VoxelGrid grid = singleObstacleGrid();
    const auto q = grid.query(grid.voxelCenter({6, 5, 5}));
    EXPECT_TRUE(q.valid);
    EXPECT_NEAR(q.distance, grid.distanceAt({6, 5, 5}), 1e-9);
}

// The analytic gradient should reproduce a central finite difference of the
// same query() field — trilinear interpolation is affine along each axis
// within a single cell, so this has zero truncation error (mismatch would
// mean the derivative formula itself is wrong, not just imprecise). p is
// chosen well inside one cell so p +/- eps never crosses a cell boundary.
TEST(VoxelGrid, QueryGradientMatchesFiniteDifference)
{
    VoxelGrid grid = singleObstacleGrid();
    const Eigen::Vector3d p(6.3, 5.7, 5.2);
    constexpr double eps = 1e-4;

    const auto q = grid.query(p);
    ASSERT_TRUE(q.valid);

    for (int axis = 0; axis < 3; ++axis) {
        Eigen::Vector3d p_plus = p, p_minus = p;
        p_plus[axis] += eps;
        p_minus[axis] -= eps;
        const auto q_plus = grid.query(p_plus);
        const auto q_minus = grid.query(p_minus);
        ASSERT_TRUE(q_plus.valid);
        ASSERT_TRUE(q_minus.valid);
        const double finite_diff = (q_plus.distance - q_minus.distance) / (2.0 * eps);
        EXPECT_NEAR(q.gradient[axis], finite_diff, 1e-6) << "axis=" << axis;
    }
}

// Simulates argus_bridge_node's use case: a grid rebuilt on the receiving
// end of a ROS2 message from a separate mapping process, with no occupancy
// data at all, only a distance field.
TEST(VoxelGrid, SetDistanceFieldMakesGridQueryable)
{
    VoxelGrid source = singleObstacleGrid();
    std::vector<double> distances;
    for (int z = 0; z < source.dims().z(); ++z)
        for (int y = 0; y < source.dims().y(); ++y)
            for (int x = 0; x < source.dims().x(); ++x)
                distances.push_back(source.distanceAt({x, y, z}));

    VoxelGrid received(source.origin(), source.dims(), source.voxelSize());
    received.setDistanceField(distances);

    EXPECT_NEAR(received.distanceAt({6, 5, 5}), source.distanceAt({6, 5, 5}), 1e-9);
    const auto q = received.query(source.voxelCenter({6, 6, 5}));
    EXPECT_TRUE(q.valid);
    EXPECT_NEAR(q.distance, source.distanceAt({6, 6, 5}), 1e-9);
}

TEST(VoxelGrid, SetDistanceFieldRejectsSizeMismatch)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 5), 1.0);
    EXPECT_THROW(grid.setDistanceField(std::vector<double>(10, 0.0)), std::invalid_argument);
}

TEST(VoxelGrid, TimingAccumulatesAcrossCalls)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(11, 11, 11), 1.0);
    const Eigen::Vector3d sensor(0.5, 0.5, 0.5);

    EXPECT_EQ(grid.insertTiming().count, 0);
    EXPECT_EQ(grid.computeEsdfTiming().count, 0);

    grid.insertPointCloud(sensor, {Eigen::Vector3d(5.5, 5.5, 5.5)});
    grid.insertPointCloud(sensor, {Eigen::Vector3d(3.5, 3.5, 3.5)});
    grid.computeEsdf();

    const auto insert_t = grid.insertTiming();
    EXPECT_EQ(insert_t.count, 2);
    EXPECT_GE(insert_t.min_ms, 0.0);
    EXPECT_LE(insert_t.min_ms, insert_t.mean_ms);
    EXPECT_LE(insert_t.mean_ms, insert_t.max_ms);

    const auto compute_t = grid.computeEsdfTiming();
    EXPECT_EQ(compute_t.count, 1);
    EXPECT_GE(compute_t.min_ms, 0.0);
    EXPECT_DOUBLE_EQ(compute_t.min_ms, compute_t.max_ms);  // single sample

    grid.resetInsertTiming();
    EXPECT_EQ(grid.insertTiming().count, 0);
    // resetInsertTiming() must not touch the other operation's stats.
    EXPECT_EQ(grid.computeEsdfTiming().count, 1);

    grid.resetComputeEsdfTiming();
    EXPECT_EQ(grid.computeEsdfTiming().count, 0);
}

TEST(VoxelGrid, QueryOutOfBoundsIsInvalid)
{
    VoxelGrid grid(Eigen::Vector3d::Zero(), Eigen::Vector3i(5, 5, 5), 1.0);
    grid.computeEsdf();
    EXPECT_FALSE(grid.query(Eigen::Vector3d(-5, -5, -5)).valid);
    EXPECT_FALSE(grid.query(Eigen::Vector3d(100, 100, 100)).valid);
}
