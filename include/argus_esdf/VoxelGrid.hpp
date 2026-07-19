#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

// Pure C++ occupancy grid + Euclidean distance field — no ROS dependency,
// same reasoning as argus_mpc::QuadrotorMpc: this is the algorithmic core,
// safe to include and unit-test from anywhere. ros2/argus_mapping is the
// thin ROS2 shim that feeds it point clouds and publishes the result.
namespace argus_esdf {

enum class VoxelState : uint8_t { kUnknown = 0, kFree = 1, kOccupied = 2 };

// Result of querying the field at an arbitrary continuous point.
// gradient points away from the nearest obstacle (direction of increasing
// distance) — this is what an OCP obstacle cost/constraint needs alongside
// the distance itself for a first-order (d0 + grad . (p - p0)) linearization.
struct EsdfQuery
{
    double distance = 0.0;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    bool valid = false;  // false if the query point is outside the grid
};

// Wall-clock timing of one operation (insertPointCloud() or computeEsdf()),
// accumulated across calls since construction or the matching reset — same
// pattern as argus_mpc::QuadrotorMpc's SolverTiming, for the same reason:
// this is the number that says whether the mapping pipeline can keep up
// with its sensor/publish rate, not just whether it's correct.
struct VoxelGridTiming
{
    double min_ms  = 0.0;
    double max_ms  = 0.0;
    double mean_ms = 0.0;
    long   count   = 0;
};

// Fixed-resolution 3-D occupancy grid with an exact Euclidean distance
// transform computed over it. Bounds are fixed at construction — this does
// not grow or rehash like an octree; it trades that flexibility for O(1)
// indexing and a straightforward exact (not approximate) distance transform.
class VoxelGrid
{
public:
    // origin: world-frame position of voxel (0,0,0)'s minimum corner.
    // dims: voxel counts per axis (must all be > 0).
    // voxel_size: edge length of one voxel, metres (must be > 0).
    VoxelGrid(const Eigen::Vector3d& origin, const Eigen::Vector3i& dims, double voxel_size);

    // Integrates one sensor scan into the occupancy grid: for each point,
    // raycasts from sensor_origin to it (3-D DDA / Amanatides-Woo), marking
    // every traversed voxel free and the voxel containing the point itself
    // occupied. Points (and ray segments) outside the grid bounds are
    // clipped, not rejected outright — a ray that starts outside but enters
    // the grid still clears/marks the portion inside it.
    //
    // Does not touch the distance field — call computeEsdf() after a batch
    // of insertions, not per point (the transform is O(N) over the whole
    // grid, not incremental in this first version; see the "advanced ESDF"
    // notes in the top-level todo file for why an incremental scheme like
    // Voxblox/FIESTA exists and when it'd be worth adopting one instead).
    void insertPointCloud(const Eigen::Vector3d& sensor_origin,
                           const std::vector<Eigen::Vector3d>& points);

    // Recomputes the exact Euclidean distance field from the current
    // occupancy grid: unsigned distance from every voxel center to the
    // nearest occupied voxel center. Unknown and free voxels are treated
    // identically as "not an obstacle" for this pass — occupancy state
    // still matters for insertPointCloud()'s raycasting, just not here.
    //
    // Implementation: the exact 1-D distance transform (Felzenszwalb &
    // Huttenlocher, "Distance Transforms of Sampled Functions") applied
    // successively along x, then y, then z on squared distances — the
    // standard separable extension to N dimensions, O(total voxels) per
    // axis pass, exact (not an approximation via chamfer/manhattan
    // distance or repeated dilation).
    void computeEsdf();

    // Trilinear-interpolated distance + gradient at an arbitrary
    // world-frame point (not required to land on a voxel center). Gradient
    // is the analytic derivative of the trilinear interpolant, not a
    // separate finite-difference pass. valid=false (distance/gradient left
    // at their default) if point falls outside the grid.
    EsdfQuery query(const Eigen::Vector3d& point) const;

    // Overwrites the distance field directly from a precomputed source —
    // e.g. one received over a ROS2 message from a separate mapping process
    // that owns the actual occupancy grid and ran computeEsdf() itself.
    // query()/distanceAt() only ever read distance_, never occupancy_, so a
    // grid built this way is fully queryable; occupancy_ stays all-kUnknown
    // and its accessors (stateAt(), insertPointCloud()) are meaningless on
    // it. distances.size() must equal dims_.x()*dims_.y()*dims_.z(), same
    // linearIndex() layout computeEsdf() itself produces.
    void setDistanceField(const std::vector<double>& distances);

    // Accumulated wall-clock timing of insertPointCloud() / computeEsdf()
    // calls made so far on this instance (or since the matching reset*()).
    // Separate stats per operation since they run at very different
    // frequencies/costs in practice (insertPointCloud() once per sensor
    // frame, computeEsdf() once per publish cycle).
    VoxelGridTiming insertTiming() const;
    VoxelGridTiming computeEsdfTiming() const;
    void resetInsertTiming();
    void resetComputeEsdfTiming();

    // ── Accessors — visualization, testing, and diagnostics ─────────────
    VoxelState stateAt(const Eigen::Vector3i& idx) const;
    double distanceAt(const Eigen::Vector3i& idx) const;
    Eigen::Vector3d voxelCenter(const Eigen::Vector3i& idx) const;
    bool inBounds(const Eigen::Vector3i& idx) const;

    const Eigen::Vector3d& origin() const { return origin_; }
    const Eigen::Vector3i& dims() const { return dims_; }
    double voxelSize() const { return voxel_size_; }

private:
    std::size_t linearIndex(const Eigen::Vector3i& idx) const;
    // World point -> containing voxel index. Does not check bounds itself —
    // callers check inBounds() (or rely on the [0,dims) clamp already
    // having been applied, e.g. by the DDA raycast walking the grid).
    Eigen::Vector3i worldToIndex(const Eigen::Vector3d& point) const;

    // Amanatides-Woo 3-D DDA: walks the grid from sensor_origin to point,
    // marking traversed voxels free (unless already occupied) and the
    // final voxel containing point occupied. Used once per point by
    // insertPointCloud() — callers with a dense cloud (e.g. a full-res
    // depth image) should downsample before calling insertPointCloud(),
    // since this walks O(ray length / voxel_size) cells per point.
    void markRay(const Eigen::Vector3d& sensor_origin, const Eigen::Vector3d& point);

    Eigen::Vector3d origin_;
    Eigen::Vector3i dims_;
    double voxel_size_;

    std::vector<VoxelState> occupancy_;
    std::vector<double> distance_;  // valid only after computeEsdf()

    double insert_min_ms_ = 0.0;
    double insert_max_ms_ = 0.0;
    double insert_sum_ms_ = 0.0;
    long   insert_count_  = 0;

    double compute_min_ms_ = 0.0;
    double compute_max_ms_ = 0.0;
    double compute_sum_ms_ = 0.0;
    long   compute_count_  = 0;
};

}  // namespace argus_esdf
