#pragma once

#include <cstdint>
#include <optional>
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

    // Marks every voxel from origin to max_range along direction free
    // (unless already kOccupied, which is left untouched) -- the
    // counterpart to insertPointCloud() for a ray that found nothing within
    // sensor range. insertPointCloud() alone cannot represent this: it
    // always marks its final point's voxel occupied, so a "miss" ray simply
    // isn't callable through it. Without this, open space a sensor actually
    // swept and found clear stays kUnknown forever -- see
    // design_decisions.md's "Known limitation" entry this resolves.
    void markFreeRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
                      double max_range);

    // Read-only counterpart to insertPointCloud()'s raycasting: walks from
    // origin along direction (need not be unit length) up to max_range
    // using the same Amanatides-Woo DDA markRay() uses for writing,
    // returning the first occupied voxel's center, or nullopt if nothing
    // occupied is found within max_range. Tolerates origin starting
    // outside the grid (same as markRay() — a ray that enters later still
    // gets a chance to hit something). Queries current occupancy only;
    // does not touch it.
    std::optional<Eigen::Vector3d> raycast(const Eigen::Vector3d& origin,
                                            const Eigen::Vector3d& direction,
                                            double max_range) const;

    // Like raycast(), but returns every in-bounds voxel index the ray
    // actually traverses (not just the final hit) -- stopping at and
    // including the first occupied voxel if one is hit before max_range,
    // otherwise everything up to max_range. For verification use (see
    // argus_perception's observable-slice check): what did a sensor sweep
    // actually have a chance to resolve, not just what it hit. Not used
    // by raycast()/insertPointCloud() themselves — those stay allocation-
    // free for the 60Hz sensor path; this is the slower, verification-only
    // sibling.
    std::vector<Eigen::Vector3i> traceRay(const Eigen::Vector3d& origin,
                                           const Eigen::Vector3d& direction,
                                           double max_range) const;

    // Recomputes the exact Euclidean distance field from the current
    // occupancy grid: unsigned distance from every voxel center to the
    // nearest seed voxel center, where "seed" is kOccupied alone by
    // default (kFree and kUnknown both "not an obstacle" for this pass —
    // occupancy state still matters for insertPointCloud()'s raycasting,
    // just not here), or kOccupied-or-kUnknown when
    // treat_unknown_as_occupied is true.
    //
    // The latter is the conservative mode an online-mapping consumer
    // should use: unexplored space reads as distance 0 (unsafe), same as
    // a confirmed obstacle, rather than the default's "unknown is free
    // until proven otherwise" — a planner reading unknown space as having
    // infinite clearance is a known real-world hazard. This flag changes
    // only what distanceAt()/query() report; stateAt() is unaffected and
    // still distinguishes kUnknown from kOccupied — callers that need to
    // tell "confirmed obstacle" apart from "unexplored" (e.g. for
    // visualization) must still use stateAt(), not infer it from distance.
    //
    // Implementation: the exact 1-D distance transform (Felzenszwalb &
    // Huttenlocher, "Distance Transforms of Sampled Functions") applied
    // successively along x, then y, then z on squared distances — the
    // standard separable extension to N dimensions, O(total voxels) per
    // axis pass, exact (not an approximation via chamfer/manhattan
    // distance or repeated dilation).
    void computeEsdf(bool treat_unknown_as_occupied = false);

    // Trilinear-interpolated distance + gradient at an arbitrary
    // world-frame point (not required to land on a voxel center). Gradient
    // is the analytic derivative of the trilinear interpolant, not a
    // separate finite-difference pass. valid=false (distance/gradient left
    // at their default) if point falls outside the grid — including, on a
    // grid with dims.z()==1 (or any axis ==1), always: interpolation needs
    // a neighbor on both sides in every axis, and a single-layer axis never
    // has one. Use distanceAt() (no interpolation) on a grid that thin.
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
