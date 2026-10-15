#pragma once

#include <vector>

#include <Eigen/Dense>

#include <argus_esdf/VoxelGrid.hpp>

// Verification helper for online mapping: "did the drone's own map end up
// correct everywhere its sensor actually had a chance to look?" -- not
// "does the own map match ground truth everywhere" (most of it is
// expected to stay kUnknown; that's correct, not a failure). See
// design_decisions.md's "Testing / verification strategy".
namespace argus_perception {

// The set of voxel indices any ray from any of `poses` (num_rays each,
// out to max_range) actually traverses against ground_truth -- both the
// occupied hit and the free cells along the way, via
// VoxelGrid::traceRay(). Deduplicated. Deliberately independent of any
// per-tick insertion calls already made elsewhere -- re-derives "what
// should be known" from scratch, so it catches real pipeline bugs
// (pose/frame mismatches, grid misalignment, missed sensor ticks) rather
// than re-testing raycast()/traceRay() themselves.
//
// Mirrors integrateLidarScan()'s actual behaviour exactly: a hit ray's
// traced path is pose->hit-point (the same path markRay() derives from a
// hit point, not pose->full-max-range along the originally sampled
// direction -- those two can disagree by a voxel near grazing/diagonal
// rays since a hit point is a voxel *center*, not the exact ray/surface
// intersection); a miss ray's traced path is pose->(pose +
// dir*max_range), matching markFreeRay()'s walk exactly.
std::vector<Eigen::Vector3i> computeObservableSlice(const argus_esdf::VoxelGrid& ground_truth,
                                                     const std::vector<Eigen::Vector3d>& poses,
                                                     int num_rays, double max_range);

struct VerificationReport
{
    int observable_count = 0;
    int matched_count    = 0;
    int mismatched_count = 0;

    bool allMatch() const { return mismatched_count == 0; }
};

// Compares own_map against ground_truth at exactly the voxels in
// observable_slice -- voxels outside the slice are not checked, since
// own_map is expected to still be kUnknown there.
//
// Compares occupied-vs-not-occupied, not raw VoxelState equality: ground
// truth is privileged/complete by construction, so its own kUnknown cells
// away from any obstacle (e.g. buildObstacleGrid()-style fixtures that
// only ever mark the obstacle's own cells, never explicitly free the
// surrounding space) mean "confirmed not occupied" just as much as an
// explicit kFree does -- there's no real ambiguity in ground truth, only
// in how little of it got explicitly written. own_map, on the other
// hand, must have actually resolved each slice voxel to kFree or
// kOccupied -- still being kUnknown inside its own observable slice is a
// real mismatch (e.g. a missed sensor tick), not something to excuse.
VerificationReport verifyAgainstGroundTruth(const argus_esdf::VoxelGrid& own_map,
                                             const argus_esdf::VoxelGrid& ground_truth,
                                             const std::vector<Eigen::Vector3i>& observable_slice);

}  // namespace argus_perception
