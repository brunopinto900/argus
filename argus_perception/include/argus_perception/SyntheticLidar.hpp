#pragma once

#include <vector>

#include <Eigen/Dense>

#include <argus_esdf/VoxelGrid.hpp>

// Synthetic 360-degree sensor (stands in for "a bunch of cameras + lidars")
// for incremental/online mapping without a real depth sensor or Gazebo.
// Pure C++, no ROS dependency, same reasoning as argus_mpc/argus_esdf/
// argus_follower -- this is the algorithmic core; a real sensor driver is
// a separate, later concern (see ros2/argus_mapping for that role today).
namespace argus_perception {

// Sweeps num_rays evenly spaced over the full 2*pi (azimuth only, z=0
// plane -- matches the dims.z()==1 "2-D-ish" grids used elsewhere in this
// repo; a true 3-D sensor would also sweep elevation, a later extension)
// against ground_truth via VoxelGrid::raycast(), from origin out to
// max_range. Returns one point per ray that actually hit something --
// shorter than num_rays whenever some rays miss (open space, or past
// max_range) -- directly usable with VoxelGrid::insertPointCloud().
//
// num_rays sizing is the caller's responsibility (not computed internally
// -- this function doesn't know the ground-truth grid's voxel_size vs.
// max_range tradeoff a particular caller cares about): pick it so adjacent
// ray endpoints at max_range are no more than about one ground-truth voxel
// apart (num_rays ~= 2*pi*max_range / voxel_size), or a thin obstacle edge
// can fall between two rays and go undetected.
std::vector<Eigen::Vector3d> simulateLidarScan(const argus_esdf::VoxelGrid& ground_truth,
                                                const Eigen::Vector3d& origin, int num_rays,
                                                double max_range);

// Sweeps the same num_rays/max_range scan as simulateLidarScan(), but
// integrates it directly into own_map instead of returning hit points:
// a ray that hits something is inserted via insertPointCloud() (marks the
// hit occupied, the path to it free); a ray that finds nothing within
// max_range is marked free along its whole length via markFreeRay().
//
// simulateLidarScan() alone only ever reports hits, so a caller using just
// that + insertPointCloud() can't represent "swept this direction, found
// it genuinely clear" -- that space stays kUnknown indefinitely even
// though the sensor looked right at it. This is the real integration step
// a sensor driver performs; see design_decisions.md's "Known limitation"
// entry this resolves.
void integrateLidarScan(const argus_esdf::VoxelGrid& ground_truth, argus_esdf::VoxelGrid& own_map,
                         const Eigen::Vector3d& origin, int num_rays, double max_range);

}  // namespace argus_perception
