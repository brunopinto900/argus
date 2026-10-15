#include <argus_perception/SyntheticLidar.hpp>

#include <cmath>
#include <stdexcept>

namespace argus_perception {

std::vector<Eigen::Vector3d> simulateLidarScan(const argus_esdf::VoxelGrid& ground_truth,
                                                const Eigen::Vector3d& origin, int num_rays,
                                                double max_range)
{
    if (num_rays <= 0) {
        throw std::invalid_argument("simulateLidarScan: num_rays must be positive");
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<std::size_t>(num_rays));

    for (int i = 0; i < num_rays; ++i) {
        const double angle = (2.0 * M_PI * i) / num_rays;
        const Eigen::Vector3d direction(std::cos(angle), std::sin(angle), 0.0);
        if (const auto hit = ground_truth.raycast(origin, direction, max_range)) {
            points.push_back(*hit);
        }
    }
    return points;
}

void integrateLidarScan(const argus_esdf::VoxelGrid& ground_truth, argus_esdf::VoxelGrid& own_map,
                         const Eigen::Vector3d& origin, int num_rays, double max_range)
{
    if (num_rays <= 0) {
        throw std::invalid_argument("integrateLidarScan: num_rays must be positive");
    }

    for (int i = 0; i < num_rays; ++i) {
        const double angle = (2.0 * M_PI * i) / num_rays;
        const Eigen::Vector3d direction(std::cos(angle), std::sin(angle), 0.0);
        if (const auto hit = ground_truth.raycast(origin, direction, max_range)) {
            own_map.insertPointCloud(origin, {*hit});
        } else {
            own_map.markFreeRay(origin, direction, max_range);
        }
    }
}

}  // namespace argus_perception
