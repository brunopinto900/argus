#include <argus_perception/ObservableSlice.hpp>

#include <cmath>
#include <set>

namespace argus_perception {

namespace {
// Eigen::Vector3i has no default std::less/std::hash -- lexicographic
// ordering is all a std::set needs here (dedup only, no iteration order
// requirement).
struct Vector3iLess
{
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const
    {
        if (a.x() != b.x()) return a.x() < b.x();
        if (a.y() != b.y()) return a.y() < b.y();
        return a.z() < b.z();
    }
};
}  // namespace

std::vector<Eigen::Vector3i> computeObservableSlice(const argus_esdf::VoxelGrid& ground_truth,
                                                     const std::vector<Eigen::Vector3d>& poses,
                                                     int num_rays, double max_range)
{
    std::set<Eigen::Vector3i, Vector3iLess> observed;

    for (const auto& pose : poses) {
        for (int i = 0; i < num_rays; ++i) {
            const double angle = (2.0 * M_PI * i) / num_rays;
            const Eigen::Vector3d dir(std::cos(angle), std::sin(angle), 0.0);

            // Matches simulateLidarScan(): nothing contributed on a miss.
            const auto hit = ground_truth.raycast(pose, dir, max_range);
            if (!hit) continue;

            // Trace pose->hit-point, not pose->full-max-range along dir --
            // the same path markRay() derives once insertPointCloud()
            // receives this hit point, which can differ from dir by
            // enough to cross a voxel boundary differently (hit point is
            // a voxel center, not the exact surface intersection).
            const Eigen::Vector3d to_hit = *hit - pose;
            for (const auto& idx : ground_truth.traceRay(pose, to_hit, to_hit.norm())) {
                observed.insert(idx);
            }
        }
    }

    return std::vector<Eigen::Vector3i>(observed.begin(), observed.end());
}

VerificationReport verifyAgainstGroundTruth(const argus_esdf::VoxelGrid& own_map,
                                             const argus_esdf::VoxelGrid& ground_truth,
                                             const std::vector<Eigen::Vector3i>& observable_slice)
{
    VerificationReport report;
    report.observable_count = static_cast<int>(observable_slice.size());

    for (const auto& idx : observable_slice) {
        const bool own_occupied = own_map.stateAt(idx) == argus_esdf::VoxelState::kOccupied;
        const bool truth_occupied = ground_truth.stateAt(idx) == argus_esdf::VoxelState::kOccupied;
        const bool own_resolved = own_map.stateAt(idx) != argus_esdf::VoxelState::kUnknown;

        if (own_resolved && own_occupied == truth_occupied) {
            ++report.matched_count;
        } else {
            ++report.mismatched_count;
        }
    }
    return report;
}

}  // namespace argus_perception
