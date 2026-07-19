#include <argus_esdf/VoxelGrid.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace argus_esdf {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// Sentinel for "no obstacle seen yet" in the distance-transform's squared-
// distance array — must be finite: the parabola-intersection formula below
// computes (f[q] - f[v[k]]) between two sentinel entries, and true IEEE
// infinity gives inf - inf = NaN there. 1e18 is safely larger than any
// (index difference)^2 a real grid produces while staying far from double's
// overflow range once squared/added.
constexpr double kNoObstacleSq = 1e18;

// Exact 1-D squared-distance transform (Felzenszwalb & Huttenlocher,
// "Distance Transforms of Sampled Functions") — lower envelope of parabolas
// rooted at each sample. f[i] is the squared distance "so far" at index i
// (0 at a source, +inf elsewhere on the first pass); overwritten in place
// with the transformed result. O(n).
void distanceTransform1D(std::vector<double>& f)
{
    const int n = static_cast<int>(f.size());
    if (n == 0) return;

    std::vector<int> v(n, 0);        // index of each parabola in the lower envelope
    std::vector<double> z(n + 1, 0); // envelope boundaries between consecutive parabolas
    std::vector<double> d(n, 0.0);

    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q) {
        double s = 0.0;
        while (true) {
            const double fq = f[q] + static_cast<double>(q) * q;
            const double fv = f[v[k]] + static_cast<double>(v[k]) * v[k];
            s = (fq - fv) / (2.0 * q - 2.0 * v[k]);
            if (s <= z[k]) {
                --k;
            } else {
                break;
            }
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<double>(q)) ++k;
        const double dq = static_cast<double>(q - v[k]);
        d[q] = dq * dq + f[v[k]];
    }
    f = std::move(d);
}
}  // namespace

VoxelGrid::VoxelGrid(const Eigen::Vector3d& origin, const Eigen::Vector3i& dims, double voxel_size)
    : origin_(origin), dims_(dims), voxel_size_(voxel_size)
{
    if (dims.x() <= 0 || dims.y() <= 0 || dims.z() <= 0) {
        throw std::invalid_argument("VoxelGrid: dims must be positive");
    }
    if (voxel_size <= 0.0) {
        throw std::invalid_argument("VoxelGrid: voxel_size must be positive");
    }
    const std::size_t n = static_cast<std::size_t>(dims_.x()) *
                           static_cast<std::size_t>(dims_.y()) *
                           static_cast<std::size_t>(dims_.z());
    occupancy_.assign(n, VoxelState::kUnknown);
    distance_.assign(n, 0.0);
}

std::size_t VoxelGrid::linearIndex(const Eigen::Vector3i& idx) const
{
    return static_cast<std::size_t>(idx.x()) +
           static_cast<std::size_t>(dims_.x()) *
               (static_cast<std::size_t>(idx.y()) +
                static_cast<std::size_t>(dims_.y()) * static_cast<std::size_t>(idx.z()));
}

Eigen::Vector3i VoxelGrid::worldToIndex(const Eigen::Vector3d& point) const
{
    const Eigen::Vector3d rel = (point - origin_) / voxel_size_;
    return Eigen::Vector3i(static_cast<int>(std::floor(rel.x())),
                            static_cast<int>(std::floor(rel.y())),
                            static_cast<int>(std::floor(rel.z())));
}

bool VoxelGrid::inBounds(const Eigen::Vector3i& idx) const
{
    return idx.x() >= 0 && idx.x() < dims_.x() &&
           idx.y() >= 0 && idx.y() < dims_.y() &&
           idx.z() >= 0 && idx.z() < dims_.z();
}

Eigen::Vector3d VoxelGrid::voxelCenter(const Eigen::Vector3i& idx) const
{
    return origin_ + (idx.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) * voxel_size_;
}

VoxelState VoxelGrid::stateAt(const Eigen::Vector3i& idx) const
{
    if (!inBounds(idx)) return VoxelState::kUnknown;
    return occupancy_[linearIndex(idx)];
}

double VoxelGrid::distanceAt(const Eigen::Vector3i& idx) const
{
    if (!inBounds(idx)) return 0.0;
    return distance_[linearIndex(idx)];
}

void VoxelGrid::markRay(const Eigen::Vector3d& sensor_origin, const Eigen::Vector3d& point)
{
    const Eigen::Vector3d delta = point - sensor_origin;
    const double ray_length = delta.norm();
    const Eigen::Vector3i hit_idx = worldToIndex(point);

    if (ray_length < 1e-9) {
        if (inBounds(hit_idx)) occupancy_[linearIndex(hit_idx)] = VoxelState::kOccupied;
        return;
    }

    const Eigen::Vector3d dir = delta / ray_length;
    Eigen::Vector3i voxel = worldToIndex(sensor_origin);

    Eigen::Vector3i step(0, 0, 0);
    Eigen::Vector3d t_max(kInf, kInf, kInf);
    Eigen::Vector3d t_delta(kInf, kInf, kInf);
    for (int axis = 0; axis < 3; ++axis) {
        if (dir[axis] > 0.0) {
            step[axis] = 1;
            const double next_boundary = origin_[axis] + (voxel[axis] + 1) * voxel_size_;
            t_max[axis] = (next_boundary - sensor_origin[axis]) / dir[axis];
            t_delta[axis] = voxel_size_ / dir[axis];
        } else if (dir[axis] < 0.0) {
            step[axis] = -1;
            const double next_boundary = origin_[axis] + voxel[axis] * voxel_size_;
            t_max[axis] = (next_boundary - sensor_origin[axis]) / dir[axis];
            t_delta[axis] = voxel_size_ / -dir[axis];
        }
    }

    double t = 0.0;
    while (t <= ray_length) {
        if (voxel == hit_idx) break;  // stop before marking the hit voxel itself free
        if (inBounds(voxel)) {
            VoxelState& state = occupancy_[linearIndex(voxel)];
            if (state != VoxelState::kOccupied) state = VoxelState::kFree;
        }
        int axis = 0;
        if (t_max[1] < t_max[axis]) axis = 1;
        if (t_max[2] < t_max[axis]) axis = 2;
        t = t_max[axis];
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }

    if (inBounds(hit_idx)) {
        occupancy_[linearIndex(hit_idx)] = VoxelState::kOccupied;
    }
}

void VoxelGrid::insertPointCloud(const Eigen::Vector3d& sensor_origin,
                                  const std::vector<Eigen::Vector3d>& points)
{
    const auto start = std::chrono::steady_clock::now();

    for (const auto& point : points) {
        markRay(sensor_origin, point);
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    insert_min_ms_ = (insert_count_ == 0) ? ms : std::min(insert_min_ms_, ms);
    insert_max_ms_ = std::max(insert_max_ms_, ms);
    insert_sum_ms_ += ms;
    ++insert_count_;
}

void VoxelGrid::computeEsdf()
{
    const auto start = std::chrono::steady_clock::now();

    std::vector<double> sq(occupancy_.size());
    for (std::size_t i = 0; i < occupancy_.size(); ++i) {
        sq[i] = (occupancy_[i] == VoxelState::kOccupied) ? 0.0 : kNoObstacleSq;
    }

    // Separable exact EDT: transform every scanline along x, then every
    // scanline of that result along y, then along z. Standard extension of
    // the 1-D transform above to N dimensions (each pass keeps every other
    // axis fixed and is itself exact), not an approximation.
    auto passAlong = [&](int axis) {
        const int n = dims_[axis];
        std::vector<double> line(static_cast<std::size_t>(n));
        Eigen::Vector3i idx;
        const int other0 = (axis + 1) % 3;
        const int other1 = (axis + 2) % 3;
        for (int b = 0; b < dims_[other1]; ++b) {
            for (int a = 0; a < dims_[other0]; ++a) {
                idx[other0] = a;
                idx[other1] = b;
                for (int i = 0; i < n; ++i) {
                    idx[axis] = i;
                    line[static_cast<std::size_t>(i)] = sq[linearIndex(idx)];
                }
                distanceTransform1D(line);
                for (int i = 0; i < n; ++i) {
                    idx[axis] = i;
                    sq[linearIndex(idx)] = line[static_cast<std::size_t>(i)];
                }
            }
        }
    };
    passAlong(0);
    passAlong(1);
    passAlong(2);

    distance_.resize(sq.size());
    for (std::size_t i = 0; i < sq.size(); ++i) {
        distance_[i] = std::sqrt(sq[i]) * voxel_size_;
    }

    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    compute_min_ms_ = (compute_count_ == 0) ? ms : std::min(compute_min_ms_, ms);
    compute_max_ms_ = std::max(compute_max_ms_, ms);
    compute_sum_ms_ += ms;
    ++compute_count_;
}

void VoxelGrid::setDistanceField(const std::vector<double>& distances)
{
    if (distances.size() != distance_.size()) {
        throw std::invalid_argument("VoxelGrid::setDistanceField: size mismatch with dims");
    }
    distance_ = distances;
}

VoxelGridTiming VoxelGrid::insertTiming() const
{
    VoxelGridTiming t;
    t.count = insert_count_;
    if (t.count > 0) {
        t.min_ms  = insert_min_ms_;
        t.max_ms  = insert_max_ms_;
        t.mean_ms = insert_sum_ms_ / static_cast<double>(t.count);
    }
    return t;
}

VoxelGridTiming VoxelGrid::computeEsdfTiming() const
{
    VoxelGridTiming t;
    t.count = compute_count_;
    if (t.count > 0) {
        t.min_ms  = compute_min_ms_;
        t.max_ms  = compute_max_ms_;
        t.mean_ms = compute_sum_ms_ / static_cast<double>(t.count);
    }
    return t;
}

void VoxelGrid::resetInsertTiming()
{
    insert_min_ms_ = 0.0;
    insert_max_ms_ = 0.0;
    insert_sum_ms_ = 0.0;
    insert_count_  = 0;
}

void VoxelGrid::resetComputeEsdfTiming()
{
    compute_min_ms_ = 0.0;
    compute_max_ms_ = 0.0;
    compute_sum_ms_ = 0.0;
    compute_count_  = 0;
}

EsdfQuery VoxelGrid::query(const Eigen::Vector3d& point) const
{
    EsdfQuery result;

    const Eigen::Vector3d rel = (point - origin_) / voxel_size_ - Eigen::Vector3d(0.5, 0.5, 0.5);
    const Eigen::Vector3i i0(static_cast<int>(std::floor(rel.x())),
                              static_cast<int>(std::floor(rel.y())),
                              static_cast<int>(std::floor(rel.z())));
    const Eigen::Vector3i i1 = i0 + Eigen::Vector3i(1, 1, 1);
    if (!inBounds(i0) || !inBounds(i1)) {
        return result;  // valid = false — no well-defined interpolant this close to the edge
    }
    const Eigen::Vector3d w = rel - i0.cast<double>();  // in [0,1) per axis

    // 8 corner distances, c[dz][dy][dx].
    double c[2][2][2];
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
                c[dz][dy][dx] = distanceAt(i0 + Eigen::Vector3i(dx, dy, dz));

    const double x = w.x(), y = w.y(), z = w.z();
    const auto lerp = [](double a, double b, double t) { return a + (b - a) * t; };

    const double c00 = lerp(c[0][0][0], c[0][0][1], x);
    const double c01 = lerp(c[0][1][0], c[0][1][1], x);
    const double c10 = lerp(c[1][0][0], c[1][0][1], x);
    const double c11 = lerp(c[1][1][0], c[1][1][1], x);
    const double c0 = lerp(c00, c01, y);
    const double c1 = lerp(c10, c11, y);
    result.distance = lerp(c0, c1, z);

    // Analytic partial derivatives of the same trilinear interpolant, not a
    // separate finite-difference pass — see VoxelGrid.hpp's query() comment.
    const double dc00_dx = c[0][0][1] - c[0][0][0];
    const double dc01_dx = c[0][1][1] - c[0][1][0];
    const double dc10_dx = c[1][0][1] - c[1][0][0];
    const double dc11_dx = c[1][1][1] - c[1][1][0];
    const double ddx = lerp(lerp(dc00_dx, dc01_dx, y), lerp(dc10_dx, dc11_dx, y), z);

    const double ddy = lerp(c01 - c00, c11 - c10, z);
    const double ddz = c1 - c0;

    // Chain rule: rel = (point - origin_) / voxel_size_ - 0.5, so
    // d(rel)/d(point) = 1 / voxel_size_.
    result.gradient = Eigen::Vector3d(ddx, ddy, ddz) / voxel_size_;
    result.valid = true;
    return result;
}

}  // namespace argus_esdf
