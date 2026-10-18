#include <argus_follower/TargetMotionSimulator.hpp>

#include <algorithm>
#include <cmath>

namespace argus_follower {

namespace {
// Matches the Python reference's sample_times(): n = round(t_end/dt) + 1
// knots at i*dt, 0 <= i < n -- exact knot placement matters since
// CubicSpline's shape (and thus every downstream value) depends on it.
std::vector<double> sampleTimes(double t_end, double sample_dt)
{
    const int n = static_cast<int>(std::round(t_end / sample_dt)) + 1;
    std::vector<double> t(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        t[static_cast<std::size_t>(i)] = i * sample_dt;
    }
    return t;
}
}  // namespace

SplineXY straightLineTarget(double x0, double y0, double speed, double t_end, double sample_dt)
{
    const std::vector<double> t = sampleTimes(t_end, sample_dt);
    std::vector<double> xs(t.size()), ys(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        xs[i] = x0;
        ys[i] = y0 + speed * t[i];
    }
    return SplineXY{CubicSpline(t, xs), CubicSpline(t, ys)};
}

SplineXY circleTarget(double cx, double cy, double radius, double angular_speed, double t_end,
                      double sample_dt)
{
    const std::vector<double> t = sampleTimes(t_end, sample_dt);
    std::vector<double> xs(t.size()), ys(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double theta = angular_speed * t[i];
        xs[i] = cx + radius * std::cos(theta);
        ys[i] = cy + radius * std::sin(theta);
    }
    return SplineXY{CubicSpline(t, xs), CubicSpline(t, ys)};
}

SplineXY zigzagTarget(double x0, double y0, double speed, double amplitude,
                      double period_distance, double t_end, double sample_dt)
{
    const std::vector<double> t = sampleTimes(t_end, sample_dt);
    std::vector<double> xs(t.size()), ys(t.size());
    const double spatial_freq = 2.0 * M_PI / period_distance;
    for (std::size_t i = 0; i < t.size(); ++i) {
        xs[i] = x0 + speed * t[i];
        ys[i] = y0 + amplitude * std::sin(spatial_freq * speed * t[i]);
    }
    return SplineXY{CubicSpline(t, xs), CubicSpline(t, ys)};
}

SplineXY straightLineWithBypassTarget(double x0, double y_start, double speed, double amplitude,
                                       double bypass_y_center, double bypass_half_width,
                                       double t_end, double sample_dt)
{
    const std::vector<double> t = sampleTimes(t_end, sample_dt);
    std::vector<double> xs(t.size()), ys(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double y = y_start + speed * t[i];
        ys[i] = y;

        const double dy = y - bypass_y_center;
        double window = 0.0;
        if (std::abs(dy) < bypass_half_width) {
            // Raised-cosine (Hann) window: 0 at the edges of
            // [-bypass_half_width, bypass_half_width], 1 at the center,
            // zero slope at both edges -- joins the straight line smoothly.
            window = 0.5 * (1.0 + std::cos(M_PI * dy / bypass_half_width));
        }
        xs[i] = x0 + amplitude * window;
    }
    return SplineXY{CubicSpline(t, xs), CubicSpline(t, ys)};
}

namespace {

double interpolateSpeed(const SpeedKeyframes& kf, double t)
{
    if (kf.empty()) return 0.0;
    if (t <= kf.front().first) return kf.front().second;
    if (t >= kf.back().first) return kf.back().second;
    for (std::size_t i = 0; i + 1 < kf.size(); ++i) {
        const double t0 = kf[i].first,     v0 = kf[i].second;
        const double t1 = kf[i + 1].first, v1 = kf[i + 1].second;
        if (t <= t1) {
            const double frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
            return v0 + frac * (v1 - v0);
        }
    }
    return kf.back().second;
}

// Point at arc-length `s` along the polyline through `wp`, clamped to the
// endpoints outside [0, total length].
std::pair<double, double> pointAtArcLength(const std::vector<std::pair<double, double>>& wp,
                                            double s)
{
    if (wp.empty()) return {0.0, 0.0};
    if (wp.size() == 1 || s <= 0.0) return wp.front();

    double remaining = s;
    for (std::size_t i = 0; i + 1 < wp.size(); ++i) {
        const double dx = wp[i + 1].first  - wp[i].first;
        const double dy = wp[i + 1].second - wp[i].second;
        const double seg_len = std::hypot(dx, dy);
        const bool last_segment = (i + 2 == wp.size());
        if (remaining <= seg_len || last_segment) {
            const double frac = seg_len > 0.0 ? std::min(1.0, remaining / seg_len) : 0.0;
            return {wp[i].first + frac * dx, wp[i].second + frac * dy};
        }
        remaining -= seg_len;
    }
    return wp.back();
}

}  // namespace

SplineXY waypointTarget(const std::vector<std::pair<double, double>>& waypoints,
                        const SpeedKeyframes& speed_keyframes, double t_end, double sample_dt)
{
    const std::vector<double> t = sampleTimes(t_end, sample_dt);
    std::vector<double> xs(t.size()), ys(t.size());

    double s = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (i > 0) {
            // Trapezoidal integration of speed(t) -- consistent with
            // sampling speed(t) densely and accumulating arc-length, same
            // spirit as this file's other generators sampling position
            // densely before fitting a spline.
            const double dt     = t[i] - t[i - 1];
            const double v_prev = interpolateSpeed(speed_keyframes, t[i - 1]);
            const double v_cur  = interpolateSpeed(speed_keyframes, t[i]);
            s += 0.5 * (v_prev + v_cur) * dt;
        }
        const auto [x, y] = pointAtArcLength(waypoints, s);
        xs[i] = x;
        ys[i] = y;
    }
    return SplineXY{CubicSpline(t, xs), CubicSpline(t, ys)};
}

}  // namespace argus_follower
