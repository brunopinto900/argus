#include <argus_follower/TargetMotionSimulator.hpp>

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

}  // namespace argus_follower
