#include <gtest/gtest.h>
#include <cmath>

#include <argus_follower/TargetMotionSimulator.hpp>

using argus_follower::circleTarget;
using argus_follower::straightLineTarget;
using argus_follower::zigzagTarget;

TEST(TargetMotionSimulator, StraightLineHasConstantXAndLinearY)
{
    const auto target = straightLineTarget(/*x0=*/1.0, /*y0=*/2.0, /*speed=*/3.0, /*t_end=*/5.0);
    for (double t : {0.0, 0.37, 2.5, 4.9}) {
        EXPECT_NEAR(target.x(t), 1.0, 1e-6) << "t=" << t;
        EXPECT_NEAR(target.y(t), 2.0 + 3.0 * t, 1e-6) << "t=" << t;
        EXPECT_NEAR(target.x.derivative(t), 0.0, 1e-6) << "t=" << t;
        EXPECT_NEAR(target.y.derivative(t), 3.0, 1e-6) << "t=" << t;
    }
}

// Not an exact check (the spline only interpolates the circle, it isn't
// one) -- but with a reasonably fine sample_dt the fit error should be
// small, so position must stay near the circle and speed near the
// constant tangential speed radius*angular_speed.
TEST(TargetMotionSimulator, CircleStaysNearRadiusAtConstantSpeed)
{
    constexpr double kRadius = 5.0;
    constexpr double kAngularSpeed = 0.5;
    const auto target = circleTarget(/*cx=*/0.0, /*cy=*/0.0, kRadius, kAngularSpeed,
                                      /*t_end=*/20.0, /*sample_dt=*/0.01);

    for (double t : {0.0, 1.3, 5.0, 10.7, 19.5}) {
        const double x = target.x(t);
        const double y = target.y(t);
        const double r = std::hypot(x, y);
        EXPECT_NEAR(r, kRadius, 1e-3) << "t=" << t;

        const double vx = target.x.derivative(t);
        const double vy = target.y.derivative(t);
        const double speed = std::hypot(vx, vy);
        EXPECT_NEAR(speed, kRadius * kAngularSpeed, 1e-2) << "t=" << t;
    }
}

TEST(TargetMotionSimulator, CircleStartsAtAngleZero)
{
    const auto target = circleTarget(0.0, 0.0, 5.0, 0.5, 20.0);
    EXPECT_NEAR(target.x(0.0), 5.0, 1e-6);
    EXPECT_NEAR(target.y(0.0), 0.0, 1e-6);
}

// x must advance monotonically (it's a pure constant-speed sweep along
// +x); y must stay bounded within the configured oscillation amplitude.
TEST(TargetMotionSimulator, ZigzagAdvancesMonotonicallyInXAndOscillatesBoundedInY)
{
    constexpr double kAmplitude = 1.5;
    const auto target = zigzagTarget(/*x0=*/0.0, /*y0=*/0.0, /*speed=*/2.0, kAmplitude,
                                      /*period_distance=*/5.0, /*t_end=*/20.0);

    double prev_x = target.x(0.0);
    for (double t = 0.1; t <= 20.0; t += 0.1) {
        const double x = target.x(t);
        EXPECT_GT(x, prev_x) << "t=" << t;
        prev_x = x;

        const double y = target.y(t);
        EXPECT_LE(std::abs(y), kAmplitude + 1e-2) << "t=" << t;
    }
}
