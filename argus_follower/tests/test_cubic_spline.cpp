#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <argus_follower/CubicSpline.hpp>

using argus_follower::CubicSpline;

// Interpolation property: regardless of curvature, the spline must pass
// through every knot exactly -- the most basic thing that could be wrong
// in a hand-rolled implementation (e.g. an off-by-one in segment lookup or
// a coefficient sign error).
TEST(CubicSpline, PassesThroughKnotsExactly)
{
    const std::vector<double> x = {0.0, 0.3, 1.0, 1.4, 2.0, 3.5};
    std::vector<double> y;
    for (double xi : x) y.push_back(std::sin(xi) + 0.2 * xi * xi);

    CubicSpline spline(x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(spline(x[i]), y[i], 1e-9) << "knot " << i;
    }
}

TEST(CubicSpline, RejectsInvalidConstruction)
{
    EXPECT_THROW(CubicSpline({0.0}, {0.0}), std::invalid_argument);                 // < 2 points
    EXPECT_THROW(CubicSpline({0.0, 1.0}, {0.0, 1.0, 2.0}), std::invalid_argument);  // size mismatch
    EXPECT_THROW(CubicSpline({0.0, 1.0, 0.5}, {0.0, 1.0, 2.0}), std::invalid_argument);  // not increasing
    EXPECT_THROW(CubicSpline({0.0, 0.0, 1.0}, {0.0, 1.0, 2.0}), std::invalid_argument);  // duplicate knot
}
