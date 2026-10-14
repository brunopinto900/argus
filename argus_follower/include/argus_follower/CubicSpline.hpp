#pragma once

#include <vector>

// Minimal natural cubic spline (second derivative = 0 at both endpoints).
// Matches the two operations the Python prototype used on a
// scipy.interpolate.CubicSpline: evaluate a position with operator(), and
// its first derivative with derivative(nu=1). Evaluating outside the knot
// range extrapolates using the boundary segment's own cubic polynomial,
// same as scipy's default extrapolate=True.
//
// Note: scipy.interpolate.CubicSpline's own default boundary condition is
// 'not-a-knot', not 'natural' -- the two only differ in the first/last
// segment's exact curvature, but if the Python prototype relied on that
// default, values very close to a spline's first/last knot won't match
// this port bit-for-bit. Pass bc_type='natural' on the Python side to
// reproduce this port's numbers exactly (done when generating the test
// fixtures under argus_follower/tests/).
namespace argus_follower {

class CubicSpline
{
public:
    // x must be strictly increasing and have at least 2 points; y must be
    // the same size.
    CubicSpline(std::vector<double> x, std::vector<double> y);

    double operator()(double t) const;
    double derivative(double t) const;

private:
    struct Segment
    {
        double x0 = 0.0;              // left knot
        double a = 0.0, b = 0.0, c = 0.0, d = 0.0;  // y = a + b*dx + c*dx^2 + d*dx^3, dx = t-x0
    };

    const Segment& segmentFor(double t) const;

    std::vector<double> x_;
    std::vector<Segment> segments_;
};

}  // namespace argus_follower
