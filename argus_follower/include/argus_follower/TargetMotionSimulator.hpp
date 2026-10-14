#pragma once

#include <argus_follower/CubicSpline.hpp>

// Synthetic target-motion generators for exercising TrailerFollower without
// Gazebo: each produces densely-sampled (t, x, y) waypoints over a time
// window, then fits a CubicSpline pair to them -- the same
// (position, velocity-via-derivative) interface
// TrailerFollower::computeGoalPoints expects from a real tracking pipeline
// (e.g. a target-motion planner's published path).
namespace argus_follower {

struct SplineXY
{
    CubicSpline x;
    CubicSpline y;
};

// Straight line along +y at constant speed, starting at (x0, y0).
SplineXY straightLineTarget(double x0, double y0, double speed, double t_end,
                             double sample_dt = 0.05);

// Circle of the given radius centered at (cx, cy), constant angular speed
// (rad/s), starting at angle 0 (i.e. at (cx + radius, cy)).
SplineXY circleTarget(double cx, double cy, double radius, double angular_speed, double t_end,
                      double sample_dt = 0.05);

// Smooth zigzag: straight-line advance along +x at constant speed, with a
// sinusoidal lateral offset along y. amplitude and period_distance (metres
// of x-travel per full oscillation) control how sharp the zigzag is.
// Smooth (C-infinity) by construction, unlike a piecewise-linear zigzag.
SplineXY zigzagTarget(double x0, double y0, double speed, double amplitude,
                      double period_distance, double t_end, double sample_dt = 0.05);

}  // namespace argus_follower
