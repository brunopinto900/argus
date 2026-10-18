#pragma once

#include <utility>

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

// Straight-line advance along +y at constant speed (same as
// straightLineTarget), with one lateral offset bump added instead of a
// periodic zigzag. x(t) = x0 + amplitude * w(y(t)), where w is a
// raised-cosine ("Hann") window: 0 outside
// [bypass_y_center - bypass_half_width, bypass_y_center + bypass_half_width],
// rising smoothly to 1 at bypass_y_center and back to 0 -- zero slope at
// both ends of the window, so the bump joins the straight line with no
// heading discontinuity. Smaller bypass_half_width (relative to amplitude)
// gives a tighter bump.
SplineXY straightLineWithBypassTarget(double x0, double y_start, double speed, double amplitude,
                                       double bypass_y_center, double bypass_half_width,
                                       double t_end, double sample_dt = 0.05);

// (t, speed) keyframes -- speed is linearly interpolated between
// consecutive keyframes (so each segment's acceleration is just its
// slope), the same "specify key values, interpolate" philosophy as the
// rest of this file, applied to speed instead of position directly. Must
// start at t=0 and be sorted by t.
using SpeedKeyframes = std::vector<std::pair<double, double>>;

// Piecewise-linear path through `waypoints` ((x,y) corners, world frame),
// traversed at a speed given by `speed_keyframes` -- distance covered by
// time t is the integral of speed(t) from 0 to t. Lets a path's shape and
// its pacing (accelerate/cruise/decelerate/stop, slow through a tight
// passage, etc.) be designed independently, unlike the constant-speed
// generators above. Holds at the final waypoint if the speed profile
// would outrun the path's total length before t_end.
SplineXY waypointTarget(const std::vector<std::pair<double, double>>& waypoints,
                        const SpeedKeyframes& speed_keyframes, double t_end,
                        double sample_dt = 0.05);

}  // namespace argus_follower
