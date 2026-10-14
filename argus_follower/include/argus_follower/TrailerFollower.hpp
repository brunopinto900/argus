#pragma once

#include <vector>

#include <argus_follower/CubicSpline.hpp>

// Trailer-like leader-follower reference generator, ported directly from a
// Python prototype (getGoalPointTrailerList). The follower's goal point
// behaves like a physical trailer towed behind the moving target at a
// fixed hitch distance -- rather than chasing the target's raw position --
// which keeps it from cutting corners on a curved target path.
namespace argus_follower {

// One call's N-step horizon: desired follower position (PDx/PDy) and
// velocity (VFx/VFy) at each predicted step, index 0 being the step closest
// to the call's own t.
struct TrailerHorizon
{
    std::vector<double> PDx;
    std::vector<double> PDy;
    std::vector<double> VFx;
    std::vector<double> VFy;
};

class TrailerFollower
{
public:
    // distance_to_target: fixed hitch distance between the target (leader)
    // and the follower -- the Python prototype's DIST_2_TARGET.
    // heading_gain: lambda, how fast the follower's heading tracks the
    // target's velocity direction -- the prototype's hardcoded lambda_ = 1.0.
    // initial_heading: starting value of the persistent heading-angle
    // state (the prototype's global a_list), radians.
    explicit TrailerFollower(double distance_to_target, double heading_gain = 1.0,
                              double initial_heading = 0.0);

    // Predicts the follower's goal position/velocity over an N-step horizon
    // starting at time t with step dt, from the target's (x, y) splines
    // (position via operator(), velocity via derivative()).
    //
    // Only the first of the N predicted steps advances this object's
    // persistent heading state; the remaining N-1 are purely predictive
    // and are discarded once the call returns -- matching the prototype's
    // own "reset a_list before leaving" behaviour, so repeated calls at
    // consecutive t integrate the heading correctly regardless of N.
    TrailerHorizon computeGoalPoints(const CubicSpline& splineX, const CubicSpline& splineY,
                                      int N, double t, double dt);

    double headingAngle() const { return heading_angle_; }

private:
    double distance_to_target_;
    double heading_gain_;
    double heading_angle_;
};

}  // namespace argus_follower
