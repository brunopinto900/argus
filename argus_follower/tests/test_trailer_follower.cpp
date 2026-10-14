#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <argus_follower/CubicSpline.hpp>
#include <argus_follower/TargetMotionSimulator.hpp>
#include <argus_follower/TrailerFollower.hpp>

using argus_follower::SplineXY;
using argus_follower::TrailerFollower;

namespace {
constexpr double kDt = 0.05;
constexpr double kDistToTarget = 1.0;
}  // namespace

TEST(TrailerFollower, HorizonHasNStepsForEachOutputList)
{
    const SplineXY target = argus_follower::straightLineTarget(0.0, 0.0, 1.0, 5.0, kDt);
    TrailerFollower follower(kDistToTarget);

    const auto horizon = follower.computeGoalPoints(target.x, target.y, /*N=*/7, 0.0, kDt);
    EXPECT_EQ(horizon.PDx.size(), 7u);
    EXPECT_EQ(horizon.PDy.size(), 7u);
    EXPECT_EQ(horizon.VFx.size(), 7u);
    EXPECT_EQ(horizon.VFy.size(), 7u);
}

TEST(TrailerFollower, RejectsNonPositiveHorizonLength)
{
    const SplineXY target = argus_follower::straightLineTarget(0.0, 0.0, 1.0, 5.0, kDt);
    TrailerFollower follower(kDistToTarget);
    EXPECT_THROW(follower.computeGoalPoints(target.x, target.y, 0, 0.0, kDt),
                 std::invalid_argument);
    EXPECT_THROW(follower.computeGoalPoints(target.x, target.y, -3, 0.0, kDt),
                 std::invalid_argument);
}

// The horizon's first predicted instant is always exactly t (the Python
// prototype's own np.linspace(t, t+dt*N, N) starts at t), so the follower's
// goal point there must sit exactly distance_to_target away from the
// leader's actual position at t -- this is the trailer-hitch geometry the
// whole algorithm is built around, checked independently of any internal
// implementation detail.
TEST(TrailerFollower, FirstGoalPointIsExactlyHitchDistanceFromLeader)
{
    const SplineXY target = argus_follower::circleTarget(0.0, 0.0, 5.0, 0.3, 20.0, kDt);
    TrailerFollower follower(kDistToTarget);

    for (double t : {0.0, 1.0, 3.7}) {
        const auto horizon = follower.computeGoalPoints(target.x, target.y, 15, t, kDt);
        const double leader_x = target.x(t);
        const double leader_y = target.y(t);
        const double dist = std::hypot(leader_x - horizon.PDx[0], leader_y - horizon.PDy[0]);
        EXPECT_NEAR(dist, kDistToTarget, 1e-9) << "t=" << t;
    }
}

// Only the first of the N predicted steps should advance the persistent
// heading state (the rest of the horizon is discarded) -- so the state
// after one call, and that call's own first-step outputs, must not depend
// on how long a horizon N was requested.
TEST(TrailerFollower, FirstStepResultIsIndependentOfHorizonLength)
{
    const SplineXY target = argus_follower::zigzagTarget(0.0, 0.0, 1.0, 1.0, 5.0, 20.0, kDt);

    TrailerFollower short_horizon(kDistToTarget);
    TrailerFollower long_horizon(kDistToTarget);

    const auto h_short = short_horizon.computeGoalPoints(target.x, target.y, 3, 2.0, kDt);
    const auto h_long = long_horizon.computeGoalPoints(target.x, target.y, 50, 2.0, kDt);

    EXPECT_NEAR(h_short.PDx[0], h_long.PDx[0], 1e-9);
    EXPECT_NEAR(h_short.PDy[0], h_long.PDy[0], 1e-9);
    EXPECT_NEAR(h_short.VFx[0], h_long.VFx[0], 1e-9);
    EXPECT_NEAR(h_short.VFy[0], h_long.VFy[0], 1e-9);
    EXPECT_NEAR(short_horizon.headingAngle(), long_horizon.headingAngle(), 1e-9);
}

// The heading state must persist (and keep integrating) across calls --
// two sequential one-step calls should land somewhere different from a
// single call made directly at the second call's time, starting fresh.
TEST(TrailerFollower, HeadingStatePersistsAcrossCalls)
{
    const SplineXY target = argus_follower::circleTarget(0.0, 0.0, 5.0, 0.5, 20.0, kDt);

    TrailerFollower stepped(kDistToTarget);
    stepped.computeGoalPoints(target.x, target.y, 10, 0.0, kDt);
    stepped.computeGoalPoints(target.x, target.y, 10, kDt, kDt);
    const double stepped_heading = stepped.headingAngle();

    TrailerFollower fresh(kDistToTarget);
    fresh.computeGoalPoints(target.x, target.y, 10, kDt, kDt);
    const double fresh_heading = fresh.headingAngle();

    EXPECT_GT(std::abs(stepped_heading - fresh_heading), 1e-6);
}

// For a leader moving in a straight line along +x, this model's own
// kinematics (WT = -sin(a)*VTx + cos(a)*VTy, linearised about a=0 for
// VTy=0 gives da/dt ~= -VTx*a) has a=0 as the locally stable equilibrium
// heading -- i.e. the follower should settle directly behind the leader
// along its line of travel. This is a property of the control law itself,
// not a copied reference number.
TEST(TrailerFollower, HeadingConvergesBehindAStraightLineLeader)
{
    const double speed = 1.0;
    // straightLineTarget (TargetMotionSimulator) advances along +y; this
    // test wants +x instead, so the equilibrium heading (a=0) has a simple
    // closed form to check against -- built by hand here rather than via
    // that helper.
    std::vector<double> t, xs, ys;
    for (int i = 0; i <= 600; ++i) {
        const double ti = i * kDt;
        t.push_back(ti);
        xs.push_back(speed * ti);
        ys.push_back(0.0);
    }
    const argus_follower::CubicSpline splineX(t, xs);
    const argus_follower::CubicSpline splineY(t, ys);

    TrailerFollower follower(kDistToTarget, /*heading_gain=*/1.0, /*initial_heading=*/0.3);  // off-equilibrium start
    double heading = follower.headingAngle();
    for (int step = 0; step < 400; ++step) {
        follower.computeGoalPoints(splineX, splineY, 10, step * kDt, kDt);
        heading = follower.headingAngle();
    }
    EXPECT_NEAR(heading, 0.0, 1e-3);
}

// Once the heading has settled (previous test), the follower's own
// commanded velocity (VFx/VFy) must match the leader's velocity, not
// oppose it -- maintaining a constant hitch distance requires the
// follower to move at the same velocity as the thing it's a fixed
// distance behind. Regression test for a sign bug where the follower's
// velocity converged to the leader's speed with the wrong sign.
TEST(TrailerFollower, FollowerVelocityConvergesToLeaderVelocity)
{
    const double speed = 1.0;
    const SplineXY target = argus_follower::straightLineTarget(0.0, 0.0, speed, 30.0, kDt);
    TrailerFollower follower(kDistToTarget);

    argus_follower::TrailerHorizon horizon;
    for (int step = 0; step < 400; ++step) {
        horizon = follower.computeGoalPoints(target.x, target.y, 10, step * kDt, kDt);
    }
    EXPECT_NEAR(horizon.VFx[0], 0.0, 1e-3);
    EXPECT_NEAR(horizon.VFy[0], speed, 1e-3);
}
