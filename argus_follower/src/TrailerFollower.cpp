#include <argus_follower/TrailerFollower.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace argus_follower {

namespace {
// N points inclusive of both ends, spanning [start, stop] -- matches the
// Python prototype's np.linspace(t, t+dt*N, N) exactly, including its
// step being slightly larger than dt (stop-start)/(N-1), not dt itself.
std::vector<double> linspaceInclusive(double start, double stop, int n)
{
    std::vector<double> out(static_cast<std::size_t>(n));
    if (n == 1) {
        out[0] = start;
        return out;
    }
    const double step = (stop - start) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = start + step * i;
    }
    return out;
}
}  // namespace

TrailerFollower::TrailerFollower(double distance_to_target, double initial_heading)
    : distance_to_target_(distance_to_target), heading_angle_(initial_heading)
{
}

TrailerHorizon TrailerFollower::computeGoalPoints(const CubicSpline& splineX,
                                                   const CubicSpline& splineY, int N, double t,
                                                   double dt)
{
    if (N <= 0) {
        throw std::invalid_argument("TrailerFollower::computeGoalPoints: N must be positive");
    }

    const double d = distance_to_target_;
    constexpr double kLambda = 1.0;

    TrailerHorizon horizon;
    horizon.PDx.reserve(static_cast<std::size_t>(N));
    horizon.PDy.reserve(static_cast<std::size_t>(N));
    horizon.VFx.reserve(static_cast<std::size_t>(N));
    horizon.VFy.reserve(static_cast<std::size_t>(N));

    const std::vector<double> taus = linspaceInclusive(t, t + dt * N, N);

    double a = heading_angle_;
    double one_step_angle = a;
    bool first = true;

    for (const double tau : taus) {
        const double PLx = splineX(tau);
        const double PLy = splineY(tau);
        const double VTx = splineX.derivative(tau);
        const double VTy = splineY.derivative(tau);

        // Both WT and VF below use this pre-update heading -- matching the
        // prototype's rotation matrix R not being recomputed after a_list
        // is reassigned mid-iteration.
        const double cos_a = std::cos(a);
        const double sin_a = std::sin(a);

        // e2 . R^T . VT: component of the target's velocity perpendicular
        // to the follower's current heading -- the trailer's angular rate.
        const double WT = -sin_a * VTx + cos_a * VTy;

        const double a_new = a + kLambda * WT * dt;

        // e1 . R . VT + d*WT: follower speed along its own heading.
        const double VF = cos_a * VTx - sin_a * VTy + d * WT;

        const double cos_a_new = std::cos(a_new);
        const double sin_a_new = std::sin(a_new);

        horizon.VFx.push_back(VF * cos_a_new);
        horizon.VFy.push_back(VF * sin_a_new);

        horizon.PDx.push_back(PLx - d * cos_a_new);
        horizon.PDy.push_back(PLy - d * sin_a_new);

        if (first) {
            one_step_angle = a_new;
            first = false;
        }
        a = a_new;
    }

    // Only the first step's heading update is a real state advance; the
    // rest of the horizon was purely predictive -- matches the prototype
    // resetting its global a_list before returning.
    heading_angle_ = one_step_angle;

    return horizon;
}

}  // namespace argus_follower
