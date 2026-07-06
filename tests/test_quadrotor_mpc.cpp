#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <argus_mpc/QuadrotorMpc.hpp>

using argus_mpc::Command;
using argus_mpc::QuadrotorMpc;
using argus_mpc::Reference;
using argus_mpc::State;
using argus_mpc::TargetState;

namespace {
std::vector<Reference> constantHorizon(const Reference& ref, int n)
{
    return std::vector<Reference>(static_cast<std::size_t>(n), ref);
}
}  // namespace

// At rest exactly at the reference (level, hover), the solver should command
// close to hover thrust and near-zero body rates. Loose tolerances: this is
// the result of a QP solve, not hand-calculable like a PID gain.
//
// The state is placed on the -x side of the world origin, nose level along
// +x: this is not arbitrary — the OCP has a FOV-cone cost (r_fov) that tries
// to point the body +x boresight at the target (passed as model.p at runtime).
// Placing the drone here means "hover in place" and "keep the origin in view"
// are simultaneously satisfied at near-zero rates; placing it elsewhere (e.g.
// directly above the origin) makes the solver command a large attitude change
// to re-aim the camera, which is expected, not a bug.
TEST(QuadrotorMpc, HoverAtZeroError)
{
    QuadrotorMpc mpc;
    State     s{};  s.x = -6.0;  // on the boresight axis looking at origin
    Reference r{};  r.x = s.x; r.y = s.y; r.z = s.z;
    TargetState tgt{};  // target at world origin

    const Command cmd = mpc.step(s, constantHorizon(r, mpc.horizonLength()), tgt);

    EXPECT_EQ(cmd.solve_status, 0);
    EXPECT_NEAR(cmd.thrust,     1.0, 0.05);
    EXPECT_NEAR(cmd.roll_rate,  0.0, 0.2);
    EXPECT_NEAR(cmd.pitch_rate, 0.0, 0.2);
    EXPECT_NEAR(cmd.yaw_rate,   0.0, 0.2);
}

// A reference ahead in +x (world frame, zero yaw = nose along +x) must
// produce a positive pitch-rate command to start tilting the vehicle
// forward — sign only, since exact magnitude depends on the solver's
// internal iterate count, not something to hand-calculate here.
// Same FOV-friendly starting pose as HoverAtZeroError (see its comment) so
// the pitch sign reflects the position error, not the FOV-cone pull.
TEST(QuadrotorMpc, ForwardReferenceProducesForwardPitch)
{
    QuadrotorMpc mpc;
    State     s{};  s.x = -6.0;
    Reference r{};  r.x = s.x + 3.0; r.y = s.y; r.z = s.z;
    TargetState tgt{};  // target at world origin

    Command cmd{};
    for (int i = 0; i < 5; ++i)
        cmd = mpc.step(s, constantHorizon(r, mpc.horizonLength()), tgt);

    EXPECT_TRUE(std::isfinite(cmd.pitch_rate));
    EXPECT_GT(cmd.pitch_rate, 0.0);
}

// horizon.size() must match horizonLength() exactly — a mismatched caller
// (e.g. after the generated model's N changes) should fail loudly, not
// silently read out of bounds or ignore extra/missing stages.
TEST(QuadrotorMpc, WrongHorizonSizeThrows)
{
    QuadrotorMpc mpc;
    State s{};
    Reference r{};
    TargetState tgt{};
    EXPECT_THROW(mpc.step(s, constantHorizon(r, mpc.horizonLength() - 1), tgt), std::invalid_argument);
}
