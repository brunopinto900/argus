#pragma once

#include <memory>
#include <vector>

// Public API for argus's acados-generated quadrotor MPC solver.
//
// This header has no acados dependency — it's safe to include from any
// consumer (e.g. Vista-Tracker) without pulling in acados include paths.
// Only QuadrotorMpc.cpp (built as part of this library) touches the raw
// solver. Consumers only need to link the `quadrotor_mpc` CMake target.
namespace argus_mpc {

// State layout matches the OCP's state vector exactly:
// [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r].
struct State
{
    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double phi = 0.0, theta = 0.0, psi = 0.0;  // roll, pitch, yaw (rad)
    double p = 0.0, q = 0.0, r = 0.0;          // body rates (rad/s)
};

// One horizon stage's position/velocity target. Attitude, FOV-cone, and
// thrust/rate references are fixed internally (level attitude, target
// inside the FOV cone, hover thrust, zero rates) — only position tracking
// is exposed to callers today.
struct Reference
{
    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
};

// First-stage optimal input, ready to command a PX4-style body-rate
// interface. thrust is normalised so 1.0 == hover (already divided by
// T_hover) — callers never need the solver's internal hover-thrust constant.
struct Command
{
    double thrust     = 1.0;
    double roll_rate  = 0.0;
    double pitch_rate = 0.0;
    double yaw_rate   = 0.0;
    int    solve_status = 0;  // acados status; 0 == success
};

class QuadrotorMpc
{
public:
    QuadrotorMpc();
    ~QuadrotorMpc();

    QuadrotorMpc(const QuadrotorMpc&)            = delete;
    QuadrotorMpc& operator=(const QuadrotorMpc&) = delete;

    // Number of horizon stages (N) the solver was generated with. `step()`
    // requires `horizon.size() == horizonLength()`.
    int horizonLength() const;

    // Solve one MPC step (SQP-RTI: one iteration) from measured state x0
    // against the given per-stage horizon, and return the first-stage
    // optimal input. Throws std::invalid_argument if horizon.size() !=
    // horizonLength().
    Command step(const State& x0, const std::vector<Reference>& horizon);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace argus_mpc
