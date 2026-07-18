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

// One horizon stage's position target for the drone.
// vx/vy/vz are unused — velocity reference is taken from TargetState.
struct Reference
{
    double x = 0.0, y = 0.0, z = 0.0;
};

// Moving target state injected before every solve.
// Position feeds model.p (FOV cost); velocity sets the drone's velocity yref
// so the MPC is incentivised to match the target's speed.
struct TargetState
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

// Wall-clock timing of the acados solve call itself (i.e. just
// quadrotor_acados_solve()), accumulated across every step() call made so
// far on this instance. Excludes state/yref packing/unpacking — this is the
// number that matters for "can this run in real time at Ts".
struct SolverTiming
{
    double min_ms  = 0.0;
    double max_ms  = 0.0;
    double mean_ms = 0.0;
    long   count   = 0;
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

    // Solve one MPC step (SQP-RTI: one iteration).
    // horizon.size() must equal horizonLength(); throws std::invalid_argument otherwise.
    // target.{x,y,z} → model.p (FOV cost).  target.{vx,vy,vz} → velocity yref.
    Command step(const State& x0, const std::vector<Reference>& horizon,
                 const TargetState& target);

    // Accumulated acados solve-time stats since construction (or since the
    // last resetSolverTiming() call).
    SolverTiming solverTiming() const;
    void resetSolverTiming();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace argus_mpc
