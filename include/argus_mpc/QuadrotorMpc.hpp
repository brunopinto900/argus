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
//
// azimuth_ux/azimuth_uy: desired horizontal DIRECTION from the leader to
// the drone (unit vector, world frame) -- e.g. "behind" means pointing
// opposite the leader's heading. Feeds r_azimuth, decoupled from x/y/z
// above: this is direction only, not a position, so it can't pull the
// drone toward a specific (possibly obstacle-colliding) point the way
// x/y/z historically did (see design_decisions.md's collision-clearance
// analysis -- same reasoning that split distance/elevation out of the
// old single trailer-point tracking term). Default (0,0) makes
// r_azimuth a harmless constant with zero gradient -- safe for any
// caller that doesn't set it.
struct Reference
{
    double x = 0.0, y = 0.0, z = 0.0;
    double azimuth_ux = 0.0, azimuth_uy = 0.0;
};

// Moving target state injected before every solve.
// Position feeds model.p (FOV cost); velocity sets the drone's velocity yref
// so the MPC is incentivised to match the target's speed.
struct TargetState
{
    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
};

// One horizon stage's local affine approximation of an ESDF around a
// linearization point p0 (planar: x,y only): distance(p) ~= distance +
// grad . (p - p0). This can't be the real nonlinear ESDF lookup itself --
// that's runtime voxel-grid data with no closed form the OCP code
// generator can differentiate at build time (see whiteboard.md). The
// caller computes this from e.g. argus_esdf::VoxelGrid::query() once per
// stage, before every solve.
//
// Default (huge distance, zero gradient) is a no-op: the collision hinge
// cost never activates, so passing no obstacle info at all (or leaving
// any particular stage at its default) is always safe, never spuriously
// penalized.
struct ObstacleLinearization
{
    double p0_x = 0.0, p0_y = 0.0;
    double distance = 1.0e6;
    double grad_x = 0.0, grad_y = 0.0;
};

// One horizon stage's local affine approximation of the ESDF's MINIMUM
// along the drone->target camera sightline (the worst occlusion point),
// NOT at the drone's own position like ObstacleLinearization above --
// feeds the occlusion-aware cost (config/quadrotor.yaml's
// occlusion_avoidance.mode), see design_decisions.md's "Occlusion-aware
// cost term" planning notes for the full derivation.
//
// distance/grad_x/grad_y: ESDF value and gradient at the worst point
// found by marching the sightline (same march occlusionClearance() in
// examples/online_mapping_demo.cpp already does for the metric, just
// against own_map's belief instead of ground truth, and also keeping
// the gradient and s_star it currently throws away).
//
// s_star in [0,1]: how far along the drone->target segment that worst
// point sits (0 = right at the drone, 1 = right at the target). The
// worst point is occ_p0 = (1-s_star)*drone_p0 + s_star*p_target -- an
// affine blend, not the drone's own position -- so moving the drone by
// delta only moves occ_p0 by (1-s_star)*delta (chain rule; the target
// end of the segment doesn't move with the drone). QuadrotorMpc::step()
// pre-multiplies grad_x/grad_y by (1-s_star) before handing them to the
// solver as model.p -- the OCP's symbolic graph just does a plain dot
// product against the drone's own linearization point (reused from
// ObstacleLinearization's p0, see generate_mpc.py), same shape as
// d_lin.
//
// Default (huge distance, zero gradient, s_star=0) is a no-op, same
// convention as ObstacleLinearization: the occlusion hinge cost never
// activates, so passing no occlusion info at all (or leaving any stage
// at its default) is always safe.
struct OcclusionLinearization
{
    double distance = 1.0e6;
    double grad_x = 0.0, grad_y = 0.0;
    double s_star = 0.0;
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
    // Largest collision-avoidance slack value (the "s" in d_lin >= d_safe
    // - s) across every stage of the just-solved horizon. Zero means the
    // constraint was satisfied everywhere in the plan with no help from
    // the slack; nonzero means the solver found at least one stage where
    // d_safe was genuinely unreachable (not a tuning artifact -- the
    // slack is an exact penalty, see design_decisions.md) and had to
    // relax it there instead of returning an infeasible solve. Log this
    // per tick -- nonzero ticks are real conflicts worth looking at.
    double max_obstacle_slack = 0.0;
    // Inner HPIPM iterations the QP subproblem took this tick ("qp_iter"
    // via ocp_nlp_get -- see QuadrotorMpc.cpp). SQP_RTI does exactly one
    // outer Newton-type step per step() call, so this is a direct per-
    // tick measure of QP-solve work, independent of wall-clock noise --
    // logged to isolate whether the slack constraint's solve-time cost is
    // from the QP actually doing more iterations (structural -- a bigger
    // problem to solve) vs. from each iteration costing more for some
    // other reason (e.g. conditioning). See design_decisions.md.
    int qp_iter = 0;
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
    // target: one TargetState per horizon stage (same size as horizon) --
    // target[k].{x,y,z} → stage k's model.p (FOV cost); target[k].{vx,vy,vz}
    // → stage k's velocity yref. Per-stage, not a single shared value,
    // since the FOV/bearing cost at stage k is evaluated against the
    // drone's *predicted* pose there -- a target held fixed at its
    // current-tick position becomes systematically stale by the time the
    // horizon reaches later stages, exactly the same staleness pattern as
    // obstacle (below). Callers tracking a known target trajectory should
    // propagate target[k] to the target's actual predicted position/
    // velocity at that stage's time (t + k*Ts), not just repeat one value;
    // callers with no better model (e.g. a momentarily-static assumption)
    // can still pass the same TargetState N times. target.size() must
    // equal horizonLength().
    // obstacle: one ObstacleLinearization per horizon stage (same size as
    // horizon), or empty to disable collision avoidance entirely (every
    // stage uses the no-op default). obstacle.size() must equal
    // horizonLength() if non-empty.
    // occlusion: one OcclusionLinearization per horizon stage, or empty
    // to disable the occlusion-aware cost entirely (every stage uses the
    // no-op default) -- safe to leave empty even when
    // occlusion_avoidance.mode='cost' was baked in at codegen time, same
    // "no-op, never spuriously penalized" convention as obstacle above.
    // occlusion.size() must equal horizonLength() if non-empty.
    Command step(const State& x0, const std::vector<Reference>& horizon,
                 const std::vector<TargetState>& target,
                 const std::vector<ObstacleLinearization>& obstacle = {},
                 const std::vector<OcclusionLinearization>& occlusion = {});

    // Accumulated acados solve-time stats since construction (or since the
    // last resetSolverTiming() call).
    SolverTiming solverTiming() const;
    void resetSolverTiming();

    // Wall-clock time of the most recent step() call's acados_solve(), in
    // milliseconds. 0 before the first call. For per-step logging (e.g. a
    // trajectory CSV's solve_time_ms column) -- solverTiming() only exposes
    // the run-long aggregate.
    double lastSolveTimeMs() const;

    // The full predicted horizon (stages 0..N, so horizonLength()+1 states)
    // from the most recent step() call -- the open-loop plan the solver
    // actually optimized, not just the single realized closed-loop state a
    // caller normally sees per tick. Empty before the first step() call.
    // For diagnosing whether a closed-loop constraint violation (e.g.
    // obstacle clearance) was already present in the solver's own plan, or
    // only appeared after low-level tracking/plant dynamics diverged from
    // it -- see design_decisions.md's collision-clearance analysis.
    std::vector<State> lastPredictedHorizon() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace argus_mpc
