#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <argus_mpc/QuadrotorMpc.hpp>
// Auto-generated from config/quadrotor.yaml — re-run generate_mpc.py to update
#include "argus_params.h"
#include "plant_dynamics.hpp"
#include "TargetSimulator.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mpc {
    constexpr double Ts             = ARGUS_MPC_TS;
    constexpr double T_hover        = ARGUS_T_HOVER;
    constexpr double hover_altitude = ARGUS_HOVER_ALTITUDE;  // drone altitude [m]
}

using PlantState = std::array<double, 12>;
using PlantInput = std::array<double, 4>;

// Generates the drone's own desired position for one horizon stage.
// `t` is the absolute simulation time "now"; `lookahead` is how far past
// `t` this stage is (j * Ts). `target` is the current FOV target sample.
class DroneReference
{
public:
    virtual ~DroneReference() = default;
    virtual argus_mpc::Reference at(double t, double lookahead,
                                     const argus_mpc::TargetState& target) const = 0;
};

// Hover directly above the target, extrapolating its ground position
// forward by the stage's look-ahead time using its current velocity.
class HoverAboveTarget : public DroneReference
{
public:
    explicit HoverAboveTarget(double altitude) : altitude_(altitude) {}

    argus_mpc::Reference at(double /*t*/, double lookahead,
                             const argus_mpc::TargetState& target) const override
    {
        return {target.x + lookahead * target.vx,
                target.y + lookahead * target.vy,
                altitude_};
    }

private:
    double altitude_;
};

// Orbit a (typically static) target at a fixed radius and angular rate.
class OrbitAroundTarget : public DroneReference
{
public:
    OrbitAroundTarget(double radius, double altitude, double period)
        : radius_(radius), altitude_(altitude), omega_(2.0 * M_PI / period) {}

    argus_mpc::Reference at(double t, double lookahead,
                             const argus_mpc::TargetState& target) const override
    {
        const double theta = omega_ * (t + lookahead);
        return {target.x + radius_ * std::cos(theta),
                target.y + radius_ * std::sin(theta),
                altitude_};
    }

private:
    double radius_, altitude_, omega_;
};

// Always returns the same world position, regardless of time, lookahead or
// target motion -- isolates pure attitude (yaw-cone) dynamics for the
// fov_edge sweep scenario by removing position-reference tracking from the
// picture entirely (drone is already sitting at its reference at t=0; the
// only error the MPC has to correct is the initial attitude offset).
class HoldPosition : public DroneReference
{
public:
    explicit HoldPosition(argus_mpc::Reference r) : r_(r) {}
    argus_mpc::Reference at(double, double, const argus_mpc::TargetState&) const override
    {
        return r_;
    }
private:
    argus_mpc::Reference r_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    // ── Parse mode: which TargetSource drives the tracked target ──────────────
    enum class Mode { Circle, MovingTarget, FovEdge };
    Mode mode = Mode::Circle;
    if (argc > 1) {
        if (std::strcmp(argv[1], "moving_target") == 0)
            mode = Mode::MovingTarget;
        else if (std::strcmp(argv[1], "fov_edge") == 0)
            mode = Mode::FovEdge;
        else if (std::strcmp(argv[1], "circle") != 0) {
            std::cerr << "Usage: argus_mpc [circle|moving_target|fov_edge]\n";
            return 1;
        }
    }
    std::cout << "Mode: " << (mode == Mode::Circle ? "circle tracking" :
                               mode == Mode::MovingTarget ? "moving target" :
                               "FOV-edge softplus_beta sweep scenario") << "\n";

    std::unique_ptr<TargetSource>    target;
    std::unique_ptr<DroneReference>  drone_ref;

    // fov_edge: static target directly ahead at the SAME altitude as the
    // drone (no pitch component -- see design_decisions.md's "softplus_beta
    // weight sweep" section), drone held at its initial position (no
    // position error to correct), initial yaw deliberately offset so
    // gamma_0 = fov_half_angle_deg + kFovEdgeMarginDeg (just outside the
    // cone) -- isolates the yaw-correction transient this sweep needs to
    // observe as softplus_beta changes.
    constexpr double kFovEdgeDistance  = 5.0;  // drone->target range [m], arbitrary (gamma is range-independent)
    constexpr double kFovEdgeMarginDeg = 5.0;  // how far outside the cone at t=0

    if (mode == Mode::Circle) {
        target    = std::make_unique<StaticTarget>(
            argus_mpc::TargetState{0.0, 0.0, ARGUS_CIRCLE_Z, 0.0, 0.0, 0.0});
        drone_ref = std::make_unique<OrbitAroundTarget>(
            ARGUS_CIRCLE_R, mpc::hover_altitude, ARGUS_CIRCLE_PERIOD);
    } else if (mode == Mode::MovingTarget) {
        const TargetConfig cfg = load_target_config(ARGUS_CONFIG_DIR "/target.yaml");
        target    = std::make_unique<TargetSimulator>(cfg);
        drone_ref = std::make_unique<HoverAboveTarget>(mpc::hover_altitude);
    } else {
        target    = std::make_unique<StaticTarget>(
            argus_mpc::TargetState{0.0, 0.0, mpc::hover_altitude, 0.0, 0.0, 0.0});
        drone_ref = std::make_unique<HoldPosition>(
            argus_mpc::Reference{-kFovEdgeDistance, 0.0, mpc::hover_altitude});
    }

    argus_mpc::QuadrotorMpc mpc;
    const int N = mpc.horizonLength();

    // ── Initial drone state ────────────────────────────────────────────────────
    const argus_mpc::TargetState t0 = target->state();
    const argus_mpc::Reference   r0 = drone_ref->at(0.0, 0.0, t0);
    const double psi0 = (mode == Mode::FovEdge)
        ? M_PI / 180.0 * (ARGUS_FOV_HALF_ANGLE_DEG + kFovEdgeMarginDeg)
        : 0.0;
    PlantState x0 = {r0.x,  r0.y,  r0.z,
                      0.0,   0.0,   0.0,
                      0.0,   0.0,   psi0,
                      0.0,   0.0,   0.0};

    PlantDynamics plant(x0, mpc::T_hover, ARGUS_MASS,
                        ARGUS_PLANT_WN_RP,  ARGUS_PLANT_ZETA_RP,
                        ARGUS_PLANT_WN_YAW, ARGUS_PLANT_ZETA_YAW);

    // ── Open CSV log ──────────────────────────────────────────────────────────
    fs::create_directories("logs");
    std::ofstream csv("logs/trajectory.csv");
    csv << std::fixed << std::setprecision(6);
    csv << "step,t,"
           "x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,"
           "T,p_cmd,q_cmd,r_cmd,"
           "xt,yt,zt,"       // drone position reference (hover point / orbit point)
           "fx,fy,fz,"       // FOV target position (the tracked target itself)
           "solve_status\n";

    // ── Simulation loop ───────────────────────────────────────────────────────
    // fov_edge only needs to capture one yaw-correction transient, not a
    // full 30s tracking run.
    const double T_sim = (mode == Mode::FovEdge) ? 3.0 : 30.0;
    const int Nsim = static_cast<int>(T_sim / mpc::Ts);

    PlantState x = x0;
    PlantInput u = {mpc::T_hover, 0.0, 0.0, 0.0};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Running " << Nsim << " MPC steps (" << T_sim << " s) ...\n\n";

    for (int step = 0; step < Nsim; ++step)
    {
        const double t = step * mpc::Ts;

        const argus_mpc::State x0_mpc{
            x[0], x[1], x[2],
            x[3], x[4], x[5],
            x[6], x[7], x[8],
            x[9], x[10], x[11],
        };

        // FOV target (model.p) is the tracked target's actual position;
        // the drone's own position reference is mode-specific (hover above
        // it, or orbit around it) via drone_ref.
        const argus_mpc::TargetState tgt = target->step(mpc::Ts);
        std::vector<argus_mpc::Reference> horizon(N);
        for (int j = 0; j < N; ++j)
            horizon[j] = drone_ref->at(t, j * mpc::Ts, tgt);

        const double xt_log = horizon[0].x, yt_log = horizon[0].y, zt_log = horizon[0].z;  // drone ref
        const double fx_log = tgt.x,        fy_log = tgt.y,        fz_log = tgt.z;          // FOV target

        const argus_mpc::Command cmd = mpc.step(x0_mpc, horizon, std::vector<argus_mpc::TargetState>(N, tgt));
        u = {cmd.thrust * mpc::T_hover, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate};

        // ── Log ───────────────────────────────────────────────────────────────
        csv << step << "," << t;
        for (double v : x) csv << "," << v;
        for (double v : u) csv << "," << v;
        csv << "," << xt_log << "," << yt_log << "," << zt_log
            << "," << fx_log << "," << fy_log << "," << fz_log
            << "," << cmd.solve_status << "\n";

        if (step % 40 == 0) {
            const double err = std::hypot(x[0] - xt_log, x[1] - yt_log);
            std::cout << "t=" << std::setw(6) << t << " s  "
                      << "drone=(" << std::setw(7) << x[0] << ", "
                                   << std::setw(7) << x[1] << ")  "
                      << "ref=("   << std::setw(6) << xt_log << ", "
                                   << std::setw(6) << yt_log << ")  "
                      << "xy_err=" << std::setw(6) << err << " m  "
                      << (cmd.solve_status == 0 ? "OK" : "WARN") << "\n";
        }

        // ── Simulate plant one step ───────────────────────────────────────────
        plant.update(u, mpc::Ts);
        x = plant.state();
    }

    csv.close();

    const argus_mpc::SolverTiming timing = mpc.solverTiming();
    std::cout << "\nSolver timing (acados_solve only, " << timing.count << " calls):\n"
              << "  min  = " << timing.min_ms  << " ms\n"
              << "  mean = " << timing.mean_ms << " ms\n"
              << "  max  = " << timing.max_ms  << " ms\n";

    std::cout << "\nDone. Trajectory saved to logs/trajectory.csv\n";
    std::cout << "Visualise with: python3 scripts/animate_xy.py\n";

    return 0;
}
