#include "plant_dynamics.hpp"
#include <cmath>

PlantDynamics::PlantDynamics(const State& x0, double T0, double mass,
                              double wn_rp,  double zeta_rp,
                              double wn_yaw, double zeta_yaw,
                              double g)
    : x_(x0)
    , T_actual_(T0)
    , mass_(mass)
    , wn_rp_(wn_rp),   zeta_rp_(zeta_rp)
    , wn_yaw_(wn_yaw), zeta_yaw_(zeta_yaw)
    , g_(g)
{}

void PlantDynamics::step2(double cmd, double& val, double& val_dot,
                          double wn2, double damp, double dt_sub)
{
    const double ddot = wn2 * (cmd - val) - damp * val_dot;
    val_dot += ddot    * dt_sub;
    val     += val_dot * dt_sub;
}

void PlantDynamics::update(const Input& u, double dt)
{
    const double T_cmd  = u[0];
    const double p_cmd  = u[1];
    const double q_cmd  = u[2];
    const double r_cmd  = u[3];
    const double dt_sub = dt / kSubSteps;

    double& p = x_[9];
    double& q = x_[10];
    double& r = x_[11];

    const double wn2_rp   = wn_rp_  * wn_rp_;
    const double damp_rp  = 2.0 * zeta_rp_  * wn_rp_;
    const double wn2_yaw  = wn_yaw_ * wn_yaw_;
    const double damp_yaw = 2.0 * zeta_yaw_ * wn_yaw_;

    // ── Second-order inner-loop (sub-stepped for numerical stability) ──────
    for (int i = 0; i < kSubSteps; ++i) {
        step2(p_cmd, p,         p_dot_, wn2_rp,  damp_rp,  dt_sub);
        step2(q_cmd, q,         q_dot_, wn2_rp,  damp_rp,  dt_sub);
        step2(r_cmd, r,         r_dot_, wn2_yaw, damp_yaw, dt_sub);
        step2(T_cmd, T_actual_, T_dot_, wn2_rp,  damp_rp,  dt_sub);
    }

    // ── Precompute trig at start-of-step attitude ──────────────────────────
    const double phi  = x_[6];
    const double theta = x_[7];
    const double psi   = x_[8];
    const double sphi  = std::sin(phi),   cphi = std::cos(phi);
    const double sth   = std::sin(theta), cth  = std::cos(theta), tth = sth / cth;
    const double spsi  = std::sin(psi),   cpsi = std::cos(psi);

    // ── Euler angle kinematics (ZYX) ──────────────────────────────────────
    x_[6] += (p + sphi*tth*q + cphi*tth*r) * dt;
    x_[7] += (cphi*q - sphi*r)             * dt;
    x_[8] += (sphi*q + cphi*r) / cth       * dt;

    // ── World-frame acceleration: R_ZYX · [0, 0, T_actual] / mass − g ─────
    const double ax = (cpsi*sth*cphi + spsi*sphi) * T_actual_ / mass_;
    const double ay = (spsi*sth*cphi - cpsi*sphi) * T_actual_ / mass_;
    const double az =  cth*cphi                   * T_actual_ / mass_ - g_;

    // ── Velocity and position integration ─────────────────────────────────
    x_[3] += ax * dt;
    x_[4] += ay * dt;
    x_[5] += az * dt;

    x_[0] += x_[3] * dt;
    x_[1] += x_[4] * dt;
    x_[2] += x_[5] * dt;
}
