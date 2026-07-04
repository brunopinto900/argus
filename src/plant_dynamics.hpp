#pragma once
#include <array>

// Second-order plant model matching PX4's body-rate inner loop.
//
// The MPC predicts with a first-order lag (tau = 1/wn).  This class
// integrates the full second-order response, so every call to update()
// is a step through a plant the MPC has never seen — a realistic
// model-plant mismatch for robustness evaluation.
//
// State layout: [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
// Input layout: [T, p_cmd, q_cmd, r_cmd]
class PlantDynamics {
public:
    using State = std::array<double, 12>;
    using Input = std::array<double, 4>;

    // x0       - initial 12-state vector
    // T0       - initial actual thrust [N] (pass T_hover for a smooth start)
    // mass     - vehicle mass [kg]
    // wn_rp    - roll/pitch rate bandwidth  [rad/s]
    // zeta_rp  - roll/pitch damping ratio
    // wn_yaw   - yaw rate bandwidth          [rad/s]
    // zeta_yaw - yaw damping ratio
    // g        - gravitational acceleration  [m/s²]
    PlantDynamics(const State& x0, double T0, double mass,
                  double wn_rp,  double zeta_rp,
                  double wn_yaw, double zeta_yaw,
                  double g = 9.81);

    // Integrate the plant forward by dt seconds.
    void update(const Input& u, double dt);

    const State& state() const { return x_; }

private:
    State  x_;
    double p_dot_    {0.0};
    double q_dot_    {0.0};
    double r_dot_    {0.0};
    double T_actual_;
    double T_dot_    {0.0};

    double mass_;
    double wn_rp_,   zeta_rp_;
    double wn_yaw_,  zeta_yaw_;
    double g_;

    // Sub-steps keep wn·dt_sub < 1 (numerical stability).
    // wn_rp=25, dt=0.05 → wn·dt=1.25; with 50 sub-steps wn·dt_sub=0.025.
    static constexpr int kSubSteps = 50;

    // One sub-step of a second-order system:  ẍ = wn²(cmd−x) − damp·ẋ
    static void step2(double cmd, double& val, double& val_dot,
                      double wn2, double damp, double dt_sub);
};
