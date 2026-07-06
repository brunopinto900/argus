#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "argus_mpc/QuadrotorMpc.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct TargetWaypoint {
    double x = 0.0, y = 0.0, z = 0.0;
    double speed = 1.0;
    double hold  = 0.0;
};

struct TargetConfig {
    double max_accel         = 1.0;
    double max_speed         = 1.5;
    double max_lateral_accel = 2.0;
    bool   loop              = false;
    std::vector<TargetWaypoint> waypoints;
};

inline TargetConfig load_target_config(const std::string& yaml_path)
{
    const YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["target"])
        throw std::runtime_error(yaml_path + ": missing 'target' section");

    const YAML::Node n = root["target"];
    TargetConfig cfg;
    if (n["max_accel"])         cfg.max_accel         = n["max_accel"].as<double>();
    if (n["max_speed"])         cfg.max_speed         = n["max_speed"].as<double>();
    if (n["max_lateral_accel"]) cfg.max_lateral_accel = n["max_lateral_accel"].as<double>();
    if (n["loop"])              cfg.loop              = n["loop"].as<bool>();

    for (const auto& wp : n["waypoints"]) {
        TargetWaypoint w;
        const auto& pos = wp["pos"];
        w.x = pos[0].as<double>();
        w.y = pos[1].as<double>();
        w.z = pos[2].as<double>();
        if (wp["speed"]) w.speed = wp["speed"].as<double>();
        if (wp["hold"])  w.hold  = wp["hold"].as<double>();
        cfg.waypoints.push_back(w);
    }
    return cfg;
}

// Common interface for anything that produces the tracked target's
// kinematic state one sample at a time — lets mpc_controller.cpp drive
// either a waypoint-following person or a parametric circle without
// branching on which one it is.
class TargetSource
{
public:
    virtual ~TargetSource()                        = default;
    virtual const argus_mpc::TargetState& state() const = 0;
    virtual argus_mpc::TargetState step(double dt)      = 0;
};

// Kinematic person simulator: trapezoidal longitudinal speed profile +
// lateral-acceleration-limited heading rate (same logic as Vista-Tracker).
class TargetSimulator : public TargetSource
{
public:
    explicit TargetSimulator(const TargetConfig& cfg) : cfg_(cfg)
    {
        if (!cfg_.waypoints.empty()) {
            state_.x = cfg_.waypoints[0].x;
            state_.y = cfg_.waypoints[0].y;
            state_.z = cfg_.waypoints[0].z;
        }
    }

    const argus_mpc::TargetState& state() const override { return state_; }

    argus_mpc::TargetState step(double dt) override
    {
        if (done_ || cfg_.waypoints.empty()) {
            state_.vx = state_.vy = state_.vz = 0.0;
            return state_;
        }

        const TargetWaypoint& wp = cfg_.waypoints[idx_];

        // Hold at waypoint
        if (holding_) {
            hold_elapsed_ += dt;
            state_.vx = state_.vy = state_.vz = 0.0;
            if (hold_elapsed_ >= wp.hold) { holding_ = false; hold_elapsed_ = 0.0; advance(); }
            return state_;
        }

        const double dx   = wp.x - state_.x;
        const double dy   = wp.y - state_.y;
        const double dz   = wp.z - state_.z;
        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (dist < kReachThreshold) {
            state_.vx = state_.vy = state_.vz = 0.0;
            if (wp.hold > 0.0) holding_ = true;
            else               advance();
            return state_;
        }

        // Trapezoidal profile: v_target = min(cruise, braking_ramp)
        const double cur_speed = std::sqrt(state_.vx*state_.vx +
                                           state_.vy*state_.vy +
                                           state_.vz*state_.vz);
        const double v_cruise  = std::min(wp.speed, cfg_.max_speed);
        const double v_brake   = std::sqrt(2.0 * cfg_.max_accel * dist);
        const double v_target  = std::min(v_cruise, v_brake);

        const double dv     = std::clamp(v_target - cur_speed,
                                         -cfg_.max_accel * dt, cfg_.max_accel * dt);
        const double new_speed = std::max(0.0, cur_speed + dv);

        // Heading: limit yaw rate via max_lateral_accel / speed
        const double desired_heading = std::atan2(dy, dx);
        double heading_err = desired_heading - heading_;
        while (heading_err >  M_PI) heading_err -= 2.0 * M_PI;
        while (heading_err < -M_PI) heading_err += 2.0 * M_PI;

        const double max_yaw_rate = cfg_.max_lateral_accel /
                                    std::max(new_speed, kMinSpeedForTurn);
        heading_ += std::clamp(heading_err, -max_yaw_rate * dt, max_yaw_rate * dt);

        const double uz = dz / dist;
        state_.x  += std::cos(heading_) * new_speed * dt;
        state_.y  += std::sin(heading_) * new_speed * dt;
        state_.z  += uz * new_speed * dt;
        state_.vx  = std::cos(heading_) * new_speed;
        state_.vy  = std::sin(heading_) * new_speed;
        state_.vz  = uz * new_speed;

        return state_;
    }

private:
    static constexpr double kReachThreshold  = 0.1;
    static constexpr double kMinSpeedForTurn = 0.05;

    TargetConfig           cfg_;
    argus_mpc::TargetState state_;
    double                 heading_      = 0.0;
    int                    idx_          = 0;
    bool                   holding_      = false;
    double                 hold_elapsed_ = 0.0;
    bool                   done_         = false;

    void advance()
    {
        if (++idx_ >= static_cast<int>(cfg_.waypoints.size()))
            idx_ = cfg_.loop ? 0 : (done_ = true, idx_);
    }
};

// Target that walks a fixed circle at constant angular rate — the
// "circle tracking" scenario expressed as a moving target rather than a
// drone-side trajectory, so it drives through the same TargetSource
// interface as TargetSimulator.
class CircleTargetSimulator : public TargetSource
{
public:
    CircleTargetSimulator(double radius, double z, double period)
        : radius_(radius), z_(z), omega_(2.0 * M_PI / period)
    {
        state_.x = radius_;
        state_.z = z_;
    }

    const argus_mpc::TargetState& state() const override { return state_; }

    argus_mpc::TargetState step(double dt) override
    {
        t_ += dt;
        const double theta = omega_ * t_;
        state_.x  =  radius_ * std::cos(theta);
        state_.y  =  radius_ * std::sin(theta);
        state_.z  =  z_;
        state_.vx = -radius_ * omega_ * std::sin(theta);
        state_.vy =  radius_ * omega_ * std::cos(theta);
        state_.vz =  0.0;
        return state_;
    }

private:
    double radius_, z_, omega_;
    double t_ = 0.0;
    argus_mpc::TargetState state_;
};
