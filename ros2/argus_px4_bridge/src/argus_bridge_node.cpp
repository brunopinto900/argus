// Bridges argus's QuadrotorMpc (see argus/include/argus_mpc/QuadrotorMpc.hpp)
// to PX4 over the native ROS2 (uXRCE-DDS) interface.
//
// State machine: INIT (stream position setpoints, request offboard + arm) ->
// TAKEOFF (climb to hover altitude on a position setpoint) -> MPC_TRACKING
// (hand control to QuadrotorMpc::step(), publish body-rate setpoints).
//
// Frame conventions: PX4's /fmu topics are NED world / FRD body. argus's
// model is ENU-ish (z-up) world / FLU body (thrust along body +z). All
// conversions go through px4_ros_com's frame_transforms rather than being
// re-derived here.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include <px4_ros_com/frame_transforms.h>

#include <argus_mpc/QuadrotorMpc.hpp>
// Auto-generated from config/quadrotor.yaml — re-run generate_mpc.py to update.
// Pulled in transitively via the quadrotor_mpc target's include dirs.
#include "argus_params.h"

using namespace std::chrono_literals;
using namespace px4_msgs::msg;
namespace ft = px4_ros_com::frame_transforms;

namespace {
constexpr double kTakeoffAltTol = 0.15;  // m
constexpr double kTakeoffVelTol = 0.3;   // m/s

// px4_ros_com's quaternion_to_euler() wraps Eigen's generic eulerAngles(),
// which branch-flips to the (roll+pi, -pitch, yaw-pi) equivalent whenever the
// yaw component crosses +-90 deg — nothing to do with the true ZYX gimbal
// lock at pitch=+-90 deg. Any vehicle spawned with a yaw past 90 deg (as the
// default gz_x500 world does, ~96 deg) hits this immediately. Use the
// standard closed-form ZYX formula instead, which is only singular at the
// real gimbal lock.
Eigen::Vector3d zyxEulerFromQuaternion(const Eigen::Quaterniond& q)
{
    const double w = q.w(), x = q.x(), y = q.y(), z = q.z();
    const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
    const double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    return {roll, pitch, yaw};
}
}  // namespace

class ArgusBridgeNode : public rclcpp::Node
{
public:
    ArgusBridgeNode() : Node("argus_bridge_node")
    {
        mpc_thr_hover_ = this->declare_parameter<double>("mpc_thr_hover", 0.6);
        N_ = mpc_.horizonLength();
        omega_ = 2.0 * M_PI / ARGUS_CIRCLE_PERIOD;

        rclcpp::QoS px4_out_qos(rclcpp::QoSInitialization(
            rmw_qos_profile_sensor_data.history, 5), rmw_qos_profile_sensor_data);

        odometry_sub_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", px4_out_qos,
            std::bind(&ArgusBridgeNode::odometryCallback, this, std::placeholders::_1));
        // VehicleStatus is schema version 4 — PX4 publishes it on the
        // version-suffixed topic, not the bare "/fmu/out/vehicle_status".
        status_sub_ = this->create_subscription<VehicleStatus>(
            "/fmu/out/vehicle_status_v4", px4_out_qos,
            std::bind(&ArgusBridgeNode::statusCallback, this, std::placeholders::_1));

        offboard_mode_pub_ = this->create_publisher<OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_pub_ = this->create_publisher<TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 10);
        rates_setpoint_pub_ = this->create_publisher<VehicleRatesSetpoint>(
            "/fmu/in/vehicle_rates_setpoint", 10);
        vehicle_command_pub_ = this->create_publisher<VehicleCommand>(
            "/fmu/in/vehicle_command", 10);

        const auto period = std::chrono::duration<double>(ARGUS_MPC_TS);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&ArgusBridgeNode::tick, this));

        RCLCPP_INFO(this->get_logger(), "argus_bridge_node started (N=%d, Ts=%.3fs, mpc_thr_hover=%.2f)",
                    N_, ARGUS_MPC_TS, mpc_thr_hover_);
    }

private:
    enum class State { kInit, kTakeoff, kMpcTracking };

    // ── Subscriptions ───────────────────────────────────────────────────────
    void odometryCallback(const VehicleOdometry::SharedPtr msg)
    {
        // Assumes NED pose/velocity frame (PX4 SITL default) — pose_frame /
        // velocity_frame are not checked per-message to keep this on the hot path.
        pos_ned_ = {msg->position[0], msg->position[1], msg->position[2]};
        vel_ned_ = {msg->velocity[0], msg->velocity[1], msg->velocity[2]};
        ang_vel_frd_ = {msg->angular_velocity[0], msg->angular_velocity[1], msg->angular_velocity[2]};
        q_px4_ = ft::utils::quaternion::array_to_eigen_quat(msg->q);
        have_odometry_ = true;
    }

    void statusCallback(const VehicleStatus::SharedPtr msg)
    {
        arming_state_ = msg->arming_state;
        nav_state_ = msg->nav_state;
    }

    // ── Main loop ────────────────────────────────────────────────────────────
    void tick()
    {
        if (!have_odometry_) return;

        const Eigen::Vector3d pos_enu = ft::ned_to_enu_local_frame(pos_ned_);
        const Eigen::Vector3d vel_enu = ft::ned_to_enu_local_frame(vel_ned_);
        const Eigen::Quaterniond q_ros = ft::px4_to_ros_orientation(q_px4_);
        const Eigen::Vector3d euler = zyxEulerFromQuaternion(q_ros);  // roll,pitch,yaw (ZYX)
        const Eigen::Vector3d ang_vel_flu = ft::aircraft_to_baselink_body_frame(ang_vel_frd_);

        const argus_mpc::State x0{
            pos_enu.x(), pos_enu.y(), pos_enu.z(),
            vel_enu.x(), vel_enu.y(), vel_enu.z(),
            euler.x(),   euler.y(),   euler.z(),
            ang_vel_flu.x(), ang_vel_flu.y(), ang_vel_flu.z(),
        };

        switch (state_) {
            case State::kInit:
                publishOffboardControlMode(/*body_rate=*/false);
                publishTakeoffSetpoint();
                if (offboard_setpoint_counter_ < 11) {
                    ++offboard_setpoint_counter_;
                } else if (offboard_setpoint_counter_ == 11) {
                    requestOffboardAndArm();
                    ++offboard_setpoint_counter_;
                }
                if (arming_state_ == VehicleStatus::ARMING_STATE_ARMED &&
                    nav_state_ == VehicleStatus::NAVIGATION_STATE_OFFBOARD) {
                    RCLCPP_INFO(this->get_logger(), "Armed + offboard — climbing to hover altitude");
                    state_ = State::kTakeoff;
                }
                break;

            case State::kTakeoff: {
                publishOffboardControlMode(/*body_rate=*/false);
                publishTakeoffSetpoint();
                const double pos_err = (pos_enu - Eigen::Vector3d(ARGUS_CIRCLE_R, 0.0, ARGUS_HOVER_ALTITUDE)).norm();
                if (pos_err < kTakeoffAltTol && vel_enu.norm() < kTakeoffVelTol) {
                    RCLCPP_INFO(this->get_logger(), "On station at circle start — beginning MPC circle tracking");
                    state_ = State::kMpcTracking;
                    track_start_time_ = this->now();
                }
                break;
            }

            case State::kMpcTracking: {
                publishOffboardControlMode(/*body_rate=*/true);
                const double t = (this->now() - track_start_time_).seconds();

                std::vector<argus_mpc::Reference> horizon(N_);
                for (int j = 0; j < N_; ++j) {
                    const double theta = omega_ * (t + j * ARGUS_MPC_TS);
                    horizon[j] = {ARGUS_CIRCLE_R * std::cos(theta),
                                  ARGUS_CIRCLE_R * std::sin(theta),
                                  ARGUS_HOVER_ALTITUDE};
                }
                const argus_mpc::TargetState target{0.0, 0.0, ARGUS_CIRCLE_Z, 0.0, 0.0, 0.0};

                const argus_mpc::Command cmd = mpc_.step(x0, horizon, target);
                if (cmd.solve_status != 0) {
                    RCLCPP_WARN(this->get_logger(),
                        "MPC solve status %d | x0: pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) "
                        "eul=(%.3f,%.3f,%.3f) rate=(%.3f,%.3f,%.3f) | href0=(%.3f,%.3f,%.3f)",
                        cmd.solve_status,
                        x0.x, x0.y, x0.z, x0.vx, x0.vy, x0.vz,
                        x0.phi, x0.theta, x0.psi, x0.p, x0.q, x0.r,
                        horizon[0].x, horizon[0].y, horizon[0].z);
                }
                publishRatesSetpoint(cmd);
                break;
            }
        }
    }

    // ── Publishers ───────────────────────────────────────────────────────────
    void publishOffboardControlMode(bool body_rate)
    {
        OffboardControlMode msg{};
        msg.position = !body_rate;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = body_rate;
        msg.thrust_and_torque = false;
        msg.direct_actuator = false;
        msg.timestamp = nowUs();
        offboard_mode_pub_->publish(msg);
    }

    // Climbs to the circle's own t=0 point (ENU (R,0,alt), nose toward the
    // origin) rather than straight up above the center. The MPC was only
    // ever validated starting already on the circle (see mpc_controller.cpp);
    // handing off from a hover above the center would present it with a full
    // radius-length position/yaw step it's never had to recover from, which
    // is what was driving the divergence seen in early SITL tests.
    // NED frame, as TrajectorySetpoint requires: x_ned=y_enu, y_ned=x_enu,
    // z_ned=-z_enu; yaw_ned = pi/2 - yaw_enu (see the NED<->ENU heading swap).
    void publishTakeoffSetpoint()
    {
        TrajectorySetpoint msg{};
        msg.position = {0.0f, static_cast<float>(ARGUS_CIRCLE_R), static_cast<float>(-ARGUS_HOVER_ALTITUDE)};
        msg.yaw = static_cast<float>(M_PI_2 - M_PI);  // nose toward origin from (R,0)
        msg.timestamp = nowUs();
        trajectory_setpoint_pub_->publish(msg);
    }

    // thrust_norm = mpc_thr_hover * cmd.thrust assumes PX4's linear thrust
    // curve (default THR_MDL_FAC=0) and that mpc_thr_hover matches the
    // vehicle's actual MPC_THR_HOVER param — verify/tune against real hover
    // behaviour in SITL before trusting XY tracking.
    void publishRatesSetpoint(const argus_mpc::Command& cmd)
    {
        // KNOWN ISSUE: commanding cmd.yaw_rate as-is causes a runaway yaw
        // spin-up (20-30 rad/s within ~2s) in SITL — confirmed by a diagnostic
        // run with yaw_rate forced to 0, which stayed bounded. Root cause not
        // yet isolated (suspect PX4 yaw-rate-controller gains vs argus's
        // assumed tau_yaw=0.25s, or a remaining sign/scale issue). Do not
        // trust yaw control until this is fixed.
        const Eigen::Vector3d rates_flu(cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate);
        const Eigen::Vector3d rates_frd = ft::baselink_to_aircraft_body_frame(rates_flu);

        const float thrust_norm = std::clamp(
            static_cast<float>(mpc_thr_hover_ * cmd.thrust), 0.0f, 1.0f);

        VehicleRatesSetpoint msg{};
        msg.roll = static_cast<float>(rates_frd.x());
        msg.pitch = static_cast<float>(rates_frd.y());
        msg.yaw = static_cast<float>(rates_frd.z());
        msg.thrust_body = {0.0f, 0.0f, -thrust_norm};
        msg.timestamp = nowUs();
        rates_setpoint_pub_->publish(msg);
    }

    void requestOffboardAndArm()
    {
        RCLCPP_INFO(this->get_logger(), "Requesting OFFBOARD mode + arm");
        publishVehicleCommand(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0);
        publishVehicleCommand(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
    }

    void publishVehicleCommand(uint32_t command, float param1 = 0.0f, float param2 = 0.0f)
    {
        VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = nowUs();
        vehicle_command_pub_->publish(msg);
    }

    uint64_t nowUs() const
    {
        return static_cast<uint64_t>(this->get_clock()->now().nanoseconds() / 1000);
    }

    // ── State ────────────────────────────────────────────────────────────────
    argus_mpc::QuadrotorMpc mpc_;
    int N_ = 0;
    double omega_ = 0.0;
    double mpc_thr_hover_ = 0.5;

    State state_ = State::kInit;
    int offboard_setpoint_counter_ = 0;
    rclcpp::Time track_start_time_;

    bool have_odometry_ = false;
    Eigen::Vector3d pos_ned_{0.0, 0.0, 0.0};
    Eigen::Vector3d vel_ned_{0.0, 0.0, 0.0};
    Eigen::Vector3d ang_vel_frd_{0.0, 0.0, 0.0};
    Eigen::Quaterniond q_px4_{1.0, 0.0, 0.0, 0.0};

    uint8_t arming_state_ = 0;
    uint8_t nav_state_ = 0;

    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_mode_pub_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<VehicleRatesSetpoint>::SharedPtr rates_setpoint_pub_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArgusBridgeNode>());
    rclcpp::shutdown();
    return 0;
}
