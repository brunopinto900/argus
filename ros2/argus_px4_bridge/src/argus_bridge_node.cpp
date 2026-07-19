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
#include <deque>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

// RViz visualization only — everything is published in a fixed "map" frame
// with no TF tree involved (Fixed Frame == frame_id, so RViz needs no lookup).
constexpr char kVizFrame[] = "map";
constexpr size_t kMaxDronePathPoints = 4000;  // ~200s of history at 20Hz
constexpr int kRefPathPoints = 100;
// 4 corners = a pyramid (proper camera-frustum look) rather than a
// many-sided cone — same half-angle geometry either way, see
// buildFovMarkers() below.
constexpr int kFovFrustumCorners = 4;

// Mirrors worlds/argus_box_world.sdf's obstacle_box_0 <pose>/<size> — RViz
// has no visibility into Gazebo's own world geometry, so this is a second,
// hand-kept copy purely for visualization. Keep in sync by hand if the SDF
// pose/size ever changes; nothing enforces it automatically.
constexpr double kObstacleBoxPos[3] = {0.3, 0.8, 0.75};
constexpr double kObstacleBoxSize[3] = {0.5, 0.5, 1.5};

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
        hover_only_ = this->declare_parameter<bool>("hover_only", false);
        // Must match config/quadrotor.yaml's camera.fov_half_angle_deg — not
        // auto-generated into argus_params.h (visualization-only, no cost-
        // formulation dependency), so keep the two in sync by hand.
        const double fov_half_angle_deg = this->declare_parameter<double>("fov_half_angle_deg", 40.0);
        fov_half_angle_rad_ = fov_half_angle_deg * M_PI / 180.0;
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

        drone_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("argus/drone_path", 10);
        reference_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("argus/reference_path", 10);
        fov_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("argus/fov_markers", 10);
        obstacle_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("argus/obstacle_markers", 10);
        initReferencePath();
        initObstacleMarkers();

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
                    if (hover_only_) {
                        RCLCPP_INFO_ONCE(this->get_logger(),
                            "On station at circle start — hover_only set, holding position "
                            "instead of starting MPC circle tracking");
                    } else {
                        RCLCPP_INFO(this->get_logger(), "On station at circle start — beginning MPC circle tracking");
                        state_ = State::kMpcTracking;
                        track_start_time_ = this->now();
                    }
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

        publishVisualization(pos_enu, euler);
    }

    // ── Visualization (RViz) ─────────────────────────────────────────────────
    // Fixed circle reference the drone tracks — computed once since it doesn't
    // depend on wall-clock time, republished (transient_local not needed: this
    // is cheap enough to just resend every tick alongside everything else).
    void initReferencePath()
    {
        reference_path_msg_.header.frame_id = kVizFrame;
        reference_path_msg_.poses.resize(kRefPathPoints + 1);
        for (int i = 0; i <= kRefPathPoints; ++i) {
            const double theta = 2.0 * M_PI * i / kRefPathPoints;
            auto& pose = reference_path_msg_.poses[i];
            pose.header.frame_id = kVizFrame;
            pose.pose.position.x = ARGUS_CIRCLE_R * std::cos(theta);
            pose.pose.position.y = ARGUS_CIRCLE_R * std::sin(theta);
            pose.pose.position.z = ARGUS_HOVER_ALTITUDE;
            pose.pose.orientation.w = 1.0;
        }
    }

    // Same rationale as initReferencePath() above — static, so built once
    // and just restamped/republished every tick alongside everything else.
    void initObstacleMarkers()
    {
        visualization_msgs::msg::Marker box;
        box.header.frame_id = kVizFrame;
        box.ns = "argus_obstacles";
        box.id = 0;
        box.type = visualization_msgs::msg::Marker::CUBE;
        box.action = visualization_msgs::msg::Marker::ADD;
        box.pose.position.x = kObstacleBoxPos[0];
        box.pose.position.y = kObstacleBoxPos[1];
        box.pose.position.z = kObstacleBoxPos[2];
        box.pose.orientation.w = 1.0;
        box.scale.x = kObstacleBoxSize[0];
        box.scale.y = kObstacleBoxSize[1];
        box.scale.z = kObstacleBoxSize[2];
        box.color.r = 0.9f;
        box.color.g = 0.2f;
        box.color.b = 0.1f;
        box.color.a = 0.9f;
        obstacle_markers_msg_.markers = {box};
    }

    void publishVisualization(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& euler)
    {
        const auto stamp = this->now();

        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = kVizFrame;
        pose.pose.position.x = pos_enu.x();
        pose.pose.position.y = pos_enu.y();
        pose.pose.position.z = pos_enu.z();
        pose.pose.orientation.w = 1.0;
        drone_path_.push_back(pose);
        if (drone_path_.size() > kMaxDronePathPoints) drone_path_.pop_front();

        nav_msgs::msg::Path drone_path_msg;
        drone_path_msg.header.stamp = stamp;
        drone_path_msg.header.frame_id = kVizFrame;
        drone_path_msg.poses.assign(drone_path_.begin(), drone_path_.end());
        drone_path_pub_->publish(drone_path_msg);

        reference_path_msg_.header.stamp = stamp;
        reference_path_pub_->publish(reference_path_msg_);

        for (auto& marker : obstacle_markers_msg_.markers) {
            marker.header.stamp = stamp;
        }
        obstacle_marker_pub_->publish(obstacle_markers_msg_);

        fov_marker_pub_->publish(buildFovMarkers(pos_enu, euler, stamp));
    }

    // Body +x (boresight) pyramidal FOV wireframe, half-angle =
    // fov_half_angle_rad_ — mirrors argus's r_fov cost (r_fov =
    // softplus(cos(alpha_fov) - d_body[0]/range), generate_mpc.py), which is
    // zero (no penalty) exactly when the target sits inside this cone.
    // r_fov itself is rotationally symmetric (a true circular cone, not
    // actually a rectangular camera frustum) — the 4-corner pyramid drawn
    // here is a simplified stand-in for that same cone (same half-angle,
    // corners instead of a smooth many-sided fan), not a literal rendering
    // of a different, non-symmetric FOV shape. Drawn out to the current
    // range to the target so it visibly reaches (or misses) it each tick.
    visualization_msgs::msg::MarkerArray buildFovMarkers(
        const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& euler, const rclcpp::Time& stamp)
    {
        const Eigen::Matrix3d R =
            (Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();

        const Eigen::Vector3d target_pos(0.0, 0.0, ARGUS_CIRCLE_Z);
        const double range = (target_pos - pos_enu).norm();

        visualization_msgs::msg::Marker frustum;
        frustum.header.frame_id = kVizFrame;
        frustum.header.stamp = stamp;
        frustum.ns = "argus_fov";
        frustum.id = 0;
        frustum.type = visualization_msgs::msg::Marker::LINE_LIST;
        frustum.action = visualization_msgs::msg::Marker::ADD;
        frustum.pose.orientation.w = 1.0;
        frustum.scale.x = 0.02;
        frustum.color.r = 1.0f;
        frustum.color.g = 0.6f;
        frustum.color.b = 0.0f;
        frustum.color.a = 0.6f;

        const auto toPoint = [](const Eigen::Vector3d& v) {
            geometry_msgs::msg::Point p;
            p.x = v.x(); p.y = v.y(); p.z = v.z();
            return p;
        };
        const geometry_msgs::msg::Point apex = toPoint(pos_enu);

        std::vector<geometry_msgs::msg::Point> tips(kFovFrustumCorners);
        for (int i = 0; i < kFovFrustumCorners; ++i) {
            const double ring = 2.0 * M_PI * i / kFovFrustumCorners;
            const Eigen::Vector3d dir_body(
                std::cos(fov_half_angle_rad_),
                std::sin(fov_half_angle_rad_) * std::cos(ring),
                std::sin(fov_half_angle_rad_) * std::sin(ring));
            tips[i] = toPoint(pos_enu + range * (R * dir_body));
            frustum.points.push_back(apex);
            frustum.points.push_back(tips[i]);
        }
        for (int i = 0; i < kFovFrustumCorners; ++i) {
            frustum.points.push_back(tips[i]);
            frustum.points.push_back(tips[(i + 1) % kFovFrustumCorners]);
        }

        visualization_msgs::msg::Marker target_marker;
        target_marker.header = frustum.header;
        target_marker.ns = "argus_fov";
        target_marker.id = 1;
        target_marker.type = visualization_msgs::msg::Marker::SPHERE;
        target_marker.action = visualization_msgs::msg::Marker::ADD;
        target_marker.pose.position = toPoint(target_pos);
        target_marker.pose.orientation.w = 1.0;
        target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.3;
        target_marker.color.r = 1.0f;
        target_marker.color.a = 1.0f;

        visualization_msgs::msg::MarkerArray markers;
        markers.markers = {frustum, target_marker};
        return markers;
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
    bool hover_only_ = false;
    double fov_half_angle_rad_ = 0.0;

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

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr drone_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr reference_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr fov_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_marker_pub_;
    nav_msgs::msg::Path reference_path_msg_;
    visualization_msgs::msg::MarkerArray obstacle_markers_msg_;
    std::deque<geometry_msgs::msg::PoseStamped> drone_path_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArgusBridgeNode>());
    rclcpp::shutdown();
    return 0;
}
