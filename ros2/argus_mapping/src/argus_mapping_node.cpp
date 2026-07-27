// Builds an argus_esdf::VoxelGrid from the PX4 bridge's depth camera point
// cloud and PX4 odometry, and periodically publishes the resulting distance
// field for argus_bridge_node (or anything else) to subscribe to and query.
//
// Deliberately decoupled from argus_px4_bridge: its own odometry
// subscription and its own NED->ENU / body-to-world math, not a shared
// library call — see the "Inter-process, pub/sub + local cache" decision in
// the top-level todo file for why (this package trades a few duplicated
// lines of frame conversion for not forcing the two nodes into lockstep
// deployment).
//
// Frame chain from a raw depth point to a world (ENU) point:
//   1. Depth camera points *should* arrive in the sensor's optical frame
//      (X-right, Y-down, Z-forward) — the standard convention gz-sim (and
//      every ROS camera driver) uses for image/point-cloud data, regardless
//      of the sensor's own <pose> rotation in the SDF (which is identity
//      here — see kCameraOffsetBody below). In practice, this simulated
//      sensor's /depth_camera/points does NOT: its z is sign-flipped
//      (negative-forward) and its x varies reciprocally (not linearly)
//      with pixel row — a real bug in the plugin's own point generation,
//      confirmed by comparing against the expected pinhole formula (see
//      the top-level todo file for the full diagnosis). z (up to sign) is
//      the one trustworthy field, so cloudCallback() ignores the message's
//      x/y entirely and reprojects each point itself from depth + pixel
//      row/col + the real intrinsics (/camera_info) — the same formula any
//      depth-image-to-point-cloud driver uses internally, just done here
//      explicitly instead of trusting a precomputed (here, broken) field.
//   2. optical -> body (FLU): a fixed 90-ish-degree axis permutation
//      (kOpticalToBody), then translate by the camera's fixed mount offset
//      in the body frame (kCameraOffsetBody, read off
//      PX4-Autopilot/Tools/simulation/gz/models/{x500_depth,OakD-Lite}/*.sdf
//      — both the camera_link joint and the depth sensor's own pose within
//      it have zero rotation, so this is a pure translation).
//   3. body (FLU) -> world (ENU): the odometry-derived rotation
//      (ft::px4_to_ros_orientation) and position (ft::ned_to_enu_local_frame).

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_ros_com/frame_transforms.h>

#include <argus_esdf/VoxelGrid.hpp>

#include <argus_mapping/msg/esdf_grid.hpp>

using namespace px4_msgs::msg;
namespace ft = px4_ros_com::frame_transforms;

namespace {
// Fixed camera extrinsics relative to base_link (FLU), from
// PX4-Autopilot/Tools/simulation/gz/models/x500_depth/model.sdf's
// CameraJoint (.12 .03 .242) plus OakD-Lite/model.sdf's StereoOV7251
// sensor pose within camera_link (0.01233 -0.03 .01878) — both zero
// rotation, so these just add.
const Eigen::Vector3d kCameraOffsetBody(0.12 + 0.01233, 0.03 - 0.03, 0.242 + 0.01878);

// Optical (X-right, Y-down, Z-forward) -> body FLU (X-forward, Y-left,
// Z-up): body_x = optical_z, body_y = -optical_x, body_z = -optical_y.
// Standard camera-optical-to-base rotation (REP 103), independent of any
// SDF <pose> rotation on the sensor itself.
Eigen::Matrix3d opticalToBodyRotation()
{
    Eigen::Matrix3d r;
    r << 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0;
    return r;
}
const Eigen::Matrix3d kOpticalToBody = opticalToBodyRotation();

// Near-obstacle (t=0, red) -> at-or-beyond esdf_viz_max_distance (t=1,
// blue) — a standard "hot=dangerous" HSV sweep (hue 0deg to 240deg at full
// saturation/value), not a perceptually-uniform colormap like viridis, but
// simple to compute with no lookup table and reads intuitively in RViz.
//
// Alpha fades linearly from max_alpha at the obstacle surface to fully
// transparent at the threshold edge, rather than a flat opacity — a flat
// ~0.8 alpha turned the whole near-obstacle volume into a near-solid block
// that visually buried the drone, its trajectory, and the FOV frustum
// underneath/inside it (RViz is real depth-buffered 3-D, not a 2-D
// painter's-algorithm canvas, so those objects were still geometrically
// "there," just invisible through opaque-looking cubes). Fading keeps the
// dangerous-close voxels legible while letting everything else read
// through the rest of the field.
std_msgs::msg::ColorRGBA distanceToColor(double distance, double max_distance, double max_alpha)
{
    const double t = std::clamp(distance / max_distance, 0.0, 1.0);
    const double h = t * 240.0;
    const double x = 1.0 - std::abs(std::fmod(h / 60.0, 2.0) - 1.0);
    double r = 0.0, g = 0.0, b = 0.0;
    if (h < 60.0)       { r = 1.0; g = x;   b = 0.0; }
    else if (h < 120.0) { r = x;   g = 1.0; b = 0.0; }
    else if (h < 180.0) { r = 0.0; g = 1.0; b = x;   }
    else                { r = 0.0; g = x;   b = 1.0; }

    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>(r);
    color.g = static_cast<float>(g);
    color.b = static_cast<float>(b);
    color.a = static_cast<float>((1.0 - t) * max_alpha);
    return color;
}
}  // namespace

class ArgusMappingNode : public rclcpp::Node
{
public:
    ArgusMappingNode() : Node("argus_mapping_node")
    {
        const auto origin = this->declare_parameter<std::vector<double>>(
            "grid_origin", {-3.0, -3.0, 0.0});
        const auto dims = this->declare_parameter<std::vector<int64_t>>(
            "grid_dims", {60, 60, 25});
        const double voxel_size = this->declare_parameter<double>("voxel_size", 0.1);
        const double esdf_period_s = this->declare_parameter<double>("esdf_update_period", 0.5);
        // Depth image is 640x480 = 307200 points/frame; a full-density
        // insertPointCloud() would raycast that many DDA walks per frame
        // for no accuracy benefit at 0.1m voxels. See VoxelGrid.hpp's
        // insertPointCloud() doc for why downsampling is the caller's job.
        point_stride_ = static_cast<int>(this->declare_parameter<int64_t>("point_stride", 16));
        // Only voxels at or below this distance get a marker — an
        // unthresholded 90k-voxel CUBE_LIST would mostly be empty free
        // space nobody needs to see; near-obstacle voxels are what matters
        // for avoidance. 1.5m comfortably covers a plausible safety margin.
        esdf_viz_max_distance_ = this->declare_parameter<double>("esdf_viz_max_distance", 1.5);
        // Peak opacity, right at the obstacle surface — fades to 0 at
        // esdf_viz_max_distance (see distanceToColor()). Kept well under 1
        // so the drone/trajectory/FOV frustum stay visible through it.
        esdf_viz_max_alpha_ = this->declare_parameter<double>("esdf_viz_max_alpha", 0.35);
        // Points at or below this world-frame Z are treated as ground, not
        // an obstacle, and dropped before ever reaching insertPointCloud().
        // A flat height threshold, not real plane fitting/RANSAC — this
        // world's ground genuinely is a flat Z=0 plane (argus_box_world.sdf
        // has no terrain), so anything more general would be solving a
        // problem that doesn't exist here. Revisit with real plane fitting
        // only if a non-flat world is ever added.
        ground_filter_height_ = this->declare_parameter<double>("ground_filter_height", 0.2);
        // Depth points farther than this from the sensor are dropped before
        // insertPointCloud() — this grid's footprint is only ~6m x 6m
        // (default grid_dims/voxel_size), so a point past its ~8.5m
        // diagonal was always going to fall outside grid bounds and get
        // silently discarded by insertPointCloud() anyway; no reason to pay
        // for the DDA walk first.
        max_point_range_ = this->declare_parameter<double>("max_point_range", 8.0);
        // Point clouds are ignored entirely until this long after odometry
        // first starts flowing. Root-caused via a real cluster of
        // near-obstacle markers with no obstacle anywhere nearby: at spawn,
        // before the vehicle's pose has settled (the default x500 spawn
        // yaw is ~96 deg — see zyxEulerFromQuaternion's comment in
        // argus_bridge_node.cpp), the camera briefly points somewhere
        // unrelated to where it'll actually fly, and a handful of transient
        // frames from that moment can compute real (non-ground,
        // in-range) points anywhere. VoxelGrid has no decay/forgetting
        // (see the "advanced ESDF" todo notes on map staleness), so even
        // one bad frame's occupied voxels are permanent — confirmed by
        // restarting the node mid-flight (skipping the bad startup window
        // entirely) and seeing the contamination vanish. argus_bridge_node
        // has the same underlying settling concern (see its own "Armed +
        // offboard" 5s buffer) but the two nodes don't share state, so this
        // needed its own delay.
        mapping_start_delay_s_ = this->declare_parameter<double>("mapping_start_delay", 3.0);
        // Timing is cheap to accumulate but noisy to print every cycle —
        // report every N computeEsdf() cycles instead (~5s at the default
        // 0.5s esdf_update_period).
        timing_report_period_ = static_cast<int>(this->declare_parameter<int64_t>("timing_report_period", 10));

        if (origin.size() != 3 || dims.size() != 3) {
            throw std::invalid_argument("argus_mapping_node: grid_origin/grid_dims must have 3 elements");
        }
        grid_ = std::make_unique<argus_esdf::VoxelGrid>(
            Eigen::Vector3d(origin[0], origin[1], origin[2]),
            Eigen::Vector3i(static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2])),
            voxel_size);

        rclcpp::QoS px4_out_qos(rclcpp::QoSInitialization(
            rmw_qos_profile_sensor_data.history, 5), rmw_qos_profile_sensor_data);

        odometry_sub_ = this->create_subscription<VehicleOdometry>(
            "/fmu/out/vehicle_odometry", px4_out_qos,
            std::bind(&ArgusMappingNode::odometryCallback, this, std::placeholders::_1));

        // Real intrinsics for the reprojection in cloudCallback() — see the
        // file-header comment on why we don't trust the depth cloud's own
        // x/y fields. Static for a fixed camera, so only the first message
        // is used (see cameraInfoCallback()).
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS(),
            std::bind(&ArgusMappingNode::cameraInfoCallback, this, std::placeholders::_1));

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/depth_camera/points", rclcpp::SensorDataQoS(),
            std::bind(&ArgusMappingNode::cloudCallback, this, std::placeholders::_1));

        grid_pub_ = this->create_publisher<argus_mapping::msg::EsdfGrid>("/argus_mapping/esdf_grid", 1);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/argus_mapping/esdf_markers", 1);

        esdf_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(esdf_period_s)),
            std::bind(&ArgusMappingNode::computeAndPublish, this));

        RCLCPP_INFO(this->get_logger(),
                    "argus_mapping_node started (dims=%ldx%ldx%ld, voxel_size=%.2fm, stride=%d)",
                    dims[0], dims[1], dims[2], voxel_size, point_stride_);
    }

private:
    void odometryCallback(const VehicleOdometry::SharedPtr msg)
    {
        pos_ned_ = {msg->position[0], msg->position[1], msg->position[2]};
        q_px4_ = ft::utils::quaternion::array_to_eigen_quat(msg->q);
        if (!have_odometry_) first_odometry_time_ = this->now();
        have_odometry_ = true;
    }

    // K = [fx 0 cx; 0 fy cy; 0 0 1], row-major — the intrinsics cloudCallback()
    // uses to reproject points itself instead of trusting the depth cloud's
    // own (broken) x/y fields. Fixed for the lifetime of the sim's static
    // camera, so only the first message is kept.
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (have_camera_info_) return;
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        have_camera_info_ = true;
        RCLCPP_INFO(this->get_logger(),
                    "argus_mapping_node: camera intrinsics received (fx=%.2f fy=%.2f cx=%.2f cy=%.2f)",
                    fx_, fy_, cx_, cy_);
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!have_odometry_ || !have_camera_info_) return;
        if ((this->now() - first_odometry_time_).seconds() < mapping_start_delay_s_) return;

        const Eigen::Vector3d pos_enu = ft::ned_to_enu_local_frame(pos_ned_);
        const Eigen::Matrix3d r_body_to_world = ft::px4_to_ros_orientation(q_px4_).toRotationMatrix();
        const Eigen::Vector3d sensor_origin_world =
            r_body_to_world * kCameraOffsetBody + pos_enu;
        const Eigen::Matrix3d r_optical_to_world = r_body_to_world * kOpticalToBody;

        // Only z is read from the message itself — see the file-header
        // comment: this sensor's x/y fields don't follow the pinhole
        // formula at all, but z (up to a sign flip) does behave like real
        // depth, so every point's optical-frame (X, Y) is rebuilt here from
        // that depth plus its own pixel row/col and the real intrinsics
        // from /camera_info, the same formula any depth-image-to-point
        // driver uses internally.
        sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
        const int width = static_cast<int>(msg->width);

        std::vector<Eigen::Vector3d> points_world;
        points_world.reserve(msg->width * msg->height / std::max(1, point_stride_) + 1);

        int i = 0;
        for (; it_z != it_z.end(); ++it_z, ++i) {
            if (i % point_stride_ != 0) continue;
            const float z = *it_z;
            if (!std::isfinite(z)) continue;  // no-return depth

            const double depth = -static_cast<double>(z);  // sign-flipped in this sim's sensor
            if (depth <= 0.0) continue;
            const int row = i / width;
            const int col = i % width;
            const double optical_x = depth * (col - cx_) / fx_;
            const double optical_y = depth * (row - cy_) / fy_;

            const Eigen::Vector3d point_world =
                r_optical_to_world * Eigen::Vector3d(optical_x, optical_y, depth) + sensor_origin_world;
            if ((point_world - sensor_origin_world).norm() > max_point_range_) continue;
            if (point_world.z() <= ground_filter_height_) continue;  // ground hit, not an obstacle
            points_world.push_back(point_world);
        }

        grid_->insertPointCloud(sensor_origin_world, points_world);
    }

    void computeAndPublish()
    {
        grid_->computeEsdf();

        argus_mapping::msg::EsdfGrid msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "map";
        msg.origin.x = grid_->origin().x();
        msg.origin.y = grid_->origin().y();
        msg.origin.z = grid_->origin().z();
        msg.dims = {grid_->dims().x(), grid_->dims().y(), grid_->dims().z()};
        msg.voxel_size = grid_->voxelSize();

        const int n = grid_->dims().x() * grid_->dims().y() * grid_->dims().z();
        msg.distances.resize(n);
        for (int z = 0; z < grid_->dims().z(); ++z) {
            for (int y = 0; y < grid_->dims().y(); ++y) {
                for (int x = 0; x < grid_->dims().x(); ++x) {
                    const int idx = x + grid_->dims().x() * (y + grid_->dims().y() * z);
                    msg.distances[idx] = static_cast<float>(grid_->distanceAt({x, y, z}));
                }
            }
        }
        grid_pub_->publish(msg);

        publishEsdfMarkers(msg.header.stamp);
        reportTimingIfDue();
    }

    // A single CUBE_LIST marker (one draw call regardless of point count)
    // of every voxel within esdf_viz_max_distance of an obstacle, colored
    // near(red)->far(blue) — see distanceToColor(). Occupied voxels
    // (distance==0) show up as solid red, giving a direct visual read of
    // both the occupancy grid and the distance field it produced.
    void publishEsdfMarkers(const builtin_interfaces::msg::Time& stamp)
    {
        visualization_msgs::msg::Marker cubes;
        cubes.header.frame_id = "map";
        cubes.header.stamp = stamp;
        cubes.ns = "esdf";
        cubes.id = 0;
        cubes.type = visualization_msgs::msg::Marker::CUBE_LIST;
        cubes.action = visualization_msgs::msg::Marker::ADD;
        cubes.pose.orientation.w = 1.0;
        cubes.scale.x = cubes.scale.y = cubes.scale.z = grid_->voxelSize();

        const Eigen::Vector3i dims = grid_->dims();
        for (int z = 0; z < dims.z(); ++z) {
            for (int y = 0; y < dims.y(); ++y) {
                for (int x = 0; x < dims.x(); ++x) {
                    const double d = grid_->distanceAt({x, y, z});
                    if (d > esdf_viz_max_distance_) continue;
                    const Eigen::Vector3d c = grid_->voxelCenter({x, y, z});
                    geometry_msgs::msg::Point p;
                    p.x = c.x(); p.y = c.y(); p.z = c.z();
                    cubes.points.push_back(p);
                    cubes.colors.push_back(distanceToColor(d, esdf_viz_max_distance_, esdf_viz_max_alpha_));
                }
            }
        }

        visualization_msgs::msg::MarkerArray array;
        array.markers.push_back(cubes);
        marker_pub_->publish(array);
    }

    void reportTimingIfDue()
    {
        if (++timing_report_counter_ < timing_report_period_) return;
        timing_report_counter_ = 0;

        const auto insert_t = grid_->insertTiming();
        const auto compute_t = grid_->computeEsdfTiming();
        RCLCPP_INFO(this->get_logger(),
                    "ESDF timing — insertPointCloud: min/mean/max = %.2f/%.2f/%.2fms (n=%ld) | "
                    "computeEsdf: min/mean/max = %.2f/%.2f/%.2fms (n=%ld)",
                    insert_t.min_ms, insert_t.mean_ms, insert_t.max_ms, insert_t.count,
                    compute_t.min_ms, compute_t.mean_ms, compute_t.max_ms, compute_t.count);
    }

    std::unique_ptr<argus_esdf::VoxelGrid> grid_;
    int point_stride_ = 16;
    double esdf_viz_max_distance_ = 1.5;
    double esdf_viz_max_alpha_ = 0.35;
    double ground_filter_height_ = 0.2;
    double max_point_range_ = 8.0;
    double mapping_start_delay_s_ = 3.0;
    int timing_report_period_ = 10;
    int timing_report_counter_ = 0;

    bool have_odometry_ = false;
    rclcpp::Time first_odometry_time_;
    Eigen::Vector3d pos_ned_{0.0, 0.0, 0.0};
    Eigen::Quaterniond q_px4_{1.0, 0.0, 0.0, 0.0};

    bool have_camera_info_ = false;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;

    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<argus_mapping::msg::EsdfGrid>::SharedPtr grid_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr esdf_timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArgusMappingNode>());
    rclcpp::shutdown();
    return 0;
}
