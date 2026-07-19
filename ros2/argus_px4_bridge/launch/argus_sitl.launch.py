"""Launches Micro-XRCE-DDS-Agent + PX4 SITL (Gazebo, gz_x500_depth) + argus_bridge_node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessIO, OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# PX4's interactive `pxh>` shell reads commands from its own stdin. Piping
# through this FIFO (kept open for writing via the `exec 3<>` trick in the
# SITL command below, so the shell doesn't see EOF and exit its command loop
# after the first write) lets a later step push `param set` commands into
# the already-running instance once it's actually ready for them.
PX4_STDIN_FIFO = "/tmp/argus_px4_bridge_stdin"


def _launch_setup(context, *args, **kwargs):
    # Resolved once, here, as plain strings — LaunchConfiguration objects
    # can't be embedded directly in an f-string (that stringifies the
    # Substitution object itself, not its runtime value).
    px4_dir = LaunchConfiguration("px4_dir").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    mpc_thr_hover = LaunchConfiguration("mpc_thr_hover").perform(context)
    rviz = LaunchConfiguration("rviz").perform(context)
    launch_gazebo = LaunchConfiguration("launch_gazebo").perform(context)
    px4_model = LaunchConfiguration("px4_model").perform(context)
    px4_world = LaunchConfiguration("px4_world").perform(context)
    hover_only = LaunchConfiguration("hover_only").perform(context)

    agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
        name="micro_xrce_dds_agent",
        output="screen",
    )

    # PX4's gz_env.sh unconditionally re-exports PX4_GZ_WORLDS to PX4-Autopilot's
    # own Tools/simulation/gz/worlds — an externally-set PX4_GZ_WORLDS gets
    # clobbered before px4-rc.gzsim ever reads it. So a world sourced from this
    # package can't be pointed to by directory; it has to physically live under
    # px4_dir. Symlinking it in (re-linked every launch, so edits here take
    # effect without a PX4 rebuild) keeps this repo as the single source of
    # truth instead of hand-copying it into the PX4-Autopilot checkout.
    world_src = os.path.join(
        get_package_share_directory("argus_px4_bridge"), "worlds", f"{px4_world}.sdf")
    world_dst = os.path.join(px4_dir, "Tools", "simulation", "gz", "worlds", f"{px4_world}.sdf")
    if os.path.isfile(world_src):
        if os.path.islink(world_dst) or os.path.exists(world_dst):
            os.remove(world_dst)
        os.symlink(world_src, world_dst)

    # PX4_GZ_WORLD (the world *name*, unlike PX4_GZ_WORLDS above) isn't
    # touched by gz_env.sh, so setting it here is honored as-is.
    #
    # PX4's own px4-rc.gzsim decides whether to launch the Gazebo GUI with
    # `[ -z "${HEADLESS}" ]` — i.e. it checks whether the var is *unset/empty*,
    # not whether it equals "0". Setting HEADLESS=0 is non-empty, so it's
    # treated identically to HEADLESS=1 (no GUI). To actually show the GUI,
    # the env var must be left unset entirely.
    headless_env = f"HEADLESS={headless} " if headless not in ("0", "", "false", "False") else ""
    px4_sitl = ExecuteProcess(
        cmd=["bash", "-c",
             f"mkfifo {PX4_STDIN_FIFO} 2>/dev/null; "
             f"exec 3<>{PX4_STDIN_FIFO}; "
             f"cd '{px4_dir}' && PX4_GZ_WORLD={px4_world} {headless_env}make px4_sitl {px4_model} <&3"],
        name="px4_sitl_gz",
        output="screen",
    )

    # SITL-only preflight checks that would otherwise block arming (no
    # simulated battery/GCS present) — see powerCheck.cpp /
    # rcAndDataLinkCheck.cpp in PX4. Fired the moment the shell prompt
    # actually appears in the SITL process's own stdout, instead of guessing
    # a delay. Guarded so it only fires once — "pxh>" reappears constantly
    # (the shell redraws its prompt on every line).
    bypass_fired = {"done": False}

    def _on_px4_stdout(event):
        if bypass_fired["done"] or b"pxh>" not in event.text:
            return None
        bypass_fired["done"] = True
        return ExecuteProcess(
            cmd=["bash", "-c",
                 "printf 'param set CBRK_SUPPLY_CHK 894281\\n"
                 f"param set NAV_DLL_ACT 0\\n' > {PX4_STDIN_FIFO}"],
            name="px4_param_bypass",
        )

    set_arming_bypass_params = RegisterEventHandler(
        OnProcessIO(target_action=px4_sitl, on_stdout=_on_px4_stdout)
    )

    # Real readiness check: poll for an actual VehicleOdometry message rather
    # than assume a fixed boot time — this is the earliest point at which
    # PX4, Gazebo, and the DDS bridge are all confirmed up. `timeout 90`
    # bounds the whole wait so a broken setup fails visibly instead of
    # hanging the launch forever.
    wait_for_odometry = ExecuteProcess(
        cmd=["bash", "-c",
             "timeout 90 bash -c '"
             "until timeout 2 ros2 topic echo /fmu/out/vehicle_odometry --once >/dev/null 2>&1; "
             "do sleep 1; done'"],
        name="wait_for_px4_odometry",
        output="screen",
    )

    # Odometry flowing doesn't by itself mean the EKF/sensors have settled —
    # arming immediately on first sample was observed to trigger sensor
    # TIMEOUT failsafes in SITL. A short buffer after real data starts
    # flowing is cheap insurance and far tighter than guessing total boot time.
    bridge_node = Node(
        package="argus_px4_bridge",
        executable="argus_bridge_node",
        name="argus_bridge_node",
        output="screen",
        parameters=[{
            "mpc_thr_hover": float(mpc_thr_hover),
            "hover_only": hover_only not in ("0", "", "false", "False"),
        }],
    )

    # x500_depth's OakD-Lite sensor names these topics explicitly in its SDF
    # (see PX4-Autopilot/Tools/simulation/gz/models/OakD-Lite/model.sdf) —
    # unlike PX4's other gz topics they aren't namespaced under
    # /world/<world>/model/<model>/..., so the names below are stable
    # regardless of world name or vehicle instance.
    gz_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_camera_bridge",
        output="screen",
        arguments=[
            "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image",
            "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    def _on_wait_exit(event, _context):
        if event.returncode == 0:
            return [TimerAction(period=5.0, actions=[bridge_node])]
        # Times out (exit 124) if PX4/Gazebo/the DDS bridge never came up —
        # surface that clearly instead of silently starting the node anyway
        # (OnProcessExit fires on any exit code, success or not).
        return [ExecuteProcess(
            cmd=["bash", "-c",
                 "echo 'ERROR: PX4 odometry never appeared within 90s — "
                 "not starting argus_bridge_node. Check px4_sitl_gz output above.' >&2"],
            name="odometry_wait_failed",
        )]

    start_bridge_when_ready = RegisterEventHandler(
        OnProcessExit(target_action=wait_for_odometry, on_exit=_on_wait_exit)
    )

    # launch_gazebo=0 assumes the Agent + PX4 SITL/Gazebo are already running
    # externally (e.g. started by hand per the README's "Manual" section, or
    # a previous launch left running) — skip spawning new ones and just wait
    # for odometry from whatever's already up, then start the bridge (+ RViz).
    actions = [wait_for_odometry, start_bridge_when_ready, gz_camera_bridge]
    if launch_gazebo not in ("0", "", "false", "False"):
        actions = [agent, px4_sitl, set_arming_bypass_params] + actions

    if rviz not in ("0", "", "false", "False"):
        rviz_config = os.path.join(
            get_package_share_directory("argus_px4_bridge"), "rviz", "argus.rviz")
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
        ))

    # if headless in ("0", "", "false", "False"):
    #     # PX4's px4-rc.gzsim sends a one-shot "follow this model" camera
    #     # command right after spawning it (over the /gui/track topic, not a
    #     # latched/persistent state). If the GUI client (gz sim -g, started
    #     # separately and takes a few seconds to initialize rendering and
    #     # subscribe) isn't up and subscribed by that exact moment, it just
    #     # misses the message and the camera never finds the drone. There's
    #     # no clean external readiness signal for "GUI has subscribed", so
    #     # instead of guessing one delay, just resend the same command a few
    #     # times over a window comfortably longer than typical GUI startup —
    #     # harmless to repeat, and only one attempt needs to land after the
    #     # subscription is live.
    #     resend_camera_follow = ExecuteProcess(
    #         cmd=["bash", "-c",
    #              "for i in 1 2 3 4 5; do sleep 2; "
    #              "gz topic -t /gui/track -m gz.msgs.CameraTrack -p "
    #              "\"track_mode: FOLLOW, follow_target: {name: 'x500_0'}, "
    #              "follow_offset: {x: -2.0, y: -2.0, z: 2.0}, "
    #              "follow_pgain: 1.0, track_pgain: 1.0\" >/dev/null 2>&1; done"],
    #         name="resend_camera_follow",
    #     )
    #     actions.append(resend_camera_follow)

    return actions


def generate_launch_description():
    px4_dir_arg = DeclareLaunchArgument(
        "px4_dir",
        default_value=os.path.expanduser("~/PX4-Autopilot"),
        description="Path to the PX4-Autopilot source tree",
    )
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="0",
        description="1 to run Gazebo without a GUI, 0 to show it",
    )
    mpc_thr_hover_arg = DeclareLaunchArgument(
        "mpc_thr_hover",
        default_value="0.6",
        description="Normalized PX4 thrust at hover (must match the vehicle's MPC_THR_HOVER param)",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="1",
        description="1 to launch RViz (circle reference, drone trajectory, FOV frustum), 0 to skip",
    )
    launch_gazebo_arg = DeclareLaunchArgument(
        "launch_gazebo",
        default_value="1",
        description="1 to start MicroXRCEAgent + PX4 SITL/Gazebo, 0 to assume they're "
                     "already running externally and only start the bridge node (+ RViz)",
    )
    px4_model_arg = DeclareLaunchArgument(
        "px4_model",
        default_value="gz_x500_depth",
        description="PX4 `make px4_sitl <target>` model target, e.g. gz_x500 (no camera) or "
                     "gz_x500_depth (adds a downward-forward OakD-Lite RGB + depth camera)",
    )
    px4_world_arg = DeclareLaunchArgument(
        "px4_world",
        default_value="argus_box_world",
        description="Gazebo world name (no .sdf) — looked up in this package's worlds/ dir "
                     "and symlinked into px4_dir. argus_box_world is PX4's default.sdf plus "
                     "one static box obstacle on the circle-start hover point's boresight.",
    )
    hover_only_arg = DeclareLaunchArgument(
        "hover_only",
        default_value="1",
        description="1 to hold position at the circle-start point after takeoff instead of "
                     "handing off to MPC circle tracking — for obstacle/depth-camera testing "
                     "where the drone should stay put near argus_box_world's box",
    )

    return LaunchDescription([
        px4_dir_arg,
        headless_arg,
        mpc_thr_hover_arg,
        rviz_arg,
        launch_gazebo_arg,
        px4_model_arg,
        px4_world_arg,
        hover_only_arg,
        OpaqueFunction(function=_launch_setup),
    ])
