"""Launches Micro-XRCE-DDS-Agent + PX4 SITL (Gazebo, gz_x500) + argus_bridge_node."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    px4_dir_arg = DeclareLaunchArgument(
        "px4_dir",
        default_value=os.path.expanduser("~/PX4-Autopilot"),
        description="Path to the PX4-Autopilot source tree",
    )
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="1",
        description="1 to run Gazebo without a GUI, 0 to show it",
    )
    mpc_thr_hover_arg = DeclareLaunchArgument(
        "mpc_thr_hover",
        default_value="0.6",
        description="Normalized PX4 thrust at hover (must match the vehicle's MPC_THR_HOVER param)",
    )

    agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
        name="micro_xrce_dds_agent",
        output="screen",
    )

    px4_sitl = ExecuteProcess(
        cmd=["make", "px4_sitl", "gz_x500"],
        cwd=LaunchConfiguration("px4_dir"),
        additional_env={"HEADLESS": LaunchConfiguration("headless")},
        name="px4_sitl_gz",
        output="screen",
    )

    # Give the Agent a moment to bind its UDP port, and SITL time to boot
    # and connect over uXRCE-DDS, before the bridge node starts publishing.
    bridge_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="argus_px4_bridge",
                executable="argus_bridge_node",
                name="argus_bridge_node",
                output="screen",
                parameters=[{"mpc_thr_hover": LaunchConfiguration("mpc_thr_hover")}],
            )
        ],
    )

    return LaunchDescription([
        px4_dir_arg,
        headless_arg,
        mpc_thr_hover_arg,
        agent,
        px4_sitl,
        bridge_node,
    ])
