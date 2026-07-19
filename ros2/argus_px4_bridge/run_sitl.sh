#!/usr/bin/env bash
# Kill stale SITL/bridge processes, build argus_px4_bridge, and launch it.
#
# Usage:
#   ./run_sitl.sh [launch args...]
#
# Any arguments are forwarded to `ros2 launch argus_sitl.launch.py`, e.g.:
#   ./run_sitl.sh headless:=1 hover_only:=0
set -o pipefail
# Not -e: the process-cleanup greps below are expected to "fail" (no match)
# on a clean machine. Not -u: ROS2/colcon's generated setup.bash files
# reference unset variables (e.g. COLCON_TRACE) internally and aren't
# written to survive `nounset` — sourcing them under -u fails immediately.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGUS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# A previous run's gz sim/PX4 process not fully exiting gets silently
# reattached by PX4's own "world already running" detection on the next
# launch (wrong world/model loads, no error printed) — see the README's
# "Re-launching: check for stale processes first" section. pkill/kill have
# also been observed to report success without the process actually dying,
# so this force-kills by PID and verifies with `kill -0` rather than
# trusting exit codes alone. Kills anything matching system-wide, not just
# processes this script itself started.
echo "==> Killing stale processes..."
patterns=(
  "gz sim"
  "px4_sitl_default/bin/px4"
  "cmake -E env PX4_SIM_MODEL"
  "MicroXRCEAgent"
  "ros2 launch argus_px4_bridge"
  "ros_gz_bridge/parameter_bridge"
  "argus_px4_bridge/lib/argus_px4_bridge/argus_bridge_node"
  "rviz2/lib/rviz2/rviz2"
)
for pattern in "${patterns[@]}"; do
  pids=$(pgrep -f "$pattern" || true)
  [ -z "$pids" ] && continue
  echo "    killing '$pattern': $pids"
  kill -9 $pids 2>/dev/null || true
done
sleep 2
for pattern in "${patterns[@]}"; do
  pids=$(pgrep -f "$pattern" || true)
  if [ -n "$pids" ]; then
    echo "    WARNING: still alive after kill -9: '$pattern' ($pids)" >&2
  fi
done

echo "==> Sourcing ROS2 environment..."
# shellcheck disable=SC1090
source ~/ros2_jazzy/install/setup.bash
# shellcheck disable=SC1090
source ~/ros2_ws/install/setup.bash

echo "==> Building argus_px4_bridge..."
cd ~/ros2_ws
colcon build --packages-select argus_px4_bridge --symlink-install || exit 1

echo "==> Launching..."
cd "$ARGUS_DIR"
exec ros2 launch argus_px4_bridge argus_sitl.launch.py "$@"
