#!/usr/bin/env bash
# Build, simulate, and visualise in one shot.
#
# Usage:
#   ./run.sh [mode] [options]
#
#   mode (default: moving_target):
#     circle_tracking  — ground target walks a fixed circle
#     moving_target    — ground target follows waypoints
#
#   options:
#     --skip-gen    skip solver regeneration (model unchanged)
#     --skip-build  skip cmake build (binary already up to date)
set -euo pipefail

cd "$(dirname "$0")"

MODE="moving_target"
SKIP_GEN=0
SKIP_BUILD=0

for arg in "$@"; do
  case "$arg" in
    circle_tracking)  MODE="circle_tracking" ;;
    moving_target)    MODE="moving_target"   ;;
    --skip-gen)       SKIP_GEN=1 ;;
    --skip-build)     SKIP_BUILD=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

case "$MODE" in
  circle_tracking) BIN_ARG="circle"        ;;
  moving_target)   BIN_ARG="moving_target" ;;
esac

# ACADOS shared libs must be visible to both the Python generator and the binary
export LD_LIBRARY_PATH="$HOME/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "$SKIP_GEN" -eq 0 ]; then
  echo "==> Generating C solver (scripts/generate_mpc.py)..."
  python3 scripts/generate_mpc.py
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "==> Configuring and building..."
  cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF
  cmake --build build --parallel
fi

echo "==> Running argus_mpc ($MODE)..."
./build/argus_mpc "$BIN_ARG"

echo "==> Visualising (close the window to exit)..."
python3 scripts/animate_xy.py
