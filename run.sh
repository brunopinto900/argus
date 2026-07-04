#!/usr/bin/env bash
# Build, simulate, and visualise in one shot.
#
# Usage:
#   ./run.sh                  # full pipeline: generate → build → run → plot
#   ./run.sh --skip-gen       # skip solver regeneration (model unchanged)
#   ./run.sh --skip-build     # skip cmake build (binary already up to date)
#   ./run.sh --skip-gen --skip-build   # run + plot only
set -euo pipefail

cd "$(dirname "$0")"

SKIP_GEN=0
SKIP_BUILD=0

for arg in "$@"; do
  case "$arg" in
    --skip-gen)   SKIP_GEN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

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

echo "==> Running simulation..."
./build/argus_mpc

echo "==> Visualising (close the window to exit)..."
python3 scripts/animate_xy.py
