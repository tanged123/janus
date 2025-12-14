#!/usr/bin/env bash
set -e

# Ensure we have a build
if [ ! -d "build" ]; then
    echo "Build directory not found. Building..."
    ./scripts/build.sh
fi

# Rebuild to ensure latest changes
ninja -C build

echo "========================================"
echo "Running Examples"
echo "========================================"

echo "[1/4] Drag Coefficient"
./build/examples/drag_coefficient
echo ""

echo "[2/4] Energy Intro"
./build/examples/energy_intro
echo ""

echo "[3/4] Numeric Intro"
./build/examples/numeric_intro
echo ""

echo "[4/4] Branching Logic"
./build/examples/branching_logic
echo ""

echo "All examples ran successfully."
