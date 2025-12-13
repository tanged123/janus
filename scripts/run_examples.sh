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

echo "[1/3] Drag Coefficient"
./build/examples/drag_coefficient
echo ""

echo "[2/3] Energy Intro"
./build/examples/energy_intro
echo ""

echo "[3/3] Numeric Intro"
./build/examples/numeric_intro
echo ""

echo "All examples ran successfully."
