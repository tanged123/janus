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

echo "[1/10] Drag Coefficient"
./build/examples/drag_coefficient
echo ""

echo "[2/10] Energy Intro"
./build/examples/energy_intro
echo ""

echo "[3/10] Numeric Intro"
./build/examples/numeric_intro
echo ""

echo "[4/10] Branching Logic"
./build/examples/branching_logic
echo ""

echo "[5/10] Loop Patterns"
./build/examples/loop_patterns
echo ""

echo "[6/10] Hybrid Simulation"
./build/examples/hybrid_sim
echo ""

echo "[7/10] Smooth Trajectory (Phase 3)"
./build/examples/smooth_trajectory
echo ""

echo "[8/10] Aircraft Attitudes (Phase 3)"
./build/examples/aircraft_attitudes
echo ""

echo "[9/10] Brachistochrone ODE (Phase 4)"
./build/examples/brachistochrone
echo ""

echo "[10/10] Graph Visualization (Phase 4)"
./build/examples/graph_visualization
echo ""

echo "All examples ran successfully."

