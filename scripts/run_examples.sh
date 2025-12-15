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

echo "[1/12] Drag Coefficient"
./build/examples/drag_coefficient
echo ""

echo "[2/12] Energy Intro"
./build/examples/energy_intro
echo ""

echo "[3/12] Numeric Intro"
./build/examples/numeric_intro
echo ""

echo "[4/12] Branching Logic"
./build/examples/branching_logic
echo ""

echo "[5/12] Loop Patterns"
./build/examples/loop_patterns
echo ""

echo "[6/12] Hybrid Simulation"
./build/examples/hybrid_sim
echo ""

echo "[7/12] Smooth Trajectory (Phase 3)"
./build/examples/smooth_trajectory
echo ""

echo "[8/12] Aircraft Attitudes (Phase 3)"
./build/examples/aircraft_attitudes
echo ""

echo "[9/12] Brachistochrone ODE (Phase 4)"
./build/examples/brachistochrone
echo ""

echo "[10/12] Graph Visualization (Phase 4)"
./build/examples/graph_visualization
echo ""

echo "[11/12] N-Dimensional Interpolation (Phase 5)"
./build/examples/nd_interpolation_demo
echo ""

echo "[12/12] Root Finding (Phase 5)"
./build/examples/rootfinding_demo
echo ""

echo "All examples ran successfully."
