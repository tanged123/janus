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
echo "Running Janus Examples"
echo "========================================"

# =============================================================================
# Intro Examples
# =============================================================================
echo ""
echo "=== INTRO EXAMPLES ==="
echo ""

echo "[intro/1] Energy Intro"
./build/examples/energy_intro
echo ""

echo "[intro/2] Numeric Intro"
./build/examples/numeric_intro
echo ""

echo "[intro/3] Print Intro"
./build/examples/print_example
echo ""

echo "[intro/4] Sparsity Intro"
./build/examples/sparsity_intro
echo ""

# =============================================================================
# Math Examples
# =============================================================================
echo ""
echo "=== MATH EXAMPLES ==="
echo ""

echo "[math/1] Branching Logic"
./build/examples/branching_logic
echo ""

echo "[math/2] Loop Patterns"
./build/examples/loop_patterns
echo ""

echo "[math/3] Graph Visualization"
./build/examples/graph_visualization
echo ""

# =============================================================================
# Interpolation Examples
# =============================================================================
echo ""
echo "=== INTERPOLATION EXAMPLES ==="
echo ""

echo "[interp/1] N-Dimensional Interpolation"
./build/examples/nd_interpolation_demo
echo ""

echo "[interp/2] Root Finding"
./build/examples/rootfinding_demo
echo ""

# =============================================================================
# Simulation Examples
# =============================================================================
echo ""
echo "=== SIMULATION EXAMPLES ==="
echo ""

echo "[sim/1] Drag Coefficient"
./build/examples/drag_coefficient
echo ""

echo "[sim/2] Hybrid Simulation"
./build/examples/hybrid_sim
echo ""

echo "[sim/3] Smooth Trajectory"
./build/examples/smooth_trajectory
echo ""

echo "[sim/4] Aircraft Attitudes"
./build/examples/aircraft_attitudes
echo ""

echo "[sim/5] Brachistochrone ODE"
./build/examples/brachistochrone
echo ""

# =============================================================================
# Optimization Examples
# =============================================================================
echo ""
echo "=== OPTIMIZATION EXAMPLES ==="
echo ""

echo "[opt/1] Rosenbrock Benchmark"
./build/examples/rosenbrock
echo ""

echo "[opt/2] Brachistochrone Trajectory Optimization"
./build/examples/brachistochrone_opti
echo ""

echo "[opt/3] Beam Deflection (Structural)"
./build/examples/beam_deflection
echo ""

echo "[opt/4] Drag Optimization (Aerodynamics)"
./build/examples/drag_optimization
echo ""

echo "========================================"
echo "All examples ran successfully!"
echo "========================================"

