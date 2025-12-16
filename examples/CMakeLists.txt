# =============================================================================
# Intro Examples
# =============================================================================
add_executable(numeric_intro intro/numeric_intro.cpp)
target_link_libraries(numeric_intro PRIVATE janus)

add_executable(energy_intro intro/energy_intro.cpp)
target_link_libraries(energy_intro PRIVATE janus)

add_executable(print_example intro/print_example.cpp)
target_link_libraries(print_example PRIVATE janus)

add_executable(sparsity_intro intro/sparsity_intro.cpp)
target_link_libraries(sparsity_intro PRIVATE janus)

# =============================================================================
# Math Examples
# =============================================================================
add_executable(branching_logic math/branching_logic.cpp)
target_link_libraries(branching_logic PRIVATE janus)

add_executable(loop_patterns math/loop_patterns.cpp)
target_link_libraries(loop_patterns PRIVATE janus)

add_executable(graph_visualization math/graph_visualization.cpp)
target_link_libraries(graph_visualization PRIVATE janus)

# =============================================================================
# Interpolation Examples
# =============================================================================
add_executable(nd_interpolation_demo interpolation/nd_interpolation_demo.cpp)
target_link_libraries(nd_interpolation_demo PRIVATE janus)

add_executable(rootfinding_demo interpolation/rootfinding_demo.cpp)
target_link_libraries(rootfinding_demo PRIVATE janus)

# =============================================================================
# Simulation Examples
# =============================================================================
add_executable(drag_coefficient simulation/drag_coefficient.cpp)
target_link_libraries(drag_coefficient PRIVATE janus)

add_executable(hybrid_sim simulation/hybrid_sim.cpp)
target_link_libraries(hybrid_sim PRIVATE janus)

add_executable(smooth_trajectory simulation/smooth_trajectory.cpp)
target_link_libraries(smooth_trajectory PRIVATE janus)

add_executable(aircraft_attitudes simulation/aircraft_attitudes.cpp)
target_link_libraries(aircraft_attitudes PRIVATE janus)

add_executable(brachistochrone simulation/brachistochrone.cpp)
target_link_libraries(brachistochrone PRIVATE janus)

# =============================================================================
# Optimization Examples (Phase 6)
# =============================================================================
add_executable(rosenbrock optimization/rosenbrock.cpp)
target_link_libraries(rosenbrock PRIVATE janus)

add_executable(brachistochrone_opti optimization/brachistochrone_opti.cpp)
target_link_libraries(brachistochrone_opti PRIVATE janus)

add_executable(beam_deflection optimization/beam_deflection.cpp)
target_link_libraries(beam_deflection PRIVATE janus)

add_executable(drag_optimization optimization/drag_optimization.cpp)
target_link_libraries(drag_optimization PRIVATE janus)



