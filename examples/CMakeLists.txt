add_executable(drag_coefficient drag_coefficient.cpp)
target_link_libraries(drag_coefficient PRIVATE janus)

add_executable(energy_intro energy_intro.cpp)
target_link_libraries(energy_intro PRIVATE janus)

add_executable(numeric_intro numeric_intro.cpp)
target_link_libraries(numeric_intro PRIVATE janus)

add_executable(branching_logic branching_logic.cpp)
target_link_libraries(branching_logic PRIVATE janus)

add_executable(loop_patterns loop_patterns.cpp)
target_link_libraries(loop_patterns PRIVATE janus)

add_executable(hybrid_sim hybrid_sim.cpp)
target_link_libraries(hybrid_sim PRIVATE janus)

add_executable(print_example print_example.cpp)
target_link_libraries(print_example PRIVATE janus)

# Phase 3 Graduation Examples
add_executable(smooth_trajectory smooth_trajectory.cpp)
target_link_libraries(smooth_trajectory PRIVATE janus)

add_executable(aircraft_attitudes aircraft_attitudes.cpp)
target_link_libraries(aircraft_attitudes PRIVATE janus)

# Phase 4 Examples
add_executable(brachistochrone brachistochrone.cpp)
target_link_libraries(brachistochrone PRIVATE janus)

add_executable(graph_visualization graph_visualization.cpp)
target_link_libraries(graph_visualization PRIVATE janus)

# Phase 5 Demonstrations
add_executable(nd_interpolation_demo nd_interpolation_demo.cpp)
target_link_libraries(nd_interpolation_demo PRIVATE janus)

add_executable(rootfinding_demo rootfinding_demo.cpp)
target_link_libraries(rootfinding_demo PRIVATE janus)
