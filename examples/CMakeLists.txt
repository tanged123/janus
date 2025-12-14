add_executable(drag_coefficient drag_coefficient.cpp)
target_link_libraries(drag_coefficient PRIVATE janus)

add_executable(energy_intro energy_intro.cpp)
target_link_libraries(energy_intro PRIVATE janus)

add_executable(numeric_intro numeric_intro.cpp)
target_link_libraries(numeric_intro PRIVATE janus)

add_executable(branching_logic branching_logic.cpp)
target_link_libraries(branching_logic PRIVATE janus)
