add_executable(drag_coefficient drag_coefficient.cpp)
target_link_libraries(drag_coefficient PRIVATE janus)

add_executable(energy_intro energy_intro.cpp)
target_link_libraries(energy_intro PRIVATE janus)
