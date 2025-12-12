cmake_minimum_required(VERSION 3.20)
project(
  janus
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Eigen3 3.4 REQUIRED)
find_package(casadi REQUIRED)

# --- Main Library ---
add_library(janus INTERFACE)
target_include_directories(janus INTERFACE include)

# Link Dependencies
target_link_libraries(janus INTERFACE Eigen3::Eigen casadi)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
