cmake_minimum_required(VERSION 3.20)
project(
  janus
  VERSION 1.3.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Eigen3 3.4 REQUIRED)
find_package(casadi REQUIRED)

# --- Main Library ---
add_library(janus INTERFACE)
target_include_directories(
  janus INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link Dependencies
target_link_libraries(janus INTERFACE Eigen3::Eigen casadi)

# --- Coverage ---
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
  else()
    message(WARNING "Coverage not supported for this compiler")
  endif()
endif()

# --- Install ---
include(GNUInstallDirs)

install(
  TARGETS janus
  EXPORT janusTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT janusTargets
  FILE janusConfig.cmake
  NAMESPACE janus::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/janus)

# --- Testing ---
enable_testing()
add_subdirectory(tests)

# --- Examples ---
add_subdirectory(examples)
