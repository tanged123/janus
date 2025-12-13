find_package(GTest REQUIRED)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE janus GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(test_math 
    math/test_arithmetic.cpp
    math/test_trig.cpp
    math/test_logic.cpp
    math/test_diffops.cpp
    math/test_linalg.cpp
    math/test_interpolate.cpp
    math/test_geometry.cpp
)
target_link_libraries(test_math PRIVATE janus GTest::gtest_main)
gtest_discover_tests(test_math)
