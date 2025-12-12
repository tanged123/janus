find_package(GTest REQUIRED)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE janus GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE janus GTest::gtest_main)
gtest_discover_tests(test_math)
