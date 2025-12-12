find_package(GTest REQUIRED)

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE janus GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_core)
