find_package(GTest REQUIRED)

add_executable(test_core 
    test_core.cpp
    core/test_janus_io.cpp
    core/test_janus_types.cpp
    core/test_io_coverage.cpp
    core/test_sparsity.cpp
)
target_link_libraries(test_core PRIVATE janus GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(test_function core/test_function.cpp)
target_link_libraries(test_function PRIVATE janus GTest::gtest_main)
gtest_discover_tests(test_function)

add_executable(test_math 
    math/test_arithmetic.cpp
    math/test_trig.cpp
    math/test_logic.cpp
    math/test_calculus.cpp
    math/test_autodiff.cpp
    math/test_linalg.cpp
    math/test_interpolate.cpp
    math/test_scattered_interpolator.cpp
    math/test_rotations.cpp
    math/test_surrogate.cpp
    math/test_finite_diff.cpp
    math/test_integrate_discrete.cpp
    math/test_integrate.cpp
    math/test_integrator_step.cpp
    math/test_spacing.cpp
    math/test_quaternion.cpp
    math/test_coverage_gaps.cpp
    math/test_rootfinding.cpp
)
target_link_libraries(test_math PRIVATE janus GTest::gtest_main)
gtest_discover_tests(test_math)

add_executable(test_opti
    optimization/test_opti.cpp
    optimization/test_opti_cache.cpp
    optimization/test_opti_solvers.cpp
    optimization/test_collocation.cpp
    optimization/test_multishoot.cpp
)
target_link_libraries(test_opti PRIVATE janus GTest::gtest_main)
gtest_discover_tests(test_opti)

