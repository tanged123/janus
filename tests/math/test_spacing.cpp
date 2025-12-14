#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusIO.hpp> // for janus::eval
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Spacing.hpp>
#include <numbers>

template <typename Scalar> void test_spacing_funcs() {
    using VectorType = janus::JanusVector<Scalar>;

    Scalar start = 0.0;
    Scalar end = 10.0;
    int n = 5;

    // --- linspace ---
    VectorType res_lin = janus::linspace(start, end, n);

    // --- cosine_spacing ---
    VectorType res_cos = janus::cosine_spacing(start, end, n);

    // --- sinspace ---
    Scalar sin_start = 0.0;
    Scalar sin_end = 1.0;
    VectorType res_sin = janus::sinspace(sin_start, sin_end, n);
    VectorType res_sin_rev = janus::sinspace(sin_start, sin_end, n, true);

    // --- logspace ---
    Scalar log_start = 0.0; // 10^0 = 1
    Scalar log_end = 2.0;   // 10^2 = 100
    VectorType res_log = janus::logspace(log_start, log_end, 3);

    // --- geomspace ---
    Scalar geom_start = 1.0;
    Scalar geom_end = 100.0;
    VectorType res_geom = janus::geomspace(geom_start, geom_end, 3);

    if constexpr (std::is_same_v<Scalar, double>) {
        // linspace
        EXPECT_EQ(res_lin.size(), 5);
        EXPECT_NEAR(res_lin(2), 5.0, 1e-9);
        EXPECT_NEAR(res_lin(4), 10.0, 1e-9);

        // cosine_spacing
        EXPECT_NEAR(res_cos(0), 0.0, 1e-9);
        EXPECT_NEAR(res_cos(2), 5.0, 1e-9);

        // sinspace
        EXPECT_NEAR(res_sin(0), 0.0, 1e-9);
        EXPECT_NEAR(res_sin(n - 1), 1.0, 1e-9);
        EXPECT_LT(res_sin(1) - res_sin(0), res_sin(n - 1) - res_sin(n - 2));

        EXPECT_GT(res_sin_rev(1) - res_sin_rev(0), res_sin_rev(n - 1) - res_sin_rev(n - 2));

        // logspace
        EXPECT_NEAR(res_log(0), 1.0, 1e-9);
        EXPECT_NEAR(res_log(1), 10.0, 1e-9);
        EXPECT_NEAR(res_log(2), 100.0, 1e-9);

        // geomspace
        EXPECT_NEAR(res_geom(0), 1.0, 1e-9);
        EXPECT_NEAR(res_geom(1), 10.0, 1e-9);
        EXPECT_NEAR(res_geom(2), 100.0, 1e-9);

    } else {
        // Validation logic for Symbolic
        auto num_lin = janus::eval(res_lin);
        EXPECT_EQ(num_lin.size(), 5);
        EXPECT_NEAR(num_lin(2), 5.0, 1e-9);
        EXPECT_NEAR(num_lin(4), 10.0, 1e-9);

        auto num_cos = janus::eval(res_cos);
        EXPECT_NEAR(num_cos(0), 0.0, 1e-9);
        EXPECT_NEAR(num_cos(2), 5.0, 1e-9);

        Eigen::VectorXd num_sin = janus::eval(res_sin);
        EXPECT_NEAR(num_sin(0), 0.0, 1e-9);
        EXPECT_NEAR(num_sin(n - 1), 1.0, 1e-9);
        EXPECT_LT(num_sin(1) - num_sin(0), num_sin(n - 1) - num_sin(n - 2));

        Eigen::VectorXd num_sin_rev = janus::eval(res_sin_rev);
        EXPECT_GT(num_sin_rev(1) - num_sin_rev(0), num_sin_rev(n - 1) - num_sin_rev(n - 2));

        Eigen::VectorXd num_log = janus::eval(res_log);
        EXPECT_NEAR(num_log(0), 1.0, 1e-9);
        EXPECT_NEAR(num_log(1), 10.0, 1e-9);
        EXPECT_NEAR(num_log(2), 100.0, 1e-9);

        Eigen::VectorXd num_geom = janus::eval(res_geom);
        EXPECT_NEAR(num_geom(0), 1.0, 1e-9);
        EXPECT_NEAR(num_geom(1), 10.0, 1e-9);
        EXPECT_NEAR(num_geom(2), 100.0, 1e-9);
    }
}

TEST(SpacingTests, Numeric) { test_spacing_funcs<double>(); }

TEST(SpacingTests, Symbolic) { test_spacing_funcs<janus::SymbolicScalar>(); }
