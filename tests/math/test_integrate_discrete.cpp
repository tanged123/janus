#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/janus.hpp>

template <typename Scalar> void test_integrate_core() {
    // x = [0, 1, 2]
    // f = [0, 2, 4] (linear 2x)

    // Intervals: [0, 1], [1, 2]. dx = 1.

    using Vector = janus::JanusVector<Scalar>;
    Vector x(3);
    x << 0.0, 1.0, 2.0;
    Vector f(3);
    f << 0.0, 2.0, 4.0;

    // Trapezoidal: Exact for linear.
    // Int 1: (0+2)/2 * 1 = 1.
    // Int 2: (2+4)/2 * 1 = 3.
    auto trapz = janus::integrate_discrete_intervals(f, x, true, "trapezoidal");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(trapz.size(), 2);
        EXPECT_NEAR(trapz(0), 1.0, 1e-9);
        EXPECT_NEAR(trapz(1), 3.0, 1e-9);
    } else {
        auto t = janus::eval(trapz);
        EXPECT_NEAR(t(0), 1.0, 1e-9);
        EXPECT_NEAR(t(1), 3.0, 1e-9);
    }

    // Euler
    auto fwd = janus::integrate_discrete_intervals(f, x, true, "forward_euler");
    // [0, 2] * 1 = [0, 2]
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(fwd(0), 0.0, 1e-9);
        EXPECT_NEAR(fwd(1), 2.0, 1e-9);
    } else {
        auto t = janus::eval(fwd);
        EXPECT_NEAR(t(0), 0.0, 1e-9);
        EXPECT_NEAR(t(1), 2.0, 1e-9);
    }

    auto bwd = janus::integrate_discrete_intervals(f, x, true, "backward_euler");
    // [2, 4] * 1 = [2, 4]
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(bwd(0), 2.0, 1e-9);
        EXPECT_NEAR(bwd(1), 4.0, 1e-9);
    } else {
        auto t = janus::eval(bwd);
        EXPECT_NEAR(t(0), 2.0, 1e-9);
        EXPECT_NEAR(t(1), 4.0, 1e-9);
    }

    // Simpson (Quadratic exactness)
    // Use quadratic data: f(x) = x^2 -> f = [0, 1, 4]
    Vector f_quad(3);
    f_quad << 0.0, 1.0, 4.0;

    // forward_simpson calculates [0, 1] exactly (1/3).
    // remaining_end = 1, so recurse on [1, 2] with Trapezoidal.
    // Trapz on [1, 2]: (1+4)/2 * 1 = 2.5.
    auto simpson_fwd =
        janus::integrate_discrete_intervals(f_quad, x, true, "forward_simpson", "lower_order");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(simpson_fwd.size(), 2);
        EXPECT_NEAR(simpson_fwd(0), 1.0 / 3.0, 1e-9); // Simpson exact
        EXPECT_NEAR(simpson_fwd(1), 2.5, 1e-9);       // Trapz fallback
    } else {
        auto t = janus::eval(simpson_fwd);
        EXPECT_NEAR(t(0), 1.0 / 3.0, 1e-9);
        EXPECT_NEAR(t(1), 2.5, 1e-9);
    }

    // Cubic method test (needs 4+ points)
    // x = [0, 1, 2, 3], f(x) = x^3 -> f = [0, 1, 8, 27]
    // Cubic should be exact for cubic polynomials
    Vector x_cubic(4);
    x_cubic << 0.0, 1.0, 2.0, 3.0;
    Vector f_cubic(4);
    f_cubic << 0.0, 1.0, 8.0, 27.0; // x^3

    // Integral of x^3 = x^4/4
    // Exact: [0,1] = 0.25, [1,2] = 3.75, [2,3] = 16.25
    // But endpoints use trapezoidal fallback (lower_order):
    // Trapz [0,1]: (0+1)/2 = 0.5
    // Cubic [1,2]: exact = 3.75
    // Trapz [2,3]: (8+27)/2 = 17.5
    auto cubic_res =
        janus::integrate_discrete_intervals(f_cubic, x_cubic, true, "cubic", "lower_order");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(cubic_res.size(), 3);
        EXPECT_NEAR(cubic_res(0), 0.5, 1e-6);  // First interval (trapz fallback)
        EXPECT_NEAR(cubic_res(1), 3.75, 1e-6); // Middle interval (cubic exact)
        EXPECT_NEAR(cubic_res(2), 17.5, 1e-6); // Last interval (trapz fallback)
    } else {
        auto t = janus::eval(cubic_res);
        EXPECT_NEAR(t(0), 0.5, 1e-6);
        EXPECT_NEAR(t(1), 3.75, 1e-6);
        EXPECT_NEAR(t(2), 17.5, 1e-6);
    }

    // Test squared curvature (for regularization)
    // f(x) = x^2 has f''(x) = 2, so ∫(f'')² dx = 4 * dx
    // For x = [0, 1, 2], intervals are 1 each, so total = 4 + 4 = 8
    auto curv_res = janus::integrate_discrete_squared_curvature(f_quad, x, "simpson");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(curv_res.size(), 2);
        // Curvature should be approximately 4 per interval for f(x)=x^2
        EXPECT_GT(curv_res(0), 0.0); // Positive curvature
        EXPECT_GT(curv_res(1), 0.0);
    } else {
        auto t = janus::eval(curv_res);
        EXPECT_GT(t(0), 0.0);
        EXPECT_GT(t(1), 0.0);
    }
}

TEST(IntegrateDiscrete, CoreNumeric) { test_integrate_core<double>(); }
TEST(IntegrateDiscrete, CoreSymbolic) { test_integrate_core<janus::SymbolicScalar>(); }
