#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Calculus.hpp>
#include <janus/math/Linalg.hpp>

template <typename Scalar> void test_gradient_uniform() {
    using Vector = janus::JanusVector<Scalar>;

    // Test case 1: Linear function y = 2x
    // dy/dx should be 2 everywhere
    Vector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;
    Vector y = 2.0 * x.array();

    auto grad = janus::gradient(y, 1.0, 1, 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        // All points should have gradient = 2
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), 2.0, 1e-10);
        }
    } else {
        auto grad_eval = janus::eval(grad);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), 2.0, 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_quadratic() {
    using Vector = janus::JanusVector<Scalar>;

    // Test case 2: Quadratic y = x^2
    // dy/dx = 2x
    Vector x(11);
    for (int i = 0; i < 11; ++i) {
        x(i) = static_cast<Scalar>(i - 5);
    }
    Vector y = x.array().square();

    // Test with edge_order = 2 for better boundary accuracy
    auto grad = janus::gradient(y, 1.0, 2, 1);
    Vector expected = 2.0 * x.array();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), expected(i), 1e-10);
        }
    } else {
        auto grad_eval = janus::eval(grad);
        auto expected_eval = janus::eval(expected);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_second_derivative() {
    using Vector = janus::JanusVector<Scalar>;

    // Test case 3: Quadratic y = x^2
    // d^2y/dx^2 = 2 everywhere
    Vector x(11);
    for (int i = 0; i < 11; ++i) {
        x(i) = static_cast<Scalar>(i);
    }
    Vector y = x.array().square();

    auto grad2 = janus::gradient(y, 1.0, 1, 2);

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad2.size(); ++i) {
            EXPECT_NEAR(grad2(i), 2.0, 1e-10);
        }
    } else {
        auto grad2_eval = janus::eval(grad2);
        for (int i = 0; i < grad2_eval.size(); ++i) {
            EXPECT_NEAR(grad2_eval(i), 2.0, 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_nonuniform() {
    using Vector = janus::JanusVector<Scalar>;

    // Test case 4: Non-uniform grid
    Vector x(5);
    x << 0.0, 1.0, 3.0, 6.0, 10.0;
    Vector y = x.array().square();

    // gradient should handle non-uniform spacing via x vector
    auto grad = janus::gradient(y, x, 2, 1);
    Vector expected = 2.0 * x.array();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), expected(i), 1e-8);
        }
    } else {
        auto grad_eval = janus::eval(grad);
        auto expected_eval = janus::eval(expected);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 1e-8);
        }
    }
}

template <typename Scalar> void test_gradient_cubic() {
    using Vector = janus::JanusVector<Scalar>;

    // Test case 5: Cubic y = x^3
    // dy/dx = 3x^2
    Vector x(9);
    for (int i = 0; i < 9; ++i) {
        x(i) = static_cast<Scalar>(i - 4);
    }
    Vector y = x.array().cube();

    auto grad = janus::gradient(y, 1.0, 2, 1);
    Vector expected = 3.0 * x.array().square();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 1; i < grad.size() - 1; ++i) {
            // Interior points: second-order formula has O(h^2) error for cubics
            // With h=1, we expect errors ~1
            EXPECT_NEAR(grad(i), expected(i), 2.0);
        }
        // Boundaries have larger error
        EXPECT_NEAR(grad(0), expected(0), 5.0);
        EXPECT_NEAR(grad(8), expected(8), 5.0);
    } else {
        auto grad_eval = janus::eval(grad);
        auto expected_eval = janus::eval(expected);
        for (int i = 1; i < grad_eval.size() - 1; ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 2.0);
        }
    }
}

template <typename Scalar> void test_gradient_edge_cases() {
    using Vector = janus::JanusVector<Scalar>;

    // Test with 2 points
    Vector x2(2);
    x2 << 0.0, 1.0;
    Vector y2 = x2.array() * 3.0;

    auto grad2 = janus::gradient(y2, 1.0, 1, 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(grad2(0), 3.0, 1e-10);
        EXPECT_NEAR(grad2(1), 3.0, 1e-10);
    } else {
        auto grad2_eval = janus::eval(grad2);
        EXPECT_NEAR(grad2_eval(0), 3.0, 1e-9);
        EXPECT_NEAR(grad2_eval(1), 3.0, 1e-9);
    }
}

TEST(CalculusTests, GradientUniformNumeric) { test_gradient_uniform<double>(); }

TEST(CalculusTests, GradientUniformSymbolic) { test_gradient_uniform<janus::SymbolicScalar>(); }

TEST(CalculusTests, GradientQuadraticNumeric) { test_gradient_quadratic<double>(); }

TEST(CalculusTests, GradientQuadraticSymbolic) { test_gradient_quadratic<janus::SymbolicScalar>(); }

TEST(CalculusTests, GradientSecondDerivativeNumeric) { test_gradient_second_derivative<double>(); }

TEST(CalculusTests, GradientSecondDerivativeSymbolic) {
    test_gradient_second_derivative<janus::SymbolicScalar>();
}

TEST(CalculusTests, GradientNonuniformNumeric) { test_gradient_nonuniform<double>(); }

TEST(CalculusTests, GradientNonuniformSymbolic) {
    test_gradient_nonuniform<janus::SymbolicScalar>();
}

TEST(CalculusTests, GradientCubicNumeric) { test_gradient_cubic<double>(); }

TEST(CalculusTests, GradientCubicSymbolic) { test_gradient_cubic<janus::SymbolicScalar>(); }

TEST(CalculusTests, GradientEdgeCasesNumeric) { test_gradient_edge_cases<double>(); }

TEST(CalculusTests, GradientEdgeCasesSymbolic) {
    test_gradient_edge_cases<janus::SymbolicScalar>();
}

// --- Tests for diff, trapz, gradient_1d ---

template <typename Scalar> void test_diff_trapz_gradient1d() {
    using Vector = janus::JanusVector<Scalar>;

    // Test diff
    Vector v(4);
    v << 0.0, 1.0, 4.0, 9.0;
    auto res_diff = janus::diff(v); // [1, 3, 5]

    // Test trapz
    Vector y(2);
    y << 1.0, 1.0;
    Vector x(2);
    x << 0.0, 1.0;
    auto res_trapz = janus::trapz(y, x);

    // Test gradient_1d
    Vector x_grad(5);
    x_grad << 0.0, 1.0, 2.0, 3.0, 4.0;
    Vector y_grad(5);
    y_grad << 0.0, 1.0, 4.0, 9.0, 16.0;
    auto res_grad = janus::gradient_1d(y_grad, x_grad);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(res_diff.size(), 3);
        EXPECT_DOUBLE_EQ(res_diff(0), 1.0);
        EXPECT_DOUBLE_EQ(res_diff(1), 3.0);

        EXPECT_DOUBLE_EQ(res_trapz, 1.0);

        EXPECT_DOUBLE_EQ(res_grad(1), 2.0); // Exact for quadratic
        EXPECT_DOUBLE_EQ(res_grad(2), 4.0);
    } else {
        EXPECT_EQ(res_diff.size(), 3);
        auto res_diff_eval = janus::eval(res_diff);
        EXPECT_DOUBLE_EQ(res_diff_eval(0), 1.0);
        EXPECT_DOUBLE_EQ(res_diff_eval(1), 3.0);

        EXPECT_DOUBLE_EQ(janus::eval(res_trapz), 1.0);

        auto res_grad_eval = janus::eval(res_grad);
        EXPECT_DOUBLE_EQ(res_grad_eval(1), 2.0);
        EXPECT_DOUBLE_EQ(res_grad_eval(2), 4.0);
    }
}

TEST(CalculusTests, DiffTrapzGradient1dNumeric) { test_diff_trapz_gradient1d<double>(); }

TEST(CalculusTests, DiffTrapzGradient1dSymbolic) {
    test_diff_trapz_gradient1d<janus::SymbolicScalar>();
}

// --- Periodic and Error Tests ---

template <typename Scalar> void test_gradient_periodic_wrapper() {
    using Vector = janus::JanusVector<Scalar>;
    Vector x(5);
    x << 0, 90, 180, 270, 360; // Degrees
    Vector y(5);
    // sin(x)
    y << 0, 1, 0, -1, 0;

    // This just dispatches to gradient() for now, but we want to cover the call
    auto grad = janus::gradient_periodic(y, 90.0, 360.0);

    // Check sizes
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(grad.size(), 5);
    } else {
        auto g = janus::eval(grad);
        EXPECT_EQ(g.size(), 5);
    }
}

TEST(CalculusTests, GradientPeriodic) {
    test_gradient_periodic_wrapper<double>();
    test_gradient_periodic_wrapper<janus::SymbolicScalar>();
}

TEST(CalculusTests, Errors) {
    janus::JanusVector<double> x(5);
    x.setZero();
    janus::JanusVector<double> y = x;

    // Invalid dx size (must be scalar, N, or N-1)
    janus::JanusVector<double> bad_dx(2);
    EXPECT_THROW(janus::gradient(y, bad_dx), std::invalid_argument);

    // Invalid edge_order
    EXPECT_THROW(janus::gradient(y, 1.0, 3), std::invalid_argument);

    // Invalid n (derivative order)
    EXPECT_THROW(janus::gradient(y, 1.0, 1, 3), std::invalid_argument);
}
