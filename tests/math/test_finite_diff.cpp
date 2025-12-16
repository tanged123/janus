#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/janus.hpp>

template <typename Scalar> void test_finite_difference() {
    using Vector = janus::JanusVector<Scalar>;

    // Case 1: Central difference for 1st derivative
    // Stencil: [-1, 0, 1] at x0=0
    Vector x(3);
    x << -1.0, 0.0, 1.0;

    auto coeffs = janus::finite_difference_coefficients(x, Scalar(0.0), 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(coeffs.size(), 3);
        EXPECT_NEAR(coeffs(0), -0.5, 1e-9);
        EXPECT_NEAR(coeffs(1), 0.0, 1e-9);
        EXPECT_NEAR(coeffs(2), 0.5, 1e-9);
    } else {
        auto coeffs_eval = janus::eval(coeffs);
        EXPECT_EQ(coeffs_eval.size(), 3);
        EXPECT_NEAR(coeffs_eval(0), -0.5, 1e-9);
        EXPECT_NEAR(coeffs_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(coeffs_eval(2), 0.5, 1e-9);
    }

    // Case 2: Central difference for 2nd derivative
    // Stencil: [-1, 0, 1] at x0=0 -> [1, -2, 1]
    auto coeffs2 = janus::finite_difference_coefficients(x, Scalar(0.0), 2);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs2(0), 1.0, 1e-9);
        EXPECT_NEAR(coeffs2(1), -2.0, 1e-9);
        EXPECT_NEAR(coeffs2(2), 1.0, 1e-9);
    } else {
        auto coeffs2_eval = janus::eval(coeffs2);
        EXPECT_NEAR(coeffs2_eval(0), 1.0, 1e-9);
        EXPECT_NEAR(coeffs2_eval(1), -2.0, 1e-9);
        EXPECT_NEAR(coeffs2_eval(2), 1.0, 1e-9);
    }

    // Case 3: Forward difference (one-sided)
    // Stencil: [0, 1, 2] at x0=0 -> [-1.5, 2, -0.5]
    Vector x_fwd(3);
    x_fwd << 0.0, 1.0, 2.0;
    auto coeffs_fwd = janus::finite_difference_coefficients(x_fwd, Scalar(0.0), 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs_fwd(0), -1.5, 1e-9);
        EXPECT_NEAR(coeffs_fwd(1), 2.0, 1e-9);
        EXPECT_NEAR(coeffs_fwd(2), -0.5, 1e-9);
    } else {
        auto coeffs_fwd_eval = janus::eval(coeffs_fwd);
        EXPECT_NEAR(coeffs_fwd_eval(0), -1.5, 1e-9);
        EXPECT_NEAR(coeffs_fwd_eval(1), 2.0, 1e-9);
        EXPECT_NEAR(coeffs_fwd_eval(2), -0.5, 1e-9);
    }

    // Case 4: Non-uniform grid
    // x = [0, 1, 3] at x0=0
    // Analytic for f(x) = x^2 (derivative 2x at 0 is 0)
    // f(0)=0, f(1)=1, f(3)=9
    // coeffs dot [0, 1, 9] should be derivative approximations.
    // 1st derivative of x^2 is 2x -> at 0 is 0.
    // 2nd derivative is 2 -> at 0 is 2.

    // Let's check coefficients directly for 1st derivative?
    // Or check against property.
    Vector x_nonuni(3);
    x_nonuni << 0.0, 1.0, 3.0;
    auto coeffs_nu = janus::finite_difference_coefficients(x_nonuni, Scalar(0.0), 1);

    // Check property on f(x) = x
    // f(0)=0, f(1)=1, f(3)=3 -> approx deriv = c0*0 + c1*1 + c2*3 = 1
    // f(x) = 1 -> c0+c1+c2 = 0
    // f(x) = x^2 -> c0*0 + c1*1 + c2*9 = 0 (deriv at 0 is 0)

    // c1 + 3c2 = 1
    // c1 + 9c2 = 0
    // -> 6c2 = -1 -> c2 = -1/6
    // c1 = 1 - 3(-1/6) = 1.5
    // c0 = -c1 - c2 = -1.5 + 0.1666 = -1.333
    // c0 = -3/2 - (-1/6) = -9/6 + 1/6 = -8/6 = -4/3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs_nu(0), -4.0 / 3.0, 1e-9);
        EXPECT_NEAR(coeffs_nu(1), 1.5, 1e-9);
        EXPECT_NEAR(coeffs_nu(2), -1.0 / 6.0, 1e-9);
    } else {
        auto c = janus::eval(coeffs_nu);
        EXPECT_NEAR(c(0), -4.0 / 3.0, 1e-9);
        EXPECT_NEAR(c(1), 1.5, 1e-9);
        EXPECT_NEAR(c(2), -1.0 / 6.0, 1e-9);
    }
}

TEST(FiniteDiffTests, Numeric) { test_finite_difference<double>(); }
TEST(FiniteDiffTests, Symbolic) { test_finite_difference<janus::SymbolicScalar>(); }

// =============================================================================
// Parse Integration Method Tests
// =============================================================================

TEST(FiniteDiffTests, ParseIntegrationMethod_Trapezoidal) {
    EXPECT_EQ(janus::parse_integration_method("trapezoidal"),
              janus::IntegrationMethod::Trapezoidal);
    EXPECT_EQ(janus::parse_integration_method("trapezoid"), janus::IntegrationMethod::Trapezoidal);
    EXPECT_EQ(janus::parse_integration_method("midpoint"), janus::IntegrationMethod::Trapezoidal);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_ForwardEuler) {
    EXPECT_EQ(janus::parse_integration_method("forward_euler"),
              janus::IntegrationMethod::ForwardEuler);
    EXPECT_EQ(janus::parse_integration_method("forward euler"),
              janus::IntegrationMethod::ForwardEuler);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_BackwardEuler) {
    EXPECT_EQ(janus::parse_integration_method("backward_euler"),
              janus::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(janus::parse_integration_method("backward euler"),
              janus::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(janus::parse_integration_method("backwards_euler"),
              janus::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(janus::parse_integration_method("backwards euler"),
              janus::IntegrationMethod::BackwardEuler);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_Invalid) {
    EXPECT_THROW(janus::parse_integration_method("unknown"), janus::InvalidArgument);
    EXPECT_THROW(janus::parse_integration_method("runge_kutta"), janus::InvalidArgument);
}

// =============================================================================
// Weight Function Tests
// =============================================================================

TEST(FiniteDiffTests, ForwardEulerWeights) {
    auto [w0, w1] = janus::forward_euler_weights(0.1);
    EXPECT_NEAR(w0, -10.0, 1e-10);
    EXPECT_NEAR(w1, 10.0, 1e-10);
}

TEST(FiniteDiffTests, BackwardEulerWeights) {
    auto [w0, w1] = janus::backward_euler_weights(0.5);
    EXPECT_NEAR(w0, -2.0, 1e-10);
    EXPECT_NEAR(w1, 2.0, 1e-10);
}

TEST(FiniteDiffTests, CentralDifferenceWeights) {
    auto [wm, wp] = janus::central_difference_weights(1.0);
    EXPECT_NEAR(wm, -0.5, 1e-10);
    EXPECT_NEAR(wp, 0.5, 1e-10);
}

TEST(FiniteDiffTests, TrapezoidalWeights) {
    auto [w0, w1] = janus::trapezoidal_weights(2.0);
    EXPECT_NEAR(w0, 1.0, 1e-10);
    EXPECT_NEAR(w1, 1.0, 1e-10);
}

// =============================================================================
// Difference Function Tests
// =============================================================================

TEST(FiniteDiffTests, ForwardDifference) {
    janus::NumericVector f(4);
    f << 0.0, 1.0, 4.0, 9.0; // f(x) = x^2 at x = 0, 1, 2, 3

    janus::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    auto df = janus::forward_difference(f, x);

    EXPECT_EQ(df.size(), 3);
    EXPECT_NEAR(df(0), 1.0, 1e-10); // (1-0)/(1-0)
    EXPECT_NEAR(df(1), 3.0, 1e-10); // (4-1)/(2-1)
    EXPECT_NEAR(df(2), 5.0, 1e-10); // (9-4)/(3-2)
}

TEST(FiniteDiffTests, ForwardDifference_SizeMismatch) {
    janus::NumericVector f(3);
    f << 0.0, 1.0, 4.0;

    janus::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    EXPECT_THROW(janus::forward_difference(f, x), janus::InvalidArgument);
}

TEST(FiniteDiffTests, ForwardDifference_TooFewPoints) {
    janus::NumericVector f(1);
    f << 0.0;

    janus::NumericVector x(1);
    x << 0.0;

    EXPECT_THROW(janus::forward_difference(f, x), janus::InvalidArgument);
}

TEST(FiniteDiffTests, BackwardDifference) {
    janus::NumericVector f(3);
    f << 1.0, 2.0, 5.0;

    janus::NumericVector x(3);
    x << 0.0, 1.0, 3.0;

    auto df = janus::backward_difference(f, x);

    EXPECT_EQ(df.size(), 2);
    EXPECT_NEAR(df(0), 1.0, 1e-10); // (2-1)/(1-0)
    EXPECT_NEAR(df(1), 1.5, 1e-10); // (5-2)/(3-1)
}

TEST(FiniteDiffTests, CentralDifference) {
    janus::NumericVector f(5);
    f << 0.0, 1.0, 4.0, 9.0, 16.0; // f(x) = x^2

    janus::NumericVector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;

    auto df = janus::central_difference(f, x);

    EXPECT_EQ(df.size(), 3);
    EXPECT_NEAR(df(0), 2.0, 1e-10); // (4-0)/(2-0)
    EXPECT_NEAR(df(1), 4.0, 1e-10); // (9-1)/(3-1)
    EXPECT_NEAR(df(2), 6.0, 1e-10); // (16-4)/(4-2)
}

TEST(FiniteDiffTests, CentralDifference_SizeMismatch) {
    janus::NumericVector f(3);
    f << 0.0, 1.0, 4.0;

    janus::NumericVector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;

    EXPECT_THROW(janus::central_difference(f, x), janus::InvalidArgument);
}

TEST(FiniteDiffTests, CentralDifference_TooFewPoints) {
    janus::NumericVector f(2);
    f << 0.0, 1.0;

    janus::NumericVector x(2);
    x << 0.0, 1.0;

    EXPECT_THROW(janus::central_difference(f, x), janus::InvalidArgument);
}

// =============================================================================
// Integration Defects Tests
// =============================================================================

TEST(FiniteDiffTests, IntegrationDefects_Trapezoidal) {
    // Test: x(t) = t^2, xdot(t) = 2t
    janus::NumericVector t(4);
    t << 0.0, 1.0, 2.0, 3.0;

    janus::NumericVector x(4);
    x << 0.0, 1.0, 4.0, 9.0;

    janus::NumericVector xdot(4);
    xdot << 0.0, 2.0, 4.0, 6.0;

    auto defects = janus::integration_defects(x, xdot, t, janus::IntegrationMethod::Trapezoidal);

    EXPECT_EQ(defects.size(), 3);
    // Trapezoidal should be exact for linear xdot integrated
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
    EXPECT_NEAR(defects(2), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_ForwardEuler) {
    // x(t) = t, xdot(t) = 1
    janus::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    janus::NumericVector x(3);
    x << 0.0, 1.0, 2.0;

    janus::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    auto defects = janus::integration_defects(x, xdot, t, janus::IntegrationMethod::ForwardEuler);

    EXPECT_EQ(defects.size(), 2);
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_BackwardEuler) {
    // x(t) = t, xdot(t) = 1
    janus::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    janus::NumericVector x(3);
    x << 0.0, 1.0, 2.0;

    janus::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    auto defects = janus::integration_defects(x, xdot, t, janus::IntegrationMethod::BackwardEuler);

    EXPECT_EQ(defects.size(), 2);
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_SizeMismatch) {
    janus::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    janus::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    janus::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    EXPECT_THROW(janus::integration_defects(x, xdot, t), janus::InvalidArgument);
}

TEST(FiniteDiffTests, IntegrationDefects_TooFewPoints) {
    janus::NumericVector t(1);
    t << 0.0;

    janus::NumericVector x(1);
    x << 0.0;

    janus::NumericVector xdot(1);
    xdot << 1.0;

    EXPECT_THROW(janus::integration_defects(x, xdot, t), janus::InvalidArgument);
}
