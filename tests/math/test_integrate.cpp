#include "../utils/TestUtils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <janus/janus.hpp>

// ============================================================================
// Tests for quad (definite integration)
// ============================================================================

TEST(Integrate, QuadConstant) {
    // ∫ 2 dx from 0 to 3 = 6
    auto result = janus::quad([](double x) { return 2.0; }, 0.0, 3.0);
    EXPECT_NEAR(result.value, 6.0, 1e-10);
}

TEST(Integrate, QuadLinear) {
    // ∫ x dx from 0 to 2 = x^2/2 |_0^2 = 2
    auto result = janus::quad([](double x) { return x; }, 0.0, 2.0);
    EXPECT_NEAR(result.value, 2.0, 1e-10);
}

TEST(Integrate, QuadQuadratic) {
    // ∫ x^2 dx from 0 to 1 = 1/3
    auto result = janus::quad([](double x) { return x * x; }, 0.0, 1.0);
    EXPECT_NEAR(result.value, 1.0 / 3.0, 1e-10);
}

TEST(Integrate, QuadCubic) {
    // ∫ x^3 dx from 0 to 2 = x^4/4 |_0^2 = 4
    auto result = janus::quad([](double x) { return x * x * x; }, 0.0, 2.0);
    EXPECT_NEAR(result.value, 4.0, 1e-10);
}

TEST(Integrate, QuadTrig) {
    // ∫ sin(x) dx from 0 to π = -cos(x) |_0^π = -(-1) - (-1) = 2
    auto result = janus::quad([](double x) { return std::sin(x); }, 0.0, M_PI);
    EXPECT_NEAR(result.value, 2.0, 1e-10);
}

TEST(Integrate, QuadExp) {
    // ∫ e^x dx from 0 to 1 = e - 1 ≈ 1.718281828
    auto result = janus::quad([](double x) { return std::exp(x); }, 0.0, 1.0);
    EXPECT_NEAR(result.value, std::exp(1.0) - 1.0, 1e-10);
}

TEST(Integrate, QuadGaussian) {
    // ∫ e^(-x^2) dx from -3 to 3 ≈ √π ≈ 1.7724538509
    auto result = janus::quad([](double x) { return std::exp(-x * x); }, -3.0, 3.0);
    EXPECT_NEAR(result.value, std::sqrt(M_PI), 1e-4); // Less accurate for unbounded
}

// ============================================================================
// Tests for quad (symbolic)
// ============================================================================

TEST(IntegrateSymbolic, QuadBasic) {
    // ∫ x dx from 0 to 1 = 0.5 (symbolic)
    auto x = janus::sym("x");
    auto expr = x; // f(x) = x

    auto result = janus::quad(expr, x, 0.0, 1.0);

    // Use janus::eval for clean evaluation
    double val = janus::eval(result.value);
    EXPECT_NEAR(val, 0.5, 1e-6);
}

TEST(IntegrateSymbolic, QuadSquare) {
    // ∫ x^2 dx from 0 to 1 = 1/3 (symbolic)
    auto x = janus::sym("x");
    auto expr = x * x;

    auto result = janus::quad(expr, x, 0.0, 1.0);

    // Use janus::eval for clean evaluation
    double val = janus::eval(result.value);
    EXPECT_NEAR(val, 1.0 / 3.0, 1e-6);
}

// ============================================================================
// Tests for solve_ivp (numeric) - Using Janus native types
// ============================================================================

TEST(Integrate, SolveIvpExponentialDecay) {
    // dy/dt = -λy, y(0) = y0
    // Exact: y(t) = y0 * e^(-λt)
    double lambda = 0.5;
    double y0_val = 2.5;

    // Use initializer list for clean API
    auto sol =
        janus::solve_ivp([lambda](double t, const janus::NumericVector &y) { return -lambda * y; },
                         {0.0, 4.0}, {y0_val}, // initializer list syntax
                         50);

    EXPECT_TRUE(sol.success);
    EXPECT_EQ(sol.t.size(), 50);
    EXPECT_EQ(sol.y.cols(), 50);
    EXPECT_EQ(sol.y.rows(), 1);

    // Check initial condition
    EXPECT_NEAR(sol.y(0, 0), y0_val, 1e-10);

    // Check solution at t=4
    double t_final = sol.t(49);
    double y_exact = y0_val * std::exp(-lambda * t_final);
    EXPECT_NEAR(sol.y(0, 49), y_exact, 1e-4);
}

TEST(Integrate, SolveIvpHarmonicOscillator) {
    // d²y/dt² = -ω²y  =>  y' = v, v' = -ω²y
    // State: [y, v]
    // y(0) = 1, v(0) = 0  =>  y(t) = cos(ωt)
    double omega = 2.0;

    // Use initializer list for multi-state initial condition
    auto sol = janus::solve_ivp(
        [omega](double t, const janus::NumericVector &state) {
            janus::NumericVector dydt(2);
            dydt << state(1), -omega * omega * state(0);
            return dydt;
        },
        {0.0, M_PI / omega}, // One half period
        {1.0, 0.0},          // y=1, v=0 as initializer list
        100);

    EXPECT_TRUE(sol.success);

    // At t = π/ω, y should be -1 (half period of cosine)
    EXPECT_NEAR(sol.y(0, 99), -1.0, 1e-3);
    // Velocity should be ~0 at extremum
    EXPECT_NEAR(sol.y(1, 99), 0.0, 1e-2);
}

TEST(Integrate, SolveIvpLogistic) {
    // dy/dt = ry(1 - y/K)
    // With y(0) = 0.1, r = 1, K = 1
    // As t -> inf, y -> K = 1
    double r = 1.0;
    double K = 1.0;
    double y0_val = 0.1;

    auto sol = janus::solve_ivp(
        [r, K](double t, const janus::NumericVector &y) {
            janus::NumericVector dydt(1);
            dydt << r * y(0) * (1 - y(0) / K);
            return dydt;
        },
        {0.0, 10.0}, {y0_val}, 100);

    EXPECT_TRUE(sol.success);

    // At t=10, should be very close to carrying capacity K=1
    EXPECT_NEAR(sol.y(0, 99), K, 0.01);
}

// ============================================================================
// Tests for solve_ivp_expr (expression-based symbolic)
// ============================================================================

TEST(IntegrateSymbolic, SolveIvpExprExponential) {
    // dy/dt = -0.5*y, y(0) = 2.5
    // Exact: y(t) = 2.5 * e^(-0.5t)
    double lambda = 0.5;
    double y0_val = 2.5;

    auto t = janus::sym("t");
    auto y = janus::sym("y");
    auto ode = -lambda * y;

    // Use NumericVector
    janus::NumericVector y0(1);
    y0(0) = y0_val;

    auto sol = janus::solve_ivp_expr(ode, t, y, {0.0, 4.0}, y0, 50);

    EXPECT_TRUE(sol.success);

    // Check final value
    double t_final = sol.t(49);
    double y_exact = y0_val * std::exp(-lambda * t_final);
    EXPECT_NEAR(sol.y(0, 49), y_exact, 1e-4);
}

TEST(IntegrateSymbolic, SolveIvpExprOscillator) {
    // y'' = -ω²y  =>  y' = v, v' = -ω²y
    double omega = 2.0;

    auto t = janus::sym("t");
    auto state = janus::sym("state", 2); // [y, v]

    janus::SymbolicScalar ode_y = state(1);                  // dy/dt = v
    janus::SymbolicScalar ode_v = -omega * omega * state(0); // dv/dt = -ω²y

    casadi::MX ode = casadi::MX::vertcat({ode_y, ode_v});

    janus::NumericVector y0(2);
    y0 << 1.0, 0.0;

    auto sol = janus::solve_ivp_expr(ode, t, state, {0.0, M_PI / omega}, y0, 100);

    EXPECT_TRUE(sol.success);

    // At half period, y ≈ -1
    EXPECT_NEAR(sol.y(0, 99), -1.0, 1e-3);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(Integrate, QuadZeroInterval) {
    // ∫ f(x) dx from a to a = 0
    auto result = janus::quad([](double x) { return x * x + 1; }, 2.0, 2.0);
    EXPECT_NEAR(result.value, 0.0, 1e-14);
}

TEST(Integrate, QuadNegativeInterval) {
    // ∫ x dx from 2 to 0 = -∫ x dx from 0 to 2 = -2
    auto result = janus::quad([](double x) { return x; }, 2.0, 0.0);
    EXPECT_NEAR(result.value, -2.0, 1e-10);
}

TEST(Integrate, SolveIvpSingleStep) {
    // Minimal case: 2 output points using initializer list
    auto sol = janus::solve_ivp([](double t, const janus::NumericVector &y) { return -y; },
                                {0.0, 1.0}, {1.0}, 2);

    EXPECT_TRUE(sol.success);
    EXPECT_EQ(sol.t.size(), 2);
    EXPECT_NEAR(sol.y(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(sol.y(0, 1), std::exp(-1.0), 1e-3);
}
