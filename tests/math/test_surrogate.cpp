#include "../utils/TestUtils.hpp" // specific path to TestUtils
#include "janus/janus.hpp"
#include "janus/math/SurrogateModel.hpp"
#include <gtest/gtest.h>

namespace janus {
namespace test {
namespace {

template <typename Fn> double forward_difference(Fn &&fn, double x, double h) {
    return (fn(x + h) - fn(x)) / h;
}

template <typename Fn> double backward_difference(Fn &&fn, double x, double h) {
    return (fn(x) - fn(x - h)) / h;
}

template <typename Fn> double forward_second_difference(Fn &&fn, double x, double h) {
    return (fn(x + 2.0 * h) - 2.0 * fn(x + h) + fn(x)) / (h * h);
}

template <typename Fn> double backward_second_difference(Fn &&fn, double x, double h) {
    return (fn(x) - 2.0 * fn(x - h) + fn(x - 2.0 * h)) / (h * h);
}

double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }

} // namespace

// ======================================================================
// Softmax Tests
// ======================================================================

TEST(SurrogateTests, SoftmaxNumeric) {
    std::vector<double> vals = {1.0, 2.0, 3.0};

    // With softness -> 0 (hard max), should approach max(vals) = 3.0
    // But implementation divides by softness, so we can't use 0.
    // Use small softness.
    double hardish = softmax(vals, 0.01);
    EXPECT_NEAR(hardish, 3.0, 1e-2);

    // With large softness, should be smoother/larger
    double soft = softmax(vals, 1.0);
    // Formula: 3 + 1 * log(exp(-2) + exp(-1) + exp(0))
    // exp(0)=1, exp(-1)~0.368, exp(-2)~0.135 -> sum ~ 1.503
    // log(1.503) ~ 0.407
    // result ~ 3.407
    EXPECT_GT(soft, 3.0);
    EXPECT_NEAR(soft, 3.0 + std::log(std::exp(-2) + std::exp(-1) + std::exp(0)), 1e-5);
}

TEST(SurrogateTests, SoftmaxSymbolic) {
    // Symbolic check
    casadi::MX x = casadi::MX::sym("x");
    casadi::MX y = casadi::MX::sym("y");
    std::vector<casadi::MX> args = {x, y};

    // Just structural check that it compiles and returns an expression
    auto res = softmax(args);

    // Evaluate
    janus::Function f({x, y}, {res});
    // f(1, 3) with default softness 1.0
    // max=3. sum= exp(-2)+exp(0) = 0.135+1 = 1.135. log(1.135)~0.126. res 3.126
    auto val = f.eval(1.0, 3.0);
    EXPECT_NEAR(static_cast<double>(val(0, 0)), 3.0 + std::log(std::exp(-2) + 1.0), 1e-5);
}

TEST(SurrogateTests, Softmax2Arg) {
    // Convenience overload
    double val = softmax(1.0, 5.0, 0.1);
    EXPECT_NEAR(val, 5.0, 0.1); // Close to max
}

// ======================================================================
// Softmin Tests
// ======================================================================

TEST(SurrogateTests, SoftminNumeric) {
    std::vector<double> vals = {1.0, 2.0, 3.0};

    // Should be close to min = 1.0
    double val = softmin(vals, 0.01);
    EXPECT_NEAR(val, 1.0, 1e-2);
}

TEST(SurrogateTests, SoftminSymbolic) {
    auto x = janus::sym("x");
    std::vector<casadi::MX> args = {x, casadi::MX(2.0)};
    auto expr = softmin(args, 0.1);

    // softmin(1.0, 2.0) ~ min(1,2) = 1
    double val = janus::Function({x}, {expr}).eval(1.0)(0, 0);
    EXPECT_NEAR(val, 1.0, 0.1);
}

// ======================================================================
// Softplus Tests
// ======================================================================

TEST(SurrogateTests, SoftplusNumeric) {
    // softplus(0) = log(1+1) = log(2) ~ 0.693 (beta=1)
    EXPECT_NEAR(softplus(0.0), std::log(2.0), 1e-5);

    // Large negative -> 0
    EXPECT_NEAR(softplus(-100.0), 0.0, 1e-5);

    // Large positive -> linear
    EXPECT_NEAR(softplus(1000.0), 1000.0, 1e-5);
}

TEST(SurrogateTests, SoftplusSymbolic) {
    auto x = janus::sym("x");
    auto expr = softplus(x);
    janus::Function f({x}, {expr});

    EXPECT_NEAR(f.eval(0.0)(0, 0), std::log(2.0), 1e-5);
    EXPECT_NEAR(f.eval(1000.0)(0, 0), 1000.0, 1e-5);
}

TEST(SurrogateTests, SoftplusNumericDerivativesRemainSmoothAcrossLegacyThreshold) {
    constexpr double beta = 3.0;
    constexpr double threshold = 2.0;
    constexpr double transition = threshold / beta;
    constexpr double h = 1e-4;

    auto fn = [&](double x) { return softplus(x, beta, threshold); };

    const double d1_left = backward_difference(fn, transition, h);
    const double d1_right = forward_difference(fn, transition, h);
    const double d2_left = backward_second_difference(fn, transition, h);
    const double d2_right = forward_second_difference(fn, transition, h);

    const double sigma = logistic(beta * transition);
    const double expected_second = beta * sigma * (1.0 - sigma);

    EXPECT_NEAR(d1_left, sigma, 5e-4);
    EXPECT_NEAR(d1_right, sigma, 5e-4);
    EXPECT_NEAR(d1_left, d1_right, 5e-4);

    EXPECT_NEAR(d2_left, expected_second, 5e-3);
    EXPECT_NEAR(d2_right, expected_second, 5e-3);
    EXPECT_NEAR(d2_left, d2_right, 5e-3);
}

TEST(SurrogateTests, SoftplusSymbolicDerivativesRemainSmoothAcrossLegacyThreshold) {
    constexpr double beta = 3.0;
    constexpr double threshold = 2.0;
    constexpr double transition = threshold / beta;
    constexpr double h = 1e-4;

    auto x = janus::sym("x");
    auto expr = softplus(x, beta, threshold);
    auto first = janus::jacobian(expr, x);
    auto second = janus::hessian(expr, x);

    janus::Function first_fn({x}, {first});
    janus::Function second_fn({x}, {second});

    const double d1_left = first_fn.eval(transition - h)(0, 0);
    const double d1_right = first_fn.eval(transition + h)(0, 0);
    const double d2_left = second_fn.eval(transition - h)(0, 0);
    const double d2_right = second_fn.eval(transition + h)(0, 0);

    const double sigma_left = logistic(beta * (transition - h));
    const double sigma_right = logistic(beta * (transition + h));
    const double expected_second_left = beta * sigma_left * (1.0 - sigma_left);
    const double expected_second_right = beta * sigma_right * (1.0 - sigma_right);

    EXPECT_NEAR(d1_left, sigma_left, 1e-9);
    EXPECT_NEAR(d1_right, sigma_right, 1e-9);
    EXPECT_NEAR(d1_left, d1_right, 1e-3);

    EXPECT_NEAR(d2_left, expected_second_left, 1e-9);
    EXPECT_NEAR(d2_right, expected_second_right, 1e-9);
    EXPECT_NEAR(d2_left, d2_right, 1e-3);
}

// ======================================================================
// Sigmoid Tests
// ======================================================================

TEST(SurrogateTests, SigmoidNumeric) {
    // Tanh type (default): tanh(0) = 0. Normalized to [0, 1] -> 0.5
    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-5);

    // Large pos -> 1.0
    EXPECT_NEAR(sigmoid(100.0), 1.0, 1e-5);

    // Large neg -> 0.0
    EXPECT_NEAR(sigmoid(-100.0), 0.0, 1e-5);

    // Custom range [-1, 1]
    EXPECT_NEAR(sigmoid(0.0, SigmoidType::Tanh, -1.0, 1.0), 0.0, 1e-5);
}

TEST(SurrogateTests, SigmoidSymbolic) {
    auto x = janus::sym("x");
    auto expr = sigmoid(x); // default 0 to 1
    double val = janus::Function({x}, {expr}).eval(0.0)(0, 0);
    EXPECT_NEAR(val, 0.5, 1e-5);
}

// ======================================================================
// Swish Tests
// ======================================================================

TEST(SurrogateTests, SwishNumeric) {
    // swish(0) = 0 / ... = 0
    EXPECT_NEAR(swish(0.0), 0.0, 1e-5);

    // swish(large) -> large
    EXPECT_NEAR(swish(10.0), 10.0 / (1.0 + std::exp(-10.0)), 1e-5);
}

TEST(SurrogateTests, SwishSymbolic) {
    auto x = janus::sym("x");
    auto expr = swish(x);
    double val = janus::Function({x}, {expr}).eval(2.0)(0, 0);
    // 2 / (1 + exp(-2))
    EXPECT_NEAR(val, 2.0 / (1.0 + std::exp(-2.0)), 1e-5);
}

// ======================================================================
// Blend Tests
// ======================================================================

TEST(SurrogateTests, BlendNumeric) {
    double low = 0.0;
    double high = 10.0;

    // switch=0 -> roughly average depending on sigmoid shape.
    // sigmoid(0) is 0.5. blend = 10*0.5 + 0*0.5 = 5.
    EXPECT_NEAR(blend(0.0, high, low), 5.0, 1e-5);

    // switch large positive -> high
    EXPECT_NEAR(blend(10.0, high, low), 10.0, 1e-2);

    // switch large negative -> low
    EXPECT_NEAR(blend(-10.0, high, low), 0.0, 1e-2);
}

TEST(SurrogateTests, BlendSymbolic) {
    auto s = janus::sym("s");
    auto expr = blend(s, 10.0, 0.0);

    // s=0 -> 5.0
    double val = janus::Function({s}, {expr}).eval(0.0)(0, 0);
    EXPECT_NEAR(val, 5.0, 1e-5);
}

TEST(SurrogateTests, CoverageErrors) {
    std::vector<double> empty;
    EXPECT_THROW(janus::softmax(empty), janus::InvalidArgument);

    std::vector<double> valid = {1.0, 2.0};
    EXPECT_THROW(janus::softmax(valid, -1.0), janus::InvalidArgument); // Invalid softness
    EXPECT_THROW(janus::softmax(valid, 0.0), janus::InvalidArgument);  // Invalid softness
}

} // namespace test
} // namespace janus
