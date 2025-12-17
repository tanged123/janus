#include <gtest/gtest.h>
#include <janus/janus.hpp>
#include <vector>

TEST(FunctionTests, BasicEvaluation) {
    // f(x, y) = x * y + x
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto f_sym = x * y + x;

    janus::Function f("f", {x, y}, {f_sym});

    // Test variadic scalar evaluation
    auto res1 = f(2.0, 3.0);
    // 2*3 + 2 = 8
    ASSERT_EQ(res1.size(), 1);
    EXPECT_NEAR(res1[0](0, 0), 8.0, 1e-9);

    // Test vector evaluation
    std::vector<double> args = {3.0, 4.0};
    auto res2 = f(args);
    // 3*4 + 3 = 15
    ASSERT_EQ(res2.size(), 1);
    EXPECT_NEAR(res2[0](0, 0), 15.0, 1e-9);
}

TEST(FunctionTests, VectorIO) {
    // f([x, y]) = [x+y, x-y]
    auto x = janus::sym("x");
    auto y = janus::sym("y");

    // Let's test explicit vector inputs as separate symbolic variables for now
    // to match signature: Function(..., {inputs}, {outputs})

    auto out1 = x + y;
    auto out2 = x - y;

    janus::Function f("f_vec", {x, y}, {out1, out2});

    auto res = f(10.0, 5.0);
    ASSERT_EQ(res.size(), 2);
    // out1 = 15
    EXPECT_NEAR(res[0](0, 0), 15.0, 1e-9);
    // out2 = 5
    EXPECT_NEAR(res[1](0, 0), 5.0, 1e-9);
}

TEST(FunctionTests, MatrixOutput) {
    // f(x) = [x, 2x; 3x, 4x]
    auto x = janus::sym("x");

    janus::SymbolicMatrix M(2, 2);
    M(0, 0) = x;
    M(0, 1) = 2.0 * x;
    M(1, 0) = 3.0 * x;
    M(1, 1) = 4.0 * x;

    // Convert Eigen matrix of MX to single MX for Function output
    janus::Function f("f_mat", {x}, {janus::to_mx(M)});

    auto res = f(2.0);
    ASSERT_EQ(res.size(), 1);
    const auto &R = res[0];
    EXPECT_EQ(R.rows(), 2);
    EXPECT_EQ(R.cols(), 2);

    EXPECT_NEAR(R(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(R(0, 1), 4.0, 1e-9);
    EXPECT_NEAR(R(1, 0), 6.0, 1e-9);
    EXPECT_NEAR(R(1, 1), 8.0, 1e-9);
}

TEST(FunctionTests, JacobianWrapper) {
    // Test the jacobian helper integration with Function
    auto x = janus::sym("x");
    auto fx = janus::pow(x, 2.0);    // x^2
    auto J = janus::jacobian(fx, x); // 2x

    janus::Function f_J("J", {x}, {J});

    auto res = f_J(3.0);
    EXPECT_NEAR(res[0](0, 0), 6.0, 1e-9);
}

// =============================================================================
// Lambda-Style Function Tests (make_function)
// =============================================================================

TEST(FunctionTests, LambdaSingleInSingleOut) {
    // f(x) = x^2
    auto f = janus::make_function<1, 1>("square", [](auto x) { return x * x; });

    auto res = f(3.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 9.0, 1e-9);

    // Also test with negative number
    auto res2 = f(-4.0);
    EXPECT_NEAR(res2[0](0, 0), 16.0, 1e-9);
}

TEST(FunctionTests, LambdaMultiInSingleOut) {
    // f(x, y) = x + y
    auto f = janus::make_function<2, 1>("sum", [](auto x, auto y) { return x + y; });

    auto res = f(3.0, 7.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 10.0, 1e-9);
}

TEST(FunctionTests, LambdaMultiInMultiOut) {
    // f(x, y) = (x+y, x-y)
    auto f = janus::make_function<2, 2>(
        "add_sub", [](auto x, auto y) { return std::make_tuple(x + y, x - y); });

    auto res = f(10.0, 3.0);
    ASSERT_EQ(res.size(), 2);
    EXPECT_NEAR(res[0](0, 0), 13.0, 1e-9); // x + y = 13
    EXPECT_NEAR(res[1](0, 0), 7.0, 1e-9);  // x - y = 7
}

TEST(FunctionTests, LambdaNamedInputs) {
    // f(x, y) = x * y using named inputs
    auto f = janus::make_function<2>("product", {"x", "y"}, [](auto x, auto y) { return x * y; });

    auto res = f(4.0, 5.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 20.0, 1e-9);
}

TEST(FunctionTests, LambdaNamedMultiOut) {
    // g(a, b, c) = (a+b+c, a*b*c) with named inputs
    auto g = janus::make_function<3>("triple", {"a", "b", "c"}, [](auto a, auto b, auto c) {
        return std::make_tuple(a + b + c, a * b * c);
    });

    auto res = g(2.0, 3.0, 4.0);
    ASSERT_EQ(res.size(), 2);
    EXPECT_NEAR(res[0](0, 0), 9.0, 1e-9);  // 2 + 3 + 4 = 9
    EXPECT_NEAR(res[1](0, 0), 24.0, 1e-9); // 2 * 3 * 4 = 24
}

TEST(FunctionTests, LambdaWithJanusMath) {
    // Test using janus:: math functions inside lambda
    auto f = janus::make_function<1, 1>("exp_plus_one", [](auto x) { return janus::exp(x) + 1.0; });

    auto res = f(0.0);
    EXPECT_NEAR(res[0](0, 0), 2.0, 1e-9); // exp(0) + 1 = 2

    auto res2 = f(1.0);
    EXPECT_NEAR(res2[0](0, 0), std::exp(1.0) + 1.0, 1e-9);
}
