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
