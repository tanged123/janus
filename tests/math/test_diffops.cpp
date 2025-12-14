#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/DiffOps.hpp>
#include <janus/math/Linalg.hpp> // needed for to_mx

template <typename Scalar> void test_diffops() {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Test diff
    Vector v(4);
    v(0) = 0.0;
    v(1) = 1.0;
    v(2) = 4.0;
    v(3) = 9.0;
    auto res_diff = janus::diff(v); // [1, 3, 5]

    // Test trapz
    Vector y(2);
    y(0) = 1.0;
    y(1) = 1.0;
    Vector x(2);
    x(0) = 0.0;
    x(1) = 1.0;
    auto res_trapz = janus::trapz(y, x);

    // Test gradient
    Vector x_grad(5);
    x_grad(0) = 0.0;
    x_grad(1) = 1.0;
    x_grad(2) = 2.0;
    x_grad(3) = 3.0;
    x_grad(4) = 4.0;
    Vector y_grad(5);
    y_grad(0) = 0.0;
    y_grad(1) = 1.0;
    y_grad(2) = 4.0;
    y_grad(3) = 9.0;
    y_grad(4) = 16.0;
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
        auto res_diff_eval = eval_matrix(janus::to_mx(res_diff));
        EXPECT_DOUBLE_EQ(res_diff_eval(0), 1.0);
        EXPECT_DOUBLE_EQ(res_diff_eval(1), 3.0);

        EXPECT_DOUBLE_EQ(eval_scalar(res_trapz), 1.0);

        auto res_grad_eval = eval_matrix(janus::to_mx(res_grad));
        EXPECT_DOUBLE_EQ(res_grad_eval(1), 2.0);
        EXPECT_DOUBLE_EQ(res_grad_eval(2), 4.0);
    }
}

TEST(DiffOpsTests, Numeric) { test_diffops<double>(); }

TEST(DiffOpsTests, Symbolic) { test_diffops<janus::SymbolicScalar>(); }
