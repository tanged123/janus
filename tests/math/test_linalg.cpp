#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusIO.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Linalg.hpp>

template <typename Scalar> void test_linalg_ops() {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Test solve
    // A = [[2, 1], [1, 2]]
    // b = [3, 3]
    // x = [1, 1]
    Matrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    Vector b(2);
    b(0) = 3.0;
    b(1) = 3.0;

    auto x = janus::solve(A, b);

    // Test norm
    Vector v(2);
    v(0) = 3.0;
    v(1) = 4.0;
    auto n = janus::norm(v);

    // Test outer
    Vector v1(2);
    v1(0) = 1.0;
    v1(1) = 2.0;
    Vector v2(2);
    v2(0) = 3.0;
    v2(1) = 4.0;                   // v2 = [3, 4]
    auto M = janus::outer(v1, v2); // [[3, 4], [6, 8]]

    // Test dot
    auto d = janus::dot(v, v); // 3*3 + 4*4 = 25

    // Test cross (3D)
    Vector c1(3);
    c1 << 1.0, 0.0, 0.0;
    Vector c2(3);
    c2 << 0.0, 1.0, 0.0;
    auto c3 = janus::cross(c1, c2); // [0, 0, 1]

    // Test inner
    auto i_prod = janus::inner(v, v); // 25.0

    // Test pinv
    // A singular = [[1, 1], [2, 2]]
    Matrix A_sing(2, 2);
    A_sing << 1.0, 1.0, 2.0, 2.0;
    auto A_pinv = janus::pinv(A_sing);

    // Test extended norms
    Vector v_norm(3);
    v_norm << -1.0, 2.0, -3.0;

    auto n_1 = janus::norm(v_norm, janus::NormType::L1);    // 1+2+3 = 6
    auto n_inf = janus::norm(v_norm, janus::NormType::Inf); // 3

    // Existing inv / det tests
    auto A_inv = janus::inv(A); // inv([[2, 1], [1, 2]]) = 1/3 * [[2, -1], [-1, 2]]
    auto A_det = janus::det(A); // 4 - 1 = 3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(x(0), 1.0, 1e-6);
        EXPECT_NEAR(x(1), 1.0, 1e-6);
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(M(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(d, 25.0);

        EXPECT_DOUBLE_EQ(c3(0), 0.0);
        EXPECT_DOUBLE_EQ(c3(1), 0.0);
        EXPECT_DOUBLE_EQ(c3(2), 1.0);

        EXPECT_DOUBLE_EQ(i_prod, 25.0);

        // Pinv of [[1, 1], [2, 2]]
        // SVD based, roughly [[0.1, 0.2], [0.1, 0.2]]
        // Check A * pinv * A = A
        auto recon = A_sing * A_pinv * A_sing;
        EXPECT_TRUE(recon.isApprox(A_sing, 1e-5));

        EXPECT_DOUBLE_EQ(n_1, 6.0);
        EXPECT_DOUBLE_EQ(n_inf, 3.0);

        EXPECT_NEAR(A_det, 3.0, 1e-9);
        EXPECT_NEAR(A_inv(0, 0), 2.0 / 3.0, 1e-9);
    } else {
        auto x_eval = janus::eval(x);
        EXPECT_NEAR(x_eval(0), 1.0, 1e-6);
        EXPECT_NEAR(x_eval(1), 1.0, 1e-6);

        EXPECT_DOUBLE_EQ(janus::eval(n), 5.0);

        auto M_eval = janus::eval(M);
        EXPECT_DOUBLE_EQ(M_eval(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(janus::eval(d), 25.0);

        auto c3_eval = janus::eval(c3);
        EXPECT_NEAR(c3_eval(0), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(2), 1.0, 1e-9);

        EXPECT_DOUBLE_EQ(janus::eval(i_prod), 25.0);

        // Pinv test symbolic
        auto A_pinv_inv = janus::pinv(A);
        auto A_pinv_eval = janus::eval(A_pinv_inv);
        EXPECT_NEAR(A_pinv_eval(0, 0), 2.0 / 3.0, 1e-6);

        EXPECT_DOUBLE_EQ(janus::eval(n_1), 6.0);
        EXPECT_DOUBLE_EQ(janus::eval(n_inf), 3.0);

        auto A_inv_eval = janus::eval(A_inv);
        if (std::abs(A_inv_eval(0, 0) - (2.0 / 3.0)) > 1e-9) {
            janus::print("A_inv (Symbolic)", A_inv);
            janus::print("A_inv (Evaluated)", A_inv_eval);
        }
        EXPECT_NEAR(A_inv_eval(0, 0), 2.0 / 3.0, 1e-9);
    }
}

TEST(LinalgTests, Numeric) { test_linalg_ops<double>(); }

TEST(LinalgTests, Symbolic) { test_linalg_ops<janus::SymbolicScalar>(); }

TEST(LinalgTests, CoverageEdges) {
    // 1. Empty to_mx coverage
    Eigen::MatrixXd empty(0, 0);
    casadi::MX empty_mx = janus::to_mx(empty);
    EXPECT_TRUE(empty_mx.is_empty());

    // 2. Numeric Norm Edges
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 2.0; // norm = 3

    // Frobenius (same as L2 for vectors)
    EXPECT_DOUBLE_EQ(janus::norm(v, janus::NormType::Frobenius), 3.0);

    // Default (invalid enum)
    EXPECT_DOUBLE_EQ(janus::norm(v, static_cast<janus::NormType>(999)), 3.0);

    // 3. Symbolic Norm Edges (Default branch)
    janus::SymbolicVector vs = janus::as_vector(janus::to_mx(v));
    auto n_def = janus::norm(vs, static_cast<janus::NormType>(999));
    EXPECT_DOUBLE_EQ(janus::eval(n_def), 3.0);
}
