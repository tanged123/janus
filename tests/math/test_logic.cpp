#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Linalg.hpp> // for to_mx
#include <janus/math/Logic.hpp>

template <typename Scalar> void test_logic_ops() {
    Scalar a = 1.0;
    Scalar b = 2.0;

    // Test where with comparison
    auto res_where = janus::where(a < b, a, b);

    // Test min/max/clamp
    auto res_min = janus::min(a, b);
    auto res_max = janus::max(a, b);

    Scalar val = 5.0;
    Scalar low = 0.0;
    Scalar high = 3.0;
    auto res_clamp = janus::clamp(val, low, high); // Should be 3.0

    // Test sigmoid blend
    Scalar val_low = 10.0;
    Scalar val_high = 20.0;
    auto blend_low = janus::sigmoid_blend(static_cast<Scalar>(-10.0), val_low, val_high);
    auto blend_high = janus::sigmoid_blend(static_cast<Scalar>(10.0), val_low, val_high);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_where, 1.0);
        EXPECT_DOUBLE_EQ(res_min, 1.0);
        EXPECT_DOUBLE_EQ(res_max, 2.0);
        EXPECT_DOUBLE_EQ(res_clamp, 3.0);
        EXPECT_NEAR(blend_low, 10.0, 1e-3);
        EXPECT_NEAR(blend_high, 20.0, 1e-3);
    } else {
        EXPECT_DOUBLE_EQ(eval_scalar(res_where), 1.0);
        EXPECT_DOUBLE_EQ(eval_scalar(res_min), 1.0);
        EXPECT_DOUBLE_EQ(eval_scalar(res_max), 2.0);
        EXPECT_DOUBLE_EQ(eval_scalar(res_clamp), 3.0);
        EXPECT_NEAR(eval_scalar(blend_low), 10.0, 1e-3);
        EXPECT_NEAR(eval_scalar(blend_high), 20.0, 1e-3);
    }
}

template <typename Scalar> void test_logic_matrix() {
    using Matrix = janus::JanusMatrix<Scalar>;
    Matrix A(2, 2);
    A << 1.0, 4.0, 2.0, 5.0;
    Matrix B(2, 2);
    B << 3.0, 2.0, 1.0, 6.0;

    // Element-wise min: [1, 2], [1, 5]
    auto M = janus::min(A, B);
    auto cond = janus::lt(A, B);

    // For numeric, cond is Array<bool>. For symbolic, Matrix<MX>.
    // janus::where expects ArrayBase.
    // If Matrix<MX>, .array() makes it ArrayWrapper, which is ArrayBase.
    // If Array<bool>, it is ArrayBase.

    // However, cond.array() works for both if we ensure cond is expression that supports .array().
    // Eigen binaryExpr returns CwiseBinaryOp which supports .array() if it's Matrix expression?
    // Actually, binaryExpr on Matrix returns Matrix.
    // Comparison on Array returns Array.

    // Let's use auto and .array() or pass derived if compatible.
    // Our where loop calls .coeff().

    auto where_mat = janus::where(cond.array(), A, B);

    // Test new comparisons
    // A: [[1, 4], [2, 5]]
    // B: [[3, 2], [1, 6]]

    // A > B: [[F, T], [T, F]] -> where(A>B, 10, -10)
    auto cond_gt = janus::gt(A, B);
    Matrix Ones = Matrix::Ones(2, 2) * 10;
    Matrix NegOnes = Matrix::Ones(2, 2) * -10;
    auto check_gt = janus::where(cond_gt.array(), Ones, NegOnes);

    // A <= B: [[T, F], [F, T]] (Inverse of >)
    auto cond_le = janus::le(A, B);
    auto check_le = janus::where(cond_le.array(), Ones, NegOnes);

    // A == A
    auto cond_eq = janus::eq(A, A);
    auto check_eq = janus::where(cond_eq.array(), Ones, NegOnes);

    // A != B (All true as scalars different everywhere)
    auto cond_neq = janus::neq(A, B);
    auto check_neq = janus::where(cond_neq.array(), Ones, NegOnes);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(M(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(M(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(M(1, 0), 1.0);
        EXPECT_DOUBLE_EQ(M(1, 1), 5.0);

        EXPECT_DOUBLE_EQ(where_mat(0, 1), 2.0);

        // gt: [F, T], [T, F] -> [-10, 10], [10, -10]
        EXPECT_DOUBLE_EQ(check_gt(0, 0), -10.0);
        EXPECT_DOUBLE_EQ(check_gt(0, 1), 10.0);
        EXPECT_DOUBLE_EQ(check_gt(1, 0), 10.0);
        EXPECT_DOUBLE_EQ(check_gt(1, 1), -10.0);

        // le: Inverse of gt
        EXPECT_DOUBLE_EQ(check_le(0, 0), 10.0);
        EXPECT_DOUBLE_EQ(check_le(0, 1), -10.0);

        // eq: All true
        EXPECT_DOUBLE_EQ(check_eq(0, 0), 10.0);

        // neq: All true
        EXPECT_DOUBLE_EQ(check_neq(0, 0), 10.0);
    } else {
        auto M_eval = eval_matrix(janus::to_mx(M));
        EXPECT_DOUBLE_EQ(M_eval(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(M_eval(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 0), 1.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 1), 5.0);

        auto W_eval = eval_matrix(janus::to_mx(where_mat));
        EXPECT_DOUBLE_EQ(W_eval(0, 1), 2.0);

        auto G_eval = eval_matrix(janus::to_mx(check_gt));
        EXPECT_DOUBLE_EQ(G_eval(0, 0), -10.0);
        EXPECT_DOUBLE_EQ(G_eval(0, 1), 10.0);

        auto L_eval = eval_matrix(janus::to_mx(check_le));
        EXPECT_DOUBLE_EQ(L_eval(0, 0), 10.0);
        EXPECT_DOUBLE_EQ(L_eval(0, 1), -10.0);

        auto E_eval = eval_matrix(janus::to_mx(check_eq));
        EXPECT_DOUBLE_EQ(E_eval(0, 0), 10.0);

        auto N_eval = eval_matrix(janus::to_mx(check_neq));
        EXPECT_DOUBLE_EQ(N_eval(0, 0), 10.0);
    }
}

TEST(LogicTests, Numeric) {
    test_logic_ops<double>();
    test_logic_matrix<double>();
}

TEST(LogicTests, Symbolic) {
    test_logic_ops<janus::SymbolicScalar>();
    test_logic_matrix<janus::SymbolicScalar>();
}
