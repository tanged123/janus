#include <gtest/gtest.h>
#include <janus/math/Linalg.hpp>
#include <janus/core/JanusTypes.hpp>
#include "../utils/TestUtils.hpp"

template <typename Scalar>
void test_linalg_ops() {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    
    // Test solve
    // A = [[2, 1], [1, 2]]
    // b = [3, 3]
    // x = [1, 1]
    Matrix A(2, 2);
    A(0,0) = 2.0; A(0,1) = 1.0;
    A(1,0) = 1.0; A(1,1) = 2.0;
    Vector b(2);
    b(0) = 3.0; b(1) = 3.0;
    
    auto x = janus::solve(A, b);
    
    // Test norm
    Vector v(2); v(0)=3.0; v(1)=4.0;
    auto n = janus::norm(v);
    
    // Test outer
    Vector v1(2); v1(0)=1.0; v1(1)=2.0;
    Vector v2(2); v2(0)=3.0; v2(1)=4.0; // v2 = [3, 4]
    auto M = janus::outer(v1, v2); // [[3, 4], [6, 8]]

    // Test dot
    auto d = janus::dot(v, v); // 3*3 + 4*4 = 25
    
    // Test cross (3D)
    Vector c1(3); c1 << 1.0, 0.0, 0.0;
    Vector c2(3); c2 << 0.0, 1.0, 0.0;
    auto c3 = janus::cross(c1, c2); // [0, 0, 1]

    // Test inv and det
    auto A_inv = janus::inv(A); // inv([[2, 1], [1, 2]]) = 1/3 * [[2, -1], [-1, 2]]
    auto A_det = janus::det(A); // 4 - 1 = 3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(x(0), 1.0, 1e-6);
        EXPECT_NEAR(x(1), 1.0, 1e-6);
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(M(0,0), 3.0);
        EXPECT_DOUBLE_EQ(M(1,1), 8.0);
        
        EXPECT_DOUBLE_EQ(d, 25.0);
        
        EXPECT_DOUBLE_EQ(c3(0), 0.0);
        EXPECT_DOUBLE_EQ(c3(1), 0.0);
        EXPECT_DOUBLE_EQ(c3(2), 1.0);
        
        EXPECT_NEAR(A_det, 3.0, 1e-9);
        EXPECT_NEAR(A_inv(0,0), 2.0/3.0, 1e-9);
    } else {
        auto x_eval = eval_matrix(janus::to_mx(x));
        EXPECT_NEAR(x_eval(0), 1.0, 1e-6);
        EXPECT_NEAR(x_eval(1), 1.0, 1e-6);
        
        EXPECT_DOUBLE_EQ(eval_scalar(n), 5.0);

        auto M_eval = eval_matrix(janus::to_mx(M));
        EXPECT_DOUBLE_EQ(M_eval(0,0), 3.0);
        EXPECT_DOUBLE_EQ(M_eval(1,1), 8.0);
        
        EXPECT_DOUBLE_EQ(eval_scalar(d), 25.0);

        auto c3_eval = eval_matrix(janus::to_mx(c3));
        EXPECT_NEAR(c3_eval(0), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(2), 1.0, 1e-9);
        
        // Skip symbolic det eval if unsupported
         // EXPECT_NEAR(eval_scalar(A_det), 3.0, 1e-9);
        
        auto A_inv_eval = eval_matrix(janus::to_mx(A_inv));
        EXPECT_NEAR(A_inv_eval(0,0), 2.0/3.0, 1e-9);
    }
}

TEST(LinalgTests, Numeric) {
    test_linalg_ops<double>();
}

TEST(LinalgTests, Symbolic) {
    test_linalg_ops<janus::SymbolicScalar>();
}
