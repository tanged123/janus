#include <gtest/gtest.h>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Trig.hpp>
#include <janus/math/Logic.hpp>
#include <janus/math/DiffOps.hpp>
#include <janus/math/Linalg.hpp>
#include <janus/math/Interpolate.hpp>
#include <janus/math/Spacing.hpp>
#include <janus/math/Rotations.hpp>
#include <janus/core/JanusTypes.hpp> 
#include <numbers>

// Test Arithmetic Functions
template <typename Scalar>
void test_arithmetic() {
    Scalar val = -4.0;
    auto res_abs = janus::abs(val);
    
    val = 16.0;
    auto res_sqrt = janus::sqrt(val);
    
    Scalar base = 2.0;
    Scalar exp = 3.0;
    auto res_pow = janus::pow(base, exp);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_abs, 4.0);
        EXPECT_DOUBLE_EQ(res_sqrt, 4.0);
        EXPECT_DOUBLE_EQ(res_pow, 8.0);
    } else {
        EXPECT_FALSE(res_abs.is_empty());
        EXPECT_FALSE(res_sqrt.is_empty());
        EXPECT_FALSE(res_pow.is_empty());
    }
}

// Test Trigonometry Functions
template <typename Scalar>
void test_trig() {
    Scalar val = 0.0;
    auto res_sin = janus::sin(val);
    auto res_cos = janus::cos(val);
    
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_sin, 0.0);
        EXPECT_DOUBLE_EQ(res_cos, 1.0);
    } else {
        EXPECT_FALSE(res_sin.is_empty());
        EXPECT_FALSE(res_cos.is_empty());
    }
}

// Test Logic Functions
template <typename Scalar>
void test_logic() {
    Scalar a = 1.0;
    Scalar b = 2.0;
    
    // Test where with comparison
    auto res = janus::where(a < b, a, b);
    
    // Test sigmoid blend
    // blend from 10 to 20. 
    // x = -10 -> closer to 10
    Scalar val_low = 10.0;
    Scalar val_high = 20.0;
    auto blend_low = janus::sigmoid_blend(static_cast<Scalar>(-10.0), val_low, val_high);
    auto blend_high = janus::sigmoid_blend(static_cast<Scalar>(10.0), val_low, val_high);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res, 1.0);
        EXPECT_NEAR(blend_low, 10.0, 1e-3);
        EXPECT_NEAR(blend_high, 20.0, 1e-3);
    } else {
        EXPECT_FALSE(res.is_empty());
        EXPECT_FALSE(blend_low.is_empty());
        EXPECT_FALSE(blend_high.is_empty());
    }
}

// Test DiffOps Functions
template <typename Scalar>
void test_diffops() {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Test diff
    Vector v(4);
    v(0) = 0.0; v(1) = 1.0; v(2) = 4.0; v(3) = 9.0;
    auto res_diff = janus::diff(v); // [1, 3, 5]

    // Test trapz
    Vector y(2); y(0) = 1.0; y(1) = 1.0;
    Vector x(2); x(0) = 0.0; x(1) = 1.0;
    auto res_trapz = janus::trapz(y, x);

    // Test gradient
    Vector x_grad(5);
    x_grad(0) = 0.0; x_grad(1) = 1.0; x_grad(2) = 2.0; x_grad(3) = 3.0; x_grad(4) = 4.0;
    Vector y_grad(5);
    y_grad(0) = 0.0; y_grad(1) = 1.0; y_grad(2) = 4.0; y_grad(3) = 9.0; y_grad(4) = 16.0;
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
        EXPECT_FALSE(res_diff(0).is_empty());
        EXPECT_FALSE(res_trapz.is_empty());
        EXPECT_FALSE(res_grad(1).is_empty());
    }
}

// Test Linalg Functions
template <typename Scalar>
void test_linalg() {
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
    Vector v2(2); v2(0)=3.0; v2(1)=4.0;
    auto M = janus::outer(v1, v2);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(x(0), 1.0, 1e-6);
        EXPECT_NEAR(x(1), 1.0, 1e-6);
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(M(0,0), 3.0);
        EXPECT_DOUBLE_EQ(M(1,1), 8.0);
    } else {
        EXPECT_EQ(x.rows(), 2);
        EXPECT_FALSE(x(0).is_empty());
        EXPECT_FALSE(n.is_empty());
        EXPECT_FALSE(M(0,0).is_empty());
    }
}

// Test Interpolation
template <typename Scalar>
void test_interpolate() {
    // x = [0, 1, 2]
    // y = [0, 10, 0]
    Eigen::VectorXd x(3); x << 0.0, 1.0, 2.0;
    Eigen::VectorXd y(3); y << 0.0, 10.0, 0.0;
    
    janus::JanusInterpolator interp(x, y);
    
    Scalar query_mid = 0.5;   // Expect 5.0
    auto res_mid = interp(query_mid);

    Scalar query_right = 1.5; // Expect 5.0
    auto res_right = interp(query_right);
    
    Scalar query_extrap = 3.0; // Slope from last segment (1->2 is -10)
    // val at 2 is 0. slope is -10. 3 is 2+1. so 0 - 10 = -10.
    auto res_extrap = interp(query_extrap);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_mid, 5.0);
        EXPECT_DOUBLE_EQ(res_right, 5.0);
        EXPECT_DOUBLE_EQ(res_extrap, -10.0);
    } else {
        EXPECT_FALSE(res_mid.is_empty());
        EXPECT_FALSE(res_right.is_empty());
        EXPECT_FALSE(res_extrap.is_empty());
    }
}

// Test Spacing and Rotations
template <typename Scalar>
void test_extras() {
    // Spacing
    Scalar start = 0.0;
    Scalar end = 10.0;
    auto lin = janus::linspace(start, end, 5);
    auto cos = janus::cosine_spacing(start, end, 5);
    
    // Rotations (2D)
    Scalar theta = std::numbers::pi_v<double> / 2.0;
    auto R2 = janus::rotation_matrix_2d(theta);
    
    // Rotations (3D Z-axis)
    auto R3 = janus::rotation_matrix_3d(theta, 2);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(lin.size(), 5);
        EXPECT_NEAR(lin(2), 5.0, 1e-9);
        EXPECT_NEAR(lin(4), 10.0, 1e-9);

        // Cos check: 5 points. i=0 -> angle=0 -> cos=1 -> x = center - radius = 5 - 5 = 0. Correct.
        EXPECT_NEAR(cos(0), 0.0, 1e-9);
        
        // R2: pi/2 -> [0 -1; 1 0]
        EXPECT_NEAR(R2(0,0), 0.0, 1e-9);
        EXPECT_NEAR(R2(0,1), -1.0, 1e-9);
        
        // R3: Z axis similar to 2D in top-left
        EXPECT_NEAR(R3(0,0), 0.0, 1e-9);
        EXPECT_NEAR(R3(0,1), -1.0, 1e-9);
        EXPECT_NEAR(R3(2,2), 1.0, 1e-9);
    } else {
         EXPECT_EQ(lin.size(), 5);
         EXPECT_FALSE(lin(0).is_empty());
         EXPECT_FALSE(cos(0).is_empty());
         EXPECT_FALSE(R2(0,0).is_empty());
         EXPECT_FALSE(R3(0,0).is_empty());
    }
}

TEST(MathTests, Numeric) {
    std::cout << "Testing Arithmetic..." << std::endl;
    test_arithmetic<double>();
    std::cout << "Testing Trig..." << std::endl;
    test_trig<double>();
    std::cout << "Testing Logic..." << std::endl;
    test_logic<double>();
    std::cout << "Testing DiffOps..." << std::endl;
    test_diffops<double>();
    std::cout << "Testing Linalg..." << std::endl;
    test_linalg<double>();
    std::cout << "Testing Interpolate..." << std::endl;
    test_interpolate<double>();
    std::cout << "Testing Extras..." << std::endl;
    test_extras<double>();
    std::cout << "All Numeric tests passed." << std::endl;
}

TEST(MathTests, Symbolic) {
    test_arithmetic<janus::SymbolicScalar>();
    test_trig<janus::SymbolicScalar>();
    test_logic<janus::SymbolicScalar>();
    test_diffops<janus::SymbolicScalar>();
    test_linalg<janus::SymbolicScalar>();
    test_interpolate<janus::SymbolicScalar>();
    test_extras<janus::SymbolicScalar>();
}
