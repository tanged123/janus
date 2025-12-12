#include <gtest/gtest.h>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Trig.hpp>
#include <janus/core/JanusTypes.hpp> 

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

TEST(MathTests, Numeric) {
    test_arithmetic<double>();
    test_trig<double>();
}

TEST(MathTests, Symbolic) {
    // Basic compilation check for Symbolic types
    test_arithmetic<janus::SymbolicScalar>();
    test_trig<janus::SymbolicScalar>();
}
