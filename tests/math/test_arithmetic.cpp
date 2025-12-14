#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Arithmetic.hpp>

template <typename Scalar> void test_arithmetic_ops() {
    Scalar val = -4.0;
    auto res_abs = janus::abs(val);

    val = 16.0;
    auto res_sqrt = janus::sqrt(val);

    Scalar base = 2.0;
    Scalar exp = 3.0;
    auto res_pow = janus::pow(base, exp);
    auto res_pow_double = janus::pow(base, 3.0); // Test overload

    Scalar v_hyp = 1.0;
    auto res_sinh = janus::sinh(v_hyp);
    auto res_cosh = janus::cosh(v_hyp);
    auto res_tanh = janus::tanh(v_hyp);

    Scalar v_float = 2.7;
    auto res_floor = janus::floor(v_float);
    auto res_ceil = janus::ceil(v_float);

    Scalar v_neg = -5.5;
    auto res_sign_neg = janus::sign(v_neg);
    auto res_sign_pos = janus::sign(v_float);

    Scalar v_mod_a = 5.3;
    Scalar v_mod_b = 2.0;
    auto res_mod = janus::fmod(v_mod_a, v_mod_b);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_abs, 4.0);
        EXPECT_DOUBLE_EQ(res_sqrt, 4.0);
        EXPECT_DOUBLE_EQ(res_pow, 8.0);
        EXPECT_DOUBLE_EQ(res_pow_double, 8.0);

        EXPECT_NEAR(res_sinh, std::sinh(1.0), 1e-9);
        EXPECT_NEAR(res_cosh, std::cosh(1.0), 1e-9);
        EXPECT_NEAR(res_tanh, std::tanh(1.0), 1e-9);

        EXPECT_DOUBLE_EQ(res_floor, 2.0);
        EXPECT_DOUBLE_EQ(res_ceil, 3.0);
        EXPECT_DOUBLE_EQ(res_sign_neg, -1.0);
        EXPECT_DOUBLE_EQ(res_sign_pos, 1.0);

        EXPECT_NEAR(res_mod, 1.3, 1e-9);
    } else {
        EXPECT_FALSE(res_abs.is_empty());
        EXPECT_DOUBLE_EQ(janus::eval(res_abs), 4.0);

        EXPECT_FALSE(res_sqrt.is_empty());
        EXPECT_DOUBLE_EQ(janus::eval(res_sqrt), 4.0);

        EXPECT_FALSE(res_pow.is_empty());
        EXPECT_DOUBLE_EQ(janus::eval(res_pow), 8.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_pow_double), 8.0);

        EXPECT_NEAR(janus::eval(res_sinh), std::sinh(1.0), 1e-9);
        EXPECT_NEAR(janus::eval(res_cosh), std::cosh(1.0), 1e-9);
        EXPECT_NEAR(janus::eval(res_tanh), std::tanh(1.0), 1e-9);

        EXPECT_DOUBLE_EQ(janus::eval(res_floor), 2.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_ceil), 3.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_sign_neg), -1.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_sign_pos), 1.0);

        EXPECT_NEAR(janus::eval(res_mod), 1.3, 1e-9);
    }
}

TEST(ArithmeticTests, Numeric) { test_arithmetic_ops<double>(); }

TEST(ArithmeticTests, Symbolic) { test_arithmetic_ops<janus::SymbolicScalar>(); }
