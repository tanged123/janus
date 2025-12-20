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

TEST(ArithmeticTests, MatrixOps) {
    // Numeric Matrix Sqrt
    janus::NumericMatrix M(2, 2);
    M << 4.0, 9.0, 16.0, 25.0;
    auto S = janus::sqrt(M);
    EXPECT_DOUBLE_EQ(S(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(S(0, 1), 3.0);

    // Symbolic Matrix Sqrt
    janus::SymbolicMatrix Ms = janus::to_eigen(janus::to_mx(M));
    auto Ss = janus::sqrt(Ms);
    auto Ss_eval = janus::eval(Ss);
    EXPECT_DOUBLE_EQ(Ss_eval(0, 0), 2.0);
}

TEST(ArithmeticTests, MixedTypePow) {
    // Test pow(double, MX)
    double base = 2.0;
    janus::SymbolicScalar exp(3.0);
    auto res = janus::pow(base, exp);
    EXPECT_DOUBLE_EQ(janus::eval(res), 8.0);

    // Test pow(MX, double) - already covered but reinforcing
    janus::SymbolicScalar base_s(2.0);
    double exp_d = 3.0;
    auto res2 = janus::pow(base_s, exp_d);
    EXPECT_DOUBLE_EQ(janus::eval(res2), 8.0);
}

// --- New Arithmetic Functions Tests ---

template <typename Scalar> void test_log2_exp2_ops() {
    // log2: base-2 logarithm
    Scalar val_log2 = 8.0;
    auto res_log2 = janus::log2(val_log2);

    // exp2: 2^x
    Scalar val_exp2 = 3.0;
    auto res_exp2 = janus::exp2(val_exp2);

    // Verify inverse relationship: log2(exp2(x)) == x
    Scalar val_round_trip = 5.0;
    auto res_round_trip = janus::log2(janus::exp2(val_round_trip));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_log2, 3.0);        // log2(8) = 3
        EXPECT_DOUBLE_EQ(res_exp2, 8.0);        // 2^3 = 8
        EXPECT_NEAR(res_round_trip, 5.0, 1e-9); // log2(2^5) = 5
    } else {
        EXPECT_NEAR(janus::eval(res_log2), 3.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_exp2), 8.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_round_trip), 5.0, 1e-9);
    }
}

TEST(ArithmeticTests, Log2Exp2Numeric) { test_log2_exp2_ops<double>(); }

TEST(ArithmeticTests, Log2Exp2Symbolic) { test_log2_exp2_ops<janus::SymbolicScalar>(); }

template <typename Scalar> void test_cbrt_ops() {
    // Positive cube root
    Scalar val_pos = 27.0;
    auto res_pos = janus::cbrt(val_pos);

    // Negative cube root (cbrt handles negative values correctly)
    Scalar val_neg = -8.0;
    auto res_neg = janus::cbrt(val_neg);

    // Zero
    Scalar val_zero = 0.0;
    auto res_zero = janus::cbrt(val_zero);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_pos, 3.0);  // cbrt(27) = 3
        EXPECT_DOUBLE_EQ(res_neg, -2.0); // cbrt(-8) = -2
        EXPECT_DOUBLE_EQ(res_zero, 0.0); // cbrt(0) = 0
    } else {
        EXPECT_NEAR(janus::eval(res_pos), 3.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_neg), -2.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_zero), 0.0, 1e-9);
    }
}

TEST(ArithmeticTests, CbrtNumeric) { test_cbrt_ops<double>(); }

TEST(ArithmeticTests, CbrtSymbolic) { test_cbrt_ops<janus::SymbolicScalar>(); }

template <typename Scalar> void test_round_trunc_ops() {
    // Round: nearest integer
    Scalar val_up = 2.7;
    Scalar val_down = 2.3;
    Scalar val_half = 2.5;
    Scalar val_neg = -2.7;

    auto res_round_up = janus::round(val_up);
    auto res_round_down = janus::round(val_down);
    auto res_round_half = janus::round(val_half);
    auto res_round_neg = janus::round(val_neg);

    // Trunc: toward zero
    auto res_trunc_pos = janus::trunc(val_up);
    auto res_trunc_neg = janus::trunc(val_neg);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_round_up, 3.0);
        EXPECT_DOUBLE_EQ(res_round_down, 2.0);
        EXPECT_DOUBLE_EQ(res_round_half, 3.0); // round away from zero for .5
        EXPECT_DOUBLE_EQ(res_round_neg, -3.0);

        EXPECT_DOUBLE_EQ(res_trunc_pos, 2.0);  // truncate 2.7 -> 2
        EXPECT_DOUBLE_EQ(res_trunc_neg, -2.0); // truncate -2.7 -> -2 (toward zero)
    } else {
        EXPECT_NEAR(janus::eval(res_round_up), 3.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_round_down), 2.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_round_half), 3.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_round_neg), -3.0, 1e-9);

        EXPECT_NEAR(janus::eval(res_trunc_pos), 2.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_trunc_neg), -2.0, 1e-9);
    }
}

TEST(ArithmeticTests, RoundTruncNumeric) { test_round_trunc_ops<double>(); }

TEST(ArithmeticTests, RoundTruncSymbolic) { test_round_trunc_ops<janus::SymbolicScalar>(); }

template <typename Scalar> void test_hypot_ops() {
    // Standard 3-4-5 triangle
    Scalar x = 3.0;
    Scalar y = 4.0;
    auto res = janus::hypot(x, y);

    // Zero case
    Scalar zero = 0.0;
    auto res_zero = janus::hypot(zero, zero);

    // Negative values (hypot takes absolute)
    Scalar neg_x = -3.0;
    auto res_neg = janus::hypot(neg_x, y);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res, 5.0); // sqrt(9 + 16) = 5
        EXPECT_DOUBLE_EQ(res_zero, 0.0);
        EXPECT_DOUBLE_EQ(res_neg, 5.0); // sqrt(9 + 16) = 5
    } else {
        EXPECT_NEAR(janus::eval(res), 5.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_zero), 0.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_neg), 5.0, 1e-9);
    }
}

TEST(ArithmeticTests, HypotNumeric) { test_hypot_ops<double>(); }

TEST(ArithmeticTests, HypotSymbolic) { test_hypot_ops<janus::SymbolicScalar>(); }

TEST(ArithmeticTests, HypotMixedTypes) {
    // Test hypot(double, MX)
    double x = 3.0;
    janus::SymbolicScalar y(4.0);
    auto res = janus::hypot(x, y);
    EXPECT_NEAR(janus::eval(res), 5.0, 1e-9);

    // Test hypot(MX, double)
    janus::SymbolicScalar x_s(3.0);
    double y_d = 4.0;
    auto res2 = janus::hypot(x_s, y_d);
    EXPECT_NEAR(janus::eval(res2), 5.0, 1e-9);
}

template <typename Scalar> void test_expm1_log1p_ops() {
    // expm1: exp(x) - 1, accurate for small x
    Scalar small_val = 1e-10;
    auto res_expm1_small = janus::expm1(small_val);

    Scalar val_1 = 1.0;
    auto res_expm1 = janus::expm1(val_1);

    // log1p: log(1 + x), accurate for small x
    auto res_log1p_small = janus::log1p(small_val);
    auto res_log1p = janus::log1p(val_1);

    // Inverse relationship: log1p(expm1(x)) ≈ x for reasonable x
    Scalar test_val = 0.5;
    auto res_round_trip = janus::log1p(janus::expm1(test_val));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(res_expm1_small, std::expm1(1e-10), 1e-15);
        EXPECT_NEAR(res_expm1, std::expm1(1.0), 1e-9);
        EXPECT_NEAR(res_log1p_small, std::log1p(1e-10), 1e-15);
        EXPECT_NEAR(res_log1p, std::log1p(1.0), 1e-9); // log(2)
        EXPECT_NEAR(res_round_trip, 0.5, 1e-9);
    } else {
        EXPECT_NEAR(janus::eval(res_expm1_small), std::expm1(1e-10), 1e-9);
        EXPECT_NEAR(janus::eval(res_expm1), std::expm1(1.0), 1e-9);
        EXPECT_NEAR(janus::eval(res_log1p_small), std::log1p(1e-10), 1e-9);
        EXPECT_NEAR(janus::eval(res_log1p), std::log1p(1.0), 1e-9);
        EXPECT_NEAR(janus::eval(res_round_trip), 0.5, 1e-9);
    }
}

TEST(ArithmeticTests, Expm1Log1pNumeric) { test_expm1_log1p_ops<double>(); }

TEST(ArithmeticTests, Expm1Log1pSymbolic) { test_expm1_log1p_ops<janus::SymbolicScalar>(); }

template <typename Scalar> void test_copysign_ops() {
    Scalar mag = 5.0;
    Scalar pos_sign = 1.0;
    Scalar neg_sign = -1.0;

    auto res_pos = janus::copysign(mag, pos_sign);
    auto res_neg = janus::copysign(mag, neg_sign);

    // Negative magnitude, positive sign
    Scalar neg_mag = -5.0;
    auto res_flip = janus::copysign(neg_mag, pos_sign);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_pos, 5.0);
        EXPECT_DOUBLE_EQ(res_neg, -5.0);
        EXPECT_DOUBLE_EQ(res_flip, 5.0);
    } else {
        EXPECT_NEAR(janus::eval(res_pos), 5.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_neg), -5.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_flip), 5.0, 1e-9);
    }
}

TEST(ArithmeticTests, CopysignNumeric) { test_copysign_ops<double>(); }

TEST(ArithmeticTests, CopysignSymbolic) { test_copysign_ops<janus::SymbolicScalar>(); }

TEST(ArithmeticTests, CopysignMixedTypes) {
    // Test copysign(double, MX)
    double mag = 5.0;
    janus::SymbolicScalar neg_sign(-1.0);
    auto res = janus::copysign(mag, neg_sign);
    EXPECT_NEAR(janus::eval(res), -5.0, 1e-9);

    // Test copysign(MX, double)
    janus::SymbolicScalar mag_s(5.0);
    double neg_d = -1.0;
    auto res2 = janus::copysign(mag_s, neg_d);
    EXPECT_NEAR(janus::eval(res2), -5.0, 1e-9);
}

template <typename Scalar> void test_square_ops() {
    Scalar val_pos = 3.0;
    Scalar val_neg = -4.0;
    Scalar val_zero = 0.0;

    auto res_pos = janus::square(val_pos);
    auto res_neg = janus::square(val_neg);
    auto res_zero = janus::square(val_zero);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_pos, 9.0);
        EXPECT_DOUBLE_EQ(res_neg, 16.0);
        EXPECT_DOUBLE_EQ(res_zero, 0.0);
    } else {
        EXPECT_NEAR(janus::eval(res_pos), 9.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_neg), 16.0, 1e-9);
        EXPECT_NEAR(janus::eval(res_zero), 0.0, 1e-9);
    }
}

TEST(ArithmeticTests, SquareNumeric) { test_square_ops<double>(); }

TEST(ArithmeticTests, SquareSymbolic) { test_square_ops<janus::SymbolicScalar>(); }

// --- Matrix tests for new functions ---

TEST(ArithmeticTests, NewFunctionsMatrixOps) {
    janus::NumericMatrix M(2, 2);
    M << 4.0, 8.0, 16.0, 32.0;

    // log2
    auto log2_result = janus::log2(M);
    EXPECT_DOUBLE_EQ(log2_result(0, 0), 2.0); // log2(4) = 2
    EXPECT_DOUBLE_EQ(log2_result(0, 1), 3.0); // log2(8) = 3
    EXPECT_DOUBLE_EQ(log2_result(1, 0), 4.0); // log2(16) = 4
    EXPECT_DOUBLE_EQ(log2_result(1, 1), 5.0); // log2(32) = 5

    // exp2
    janus::NumericMatrix M_exp(2, 2);
    M_exp << 1.0, 2.0, 3.0, 4.0;
    auto exp2_result = janus::exp2(M_exp);
    EXPECT_DOUBLE_EQ(exp2_result(0, 0), 2.0);  // 2^1 = 2
    EXPECT_DOUBLE_EQ(exp2_result(0, 1), 4.0);  // 2^2 = 4
    EXPECT_DOUBLE_EQ(exp2_result(1, 0), 8.0);  // 2^3 = 8
    EXPECT_DOUBLE_EQ(exp2_result(1, 1), 16.0); // 2^4 = 16

    // cbrt
    janus::NumericMatrix M_cbrt(2, 2);
    M_cbrt << 1.0, 8.0, 27.0, -8.0;
    auto cbrt_result = janus::cbrt(M_cbrt);
    EXPECT_DOUBLE_EQ(cbrt_result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(cbrt_result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(cbrt_result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(cbrt_result(1, 1), -2.0);

    // round
    janus::NumericMatrix M_round(2, 2);
    M_round << 1.4, 1.5, 2.6, -1.5;
    auto round_result = janus::round(M_round);
    EXPECT_DOUBLE_EQ(round_result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(round_result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(round_result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(round_result(1, 1), -2.0);

    // trunc
    janus::NumericMatrix M_trunc(2, 2);
    M_trunc << 2.7, -2.7, 3.9, -3.9;
    auto trunc_result = janus::trunc(M_trunc);
    EXPECT_DOUBLE_EQ(trunc_result(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(trunc_result(0, 1), -2.0);
    EXPECT_DOUBLE_EQ(trunc_result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(trunc_result(1, 1), -3.0);

    // hypot
    janus::NumericMatrix M_x(2, 2), M_y(2, 2);
    M_x << 3.0, 5.0, 0.0, 1.0;
    M_y << 4.0, 12.0, 0.0, 1.0;
    auto hypot_result = janus::hypot(M_x, M_y);
    EXPECT_DOUBLE_EQ(hypot_result(0, 0), 5.0);  // sqrt(9+16)
    EXPECT_DOUBLE_EQ(hypot_result(0, 1), 13.0); // sqrt(25+144)
    EXPECT_DOUBLE_EQ(hypot_result(1, 0), 0.0);
    EXPECT_NEAR(hypot_result(1, 1), std::sqrt(2.0), 1e-9);

    // square
    janus::NumericMatrix M_sq(2, 2);
    M_sq << 2.0, 3.0, 4.0, 5.0;
    auto square_result = janus::square(M_sq);
    EXPECT_DOUBLE_EQ(square_result(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(square_result(0, 1), 9.0);
    EXPECT_DOUBLE_EQ(square_result(1, 0), 16.0);
    EXPECT_DOUBLE_EQ(square_result(1, 1), 25.0);
}

TEST(ArithmeticTests, NewFunctionsSymbolicMatrix) {
    janus::NumericMatrix M(2, 2);
    M << 4.0, 8.0, 16.0, 32.0;
    janus::SymbolicMatrix Ms = janus::to_eigen(janus::to_mx(M));

    // log2
    auto log2_s = janus::log2(Ms);
    auto log2_eval = janus::eval(log2_s);
    EXPECT_NEAR(log2_eval(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(log2_eval(1, 1), 5.0, 1e-9);

    // exp2
    janus::NumericMatrix M_exp(2, 2);
    M_exp << 1.0, 2.0, 3.0, 4.0;
    janus::SymbolicMatrix M_exp_s = janus::to_eigen(janus::to_mx(M_exp));
    auto exp2_s = janus::exp2(M_exp_s);
    auto exp2_eval = janus::eval(exp2_s);
    EXPECT_NEAR(exp2_eval(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(exp2_eval(1, 1), 16.0, 1e-9);

    // square
    janus::NumericMatrix M_sq(2, 2);
    M_sq << 2.0, 3.0, 4.0, 5.0;
    janus::SymbolicMatrix M_sq_s = janus::to_eigen(janus::to_mx(M_sq));
    auto square_s = janus::square(M_sq_s);
    auto square_eval = janus::eval(square_s);
    EXPECT_NEAR(square_eval(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(square_eval(1, 1), 25.0, 1e-9);
}
