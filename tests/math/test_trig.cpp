#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Trig.hpp>
#include <numbers>

template <typename Scalar> void test_trig_ops() {
    Scalar zero = 0.0;
    Scalar half_pi = std::numbers::pi_v<double> / 2.0; // 1.5707...
    Scalar pi = std::numbers::pi_v<double>;

    // Sin, Cos, Tan
    auto s = janus::sin(zero);
    auto c = janus::cos(zero);
    auto t = janus::tan(zero);

    // Inverse
    auto as = janus::asin(1.0);        // pi/2
    auto ac = janus::acos(1.0);        // 0
    auto at = janus::atan(zero);       // 0
    auto at2 = janus::atan2(1.0, 1.0); // pi/4

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(s, 0.0, 1e-9);
        EXPECT_NEAR(c, 1.0, 1e-9);
        EXPECT_NEAR(t, 0.0, 1e-9);

        EXPECT_NEAR(as, half_pi, 1e-9);
        EXPECT_NEAR(ac, 0.0, 1e-9);
        EXPECT_NEAR(at, 0.0, 1e-9);
        EXPECT_NEAR(at2, pi / 4.0, 1e-9);
    } else {
        EXPECT_NEAR(janus::eval(s), 0.0, 1e-9);
        EXPECT_NEAR(janus::eval(c), 1.0, 1e-9);
        EXPECT_NEAR(janus::eval(t), 0.0, 1e-9);

        EXPECT_NEAR(janus::eval(as), std::numbers::pi_v<double> / 2.0, 1e-9);
        EXPECT_NEAR(janus::eval(ac), 0.0, 1e-9);
        EXPECT_NEAR(janus::eval(at), 0.0, 1e-9);
        EXPECT_NEAR(janus::eval(at2), std::numbers::pi_v<double> / 4.0, 1e-9);
    }
}

TEST(TrigTests, Numeric) { test_trig_ops<double>(); }

TEST(TrigTests, Symbolic) { test_trig_ops<janus::SymbolicScalar>(); }
