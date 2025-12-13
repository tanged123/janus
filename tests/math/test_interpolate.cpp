#include <gtest/gtest.h>
#include <janus/math/Interpolate.hpp>
#include <janus/core/JanusTypes.hpp>
#include "../utils/TestUtils.hpp"

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
    
    Scalar query_extrap = 3.0; // Slope from last segment (1->2 is -10). 0 - 10 = -10.
    auto res_extrap = interp(query_extrap);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_mid, 5.0);
        EXPECT_DOUBLE_EQ(res_right, 5.0);
        EXPECT_DOUBLE_EQ(res_extrap, -10.0);
    } else {
        EXPECT_DOUBLE_EQ(eval_scalar(res_mid), 5.0);
        EXPECT_DOUBLE_EQ(eval_scalar(res_right), 5.0);
        EXPECT_DOUBLE_EQ(eval_scalar(res_extrap), -10.0);
    }
}

TEST(InterpolateTests, Numeric) {
    test_interpolate<double>();
}

TEST(InterpolateTests, Symbolic) {
    test_interpolate<janus::SymbolicScalar>();
}
