#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Interpolate.hpp>

template <typename Scalar> void test_interpolate() {
    // x = [0, 1, 2]
    // y = [0, 10, 0]
    Eigen::VectorXd x(3);
    x << 0.0, 1.0, 2.0;
    Eigen::VectorXd y(3);
    y << 0.0, 10.0, 0.0;

    janus::JanusInterpolator interp(x, y);

    Scalar query_mid = 0.5; // Expect 5.0
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
        EXPECT_DOUBLE_EQ(janus::eval(res_mid), 5.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_right), 5.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_extrap), -10.0);
    }
}

TEST(InterpolateTests, Numeric) { test_interpolate<double>(); }

TEST(InterpolateTests, Symbolic) { test_interpolate<janus::SymbolicScalar>(); }

TEST(InterpolateTests, CoverageErrorChecks) {
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(2);
    y << 0, 1;

    // Mismatched size
    EXPECT_THROW(janus::JanusInterpolator(x, y), std::invalid_argument);

    // Size < 2
    Eigen::VectorXd x1(1);
    x1 << 0;
    Eigen::VectorXd y1(1);
    y1 << 0;
    EXPECT_THROW(janus::JanusInterpolator(x1, y1), std::invalid_argument);

    // Unsorted
    Eigen::VectorXd xu(3);
    xu << 0, 2, 1;
    Eigen::VectorXd yu(3);
    yu << 0, 0, 0;
    EXPECT_THROW(janus::JanusInterpolator(xu, yu), std::invalid_argument);

    // Uninitialized use
    janus::JanusInterpolator empty;
    EXPECT_THROW(empty(1.0), std::runtime_error);

    // Uninitialized matrix use
    Eigen::MatrixXd q(1, 1);
    q << 1.0;
    EXPECT_THROW(empty(q), std::runtime_error);
}

TEST(InterpolateTests, NumericExtrapolationLow) {
    Eigen::VectorXd x(2);
    x << 0, 1;
    Eigen::VectorXd y(2);
    y << 0, 10;
    janus::JanusInterpolator interp(x, y);

    // x < 0. Slope 10. at -1 should be -10.
    EXPECT_DOUBLE_EQ(interp(-1.0), -10.0);
}
