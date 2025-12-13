#include <gtest/gtest.h>
#include <janus/math/Spacing.hpp>
#include <janus/math/Rotations.hpp>
#include <janus/math/Linalg.hpp> // to_mx
#include <janus/core/JanusTypes.hpp>
#include "../utils/TestUtils.hpp"
#include <numbers>

template <typename Scalar>
void test_spacing() {
    Scalar start = 0.0;
    Scalar end = 10.0;
    auto lin = janus::linspace(start, end, 5);
    auto cos = janus::cosine_spacing(start, end, 5);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(lin.size(), 5);
        EXPECT_NEAR(lin(2), 5.0, 1e-9);
        EXPECT_NEAR(lin(4), 10.0, 1e-9);
        
        EXPECT_NEAR(cos(0), 0.0, 1e-9);
    } else {
         EXPECT_EQ(lin.size(), 5);
         auto lin_eval = eval_matrix(janus::to_mx(lin));
         EXPECT_NEAR(lin_eval(2), 5.0, 1e-9);
         EXPECT_NEAR(lin_eval(4), 10.0, 1e-9);
         
         auto cos_eval = eval_matrix(janus::to_mx(cos));
         EXPECT_NEAR(cos_eval(0), 0.0, 1e-9);
    }
}

template <typename Scalar>
void test_rotations() {
    Scalar theta = std::numbers::pi_v<double> / 2.0;
    auto R2 = janus::rotation_matrix_2d(theta);
    auto R3 = janus::rotation_matrix_3d(theta, 2); // Z-axis

    if constexpr (std::is_same_v<Scalar, double>) {
        // R2: pi/2 -> [0 -1; 1 0]
        EXPECT_NEAR(R2(0,0), 0.0, 1e-9);
        EXPECT_NEAR(R2(0,1), -1.0, 1e-9);
        
        // R3: Z axis similar to 2D in top-left
        EXPECT_NEAR(R3(0,0), 0.0, 1e-9);
        EXPECT_NEAR(R3(0,1), -1.0, 1e-9);
        EXPECT_NEAR(R3(2,2), 1.0, 1e-9);
    } else {
         auto R2_eval = eval_matrix(janus::to_mx(R2));
         EXPECT_NEAR(R2_eval(0,0), 0.0, 1e-9);
         EXPECT_NEAR(R2_eval(0,1), -1.0, 1e-9);

         auto R3_eval = eval_matrix(janus::to_mx(R3));
         EXPECT_NEAR(R3_eval(0,0), 0.0, 1e-9);
         EXPECT_NEAR(R3_eval(0,1), -1.0, 1e-9);
         EXPECT_NEAR(R3_eval(2,2), 1.0, 1e-9);
    }
}

TEST(GeometryTests, SpacingNumeric) {
    test_spacing<double>();
}

TEST(GeometryTests, SpacingSymbolic) {
    test_spacing<janus::SymbolicScalar>();
}

TEST(GeometryTests, RotationsNumeric) {
    test_rotations<double>();
}

TEST(GeometryTests, RotationsSymbolic) {
    test_rotations<janus::SymbolicScalar>();
}
