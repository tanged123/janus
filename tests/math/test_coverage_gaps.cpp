#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/janus.hpp>

// ======================================================================
// FiniteDifference.hpp Coverage
// ======================================================================

TEST(FiniteDifferenceCoverage, ErrorChecks) {
    janus::JanusVector<double> x(3);
    x << 0, 1, 2;

    // Invalid degree
    EXPECT_THROW(janus::finite_difference_coefficients(x, 0.0, -1), janus::InvalidArgument);

    // Too few points
    // Degree 3 requires 4 points
    EXPECT_THROW(janus::finite_difference_coefficients(x, 0.0, 3), janus::InvalidArgument);
}

// ======================================================================
// Integrate.hpp Coverage
// ======================================================================

TEST(IntegrateCoverage, SymbolicLambdaError) {
    auto x = janus::sym("x");
    // Use quad(func, a, b) signature with Symbolic arguments to trigger the template runtime error
    janus::SymbolicScalar a(0.0), b(1.0);
    EXPECT_THROW(janus::quad([](janus::SymbolicScalar s) { return s; }, a, b),
                 janus::IntegrationError);
}

// ======================================================================
// Spacing.hpp Coverage
// ======================================================================

TEST(SpacingCoverage, InvalidN) {
    // n < 1 should throw for all
    EXPECT_THROW(janus::linspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::cosine_spacing(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::sinspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::logspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::geomspace(1.0, 10.0, 0), janus::InvalidArgument);
}

// ======================================================================
// Linalg.hpp Coverage
// ======================================================================

TEST(LinalgCoverage, CrossError) {
    janus::JanusVector<double> a(2);
    a << 1, 2;
    janus::JanusVector<double> b(3);
    b << 1, 2, 3;

    EXPECT_THROW(janus::cross(a, b), janus::InvalidArgument);
}
