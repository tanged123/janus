#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Interpolate.hpp>

// ============================================================================
// Interp1D Tests (1D Interpolation Class)
// ============================================================================

template <typename Scalar> void test_interp1d() {
    // x = [0, 1, 2]
    // y = [0, 10, 0]
    Eigen::VectorXd x(3);
    x << 0.0, 1.0, 2.0;
    Eigen::VectorXd y(3);
    y << 0.0, 10.0, 0.0;

    janus::Interp1D interp(x, y); // Default: Linear

    Scalar query_mid = 0.5; // Expect 5.0
    auto res_mid = interp(query_mid);

    Scalar query_right = 1.5; // Expect 5.0
    auto res_right = interp(query_right);

    // Query at boundary (clamped)
    Scalar query_bound = 2.0;
    auto res_bound = interp(query_bound);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_mid, 5.0);
        EXPECT_DOUBLE_EQ(res_right, 5.0);
        EXPECT_DOUBLE_EQ(res_bound, 0.0); // Value at x=2
    } else {
        EXPECT_DOUBLE_EQ(janus::eval(res_mid), 5.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_right), 5.0);
        EXPECT_DOUBLE_EQ(janus::eval(res_bound), 0.0);
    }
}

TEST(Interp1DTests, Numeric) { test_interp1d<double>(); }

TEST(Interp1DTests, Symbolic) { test_interp1d<janus::SymbolicScalar>(); }

TEST(Interp1DTests, CoverageErrorChecks) {
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(2);
    y << 0, 1;

    // Mismatched size
    EXPECT_THROW(janus::Interp1D(x, y), janus::InterpolationError);

    // Size < 2
    Eigen::VectorXd x1(1);
    x1 << 0;
    Eigen::VectorXd y1(1);
    y1 << 0;
    EXPECT_THROW(janus::Interp1D(x1, y1), janus::InterpolationError);

    // Unsorted
    Eigen::VectorXd xu(3);
    xu << 0, 2, 1;
    Eigen::VectorXd yu(3);
    yu << 0, 0, 0;
    EXPECT_THROW(janus::Interp1D(xu, yu), janus::InterpolationError);

    // Uninitialized use
    janus::Interp1D empty;
    EXPECT_THROW(empty(1.0), janus::InterpolationError);

    // Uninitialized matrix use
    Eigen::MatrixXd q(1, 1);
    q << 1.0;
    EXPECT_THROW(empty(q), janus::InterpolationError);
}

TEST(Interp1DTests, BoundsClamping) {
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(3);
    y << 0, 10, 20;
    janus::Interp1D interp(x, y);

    // Query outside bounds - should clamp
    EXPECT_DOUBLE_EQ(interp(-1.0), 0.0); // Clamps to x=0, y=0
    EXPECT_DOUBLE_EQ(interp(5.0), 20.0); // Clamps to x=2, y=20
}

TEST(Interp1DTests, HermiteMethod) {
    // Test Hermite (C1) interpolation
    Eigen::VectorXd x(4);
    x << 0, 1, 2, 3;
    Eigen::VectorXd y(4);
    y << 0, 1, 4, 9; // y = x^2

    janus::Interp1D interp(x, y, janus::InterpolationMethod::Hermite);
    EXPECT_EQ(interp.method(), janus::InterpolationMethod::Hermite);

    // Should produce smooth interpolation
    double result = interp(1.5);
    EXPECT_GT(result, 1.0); // Between y(1)=1 and y(2)=4
    EXPECT_LT(result, 4.0);
}

TEST(Interp1DTests, BSplineMethod) {
    // Test BSpline (C2) interpolation
    Eigen::VectorXd x(4);
    x << 0, 1, 2, 3;
    Eigen::VectorXd y(4);
    y << 1, 1, 1, 1; // Constant function

    janus::Interp1D interp(x, y, janus::InterpolationMethod::BSpline);
    EXPECT_EQ(interp.method(), janus::InterpolationMethod::BSpline);

    // Constant should interpolate exactly
    EXPECT_NEAR(interp(0.5), 1.0, 1e-10);
    EXPECT_NEAR(interp(1.5), 1.0, 1e-10);
    EXPECT_NEAR(interp(2.5), 1.0, 1e-10);
}

TEST(Interp1DTests, BSplineRequires4Points) {
    // BSpline should fail with < 4 points
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(3);
    y << 0, 1, 2;

    EXPECT_THROW(janus::Interp1D(x, y, janus::InterpolationMethod::BSpline),
                 janus::InterpolationError);
}

TEST(Interp1DTests, NearestMethod) {
    // Test Nearest neighbor
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(3);
    y << 0, 10, 20;

    janus::Interp1D interp(x, y, janus::InterpolationMethod::Nearest);

    // Nearest to x=0
    EXPECT_DOUBLE_EQ(interp(0.4), 0.0);
    // Nearest to x=1
    EXPECT_DOUBLE_EQ(interp(0.6), 10.0);
    EXPECT_DOUBLE_EQ(interp(1.4), 10.0);
    // Nearest to x=2
    EXPECT_DOUBLE_EQ(interp(1.6), 20.0);
}

TEST(Interp1DTests, HermiteSymbolicNotSupported) {
    Eigen::VectorXd x(4);
    x << 0, 1, 2, 3;
    Eigen::VectorXd y(4);
    y << 0, 1, 4, 9;

    janus::Interp1D interp(x, y, janus::InterpolationMethod::Hermite);

    // Symbolic should throw
    janus::SymbolicScalar query = casadi::MX(1.5);
    EXPECT_THROW(interp(query), janus::InterpolationError);
}

TEST(Interp1DTests, BSplineSymbolic) {
    // BSpline should work with symbolic
    Eigen::VectorXd x(4);
    x << 0, 1, 2, 3;
    Eigen::VectorXd y(4);
    y << 1, 1, 1, 1;

    janus::Interp1D interp(x, y, janus::InterpolationMethod::BSpline);

    janus::SymbolicScalar query = casadi::MX(1.5);
    auto result = interp(query);

    EXPECT_NEAR(eval_scalar(result), 1.0, 1e-9);
}

TEST(Interp1DTests, VectorizedQuery) {
    // Test vectorized queries using eval_batch
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    Eigen::VectorXd y(3);
    y << 0, 10, 20;

    janus::Interp1D interp(x, y);

    Eigen::VectorXd queries(3);
    queries << 0.5, 1.0, 1.5;

    auto results = interp.eval_batch(queries);

    EXPECT_NEAR(results(0), 5.0, 1e-10);
    EXPECT_NEAR(results(1), 10.0, 1e-10);
    EXPECT_NEAR(results(2), 15.0, 1e-10);
}

// ============================================================================
// N-Dimensional Interpolation Tests (interpn)
// ============================================================================

TEST(InterpnTests, Numeric2DLinear) {
    // 2D grid: x = [0, 1], y = [0, 1]
    // Values: z(x, y) = x + y
    // z(0,0)=0, z(1,0)=1, z(0,1)=1, z(1,1)=2
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // Values in Fortran order: (0,0), (1,0), (0,1), (1,1) -> 0, 1, 1, 2
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query at (0.5, 0.5) - should get 1.0
    Eigen::MatrixXd xi(1, 2);
    xi << 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Linear);

    EXPECT_NEAR(result(0), 1.0, 1e-10);
}

TEST(InterpnTests, Numeric2DMultiplePoints) {
    // 3x3 grid: x = [0, 1, 2], y = [0, 1, 2]
    // Values: z(x, y) = x * y
    Eigen::VectorXd x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    Eigen::VectorXd y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // Values in Fortran order:
    // (0,0)=0, (1,0)=0, (2,0)=0, (0,1)=0, (1,1)=1, (2,1)=2, (0,2)=0, (1,2)=2, (2,2)=4
    Eigen::VectorXd values(9);
    values << 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0;

    // Multiple query points
    Eigen::MatrixXd xi(3, 2);
    xi << 0.5, 0.5, // Should give ~0.25
        1.0, 1.0,   // Should give 1.0 exactly
        1.5, 1.5;   // Should give ~2.25

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Linear);

    EXPECT_NEAR(result(0), 0.25, 1e-10);
    EXPECT_NEAR(result(1), 1.0, 1e-10);
    EXPECT_NEAR(result(2), 2.25, 1e-10);
}

TEST(InterpnTests, Numeric2DBSpline) {
    // Test BSpline method - needs at least 4 points per dimension for cubic
    // 4x4 grid with constant function for easy verification
    Eigen::VectorXd x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;
    Eigen::VectorXd y_pts(4);
    y_pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // Values: z = 1 (constant for easy verification)
    Eigen::VectorXd values(16);
    values.setConstant(1.0);

    Eigen::MatrixXd xi(1, 2);
    xi << 1.5, 1.5;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::BSpline);

    // For constant function, bspline should also give 1.0
    EXPECT_NEAR(result(0), 1.0, 1e-10);
}

TEST(InterpnTests, NumericFillValue) {
    // Test out-of-bounds with fill_value
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query outside bounds
    Eigen::MatrixXd xi(2, 2);
    xi << 0.5, 0.5, // In bounds
        2.0, 0.5;   // Out of bounds in x

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Linear,
                                         std::optional<double>(-999.0));

    EXPECT_NEAR(result(0), 1.0, 1e-10);    // In bounds
    EXPECT_NEAR(result(1), -999.0, 1e-10); // Fill value
}

TEST(InterpnTests, NumericExtrapolation) {
    // Without fill_value, should clamp to bounds
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query outside bounds - should clamp
    Eigen::MatrixXd xi(1, 2);
    xi << 2.0, 0.5; // x=2 should clamp to x=1

    auto result = janus::interpn<double>(points, values, xi);

    // At (1.0, 0.5): interpolate between z(1,0)=1 and z(1,1)=2 -> 1.5
    EXPECT_NEAR(result(0), 1.5, 1e-10);
}

TEST(InterpnTests, Symbolic2DLinear) {
    // 2D grid symbolic test
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // z(x,y) = x + y
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Symbolic query - use fixed numeric values for simplicity
    // Query at (0.5, 0.5) which should give 1.0
    Eigen::Matrix<janus::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 2);
    xi(0, 0) = casadi::MX(0.5);
    xi(0, 1) = casadi::MX(0.5);

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi);

    // Evaluate the symbolic result (no variables, just constants)
    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

TEST(InterpnTests, Numeric3D) {
    // 3D interpolation: 2x2x2 grid
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;
    Eigen::VectorXd z_pts(2);
    z_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts, z_pts};

    // Values: f(x,y,z) = x + y + z
    // Fortran order: iterate x fastest, then y, then z
    // (0,0,0)=0, (1,0,0)=1, (0,1,0)=1, (1,1,0)=2, (0,0,1)=1, (1,0,1)=2, (0,1,1)=2, (1,1,1)=3
    Eigen::VectorXd values(8);
    values << 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0;

    // Query at center (0.5, 0.5, 0.5) -> should give 1.5
    Eigen::MatrixXd xi(1, 3);
    xi << 0.5, 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.5, 1e-10);
}

TEST(InterpnTests, Numeric4D) {
    // 4D interpolation: 2^4 = 16 grid points
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {pts, pts, pts, pts};

    // Values: f(x1, x2, x3, x4) = x1 + x2 + x3 + x4
    // 2^4 = 16 values in Fortran order
    Eigen::VectorXd values(16);
    int idx = 0;
    for (int i4 = 0; i4 < 2; ++i4) {
        for (int i3 = 0; i3 < 2; ++i3) {
            for (int i2 = 0; i2 < 2; ++i2) {
                for (int i1 = 0; i1 < 2; ++i1) {
                    values(idx++) = i1 + i2 + i3 + i4;
                }
            }
        }
    }

    // Query at center (0.5, 0.5, 0.5, 0.5) -> should give 2.0
    Eigen::MatrixXd xi(1, 4);
    xi << 0.5, 0.5, 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.0, 1e-10);
}

TEST(InterpnTests, Numeric5D) {
    // 5D interpolation: 2^5 = 32 grid points
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    Eigen::VectorXd values(32);
    int idx = 0;
    for (int i5 = 0; i5 < 2; ++i5) {
        for (int i4 = 0; i4 < 2; ++i4) {
            for (int i3 = 0; i3 < 2; ++i3) {
                for (int i2 = 0; i2 < 2; ++i2) {
                    for (int i1 = 0; i1 < 2; ++i1) {
                        values(idx++) = i1 + i2 + i3 + i4 + i5;
                    }
                }
            }
        }
    }

    // Query at center -> should give 2.5
    Eigen::MatrixXd xi(1, 5);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.5, 1e-10);
}

TEST(InterpnTests, Numeric6D) {
    // 6D interpolation: 2^6 = 64 grid points
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {pts, pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    Eigen::VectorXd values(64);
    int idx = 0;
    for (int i6 = 0; i6 < 2; ++i6) {
        for (int i5 = 0; i5 < 2; ++i5) {
            for (int i4 = 0; i4 < 2; ++i4) {
                for (int i3 = 0; i3 < 2; ++i3) {
                    for (int i2 = 0; i2 < 2; ++i2) {
                        for (int i1 = 0; i1 < 2; ++i1) {
                            values(idx++) = i1 + i2 + i3 + i4 + i5 + i6;
                        }
                    }
                }
            }
        }
    }

    // Query at center -> should give 3.0
    Eigen::MatrixXd xi(1, 6);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 3.0, 1e-10);
}

TEST(InterpnTests, Numeric7D) {
    // 7D interpolation: 2^7 = 128 grid points
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {pts, pts, pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    Eigen::VectorXd values(128);
    int idx = 0;
    for (int i7 = 0; i7 < 2; ++i7) {
        for (int i6 = 0; i6 < 2; ++i6) {
            for (int i5 = 0; i5 < 2; ++i5) {
                for (int i4 = 0; i4 < 2; ++i4) {
                    for (int i3 = 0; i3 < 2; ++i3) {
                        for (int i2 = 0; i2 < 2; ++i2) {
                            for (int i1 = 0; i1 < 2; ++i1) {
                                values(idx++) = i1 + i2 + i3 + i4 + i5 + i6 + i7;
                            }
                        }
                    }
                }
            }
        }
    }

    // Query at center -> should give 3.5
    Eigen::MatrixXd xi(1, 7);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 3.5, 1e-10);
}

TEST(InterpnTests, ErrorEmptyPoints) {
    std::vector<Eigen::VectorXd> points;
    Eigen::VectorXd values(1);
    values << 1.0;
    Eigen::MatrixXd xi(1, 1);
    xi << 0.5;

    EXPECT_THROW(janus::interpn<double>(points, values, xi), janus::InterpolationError);
}

TEST(InterpnTests, ErrorUnsortedPoints) {
    Eigen::VectorXd x_pts(3);
    x_pts << 0.0, 2.0, 1.0; // Not sorted!
    std::vector<Eigen::VectorXd> points = {x_pts};
    Eigen::VectorXd values(3);
    values << 1.0, 2.0, 3.0;
    Eigen::MatrixXd xi(1, 1);
    xi << 0.5;

    EXPECT_THROW(janus::interpn<double>(points, values, xi), janus::InterpolationError);
}

TEST(InterpnTests, ErrorValuesSizeMismatch) {
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;
    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // Wrong size: should be 4, not 3
    Eigen::VectorXd values(3);
    values << 1.0, 2.0, 3.0;
    Eigen::MatrixXd xi(1, 2);
    xi << 0.5, 0.5;

    EXPECT_THROW(janus::interpn<double>(points, values, xi), janus::InterpolationError);
}

// ============================================================================
// Edge and Corner Cases
// ============================================================================

TEST(InterpnTests, QueryAtGridPoints2D) {
    // Query exactly at grid points should return exact values
    Eigen::VectorXd x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    Eigen::VectorXd y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // z = x + 2*y
    // Fortran order: iterate x fastest
    Eigen::VectorXd values(9);
    values << 0, 1, 2, // y=0: (0,0)=0, (1,0)=1, (2,0)=2
        2, 3, 4,       // y=1: (0,1)=2, (1,1)=3, (2,1)=4
        4, 5, 6;       // y=2: (0,2)=4, (1,2)=5, (2,2)=6

    // Query all grid points
    Eigen::MatrixXd xi(9, 2);
    xi << 0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2;

    auto result = janus::interpn<double>(points, values, xi);

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result(i), values(i), 1e-10) << "Mismatch at grid point " << i;
    }
}

TEST(InterpnTests, QueryAtEdges2D) {
    // Query along edges (one coordinate at boundary)
    Eigen::VectorXd x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    Eigen::VectorXd y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // z = x + y (simple to verify)
    Eigen::VectorXd values(9);
    values << 0, 1, 2, // y=0
        1, 2, 3,       // y=1
        2, 3, 4;       // y=2

    // Query along bottom edge (y=0), top edge (y=2), left edge (x=0), right edge (x=2)
    Eigen::MatrixXd xi(8, 2);
    xi << 0.5, 0.0, // bottom edge
        1.5, 0.0,   // bottom edge
        0.5, 2.0,   // top edge
        1.5, 2.0,   // top edge
        0.0, 0.5,   // left edge
        0.0, 1.5,   // left edge
        2.0, 0.5,   // right edge
        2.0, 1.5;   // right edge

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.5, 1e-10); // (0.5, 0) -> 0.5
    EXPECT_NEAR(result(1), 1.5, 1e-10); // (1.5, 0) -> 1.5
    EXPECT_NEAR(result(2), 2.5, 1e-10); // (0.5, 2) -> 2.5
    EXPECT_NEAR(result(3), 3.5, 1e-10); // (1.5, 2) -> 3.5
    EXPECT_NEAR(result(4), 0.5, 1e-10); // (0, 0.5) -> 0.5
    EXPECT_NEAR(result(5), 1.5, 1e-10); // (0, 1.5) -> 1.5
    EXPECT_NEAR(result(6), 2.5, 1e-10); // (2, 0.5) -> 2.5
    EXPECT_NEAR(result(7), 3.5, 1e-10); // (2, 1.5) -> 3.5
}

TEST(InterpnTests, QueryAtCorners2D) {
    // Query exactly at all four corners
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    Eigen::VectorXd values(4);
    values << 1.0, 2.0, 3.0, 4.0; // corners: (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4

    Eigen::MatrixXd xi(4, 2);
    xi << 0, 0, 1, 0, 0, 1, 1, 1;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.0, 1e-10);
    EXPECT_NEAR(result(1), 2.0, 1e-10);
    EXPECT_NEAR(result(2), 3.0, 1e-10);
    EXPECT_NEAR(result(3), 4.0, 1e-10);
}

// ============================================================================
// Extrapolation Tests
// ============================================================================

TEST(InterpnTests, ExtrapolationAllDirections2D) {
    // Test extrapolation (clamping) in all directions
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // z = x + y
    Eigen::VectorXd values(4);
    values << 0, 1, 1, 2; // (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=2

    // Query outside in all directions (will be clamped)
    Eigen::MatrixXd xi(8, 2);
    xi << -1.0, 0.5, // left of grid
        2.0, 0.5,    // right of grid
        0.5, -1.0,   // below grid
        0.5, 2.0,    // above grid
        -1.0, -1.0,  // bottom-left corner
        2.0, -1.0,   // bottom-right corner
        -1.0, 2.0,   // top-left corner
        2.0, 2.0;    // top-right corner

    auto result = janus::interpn<double>(points, values, xi);

    // All should clamp to boundary values
    EXPECT_NEAR(result(0), 0.5, 1e-10); // clamps to (0, 0.5)
    EXPECT_NEAR(result(1), 1.5, 1e-10); // clamps to (1, 0.5)
    EXPECT_NEAR(result(2), 0.5, 1e-10); // clamps to (0.5, 0)
    EXPECT_NEAR(result(3), 1.5, 1e-10); // clamps to (0.5, 1)
    EXPECT_NEAR(result(4), 0.0, 1e-10); // clamps to (0, 0)
    EXPECT_NEAR(result(5), 1.0, 1e-10); // clamps to (1, 0)
    EXPECT_NEAR(result(6), 1.0, 1e-10); // clamps to (0, 1)
    EXPECT_NEAR(result(7), 2.0, 1e-10); // clamps to (1, 1)
}

TEST(InterpnTests, FillValueAllDirections2D) {
    // Test fill_value in all out-of-bounds directions
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};
    Eigen::VectorXd values(4);
    values << 0, 1, 1, 2;

    // Mix of in-bounds and out-of-bounds
    Eigen::MatrixXd xi(5, 2);
    xi << 0.5, 0.5, // in bounds
        -0.5, 0.5,  // out left
        1.5, 0.5,   // out right
        0.5, -0.5,  // out bottom
        0.5, 1.5;   // out top

    double fill = -999.0;
    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Linear,
                                         std::optional<double>(fill));

    EXPECT_NEAR(result(0), 1.0, 1e-10);  // in bounds
    EXPECT_NEAR(result(1), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(2), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(3), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(4), fill, 1e-10); // out of bounds
}

// ============================================================================
// Non-Uniform Grid Tests
// ============================================================================

TEST(InterpnTests, NonUniformGrid2D) {
    // Non-uniformly spaced grid
    Eigen::VectorXd x_pts(4);
    x_pts << 0.0, 0.1, 0.5, 1.0; // Clustered near 0
    Eigen::VectorXd y_pts(3);
    y_pts << 0.0, 0.8, 1.0; // Clustered near 1

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};

    // z = x * y
    Eigen::VectorXd values(12);
    int idx = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
            values(idx++) = x_pts(i) * y_pts(j);
        }
    }

    // Query at various points
    Eigen::MatrixXd xi(3, 2);
    xi << 0.05, 0.4, // in first x-cell
        0.75, 0.9,   // in last x-cell, second y-cell
        0.25, 0.5;   // between cells

    auto result = janus::interpn<double>(points, values, xi);

    // Expected: linear interpolation of x*y
    EXPECT_NEAR(result(0), 0.05 * 0.4, 0.05); // Approximate
    EXPECT_NEAR(result(1), 0.75 * 0.9, 0.05);
    EXPECT_NEAR(result(2), 0.25 * 0.5, 0.05);
}

TEST(InterpnTests, NonUniformGrid3D) {
    // Non-uniform 3D grid
    Eigen::VectorXd x_pts(3);
    x_pts << 0.0, 0.2, 1.0;
    Eigen::VectorXd y_pts(3);
    y_pts << 0.0, 0.5, 1.0;
    Eigen::VectorXd z_pts(2);
    z_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts, z_pts};

    // z = x + y + z_coord
    Eigen::VectorXd values(18); // 3*3*2 = 18
    int idx = 0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = x_pts(i) + y_pts(j) + z_pts(k);
            }
        }
    }

    // Query
    Eigen::MatrixXd xi(1, 3);
    xi << 0.1, 0.25, 0.5;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.1 + 0.25 + 0.5, 0.05);
}

// ============================================================================
// High-Dimensional Edge Cases
// ============================================================================

TEST(InterpnTests, HighDimEdgeQuery5D) {
    // Query at edge of 5D hypercube
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(5, pts);

    // f = sum of coordinates
    Eigen::VectorXd values(32);
    for (int i = 0; i < 32; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 5; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at edge: (0.5, 0.5, 0.5, 0.5, 0) - last dim at boundary
    Eigen::MatrixXd xi(1, 5);
    xi << 0.5, 0.5, 0.5, 0.5, 0.0;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.0, 1e-10); // 0.5*4 + 0 = 2.0
}

TEST(InterpnTests, HighDimCornerQuery6D) {
    // Query at corner of 6D hypercube
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(6, pts);

    // f = sum of coordinates
    Eigen::VectorXd values(64);
    for (int i = 0; i < 64; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 6; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at all-ones corner
    Eigen::MatrixXd xi(1, 6);
    xi << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 6.0, 1e-10); // sum of all 1s
}

TEST(InterpnTests, HighDimExtrapolation4D) {
    // Test extrapolation in 4D
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(4, pts);

    // f = x1 + x2 + x3 + x4
    Eigen::VectorXd values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query outside grid (should clamp)
    Eigen::MatrixXd xi(1, 4);
    xi << 2.0, 2.0, 2.0, 2.0; // All out of bounds

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 4.0, 1e-10); // clamps to (1,1,1,1)
}

// ============================================================================
// Multiple Query Points Batch Test
// ============================================================================

TEST(InterpnTests, BatchQuery100Points3D) {
    // Test with many query points
    Eigen::VectorXd pts(3);
    pts << 0.0, 0.5, 1.0;

    std::vector<Eigen::VectorXd> points(3, pts);

    // f = x + y + z
    Eigen::VectorXd values(27); // 3^3
    int idx = 0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = pts(i) + pts(j) + pts(k);
            }
        }
    }

    // 100 random-ish query points
    Eigen::MatrixXd xi(100, 3);
    for (int i = 0; i < 100; ++i) {
        double t = static_cast<double>(i) / 99.0;
        xi(i, 0) = t;
        xi(i, 1) = t * 0.8;
        xi(i, 2) = t * 0.6;
    }

    auto result = janus::interpn<double>(points, values, xi);

    // Verify a subset
    for (int i = 0; i < 100; i += 10) {
        double expected = xi(i, 0) + xi(i, 1) + xi(i, 2);
        EXPECT_NEAR(result(i), expected, 1e-10) << "Mismatch at query " << i;
    }
}

TEST(InterpnTests, TransposedXiInput) {
    // Test that transposed xi input works (n_dims x n_points instead of n_points x n_dims)
    Eigen::VectorXd x_pts(2);
    x_pts << 0.0, 1.0;
    Eigen::VectorXd y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points = {x_pts, y_pts};
    Eigen::VectorXd values(4);
    values << 0, 1, 1, 2; // z = x + y

    // xi as (n_dims, n_points) = (2, 3)
    Eigen::MatrixXd xi(2, 3);
    xi << 0.5, 0.0, 1.0, // x values
        0.5, 0.0, 1.0;   // y values

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.0, 1e-10); // (0.5, 0.5)
    EXPECT_NEAR(result(1), 0.0, 1e-10); // (0, 0)
    EXPECT_NEAR(result(2), 2.0, 1e-10); // (1, 1)
}

// ============================================================================
// Symbolic High-Dimensional Tests
// ============================================================================

TEST(InterpnTests, Symbolic3DLinear) {
    // 3D symbolic interpolation
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(3, pts);

    // f = x + y + z
    Eigen::VectorXd values(8);
    values << 0, 1, 1, 2, 1, 2, 2, 3; // Fortran order

    // Symbolic query at fixed point
    Eigen::Matrix<janus::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 3);
    xi(0, 0) = casadi::MX(0.5);
    xi(0, 1) = casadi::MX(0.5);
    xi(0, 2) = casadi::MX(0.5);

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi);

    // Should give 1.5
    EXPECT_NEAR(eval_scalar(result(0)), 1.5, 1e-9);
}

TEST(InterpnTests, Symbolic4DLinear) {
    // 4D symbolic interpolation
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(4, pts);

    // f = sum of coordinates
    Eigen::VectorXd values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Symbolic query
    Eigen::Matrix<janus::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 4);
    xi(0, 0) = casadi::MX(0.25);
    xi(0, 1) = casadi::MX(0.25);
    xi(0, 2) = casadi::MX(0.25);
    xi(0, 3) = casadi::MX(0.25);

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi);

    // Should give 1.0 (0.25 * 4)
    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

TEST(InterpnTests, Symbolic5DCorner) {
    // 5D symbolic at corner
    Eigen::VectorXd pts(2);
    pts << 0.0, 1.0;

    std::vector<Eigen::VectorXd> points(5, pts);

    // f = sum of coordinates
    Eigen::VectorXd values(32);
    for (int i = 0; i < 32; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 5; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at (1,1,1,1,1) corner
    Eigen::Matrix<janus::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 5);
    for (int d = 0; d < 5; ++d) {
        xi(0, d) = casadi::MX(1.0);
    }

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi);

    EXPECT_NEAR(eval_scalar(result(0)), 5.0, 1e-9);
}

TEST(InterpnTests, SymbolicBSpline4D) {
    // 4D B-spline symbolic interpolation (requires 4+ points per dim)
    Eigen::VectorXd pts(4);
    pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<Eigen::VectorXd> points(4, pts);

    // f = 1 (constant for easy verification)
    Eigen::VectorXd values(256); // 4^4 = 256
    values.setConstant(1.0);

    // Symbolic query at center
    Eigen::Matrix<janus::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 4);
    for (int d = 0; d < 4; ++d) {
        xi(0, d) = casadi::MX(1.5);
    }

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi,
                                                        janus::InterpolationMethod::BSpline);

    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

// ============================================================================
// Tests Using Janus Types (API Best Practice Demonstration)
// ============================================================================

TEST(InterpnTests, JanusTypesNumeric3D) {
    // Demonstrate using janus::NumericVector instead of Eigen::VectorXd
    janus::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    janus::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;
    janus::NumericVector z_pts(2);
    z_pts << 0.0, 1.0;

    // Use vector of janus::NumericVector
    std::vector<janus::NumericVector> points = {x_pts, y_pts, z_pts};

    // Values using janus::NumericVector
    janus::NumericVector values(18); // 3*3*2 = 18
    int idx = 0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = x_pts(i) + y_pts(j) + z_pts(k);
            }
        }
    }

    // Query using janus::NumericMatrix
    janus::NumericMatrix xi(2, 3);
    xi << 0.5, 0.5, 0.5, 1.0, 1.0, 0.5;

    auto result = janus::interpn<janus::NumericScalar>(points, values, xi);

    EXPECT_NEAR(result(0), 1.5, 1e-10); // 0.5 + 0.5 + 0.5
    EXPECT_NEAR(result(1), 2.5, 1e-10); // 1.0 + 1.0 + 0.5
}

TEST(InterpnTests, JanusTypesSymbolic4D) {
    // Demonstrate using janus::SymbolicMatrix for queries
    janus::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<janus::NumericVector> points(4, pts);

    // f = sum of coordinates
    janus::NumericVector values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query using janus::SymbolicMatrix
    janus::SymbolicMatrix xi(1, 4);
    for (int d = 0; d < 4; ++d) {
        xi(0, d) = casadi::MX(0.5);
    }

    auto result = janus::interpn<janus::SymbolicScalar>(points, values, xi);

    // Should give 2.0 (0.5 * 4)
    EXPECT_NEAR(eval_scalar(result(0)), 2.0, 1e-9);
}

TEST(InterpnTests, JanusTypesTemplated) {
    // Template test using JanusVector<T> and JanusMatrix<T>
    janus::JanusVector<double> pts(3);
    pts << 0.0, 0.5, 1.0;

    std::vector<janus::JanusVector<double>> points(2, pts);

    // f = x + y
    janus::JanusVector<double> values(9);
    int idx = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            values(idx++) = pts(i) + pts(j);
        }
    }

    // Query
    janus::JanusMatrix<double> xi(3, 2);
    xi << 0.25, 0.25, 0.5, 0.5, 0.75, 0.75;

    auto result = janus::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.5, 1e-10); // 0.25 + 0.25
    EXPECT_NEAR(result(1), 1.0, 1e-10); // 0.5 + 0.5
    EXPECT_NEAR(result(2), 1.5, 1e-10); // 0.75 + 0.75
}

TEST(InterpnTests, JanusTypesHighDim6D) {
    // 6D test with Janus types
    janus::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<janus::NumericVector> points(6, pts);

    // f = sum of coordinates
    janus::NumericVector values(64); // 2^6 = 64
    for (int i = 0; i < 64; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 6; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Edge query: (0.5, 0.5, 0.5, 0.5, 0.5, 0)
    janus::NumericMatrix xi(1, 6);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.0;

    auto result = janus::interpn<janus::NumericScalar>(points, values, xi);

    EXPECT_NEAR(result(0), 2.5, 1e-10); // 0.5*5 + 0
}

// ============================================================================
// Hermite (C1 Catmull-Rom) Interpolation Tests
// ============================================================================

TEST(InterpnTests, Hermite1D) {
    // 1D Hermite interpolation (via 1D grid)
    janus::NumericVector x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<janus::NumericVector> points = {x_pts};

    // y = x^2 (quadratic function to test smoothness)
    janus::NumericVector values(4);
    values << 0.0, 1.0, 4.0, 9.0;

    // Query at interior points
    janus::NumericMatrix xi(3, 1);
    xi << 0.5, 1.5, 2.5;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    // Hermite should produce smooth interpolation
    // At x=0.5, linear would give 0.5, Hermite should be closer to 0.25
    EXPECT_GT(result(0), 0.0);
    EXPECT_LT(result(0), 1.0);

    // At x=1.5, expect between 1.0 and 4.0, closer to 2.25
    EXPECT_GT(result(1), 1.0);
    EXPECT_LT(result(1), 4.0);

    // At x=2.5, expect between 4.0 and 9.0, closer to 6.25
    EXPECT_GT(result(2), 4.0);
    EXPECT_LT(result(2), 9.0);
}

TEST(InterpnTests, Hermite2DLinearFunction) {
    // 2D Hermite should exactly interpolate linear functions
    janus::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    janus::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points = {x_pts, y_pts};

    // z = x + y (linear function)
    janus::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4; // Fortran order

    janus::NumericMatrix xi(4, 2);
    xi << 0.5, 0.5, 1.0, 1.0, 0.25, 0.75, 1.5, 0.5;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    // Linear functions should be exactly interpolated
    EXPECT_NEAR(result(0), 1.0, 1e-10); // 0.5 + 0.5
    EXPECT_NEAR(result(1), 2.0, 1e-10); // 1.0 + 1.0
    EXPECT_NEAR(result(2), 1.0, 1e-10); // 0.25 + 0.75
    EXPECT_NEAR(result(3), 2.0, 1e-10); // 1.5 + 0.5
}

TEST(InterpnTests, Hermite3DSmoothness) {
    // Test 3D Hermite interpolation
    janus::NumericVector pts(3);
    pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points(3, pts);

    // f = x + y + z
    janus::NumericVector values(27);
    int idx = 0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = pts(i) + pts(j) + pts(k);
            }
        }
    }

    // Query at center
    janus::NumericMatrix xi(1, 3);
    xi << 1.0, 1.0, 1.0;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    EXPECT_NEAR(result(0), 3.0, 1e-10); // 1+1+1
}

TEST(InterpnTests, HermiteVsLinearSmoother) {
    // Hermite should produce smoother results than linear for non-linear data
    janus::NumericVector x_pts(5);
    x_pts << 0.0, 1.0, 2.0, 3.0, 4.0;

    std::vector<janus::NumericVector> points = {x_pts};

    // Sinusoidal function (highly non-linear)
    janus::NumericVector values(5);
    values << 0.0, 0.84147, 0.9093, 0.14112, -0.7568; // sin(0), sin(1), sin(2), sin(3), sin(4)

    // Query between grid points
    janus::NumericMatrix xi(1, 1);
    xi << 1.5;

    auto result_linear =
        janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Linear);
    auto result_hermite =
        janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    // sin(1.5) ≈ 0.9975
    double true_val = 0.9974949866;

    // Both should be reasonable, but we mainly check they're different
    EXPECT_NE(result_linear(0), result_hermite(0));

    // Hermite may or may not be closer depending on the function
    EXPECT_GT(result_hermite(0), 0.5); // Should be positive
    EXPECT_LT(result_hermite(0), 1.5); // Should be reasonable
}

TEST(InterpnTests, HermiteHighDim4D) {
    // 4D Hermite interpolation
    janus::NumericVector pts(3);
    pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points(4, pts);

    // f = sum of coordinates
    janus::NumericVector values(81); // 3^4
    int idx = 0;
    for (int i4 = 0; i4 < 3; ++i4) {
        for (int i3 = 0; i3 < 3; ++i3) {
            for (int i2 = 0; i2 < 3; ++i2) {
                for (int i1 = 0; i1 < 3; ++i1) {
                    values(idx++) = pts(i1) + pts(i2) + pts(i3) + pts(i4);
                }
            }
        }
    }

    // Query at center
    janus::NumericMatrix xi(1, 4);
    xi << 1.0, 1.0, 1.0, 1.0;

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    EXPECT_NEAR(result(0), 4.0, 1e-10);
}

TEST(InterpnTests, HermiteEdgesAndCorners) {
    // Hermite at grid edges and corners
    janus::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    janus::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points = {x_pts, y_pts};

    // Constant function (should interpolate exactly)
    janus::NumericVector values(9);
    values.setConstant(5.0);

    // Query at edge and corner
    janus::NumericMatrix xi(4, 2);
    xi << 0.0, 0.0, // corner
        2.0, 2.0,   // corner
        0.0, 1.0,   // edge
        1.0, 0.0;   // edge

    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result(i), 5.0, 1e-10) << "Failed at query " << i;
    }
}

TEST(InterpnTests, HermiteFillValue) {
    // Hermite with fill_value for out-of-bounds
    janus::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points = {x_pts};

    janus::NumericVector values(3);
    values << 1.0, 2.0, 3.0;

    // In-bounds and out-of-bounds queries
    janus::NumericMatrix xi(3, 1);
    xi << 0.5, -1.0, 3.0;

    double fill = -999.0;
    auto result = janus::interpn<double>(points, values, xi, janus::InterpolationMethod::Hermite,
                                         std::optional<double>(fill));

    EXPECT_GT(result(0), 1.0); // In bounds
    EXPECT_LT(result(0), 2.0);
    EXPECT_NEAR(result(1), fill, 1e-10); // Out of bounds
    EXPECT_NEAR(result(2), fill, 1e-10); // Out of bounds
}

TEST(InterpnTests, HermiteSymbolicNotSupported) {
    // Hermite should throw for symbolic types
    janus::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;

    std::vector<janus::NumericVector> points = {x_pts};
    janus::NumericVector values(3);
    values << 1.0, 2.0, 3.0;

    janus::SymbolicMatrix xi(1, 1);
    xi(0, 0) = casadi::MX(0.5);

    EXPECT_THROW(janus::interpn<janus::SymbolicScalar>(points, values, xi,
                                                       janus::InterpolationMethod::Hermite),
                 janus::InterpolationError);
}
