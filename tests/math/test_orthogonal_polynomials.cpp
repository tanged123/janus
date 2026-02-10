/**
 * @file test_orthogonal_polynomials.cpp
 * @brief Tests for Legendre/Chebyshev pseudospectral infrastructure
 */

#include <gtest/gtest.h>
#include <janus/janus.hpp>

#include <cmath>

using namespace janus;

namespace {

double exact_monomial_integral_on_minus1_1(int k) {
    if (k % 2 == 1) {
        return 0.0;
    }
    return 2.0 / static_cast<double>(k + 1);
}

} // namespace

TEST(OrthogonalPolynomialsTests, LegendreKnownValuesAndDerivatives) {
    const double x = 0.3;

    EXPECT_NEAR(legendre_poly(0, x).first, 1.0, 1e-14);
    EXPECT_NEAR(legendre_poly(1, x).first, x, 1e-14);
    EXPECT_NEAR(legendre_poly(2, x).first, 0.5 * (3.0 * x * x - 1.0), 1e-14);
    EXPECT_NEAR(legendre_poly(3, x).first, 0.5 * (5.0 * x * x * x - 3.0 * x), 1e-14);
    EXPECT_NEAR(legendre_poly(4, x).first, (35.0 * std::pow(x, 4) - 30.0 * x * x + 3.0) / 8.0,
                1e-14);
    EXPECT_NEAR(legendre_poly(5, x).first,
                (63.0 * std::pow(x, 5) - 70.0 * std::pow(x, 3) + 15.0 * x) / 8.0, 1e-14);

    EXPECT_NEAR(legendre_poly(6, 1.0).second, 21.0, 1e-12);
    EXPECT_NEAR(legendre_poly(6, -1.0).second, -21.0, 1e-12);
    EXPECT_NEAR(legendre_poly(5, -1.0).second, 15.0, 1e-12);
}

TEST(OrthogonalPolynomialsTests, LGLNodesBasicProperties) {
    auto n2 = lgl_nodes(2);
    ASSERT_EQ(n2.size(), 2);
    EXPECT_DOUBLE_EQ(n2(0), -1.0);
    EXPECT_DOUBLE_EQ(n2(1), 1.0);

    auto n5 = lgl_nodes(5);
    ASSERT_EQ(n5.size(), 5);
    EXPECT_NEAR(n5(0), -1.0, 1e-14);
    EXPECT_NEAR(n5(1), -std::sqrt(3.0 / 7.0), 1e-12);
    EXPECT_NEAR(n5(2), 0.0, 1e-12);
    EXPECT_NEAR(n5(3), std::sqrt(3.0 / 7.0), 1e-12);
    EXPECT_NEAR(n5(4), 1.0, 1e-14);

    auto n10 = lgl_nodes(10);
    const NumericVector ref10 = NumericVector{
        {-1.0, -0.9195339081664589, -0.7387738651055051, -0.4779249498104445, -0.1652789576663870,
         0.1652789576663870, 0.4779249498104445, 0.7387738651055051, 0.9195339081664589, 1.0}};
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(n10(i), ref10(i), 1e-12);
    }

    // Interior nodes are roots of P'_{N-1}
    for (int i = 1; i < 9; ++i) {
        EXPECT_NEAR(legendre_poly(9, n10(i)).second, 0.0, 1e-12);
    }
}

TEST(OrthogonalPolynomialsTests, CGLNodesMatchClosedForm) {
    const int N = 7;
    auto nodes = cgl_nodes(N);
    for (int j = 0; j < N; ++j) {
        double expected =
            -std::cos(std::acos(-1.0) * static_cast<double>(j) / static_cast<double>(N - 1));
        EXPECT_NEAR(nodes(j), expected, 1e-14);
    }
    EXPECT_NEAR(nodes(0), -1.0, 1e-14);
    EXPECT_NEAR(nodes(N - 1), 1.0, 1e-14);
}

TEST(OrthogonalPolynomialsTests, LGLWeightsIntegratePolynomialsExactly) {
    const int N = 8;
    auto nodes = lgl_nodes(N);
    auto w = lgl_weights(N, nodes);

    EXPECT_NEAR(w.sum(), 2.0, 1e-13);

    for (int k = 0; k <= 2 * N - 3; ++k) {
        double approx = 0.0;
        for (int i = 0; i < N; ++i) {
            approx += w(i) * std::pow(nodes(i), k);
        }
        EXPECT_NEAR(approx, exact_monomial_integral_on_minus1_1(k), 2e-11);
    }
}

TEST(OrthogonalPolynomialsTests, CGLWeightsIntegratePolynomialsToExpectedOrder) {
    const int N = 8;
    auto nodes = cgl_nodes(N);
    auto w = cgl_weights(N, nodes);

    EXPECT_NEAR(w.sum(), 2.0, 1e-13);

    for (int k = 0; k <= N - 1; ++k) {
        double approx = 0.0;
        for (int i = 0; i < N; ++i) {
            approx += w(i) * std::pow(nodes(i), k);
        }
        EXPECT_NEAR(approx, exact_monomial_integral_on_minus1_1(k), 2e-10);
    }
}

TEST(OrthogonalPolynomialsTests, SpectralDiffMatrixDifferentiatesAsExpected) {
    const int N = 18;

    for (const auto &nodes : {lgl_nodes(N), cgl_nodes(N)}) {
        const auto D = spectral_diff_matrix(nodes);

        NumericVector ones = NumericVector::Ones(N);
        NumericVector tau = nodes;

        NumericVector d_const = D * ones;
        NumericVector d_tau = D * tau;

        EXPECT_LT(d_const.cwiseAbs().maxCoeff(), 1e-11);
        EXPECT_LT((d_tau - NumericVector::Ones(N)).cwiseAbs().maxCoeff(), 1e-11);

        NumericVector f(N), df_exact(N);
        for (int i = 0; i < N; ++i) {
            f(i) = std::sin(nodes(i));
            df_exact(i) = std::cos(nodes(i));
        }
        NumericVector df_approx = D * f;
        EXPECT_LT((df_approx - df_exact).cwiseAbs().maxCoeff(), 1e-8);
    }
}
