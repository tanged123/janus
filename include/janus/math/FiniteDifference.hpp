#pragma once

#include "../core/JanusError.hpp"
#include "../core/JanusTypes.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace janus {

/**
 * @brief Computes finite difference coefficients for arbitrary grids
 *
 * Based on Fornberg 1988: "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids"
 *
 * @param x Grid points (vector)
 * @param x0 Evaluation point
 * @param derivative_degree Order of derivative to approximate
 * @return Vector of coefficients matching x size
 */
template <typename Derived>
auto finite_difference_coefficients(const Eigen::MatrixBase<Derived> &x,
                                    typename Derived::Scalar x0 = typename Derived::Scalar(0.0),
                                    int derivative_degree = 1) {
    using Scalar = typename Derived::Scalar;

    if (derivative_degree < 0) {
        throw InvalidArgument("finite_difference_coefficients: derivative_degree must be >= 0");
    }

    int n_points = x.size();
    if (n_points < derivative_degree + 1) {
        throw InvalidArgument(
            "finite_difference_coefficients: need at least (derivative_degree + 1) grid points");
    }

    int N = n_points - 1;
    int M = derivative_degree;

    // 3D array delta[derivative_degree + 1][N + 1][N + 1]
    // We can use a flattened std::vector
    // Indexing: delta(m, n, v) -> idx
    auto get_idx = [&](int m, int n, int v) { return m * (N + 1) * (N + 1) + n * (N + 1) + v; };

    std::vector<Scalar> delta((M + 1) * (N + 1) * (N + 1), Scalar(0.0));

    // delta[0, 0, 0] = 1
    delta[get_idx(0, 0, 0)] = Scalar(1.0);

    Scalar c1 = Scalar(1.0);

    for (int n = 1; n <= N; ++n) {
        Scalar c2 = Scalar(1.0);
        for (int v = 0; v < n; ++v) {
            Scalar c3 = x(n) - x(v);
            c2 = c2 * c3;

            for (int m = 0; m <= std::min(n, M); ++m) {
                // delta[m, n, v] = ((x[n] - x0) * delta[m, n - 1, v] - m * delta[m - 1, n - 1, v])
                // / c3
                Scalar term1 = (x(n) - x0) * delta[get_idx(m, n - 1, v)];
                Scalar term2 = Scalar(0.0);
                if (m > 0) {
                    term2 = static_cast<Scalar>(m) * delta[get_idx(m - 1, n - 1, v)];
                }
                delta[get_idx(m, n, v)] = (term1 - term2) / c3;
            }
        }

        for (int m = 0; m <= std::min(n, M); ++m) {
            // delta[m, n, n] = c1 / c2 * (m * delta[m - 1, n - 1, n - 1] - (x[n - 1] - x0) *
            // delta[m, n - 1, n - 1])
            Scalar term1 = Scalar(0.0);
            if (m > 0) {
                term1 = static_cast<Scalar>(m) * delta[get_idx(m - 1, n - 1, n - 1)];
            }
            Scalar term2 = (x(n - 1) - x0) * delta[get_idx(m, n - 1, n - 1)];

            delta[get_idx(m, n, n)] = (c1 / c2) * (term1 - term2);
        }
        c1 = c2;
    }

    // Extract result: delta[M, N, :]
    // The last row of the last matrix for the derivative degree M
    JanusVector<Scalar> coeffs(n_points);
    for (int i = 0; i < n_points; ++i) {
        coeffs(i) = delta[get_idx(M, N, i)];
    }

    return coeffs;
}

} // namespace janus
