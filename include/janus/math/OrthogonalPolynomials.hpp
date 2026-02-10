#pragma once

#include "janus/core/JanusTypes.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace janus {

/**
 * @brief Evaluate Legendre polynomial P_n(x) and derivative P'_n(x)
 */
inline std::pair<double, double> legendre_poly(int n, double x) {
    if (n < 0) {
        throw std::invalid_argument("legendre_poly: n must be >= 0");
    }

    if (n == 0) {
        return {1.0, 0.0};
    }
    if (n == 1) {
        return {x, 1.0};
    }

    double p_nm1 = 1.0;
    double p_n = x;
    for (int k = 1; k < n; ++k) {
        const double p_np1 =
            ((2.0 * static_cast<double>(k) + 1.0) * x * p_n - static_cast<double>(k) * p_nm1) /
            (static_cast<double>(k) + 1.0);
        p_nm1 = p_n;
        p_n = p_np1;
    }

    const double nn = static_cast<double>(n);
    const double endpoint_tol = 64.0 * std::numeric_limits<double>::epsilon();
    const double one_minus_abs_x = std::abs(1.0 - std::abs(x));

    double dp_n = 0.0;
    if (one_minus_abs_x < endpoint_tol) {
        const double endpoint = 0.5 * nn * (nn + 1.0);
        if (x >= 0.0) {
            dp_n = endpoint;
        } else {
            const double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0; // (-1)^(n+1)
            dp_n = sign * endpoint;
        }
    } else {
        dp_n = nn * (x * p_n - p_nm1) / (x * x - 1.0);
    }

    return {p_n, dp_n};
}

/**
 * @brief Evaluate P_n(x) at each x in vector
 */
inline NumericVector legendre_poly_vec(int n, const NumericVector &x) {
    NumericVector out(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        out(i) = legendre_poly(n, x(i)).first;
    }
    return out;
}

/**
 * @brief Compute Legendre-Gauss-Lobatto nodes on [-1, 1]
 */
inline NumericVector lgl_nodes(int N) {
    if (N < 2) {
        throw std::invalid_argument("lgl_nodes: N must be >= 2");
    }

    NumericVector nodes(N);
    nodes(0) = -1.0;
    nodes(N - 1) = 1.0;
    if (N == 2) {
        return nodes;
    }

    const int n = N - 1;
    const double pi = std::acos(-1.0);
    const double tol = 32.0 * std::numeric_limits<double>::epsilon();

    for (int j = 1; j < N - 1; ++j) {
        double x = -std::cos(pi * static_cast<double>(j) / static_cast<double>(N - 1));

        for (int iter = 0; iter < 100; ++iter) {
            const auto [pn, dpn] = legendre_poly(n, x);
            const double denom = 1.0 - x * x;
            const double ddpn = (2.0 * x * dpn - static_cast<double>(n) * (n + 1) * pn) / denom;

            const double delta = dpn / ddpn;
            x -= delta;

            if (std::abs(delta) < tol) {
                break;
            }
        }

        nodes(j) = x;
    }

    return nodes;
}

/**
 * @brief Compute Chebyshev-Gauss-Lobatto nodes on [-1, 1]
 */
inline NumericVector cgl_nodes(int N) {
    if (N < 2) {
        throw std::invalid_argument("cgl_nodes: N must be >= 2");
    }

    NumericVector nodes(N);
    const double pi = std::acos(-1.0);
    for (int j = 0; j < N; ++j) {
        nodes(j) = -std::cos(pi * static_cast<double>(j) / static_cast<double>(N - 1));
    }
    return nodes;
}

/**
 * @brief LGL quadrature weights
 */
inline NumericVector lgl_weights(int N, const NumericVector &nodes) {
    if (N < 2) {
        throw std::invalid_argument("lgl_weights: N must be >= 2");
    }
    if (nodes.size() != N) {
        throw std::invalid_argument("lgl_weights: nodes size mismatch");
    }

    NumericVector w(N);
    const double scale = 2.0 / (static_cast<double>(N) * static_cast<double>(N - 1));
    const int n = N - 1;
    for (int i = 0; i < N; ++i) {
        const double pn = legendre_poly(n, nodes(i)).first;
        w(i) = scale / (pn * pn);
    }
    return w;
}

/**
 * @brief CGL (Clenshaw-Curtis) quadrature weights on [-1, 1]
 */
inline NumericVector cgl_weights(int N, const NumericVector &nodes) {
    if (N < 2) {
        throw std::invalid_argument("cgl_weights: N must be >= 2");
    }
    if (nodes.size() != N) {
        throw std::invalid_argument("cgl_weights: nodes size mismatch");
    }

    if (N == 2) {
        NumericVector w(2);
        w(0) = 1.0;
        w(1) = 1.0;
        return w;
    }

    const int n = N - 1;
    const double pi = std::acos(-1.0);
    NumericVector theta(N);
    for (int j = 0; j < N; ++j) {
        theta(j) = pi * static_cast<double>(j) / static_cast<double>(n);
    }

    NumericVector w = NumericVector::Zero(N);
    NumericVector v = NumericVector::Ones(N - 2);

    if (n % 2 == 0) {
        const double w0 = 1.0 / (static_cast<double>(n) * static_cast<double>(n) - 1.0);
        w(0) = w0;
        w(N - 1) = w0;

        for (int k = 1; k < n / 2; ++k) {
            const double denom = 4.0 * static_cast<double>(k) * static_cast<double>(k) - 1.0;
            for (int j = 1; j < N - 1; ++j) {
                v(j - 1) -= 2.0 * std::cos(2.0 * static_cast<double>(k) * theta(j)) / denom;
            }
        }

        const double denom = static_cast<double>(n) * static_cast<double>(n) - 1.0;
        for (int j = 1; j < N - 1; ++j) {
            v(j - 1) -= std::cos(static_cast<double>(n) * theta(j)) / denom;
        }
    } else {
        const double w0 = 1.0 / (static_cast<double>(n) * static_cast<double>(n));
        w(0) = w0;
        w(N - 1) = w0;

        for (int k = 1; k <= (n - 1) / 2; ++k) {
            const double denom = 4.0 * static_cast<double>(k) * static_cast<double>(k) - 1.0;
            for (int j = 1; j < N - 1; ++j) {
                v(j - 1) -= 2.0 * std::cos(2.0 * static_cast<double>(k) * theta(j)) / denom;
            }
        }
    }

    for (int j = 1; j < N - 1; ++j) {
        w(j) = 2.0 * v(j - 1) / static_cast<double>(n);
    }

    return w;
}

/**
 * @brief Spectral differentiation matrix using barycentric weights
 */
inline NumericMatrix spectral_diff_matrix(const NumericVector &nodes) {
    const int N = static_cast<int>(nodes.size());
    if (N < 2) {
        throw std::invalid_argument("spectral_diff_matrix: need at least 2 nodes");
    }

    NumericVector lambda(N);
    for (int j = 0; j < N; ++j) {
        double prod = 1.0;
        for (int k = 0; k < N; ++k) {
            if (k == j) {
                continue;
            }
            prod *= (nodes(j) - nodes(k));
        }
        lambda(j) = 1.0 / prod;
    }

    NumericMatrix D = NumericMatrix::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            D(i, j) = (lambda(j) / lambda(i)) / (nodes(i) - nodes(j));
        }
    }

    for (int i = 0; i < N; ++i) {
        D(i, i) = -D.row(i).sum();
    }

    return D;
}

} // namespace janus
