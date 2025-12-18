#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Linalg.hpp"
#include <algorithm>
#include <casadi/casadi.hpp>
#include <cmath>
#include <optional>
#include <vector>

namespace janus {

// ============================================================================
// Interpolation Method Enum
// ============================================================================

/**
 * @brief Supported interpolation methods
 *
 * Controls the interpolation algorithm used for both 1D and N-D interpolation.
 */
enum class InterpolationMethod {
    Linear,  ///< Piecewise linear (C0 continuous) - fast, simple
    Hermite, ///< Cubic Hermite/Catmull-Rom (C1 continuous) - smooth gradients
    BSpline, ///< Cubic B-spline (C2 continuous) - smoothest, good for optimization
    Nearest  ///< Nearest neighbor - fast, non-differentiable
};

// ============================================================================
// Implementation Details
// ============================================================================

namespace detail {

/**
 * @brief Convert InterpolationMethod enum to CasADi string
 */
inline std::string method_to_casadi_string(InterpolationMethod method) {
    switch (method) {
    case InterpolationMethod::Linear:
        return "linear";
    case InterpolationMethod::BSpline:
        return "bspline";
    case InterpolationMethod::Hermite:
    case InterpolationMethod::Nearest:
        // These don't have native CasADi support, handled separately
        return "linear"; // Fallback for internal use
    default:
        return "linear";
    }
}

/**
 * @brief Flatten N-D values array in Fortran order for CasADi
 *
 * CasADi expects values flattened in column-major (Fortran) order.
 * For a 2D array with shape (m, n), this means iterating over
 * the first dimension fastest.
 */
template <typename Derived>
inline NumericVector flatten_fortran_order(const Eigen::MatrixBase<Derived> &values) {
    // For 2D matrices, transpose then flatten (equivalent to Fortran order)
    NumericMatrix transposed = values.transpose();
    return Eigen::Map<const NumericVector>(transposed.data(), transposed.size());
}

// ============================================================================
// Catmull-Rom (C1 Hermite) Interpolation Helpers
// ============================================================================

/**
 * @brief Compute Catmull-Rom slope at a point using neighboring values
 *
 * For interior points: m = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
 * For endpoints: use one-sided differences
 *
 * @param x Grid x-coordinates
 * @param y Grid y-values
 * @param i Index of point
 * @return Slope at point i
 */
inline double catmull_rom_slope(const std::vector<double> &x, const std::vector<double> &y,
                                size_t i) {
    size_t n = x.size();
    if (n < 2)
        return 0.0;

    if (i == 0) {
        // Left endpoint: forward difference
        return (y[1] - y[0]) / (x[1] - x[0]);
    } else if (i == n - 1) {
        // Right endpoint: backward difference
        return (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    } else {
        // Interior: central difference using neighbors
        return (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }
}

/**
 * @brief Evaluate 1D Catmull-Rom cubic Hermite spline (numeric only)
 *
 * Uses cubic Hermite basis functions:
 * h00(t) = 2t³ - 3t² + 1
 * h10(t) = t³ - 2t² + t
 * h01(t) = -2t³ + 3t²
 * h11(t) = t³ - t²
 *
 * p(t) = h00(t)*p0 + h10(t)*m0*h + h01(t)*p1 + h11(t)*m1*h
 * where h = x1 - x0, t = (x - x0) / h
 *
 * @param x Grid x-coordinates (sorted)
 * @param y Grid y-values
 * @param query Query point (must be numeric)
 * @return Interpolated value
 */
inline double hermite_interp_1d_numeric(const std::vector<double> &x, const std::vector<double> &y,
                                        double query) {
    size_t n = x.size();
    if (n < 2) {
        return y[0];
    }

    // Clamp query to grid bounds
    query = std::max(query, x.front());
    query = std::min(query, x.back());

    // Find interval containing query using binary search
    auto it = std::upper_bound(x.begin(), x.end(), query);
    size_t idx = 0;
    if (it == x.begin()) {
        idx = 0;
    } else if (it == x.end()) {
        idx = n - 2;
    } else {
        idx = static_cast<size_t>(std::distance(x.begin(), it)) - 1;
    }
    if (idx >= n - 1)
        idx = n - 2; // Safety clamp

    // Get interval endpoints
    double x0 = x[idx];
    double x1 = x[idx + 1];
    double y0 = y[idx];
    double y1 = y[idx + 1];
    double h = x1 - x0;

    // Compute slopes at endpoints (Catmull-Rom)
    double m0 = catmull_rom_slope(x, y, idx);
    double m1 = catmull_rom_slope(x, y, idx + 1);

    // Normalized coordinate
    double t = (query - x0) / h;
    t = std::max(0.0, std::min(1.0, t)); // Clamp to [0, 1]

    double t2 = t * t;
    double t3 = t2 * t;

    // Hermite basis functions
    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;

    // Interpolated value
    return h00 * y0 + h10 * (m0 * h) + h01 * y1 + h11 * (m1 * h);
}

/**
 * @brief Compute multi-dimensional index from flat index (Fortran order)
 */
inline std::vector<int> flat_to_multi_index(int flat_idx, const std::vector<int> &dims) {
    std::vector<int> multi(dims.size());
    int remaining = flat_idx;
    for (size_t d = 0; d < dims.size(); ++d) {
        multi[d] = remaining % dims[d];
        remaining /= dims[d];
    }
    return multi;
}

/**
 * @brief Compute flat index from multi-dimensional index (Fortran order)
 */
inline int multi_to_flat_index(const std::vector<int> &multi, const std::vector<int> &dims) {
    int flat = 0;
    int stride = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        flat += multi[d] * stride;
        stride *= dims[d];
    }
    return flat;
}

/**
 * @brief N-dimensional Catmull-Rom interpolation via tensor product (numeric only)
 *
 * Interpolates in each dimension sequentially using 1D Hermite splines.
 * This function is numeric-only because interval finding requires comparisons.
 */
inline double hermite_interpn_numeric(const std::vector<std::vector<double>> &grid,
                                      const std::vector<double> &values,
                                      const std::vector<double> &query) {
    int n_dims = static_cast<int>(grid.size());

    // Build dimension sizes
    std::vector<int> dims(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        dims[d] = static_cast<int>(grid[d].size());
    }

    // For N-D interpolation, we use recursive tensor product:
    // 1. Interpolate along first dimension to reduce to (n-1)-D
    // 2. Repeat until we have a scalar

    // Start with the full values array
    std::vector<double> current_values = values;
    std::vector<int> current_dims = dims;

    for (int d = 0; d < n_dims; ++d) {
        const std::vector<double> &x = grid[d];
        int n_x = current_dims[0]; // Size in first (current) dimension

        // Compute size of remaining dimensions
        int remaining_size = 1;
        for (size_t dd = 1; dd < current_dims.size(); ++dd) {
            remaining_size *= current_dims[dd];
        }

        // Interpolate along first dimension for each slice
        std::vector<double> new_values(remaining_size);

        for (int slice = 0; slice < remaining_size; ++slice) {
            // Extract 1D slice along first dimension
            std::vector<double> y_slice(n_x);
            for (int i = 0; i < n_x; ++i) {
                int flat_idx = i + slice * n_x;
                y_slice[i] = current_values[flat_idx];
            }

            // Interpolate this slice
            new_values[slice] = hermite_interp_1d_numeric(x, y_slice, query[d]);
        }

        // Update for next dimension
        current_values = std::move(new_values);
        current_dims.erase(current_dims.begin());
    }

    return current_values[0];
}

} // namespace detail

// ============================================================================
// Unified Interpolator Class
// ============================================================================

/**
 * @brief Unified N-dimensional interpolator
 *
 * A single class that handles interpolation in any number of dimensions.
 * The 1D case is simply N=1. Supports both numeric (double) and symbolic
 * (casadi::MX) evaluation via the same interface.
 *
 * For numeric queries, all methods (Linear, Hermite, BSpline, Nearest) are supported.
 * For symbolic queries, only Linear and BSpline are supported (Hermite and Nearest
 * require runtime interval finding which is incompatible with symbolic graphs).
 *
 * @example 1D usage:
 * ```cpp
 * NumericVector x(4), y(4);
 * x << 0, 1, 2, 3;
 * y << 0, 1, 4, 9;  // y = x^2
 *
 * janus::Interpolator interp(x, y, janus::InterpolationMethod::Linear);
 *
 * double result = interp(1.5);  // Query at single point
 * NumericVector batch = interp(query_vec);  // Batch query
 * ```
 *
 * @example N-D usage:
 * ```cpp
 * std::vector<NumericVector> grid = {x_pts, y_pts};
 * NumericVector values(4);  // Fortran order
 * values << 0, 1, 1, 2;     // z(x,y) = x + y
 *
 * janus::Interpolator interp(grid, values);
 *
 * NumericVector query(2);
 * query << 0.5, 0.5;
 * double result = interp(query);  // Query at (0.5, 0.5)
 * ```
 */
class Interpolator {
  private:
    std::vector<std::vector<double>> m_grid;
    std::vector<double> m_values;
    casadi::Function m_casadi_fn;
    InterpolationMethod m_method = InterpolationMethod::Linear;
    int m_dims = 0;
    bool m_valid = false;
    bool m_use_custom_hermite = false;

  public:
    /**
     * @brief Default constructor (invalid state)
     */
    Interpolator() = default;

    /**
     * @brief Construct N-dimensional interpolator
     *
     * @param points Vector of 1D coordinate arrays for each dimension
     * @param values Flattened values in Fortran order (column-major)
     * @param method Interpolation method (default: Linear)
     * @throw InterpolationError if inputs are invalid
     */
    Interpolator(const std::vector<NumericVector> &points, const NumericVector &values,
                 InterpolationMethod method = InterpolationMethod::Linear) {
        if (points.empty()) {
            throw InterpolationError("Interpolator: points cannot be empty");
        }

        m_dims = static_cast<int>(points.size());
        m_method = method;

        // Convert to std::vector for internal storage
        m_grid.resize(m_dims);
        int expected_size = 1;
        for (int d = 0; d < m_dims; ++d) {
            if (points[d].size() < 2) {
                throw InterpolationError("Interpolator: need at least 2 grid points in dimension " +
                                         std::to_string(d));
            }
            m_grid[d].resize(points[d].size());
            Eigen::VectorXd::Map(m_grid[d].data(), points[d].size()) = points[d];

            if (!std::is_sorted(m_grid[d].begin(), m_grid[d].end())) {
                throw InterpolationError("Interpolator: points[" + std::to_string(d) +
                                         "] must be sorted");
            }
            expected_size *= static_cast<int>(points[d].size());
        }

        // Validate values size
        if (values.size() != expected_size) {
            throw InterpolationError("Interpolator: values size mismatch. Expected " +
                                     std::to_string(expected_size) + ", got " +
                                     std::to_string(values.size()));
        }

        m_values.resize(values.size());
        Eigen::VectorXd::Map(m_values.data(), values.size()) = values;

        // Method-specific setup
        setup_method(method);
    }

    /**
     * @brief Construct 1D interpolator (convenience overload)
     *
     * This is syntactic sugar for the N-D constructor with N=1.
     *
     * @param x Grid points (must be sorted)
     * @param y Function values at grid points
     * @param method Interpolation method (default: Linear)
     * @throw InterpolationError if inputs are invalid
     */
    Interpolator(const NumericVector &x, const NumericVector &y,
                 InterpolationMethod method = InterpolationMethod::Linear) {
        if (x.size() != y.size()) {
            throw InterpolationError("Interpolator: x and y must have same size");
        }
        if (x.size() < 2) {
            throw InterpolationError("Interpolator: need at least 2 grid points");
        }

        m_dims = 1;
        m_method = method;

        // Store grid and values
        m_grid.resize(1);
        m_grid[0].resize(x.size());
        Eigen::VectorXd::Map(m_grid[0].data(), x.size()) = x;

        if (!std::is_sorted(m_grid[0].begin(), m_grid[0].end())) {
            throw InterpolationError("Interpolator: x grid must be sorted");
        }

        m_values.resize(y.size());
        Eigen::VectorXd::Map(m_values.data(), y.size()) = y;

        // Method-specific setup
        setup_method(method);
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get number of dimensions
     */
    int dims() const { return m_dims; }

    /**
     * @brief Get the interpolation method
     */
    InterpolationMethod method() const { return m_method; }

    /**
     * @brief Check if interpolator is valid (initialized)
     */
    bool valid() const { return m_valid; }

    // ========================================================================
    // Scalar Query (1D only)
    // ========================================================================

    /**
     * @brief Evaluate interpolant at a scalar point (1D only)
     * @tparam Scalar Scalar type (double or SymbolicScalar)
     * @param query Query point
     * @return Interpolated value
     * @throw InterpolationError if not 1D or not initialized
     */
    template <JanusScalar Scalar> Scalar operator()(const Scalar &query) const {
        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");
        if (m_dims != 1)
            throw InterpolationError("Interpolator: scalar query only valid for 1D");

        if constexpr (std::is_floating_point_v<Scalar>) {
            return eval_numeric_scalar(query);
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(
                    "Interpolator: Hermite method not supported for symbolic types");
            }
            return eval_symbolic_scalar(query);
        }
    }

    // ========================================================================
    // N-D Point Query
    // ========================================================================

    /**
     * @brief Evaluate N-D interpolant at a single point
     *
     * For N≥2 interpolators, pass a vector of coordinates.
     * For 1D, use scalar overload or batch query.
     *
     * @tparam Scalar Scalar type (double or SymbolicScalar)
     * @param query Query point (size must match dims())
     * @return Interpolated scalar value
     */
    template <JanusScalar Scalar> Scalar operator()(const JanusVector<Scalar> &query) const {
        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");

        if (query.size() != m_dims) {
            throw InterpolationError("Interpolator: query size must match dims. Expected " +
                                     std::to_string(m_dims) + ", got " +
                                     std::to_string(query.size()));
        }

        if constexpr (std::is_floating_point_v<Scalar>) {
            return eval_numeric_point(query);
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(
                    "Interpolator: Hermite method not supported for symbolic types");
            }
            return eval_symbolic_point(query);
        }
    }

    // ========================================================================
    // Batch Query (Multiple Points)
    // ========================================================================

    /**
     * @brief Evaluate interpolant at multiple points
     *
     * For 1D interpolators, pass a vector of query points.
     * For N-D interpolators, pass (n_points, n_dims) matrix.
     *
     * @tparam Derived Eigen expression type
     * @param queries Query points (vector for 1D, matrix for N-D)
     * @return Vector of interpolated values
     */
    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived> &queries) const
        -> JanusVector<typename Derived::Scalar> {
        using Scalar = typename Derived::Scalar;

        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");

        int n_points;
        if (m_dims == 1) {
            // 1D: flatten input and treat each element as a query point
            n_points = static_cast<int>(queries.size());
        } else {
            // N-D: determine shape
            bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
            n_points =
                is_transposed ? static_cast<int>(queries.cols()) : static_cast<int>(queries.rows());
        }

        JanusVector<Scalar> result(n_points);

        if constexpr (std::is_floating_point_v<Scalar>) {
            if (m_dims == 1) {
                // 1D: iterate over all elements using coefficient access
                for (int i = 0; i < n_points; ++i) {
                    // Use linear coefficient access (works for any Eigen expression)
                    Scalar val;
                    if (queries.cols() == 1) {
                        val = queries(i, 0);
                    } else {
                        val = queries(0, i);
                    }
                    result(i) = eval_numeric_scalar(val);
                }
            } else {
                // N-D: handle row/col orientation
                bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
                for (int i = 0; i < n_points; ++i) {
                    NumericVector point(m_dims);
                    for (int d = 0; d < m_dims; ++d) {
                        point(d) = is_transposed ? queries(d, i) : queries(i, d);
                    }
                    result(i) = eval_numeric_point(point);
                }
            }
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(
                    "Interpolator: Hermite method not supported for symbolic types");
            }
            if (m_dims == 1) {
                for (int i = 0; i < n_points; ++i) {
                    Scalar val;
                    if (queries.cols() == 1) {
                        val = queries(i, 0);
                    } else {
                        val = queries(0, i);
                    }
                    result(i) = eval_symbolic_scalar(val);
                }
            } else {
                bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
                for (int i = 0; i < n_points; ++i) {
                    SymbolicVector point(m_dims);
                    for (int d = 0; d < m_dims; ++d) {
                        point(d) = is_transposed ? queries(d, i) : queries(i, d);
                    }
                    result(i) = eval_symbolic_point(point);
                }
            }
        }

        return result;
    }

  private:
    // ========================================================================
    // Setup
    // ========================================================================

    void setup_method(InterpolationMethod method) {
        if (method == InterpolationMethod::Hermite) {
            // Hermite uses custom implementation, no CasADi interpolant needed
            m_use_custom_hermite = true;
            m_valid = true;
        } else if (method == InterpolationMethod::BSpline) {
            // BSpline requires at least 4 points per dimension for cubic
            for (int d = 0; d < m_dims; ++d) {
                if (m_grid[d].size() < 4) {
                    throw InterpolationError(
                        "Interpolator: BSpline requires at least 4 grid points in dimension " +
                        std::to_string(d));
                }
            }
            m_casadi_fn = casadi::interpolant("interp", "bspline", m_grid, m_values);
            m_valid = true;
        } else if (method == InterpolationMethod::Nearest) {
            // Nearest neighbor - use linear CasADi but round in eval
            m_casadi_fn = casadi::interpolant("interp", "linear", m_grid, m_values);
            m_valid = true;
        } else {
            // Linear (default)
            m_casadi_fn = casadi::interpolant("interp", "linear", m_grid, m_values);
            m_valid = true;
        }
    }

    // ========================================================================
    // Numeric Evaluation
    // ========================================================================

    double eval_numeric_scalar(double query) const {
        // Clamp to bounds
        double clamped = std::max(query, m_grid[0].front());
        clamped = std::min(clamped, m_grid[0].back());

        if (m_use_custom_hermite) {
            return detail::hermite_interp_1d_numeric(m_grid[0], m_values, clamped);
        }

        if (m_method == InterpolationMethod::Nearest) {
            // Snap to nearest grid point
            auto it = std::lower_bound(m_grid[0].begin(), m_grid[0].end(), clamped);
            size_t idx = std::distance(m_grid[0].begin(), it);
            if (idx > 0 && (idx == m_grid[0].size() || std::abs(clamped - m_grid[0][idx - 1]) <
                                                           std::abs(clamped - m_grid[0][idx]))) {
                idx--;
            }
            return m_values[idx];
        }

        // Linear/BSpline: Use CasADi
        std::vector<casadi::DM> args = {casadi::DM(clamped)};
        std::vector<casadi::DM> res = m_casadi_fn(args);
        return static_cast<double>(res[0]);
    }

    double eval_numeric_point(const NumericVector &query) const {
        // Build clamped query
        std::vector<double> clamped(m_dims);
        for (int d = 0; d < m_dims; ++d) {
            double val = query(d);
            val = std::max(val, m_grid[d].front());
            val = std::min(val, m_grid[d].back());
            clamped[d] = val;
        }

        if (m_use_custom_hermite) {
            return detail::hermite_interpn_numeric(m_grid, m_values, clamped);
        }

        if (m_method == InterpolationMethod::Nearest) {
            // TODO: Implement N-D nearest neighbor
            // For now, fall through to linear
        }

        // Linear/BSpline: Use CasADi
        std::vector<casadi::DM> args = {casadi::DM(clamped)};
        std::vector<casadi::DM> res = m_casadi_fn(args);
        return static_cast<double>(res[0]);
    }

    // ========================================================================
    // Symbolic Evaluation
    // ========================================================================

    SymbolicScalar eval_symbolic_scalar(const SymbolicScalar &query) const {
        // Clamp to bounds
        SymbolicScalar clamped = SymbolicScalar::fmax(query, m_grid[0].front());
        clamped = SymbolicScalar::fmin(clamped, m_grid[0].back());

        std::vector<casadi::MX> args = {clamped};
        std::vector<casadi::MX> res = m_casadi_fn(args);
        return res[0];
    }

    SymbolicScalar eval_symbolic_point(const SymbolicVector &query) const {
        // Build clamped query
        casadi::MX clamped(m_dims, 1);
        for (int d = 0; d < m_dims; ++d) {
            casadi::MX val = query(d);
            val = casadi::MX::fmax(val, m_grid[d].front());
            val = casadi::MX::fmin(val, m_grid[d].back());
            clamped(d) = val;
        }

        std::vector<casadi::MX> args = {clamped};
        std::vector<casadi::MX> res = m_casadi_fn(args);
        return res[0];
    }
};

// ============================================================================
// Backwards Compatibility
// ============================================================================

/**
 * @brief Alias for 1D interpolation (backwards compatibility)
 * @deprecated Use Interpolator directly
 */
using Interp1D = Interpolator;

/**
 * @deprecated Use Interpolator directly
 */
using JanusInterpolator = Interpolator;

// ============================================================================
// Free Function API (Backwards Compatibility)
// ============================================================================

/**
 * @brief N-dimensional interpolation on regular grids (backwards compatibility)
 *
 * @deprecated Use Interpolator class instead for better performance (cached CasADi function)
 *
 * This function creates a new Interpolator for each call. For repeated queries,
 * construct an Interpolator once and reuse it.
 *
 * @tparam Scalar Scalar type (double or casadi::MX)
 * @param points Vector of 1D coordinate arrays for each dimension
 * @param values_flat Flattened values in Fortran order (column-major)
 * @param xi Query points, shape (n_points, n_dimensions)
 * @param method Interpolation method
 * @param fill_value Optional value for out-of-bounds queries (extrapolates if nullopt)
 * @return Vector of interpolated values at query points
 */
template <typename Scalar>
JanusVector<Scalar> interpn(const std::vector<NumericVector> &points,
                            const NumericVector &values_flat, const JanusMatrix<Scalar> &xi,
                            InterpolationMethod method = InterpolationMethod::Linear,
                            std::optional<Scalar> fill_value = std::nullopt) {

    // Validate inputs
    if (points.empty()) {
        throw InterpolationError("interpn: points cannot be empty");
    }

    const int n_dims = static_cast<int>(points.size());

    // Check xi dimensions
    if (xi.cols() != n_dims && xi.rows() != n_dims) {
        throw InterpolationError(
            "interpn: xi must have shape (n_points, n_dims) or (n_dims, n_points)");
    }

    // Determine if we need to transpose xi
    bool need_transpose = (xi.rows() == n_dims && xi.cols() != n_dims);
    JanusMatrix<Scalar> xi_work;
    if (need_transpose) {
        xi_work = xi.transpose();
    } else {
        xi_work = xi;
    }
    const int n_points = static_cast<int>(xi_work.rows());

    // Build grid for bounds checking
    std::vector<std::vector<double>> grid(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        grid[d].resize(points[d].size());
        Eigen::VectorXd::Map(grid[d].data(), points[d].size()) = points[d];

        if (!std::is_sorted(grid[d].begin(), grid[d].end())) {
            throw InterpolationError("interpn: points[" + std::to_string(d) + "] must be sorted");
        }
    }

    // Validate values size
    int expected_size = 1;
    for (const auto &p : points) {
        expected_size *= static_cast<int>(p.size());
    }
    if (values_flat.size() != expected_size) {
        throw InterpolationError("interpn: values_flat size mismatch. Expected " +
                                 std::to_string(expected_size) + ", got " +
                                 std::to_string(values_flat.size()));
    }

    // Create interpolator
    Interpolator interp(points, values_flat, method);

    // Prepare result
    JanusVector<Scalar> result(n_points);

    // Handle fill_value for out-of-bounds
    if (fill_value.has_value()) {
        for (int i = 0; i < n_points; ++i) {
            bool out_of_bounds = false;
            if constexpr (std::is_floating_point_v<Scalar>) {
                for (int d = 0; d < n_dims; ++d) {
                    double val = xi_work(i, d);
                    if (val < grid[d].front() || val > grid[d].back()) {
                        out_of_bounds = true;
                        break;
                    }
                }
            }

            if (out_of_bounds) {
                result(i) = fill_value.value();
            } else {
                JanusVector<Scalar> point = xi_work.row(i).transpose();
                result(i) = interp(point);
            }
        }
    } else {
        // No fill_value, allow extrapolation (clamp)
        for (int i = 0; i < n_points; ++i) {
            JanusVector<Scalar> point = xi_work.row(i).transpose();
            result(i) = interp(point);
        }
    }

    return result;
}

/**
 * @brief Convenience overload for 2D interpolation with Eigen matrix values
 *
 * @deprecated Use Interpolator class instead
 */
template <typename Scalar>
JanusVector<Scalar> interpn2d(const NumericVector &x_points, const NumericVector &y_points,
                              const NumericMatrix &values,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &xi,
                              InterpolationMethod method = InterpolationMethod::Linear) {

    // Validate dimensions
    if (values.rows() != x_points.size() || values.cols() != y_points.size()) {
        throw InterpolationError(
            "interpn2d: values shape must match (x_points.size(), y_points.size())");
    }

    // Build points vector
    std::vector<NumericVector> points = {x_points, y_points};

    // Flatten values in Fortran order (column-major)
    NumericVector values_flat = detail::flatten_fortran_order(values);

    return interpn<Scalar>(points, values_flat, xi, method);
}

} // namespace janus
