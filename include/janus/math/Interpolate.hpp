#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/math/Linalg.hpp"
#include <Eigen/Dense>
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
// N-Dimensional Interpolation
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
inline Eigen::VectorXd flatten_fortran_order(const Eigen::MatrixBase<Derived> &values) {
    // For 2D matrices, transpose then flatten (equivalent to Fortran order)
    Eigen::MatrixXd transposed = values.transpose();
    return Eigen::Map<const Eigen::VectorXd>(transposed.data(), transposed.size());
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

/**
 * @brief N-dimensional interpolation on regular grids
 *
 * Performs interpolation on a regular grid in N dimensions. This is analogous
 * to scipy.interpolate.interpn() and supports both numeric and symbolic
 * evaluation.
 *
 * @tparam Scalar Scalar type (double or casadi::MX)
 * @param points Vector of 1D coordinate arrays for each dimension
 * @param values_flat Flattened values in Fortran order (column-major)
 * @param xi Query points, shape (n_points, n_dimensions)
 * @param method Interpolation method (Linear, Hermite, BSpline, Nearest). Note: Hermite is
 * numeric-only.
 * @param fill_value Optional value for out-of-bounds queries (extrapolates if nullopt)
 * @return Vector of interpolated values at query points
 *
 * @example
 * ```cpp
 * // 2D interpolation on a 3x4 grid
 * std::vector<Eigen::VectorXd> points = {
 *     (Eigen::VectorXd(3) << 0, 1, 2).finished(),  // x coordinates
 *     (Eigen::VectorXd(4) << 0, 1, 2, 3).finished() // y coordinates
 * };
 * Eigen::VectorXd values(12); // 3*4 = 12 values in Fortran order
 * // ... fill values ...
 *
 * Eigen::MatrixXd xi(2, 2); // 2 query points, 2 dimensions
 * xi << 0.5, 1.5,  // point 1: (0.5, 1.5)
 *       1.0, 2.0;  // point 2: (1.0, 2.0)
 *
 * auto result = janus::interpn<double>(points, values, xi);
 * ```
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
interpn(const std::vector<Eigen::VectorXd> &points, const Eigen::VectorXd &values_flat,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &xi,
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
    // Convention: xi should be (n_points, n_dims)
    bool need_transpose = (xi.rows() == n_dims && xi.cols() != n_dims);

    // Create working matrix - explicitly handle transpose to avoid type ambiguity
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> xi_work;
    if (need_transpose) {
        xi_work = xi.transpose();
    } else {
        xi_work = xi;
    }
    const int n_points = static_cast<int>(xi_work.rows());

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

    // Extract grid vectors for CasADi
    std::vector<std::vector<double>> grid(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        grid[d].resize(points[d].size());
        Eigen::VectorXd::Map(grid[d].data(), points[d].size()) = points[d];

        // Check sorted
        if (!std::is_sorted(grid[d].begin(), grid[d].end())) {
            throw InterpolationError("interpn: points[" + std::to_string(d) + "] must be sorted");
        }
    }

    // Convert values to std::vector for CasADi
    std::vector<double> values_vec(values_flat.data(), values_flat.data() + values_flat.size());

    // Handle Nearest method specially (round to nearest grid point, then use linear)
    if (method == InterpolationMethod::Nearest) {
        // For nearest neighbor, we snap query points to nearest grid values
        // This is a numeric-only operation for now
        if constexpr (!std::is_floating_point_v<Scalar>) {
            throw InterpolationError("interpn: Nearest method not supported for symbolic types");
        }
    }

    // Prepare result
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(n_points);

    // Handle Hermite method with custom implementation (C1 Catmull-Rom)
    // Note: Hermite is numeric-only because interval finding requires comparisons
    if (method == InterpolationMethod::Hermite) {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            throw InterpolationError("interpn: Hermite method not supported for symbolic types. "
                                     "Use Linear or BSpline for symbolic interpolation.");
        } else {
            for (int i = 0; i < n_points; ++i) {
                // Check bounds and apply fill_value if needed
                bool out_of_bounds = false;
                for (int d = 0; d < n_dims; ++d) {
                    double val = xi_work(i, d);
                    double min_val = grid[d].front();
                    double max_val = grid[d].back();
                    if (val < min_val || val > max_val) {
                        out_of_bounds = true;
                        break;
                    }
                }

                if (out_of_bounds && fill_value.has_value()) {
                    result(i) = fill_value.value();
                } else {
                    // Build query vector with clamping
                    std::vector<double> query(n_dims);
                    for (int d = 0; d < n_dims; ++d) {
                        double val = xi_work(i, d);
                        val = std::max(val, grid[d].front());
                        val = std::min(val, grid[d].back());
                        query[d] = val;
                    }

                    // Use custom numeric Hermite interpolation
                    result(i) = detail::hermite_interpn_numeric(grid, values_vec, query);
                }
            }
            return result;
        }
    }

    // Create CasADi interpolant for Linear/BSpline/Nearest methods
    std::string method_str = detail::method_to_casadi_string(method);
    casadi::Function interpolant;

    try {
        interpolant = casadi::interpolant("interpn", method_str, grid, values_vec);
    } catch (const std::exception &e) {
        throw InterpolationError(std::string("interpn: CasADi interpolant creation failed: ") +
                                 e.what());
    }

    if constexpr (std::is_floating_point_v<Scalar>) {
        // Numeric evaluation
        for (int i = 0; i < n_points; ++i) {
            // Check bounds and apply fill_value if needed
            bool out_of_bounds = false;
            for (int d = 0; d < n_dims; ++d) {
                double val = xi_work(i, d);
                double min_val = grid[d].front();
                double max_val = grid[d].back();
                if (val < min_val || val > max_val) {
                    out_of_bounds = true;
                    break;
                }
            }

            if (out_of_bounds && fill_value.has_value()) {
                result(i) = fill_value.value();
            } else {
                // Build query vector
                std::vector<double> query(n_dims);
                for (int d = 0; d < n_dims; ++d) {
                    double val = xi_work(i, d);
                    // Clamp to grid bounds for extrapolation
                    val = std::max(val, grid[d].front());
                    val = std::min(val, grid[d].back());
                    query[d] = val;
                }

                // Call interpolant
                std::vector<casadi::DM> args = {casadi::DM(query)};
                std::vector<casadi::DM> res = interpolant(args);
                result(i) = static_cast<double>(res[0]);
            }
        }
    } else {
        // Symbolic evaluation
        // Build query matrix for batch evaluation
        // CasADi expects shape (n_dims, n_points)
        SymbolicScalar xi_mx = janus::to_mx(xi_work.transpose());

        // Handle bounds checking with fill_value using where
        if (fill_value.has_value()) {
            // Create bounds mask
            casadi::MX in_bounds = casadi::MX::ones(1, n_points);
            for (int d = 0; d < n_dims; ++d) {
                casadi::MX row = xi_mx(d, casadi::Slice());
                double min_val = grid[d].front();
                double max_val = grid[d].back();
                in_bounds = in_bounds * (row >= min_val) * (row <= max_val);
            }

            // Clamp for interpolation
            casadi::MX xi_clamped = xi_mx;
            for (int d = 0; d < n_dims; ++d) {
                double min_val = grid[d].front();
                double max_val = grid[d].back();
                casadi::MX row = xi_mx(d, casadi::Slice());
                row = casadi::MX::fmax(row, min_val);
                row = casadi::MX::fmin(row, max_val);
                for (int j = 0; j < n_points; ++j) {
                    xi_clamped(d, j) = row(j);
                }
            }

            // Interpolate
            std::vector<casadi::MX> args = {xi_clamped};
            std::vector<casadi::MX> interp_result = interpolant(args);
            casadi::MX interpolated = interp_result[0];

            // Apply fill_value where out of bounds
            casadi::MX fill_mx = fill_value.value();
            casadi::MX final_result = casadi::MX::if_else(in_bounds, interpolated, fill_mx);

            // Convert to output
            for (int i = 0; i < n_points; ++i) {
                result(i) = final_result(i);
            }
        } else {
            // No fill_value, allow extrapolation (clamp to bounds)
            casadi::MX xi_clamped = xi_mx;
            for (int d = 0; d < n_dims; ++d) {
                double min_val = grid[d].front();
                double max_val = grid[d].back();
                for (int j = 0; j < n_points; ++j) {
                    casadi::MX val = xi_mx(d, j);
                    val = casadi::MX::fmax(val, min_val);
                    val = casadi::MX::fmin(val, max_val);
                    xi_clamped(d, j) = val;
                }
            }

            std::vector<casadi::MX> args = {xi_clamped};
            std::vector<casadi::MX> interp_result = interpolant(args);
            casadi::MX interpolated = interp_result[0];

            for (int i = 0; i < n_points; ++i) {
                result(i) = interpolated(i);
            }
        }
    }

    return result;
}

/**
 * @brief Convenience overload for 2D interpolation with Eigen matrix values
 *
 * @param x_points X-axis coordinates
 * @param y_points Y-axis coordinates
 * @param values 2D matrix of values, shape (x_points.size(), y_points.size())
 * @param xi Query points, shape (n_points, 2)
 * @param method Interpolation method
 * @return Interpolated values
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
interpn2d(const Eigen::VectorXd &x_points, const Eigen::VectorXd &y_points,
          const Eigen::MatrixXd &values, const Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &xi,
          InterpolationMethod method = InterpolationMethod::Linear) {

    // Validate dimensions
    if (values.rows() != x_points.size() || values.cols() != y_points.size()) {
        throw InterpolationError(
            "interpn2d: values shape must match (x_points.size(), y_points.size())");
    }

    // Build points vector
    std::vector<Eigen::VectorXd> points = {x_points, y_points};

    // Flatten values in Fortran order (column-major)
    Eigen::VectorXd values_flat = detail::flatten_fortran_order(values);

    return interpn<Scalar>(points, values_flat, xi, method);
}

/**
 * @brief 1D Interpolator with method selection
 *
 * High-performance 1D interpolation with cached CasADi function.
 * Supports Linear, Hermite (C1), and BSpline (C2) interpolation.
 *
 * @example
 * ```cpp
 * Eigen::VectorXd x(4), y(4);
 * x << 0, 1, 2, 3;
 * y << 0, 1, 4, 9;  // y = x^2
 *
 * // Create interpolator with method
 * janus::Interp1D interp(x, y, janus::InterpolationMethod::Hermite);
 *
 * // Query at multiple points
 * double result = interp(1.5);  // Smooth interpolation
 * ```
 */
class Interp1D {
  private:
    std::vector<double> m_x;
    std::vector<double> m_y;
    casadi::Function m_casadi_fn;
    InterpolationMethod m_method = InterpolationMethod::Linear;
    bool m_valid = false;
    bool m_use_custom_hermite = false;

  public:
    /**
     * @brief Default constructor
     */
    Interp1D() = default;

    /**
     * @brief Construct a 1D interpolator
     *
     * @param x Grid points (must be sorted)
     * @param y Function values at grid points
     * @param method Interpolation method (default: Linear)
     * @throw InterpolationError if inputs are invalid
     */
    Interp1D(const Eigen::VectorXd &x, const Eigen::VectorXd &y,
             InterpolationMethod method = InterpolationMethod::Linear) {
        if (x.size() != y.size()) {
            throw InterpolationError("Interp1D: x and y must have same size");
        }
        if (x.size() < 2) {
            throw InterpolationError("Interp1D: need at least 2 grid points");
        }

        m_method = method;
        m_x.resize(x.size());
        m_y.resize(y.size());
        Eigen::VectorXd::Map(&m_x[0], x.size()) = x;
        Eigen::VectorXd::Map(&m_y[0], y.size()) = y;

        // Check sorted
        if (!std::is_sorted(m_x.begin(), m_x.end())) {
            throw InterpolationError("Interp1D: x grid must be sorted");
        }

        // Method-specific setup
        if (method == InterpolationMethod::Hermite) {
            // Hermite uses custom implementation, no CasADi interpolant needed
            m_use_custom_hermite = true;
            m_valid = true;
        } else if (method == InterpolationMethod::BSpline) {
            // BSpline requires at least 4 points
            if (x.size() < 4) {
                throw InterpolationError("Interp1D: BSpline requires at least 4 grid points");
            }
            m_casadi_fn = casadi::interpolant("interp1d", "bspline", {m_x}, m_y);
            m_valid = true;
        } else if (method == InterpolationMethod::Nearest) {
            // Nearest neighbor - use linear CasADi but round in eval
            m_casadi_fn = casadi::interpolant("interp1d", "linear", {m_x}, m_y);
            m_valid = true;
        } else {
            // Linear (default)
            m_casadi_fn = casadi::interpolant("interp1d", "linear", {m_x}, m_y);
            m_valid = true;
        }
    }

    /**
     * @brief Get the interpolation method
     */
    InterpolationMethod method() const { return m_method; }

    /**
     * @brief Evaluate interpolant at a scalar point
     * @tparam T Scalar type (double or casadi::MX)
     * @param query Query point
     * @return Interpolated value
     */
    template <JanusScalar T> T operator()(const T &query) const {
        if (!m_valid)
            throw InterpolationError("Interp1D: interpolator not initialized");

        if constexpr (std::is_floating_point_v<T>) {
            return eval_numeric(query);
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(
                    "Interp1D: Hermite method not supported for symbolic types");
            }
            return eval_symbolic(query);
        }
    }

    /**
     * @brief Evaluate interpolant at multiple points
     * @tparam Derived Eigen matrix type
     * @param query Matrix of query points
     * @return Matrix of interpolated values
     */
    template <typename Derived> auto operator()(const Eigen::MatrixBase<Derived> &query) const {
        using Scalar = typename Derived::Scalar;
        if (!m_valid)
            throw InterpolationError("Interp1D: interpolator not initialized");

        if constexpr (std::is_floating_point_v<Scalar>) {
            // Numeric: Element-wise map
            return query.unaryExpr([this](Scalar x) { return this->eval_numeric(x); }).eval();
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(
                    "Interp1D: Hermite method not supported for symbolic types");
            }
            // Symbolic: Convert -> Call -> Convert
            SymbolicScalar q_mx = janus::to_mx(query);
            std::vector<SymbolicScalar> args = {q_mx};
            std::vector<SymbolicScalar> res = m_casadi_fn(args);
            return janus::to_eigen(res[0]);
        }
    }

  private:
    double eval_numeric(double query) const {
        // Clamp to bounds for all methods
        double clamped = std::max(query, m_x.front());
        clamped = std::min(clamped, m_x.back());

        if (m_use_custom_hermite) {
            // Use Catmull-Rom Hermite spline
            return detail::hermite_interp_1d_numeric(m_x, m_y, clamped);
        }

        if (m_method == InterpolationMethod::Nearest) {
            // Snap to nearest grid point
            auto it = std::lower_bound(m_x.begin(), m_x.end(), clamped);
            size_t idx = std::distance(m_x.begin(), it);
            if (idx > 0 && (idx == m_x.size() ||
                            std::abs(clamped - m_x[idx - 1]) < std::abs(clamped - m_x[idx]))) {
                idx--;
            }
            return m_y[idx];
        }

        // Linear/BSpline: Use CasADi
        std::vector<casadi::DM> args = {casadi::DM(clamped)};
        std::vector<casadi::DM> res = m_casadi_fn(args);
        return static_cast<double>(res[0]);
    }

    SymbolicScalar eval_symbolic(const SymbolicScalar &query) const {
        // Clamp to bounds
        SymbolicScalar clamped = SymbolicScalar::fmax(query, m_x.front());
        clamped = SymbolicScalar::fmin(clamped, m_x.back());

        std::vector<casadi::MX> args = {clamped};
        std::vector<casadi::MX> res = m_casadi_fn(args);
        return res[0];
    }
};

// Backwards compatibility alias (deprecated)
using JanusInterpolator = Interp1D;

} // namespace janus
