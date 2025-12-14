#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Linalg.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <vector>

namespace janus {

/**
 * @brief 1D Interpolator using CasADi
 *
 * Supports both numeric and symbolic evaluation.
 * Uses linear interpolation by default.
 */
class JanusInterpolator {
  private:
    std::vector<double> m_x;
    std::vector<double> m_y;
    casadi::Function m_casadi_fn;
    bool m_valid = false;

  public:
    /**
     * @brief Default constructor
     */
    JanusInterpolator() = default;

    /**
     * @brief Construct a new Janus Interpolator
     *
     * @param x Grid points (must be sorted)
     * @param y Function values
     * @throw std::invalid_argument if inputs are invalid
     */
    JanusInterpolator(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("JanusInterpolator: x and y must have same size");
        }
        if (x.size() < 2) {
            throw std::invalid_argument("JanusInterpolator: Need at least 2 points");
        }

        m_x.resize(x.size());
        m_y.resize(y.size());
        Eigen::VectorXd::Map(&m_x[0], x.size()) = x;
        Eigen::VectorXd::Map(&m_y[0], y.size()) = y;

        // rudimentary check for sorted x
        if (!std::is_sorted(m_x.begin(), m_x.end())) {
            throw std::invalid_argument("JanusInterpolator: x grid must be sorted");
        }

        // Setup CasADi interpolant
        // args: name, solver ("linear"), grid (vector of vector), values
        m_casadi_fn = casadi::interpolant("interp1d", "linear", {m_x}, m_y);
        m_valid = true;
    }

    // Scalar operator
    /**
     * @brief Evaluate interpolant at a scalar point
     * @tparam T Scalar type (double or MX)
     * @param query Query point
     * @return Interpolated value
     */
    template <typename T> T operator()(const T &query) const {
        if (!m_valid)
            throw std::runtime_error("JanusInterpolator: Uninitialized");

        if constexpr (std::is_floating_point_v<T>) {
            return eval_numeric(query);
        } else {
            return eval_symbolic(query);
        }
    }

    // Matrix operator (Vectorized)
    /**
     * @brief Evaluate interpolant at multiple points
     * @tparam Derived Eigen matrix type
     * @param query Matrix of query points
     * @return Matrix of interpolated values
     */
    template <typename Derived> auto operator()(const Eigen::MatrixBase<Derived> &query) const {
        using Scalar = typename Derived::Scalar;
        if (!m_valid)
            throw std::runtime_error("JanusInterpolator: Uninitialized");

        if constexpr (std::is_floating_point_v<Scalar>) {
            // Numeric: Element-wise map
            return query.unaryExpr([this](Scalar x) { return this->eval_numeric(x); });
        } else {
            // Symbolic: Convert -> Call -> Convert
            // Note: casadi interpolant handles vector inputs naturally
            casadi::MX q_mx = janus::to_mx(query);
            std::vector<casadi::MX> args = {q_mx};
            std::vector<casadi::MX> res = m_casadi_fn(args);
            return janus::to_eigen(res[0]);
        }
    }

  private:
    double eval_numeric(double query) const {
        // Binary search
        auto it = std::upper_bound(m_x.begin(), m_x.end(), query);

        if (it == m_x.begin()) {
            // query < first element. Extrapolate using first segment
            double slope = (m_y[1] - m_y[0]) / (m_x[1] - m_x[0]);
            return m_y[0] + slope * (query - m_x[0]);
        } else if (it == m_x.end()) {
            // query > last element. Extrapolate using last segment
            size_t n = m_x.size();
            double slope = (m_y[n - 1] - m_y[n - 2]) / (m_x[n - 1] - m_x[n - 2]);
            return m_y[n - 1] + slope * (query - m_x[n - 1]);
        } else {
            // Interior: Linear Interpolation
            size_t idx = std::distance(m_x.begin(), it);
            double x_high = m_x[idx];
            double x_low = m_x[idx - 1];
            double y_high = m_y[idx];
            double y_low = m_y[idx - 1];

            double t = (query - x_low) / (x_high - x_low);
            return y_low + t * (y_high - y_low);
        }
    }

    casadi::MX eval_symbolic(const casadi::MX &query) const {
        std::vector<casadi::MX> args = {query};
        std::vector<casadi::MX> res = m_casadi_fn(args);
        return res[0];
    }
};

} // namespace janus
