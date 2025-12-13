#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <Eigen/Dense>

namespace janus {

// --- diff(vector) ---
// Returns adjacent differences: out[i] = v[i+1] - v[i]
// Returns a vector of size N-1
template <typename Derived> auto diff(const Eigen::MatrixBase<Derived> &v) {
    // Return expression template for efficiency
    return (v.tail(v.size() - 1) - v.head(v.size() - 1));
}

// --- trapz(y, x) ---
// Trapezoidal integration: sum(0.5 * (y[i+1]+y[i]) * (x[i+1]-x[i]))
template <typename DerivedY, typename DerivedX>
auto trapz(const Eigen::MatrixBase<DerivedY> &y, const Eigen::MatrixBase<DerivedX> &x) {
    auto dx = (x.tail(x.size() - 1) - x.head(x.size() - 1));
    auto mean_y = 0.5 * (y.tail(y.size() - 1) + y.head(y.size() - 1));

    // Sum of element-wise product
    return (mean_y.array() * dx.array()).sum();
}

// --- gradient_1d(y, x) ---
// Central difference gradient
// Returns vector of same size as inputs
template <typename DerivedY, typename DerivedX>
auto gradient_1d(const Eigen::MatrixBase<DerivedY> &y, const Eigen::MatrixBase<DerivedX> &x) {
    Eigen::Index n = y.size();
    typename DerivedY::PlainObject grad(n);

    if (n < 2) {
        if (n > 0)
            grad.setZero();
        return grad;
    }

    // Interior points: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    if (n > 2) {
        grad.segment(1, n - 2) =
            (y.tail(n - 2) - y.head(n - 2)).array() / (x.tail(n - 2) - x.head(n - 2)).array();
    }

    // Boundaries (Forward/Backward difference)
    // Forward at 0
    grad(0) = (y(1) - y(0)) / (x(1) - x(0));
    // Backward at n-1
    grad(n - 1) = (y(n - 1) - y(n - 2)) / (x(n - 1) - x(n - 2));

    return grad;
}

// --- Jacobian (Symbolic) ---
// Computes Jacobian of an expression with respect to variables.
// Helper to abstract away CasADi vertcat details.
// Usage: auto J = janus::jacobian(expr, v1, v2); or janus::jacobian(expr, vector_of_vars);
template <typename Expr, typename... Vars>
auto jacobian(const Expr &expression, const Vars &...variables) {
    if constexpr (std::is_same_v<Expr, casadi::MX>) {
        // Collect pack into a vector of MX
        std::vector<casadi::MX> vars;
        (vars.push_back(variables), ...);

        casadi::MX v_cat = casadi::MX::vertcat(vars);
        return casadi::MX::jacobian(expression, v_cat);
    } else {
        // Fallback or static_assert for non-symbolic types
        // Ideally we would support numeric autodiff here later
        static_assert(std::is_same_v<Expr, casadi::MX>,
                      "Automatic Jacobian only supported for Symbolic types currently.");
        return Expr(0);
    }
}

// Helper to separate implementation
namespace detail {
inline std::vector<casadi::MX> to_mx_vector(const std::vector<SymbolicArg> &args) {
    std::vector<casadi::MX> ret;
    ret.reserve(args.size());
    for (const auto &arg : args)
        ret.push_back(arg.get());
    return ret;
}
} // namespace detail

// Overload for vector of variables (expr, {vars})
// Supports mixed types via SymbolicArg
// REMOVED due to ambiguity with ({expr}, {var}). use {expr} instead.
// inline auto jacobian(const SymbolicArg& expression, const std::vector<SymbolicArg>& variables) {
//     casadi::MX v_cat = casadi::MX::vertcat(detail::to_mx_vector(variables));
//     return casadi::MX::jacobian(expression, v_cat);
// }

// Overload for vector of expressions and variables ({exprs}, {vars})
// Supports mixed types via SymbolicArg
inline auto jacobian(const std::vector<SymbolicArg> &expressions,
                     const std::vector<SymbolicArg> &variables) {
    casadi::MX expr_cat = casadi::MX::vertcat(detail::to_mx_vector(expressions));
    casadi::MX var_cat = casadi::MX::vertcat(detail::to_mx_vector(variables));
    return casadi::MX::jacobian(expr_cat, var_cat);
}

} // namespace janus
