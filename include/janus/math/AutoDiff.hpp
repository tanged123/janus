#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusTypes.hpp"
#include <vector>

namespace janus {

// --- Jacobian (Symbolic Automatic Differentiation) ---
/**
 * @brief Computes Jacobian of an expression with respect to variables.
 *
 * Wraps CasADi's jacobian function. Automatic Jacobian only supported for Symbolic types currently.
 *
 * @tparam Expr Expression type (must be convertible to MX)
 * @tparam Vars Variable types (must be convertible to MX)
 * @param expression Expression to differentiate
 * @param variables Variables to differentiate with respect to
 * @return Jacobian matrix (symbolic)
 */
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

/**
 * @brief Computes Jacobian with vector arguments (expressions and variables)
 *
 * @param expressions Vector of symbolic expressions
 * @param variables Vector of symbolic variables
 * @return Jacobian matrix
 */
inline auto jacobian(const std::vector<SymbolicArg> &expressions,
                     const std::vector<SymbolicArg> &variables) {
    casadi::MX expr_cat = casadi::MX::vertcat(detail::to_mx_vector(expressions));
    casadi::MX var_cat = casadi::MX::vertcat(detail::to_mx_vector(variables));
    return casadi::MX::jacobian(expr_cat, var_cat);
}

} // namespace janus
