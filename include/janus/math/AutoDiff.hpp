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
    if constexpr (std::is_same_v<Expr, SymbolicScalar>) {
        // Collect pack into a vector of MX
        std::vector<SymbolicScalar> vars;
        (vars.push_back(variables), ...);

        SymbolicScalar v_cat = SymbolicScalar::vertcat(vars);
        return SymbolicScalar::jacobian(expression, v_cat);
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
inline std::vector<SymbolicScalar> to_mx_vector(const std::vector<SymbolicArg> &args) {
    std::vector<SymbolicScalar> ret;
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
    SymbolicScalar expr_cat = SymbolicScalar::vertcat(detail::to_mx_vector(expressions));
    SymbolicScalar var_cat = SymbolicScalar::vertcat(detail::to_mx_vector(variables));
    return SymbolicScalar::jacobian(expr_cat, var_cat);
}

/**
 * @brief Symbolic gradient (for scalar-output functions)
 *
 * Computes the symbolic gradient of a scalar expression with respect to variables.
 * Note: Named `sym_gradient` to distinguish from numerical `gradient()` in Calculus.hpp.
 *
 * @param expr Scalar expression to differentiate
 * @param vars Variables to differentiate with respect to
 * @return Column vector of partial derivatives
 */
inline SymbolicVector sym_gradient(const SymbolicArg &expr, const SymbolicArg &vars) {
    // CasADi gradient returns a dense vector (column or row depending on version, usually 1xN or
    // Nx1) We force it to be a column vector
    SymbolicScalar g = SymbolicScalar::gradient(expr.get(), vars.get());

    // Convert to SymbolicVector (Janus type)
    if (g.size2() > 1 && g.size1() == 1) {
        return to_eigen(g.T());
    }
    return to_eigen(g);
}

/**
 * @brief Hessian matrix (second-order derivatives)
 *
 * Computes the matrix of second derivatives: H_ij = d²f / (dx_i dx_j)
 *
 * @param expr Scalar expression
 * @param vars Variables
 * @return Symmetric Hessian matrix (SymbolicMatrix)
 */
inline SymbolicMatrix hessian(const SymbolicArg &expr, const SymbolicArg &vars) {
    // casadi::MX::hessian returns {H, g}
    // We only want H
    // Note: older CasADi versions return H directly, but modern one returns vector
    // Let's use the property that hessian(f, x) = jacobian(gradient(f, x), x) for safety/clarity
    // or use the optimized CasADi call:
    SymbolicScalar H = SymbolicScalar::hessian(expr.get(), vars.get());
    return to_eigen(H);
}

/**
 * @brief Hessian of Lagrangian for constrained optimization
 *
 * Computes ∇²L where L = f(x) + λᵀg(x)
 * This is the exact matrix needed for IPOPT's hessian_lagrangian callback.
 *
 * @param objective Objective function f(x) (scalar)
 * @param constraints Constraint functions g(x) (vector)
 * @param vars Decision variables x (vector)
 * @param multipliers Lagrange multipliers λ (vector, same size as constraints)
 * @return Hessian of Lagrangian (symmetric matrix)
 */
inline SymbolicMatrix hessian_lagrangian(const SymbolicArg &objective,
                                         const SymbolicArg &constraints, const SymbolicArg &vars,
                                         const SymbolicArg &multipliers) {
    // L = f + lambda' * g
    SymbolicScalar f = objective.get();
    SymbolicScalar g = constraints.get();
    SymbolicScalar lam = multipliers.get();

    // Robust dot product
    SymbolicScalar lagrangian = f + SymbolicScalar::dot(lam, g);

    return janus::hessian(lagrangian, vars);
}

// --- Overloads for vector inputs (convenience) ---

/**
 * @brief Symbolic gradient with vector of variables
 */
inline SymbolicVector sym_gradient(const SymbolicArg &expr, const std::vector<SymbolicArg> &vars) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return sym_gradient(expr, v_cat);
}

/**
 * @brief Hessian with vector of variables
 */
inline SymbolicMatrix hessian(const SymbolicArg &expr, const std::vector<SymbolicArg> &vars) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return janus::hessian(expr, v_cat);
}

/**
 * @brief Hessian of Lagrangian with vector inputs
 */
inline SymbolicMatrix hessian_lagrangian(const SymbolicArg &objective,
                                         const SymbolicArg &constraints,
                                         const std::vector<SymbolicArg> &vars,
                                         const SymbolicArg &multipliers) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return hessian_lagrangian(objective, constraints, v_cat, multipliers);
}

} // namespace janus
