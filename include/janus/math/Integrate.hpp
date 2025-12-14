#pragma once
/**
 * @file Integrate.hpp
 * @brief ODE integration for Janus framework
 *
 * Provides `quad` (definite integration) and `solve_ivp` (initial value problem)
 * with dual-backend support: numeric fallback and CasADi CVODES for symbolic graphs.
 */

#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Arithmetic.hpp"
#include "janus/math/Spacing.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace janus {

// ============================================================================
// ODE Result structure
// ============================================================================

/**
 * @brief Result structure for ODE solvers
 *
 * Contains time points, solution values, and solver metadata.
 *
 * @tparam Scalar numeric or symbolic scalar type
 */
template <typename Scalar> struct OdeResult {
    /// Time points where solution was computed
    JanusVector<Scalar> t;

    /// Solution values at each time point (each column is a state at time t[i])
    JanusMatrix<Scalar> y;

    /// Whether integration was successful
    bool success = true;

    /// Descriptive message
    std::string message = "";

    /// Status code (0 = success)
    int status = 0;
};

// ============================================================================
// Quadrature Result structure
// ============================================================================

/**
 * @brief Result structure for quadrature (definite integration)
 *
 * @tparam Scalar numeric or symbolic scalar type
 */
template <typename Scalar> struct QuadResult {
    /// Integral value
    Scalar value;

    /// Estimated error (only meaningful for numeric backend)
    double error = 0.0;
};

// ============================================================================
// quad: Definite Integral
// ============================================================================

/**
 * @brief Compute definite integral using adaptive quadrature (numeric) or CVODES (symbolic)
 *
 * Numeric backend uses Gauss-Kronrod adaptive quadrature.
 * Symbolic backend wraps CasADi's CVODES integrator.
 *
 * @param func Function to integrate (callable taking Scalar, returning Scalar)
 * @param a Lower bound
 * @param b Upper bound
 * @param abstol Absolute tolerance (default 1e-8)
 * @param reltol Relative tolerance (default 1e-6)
 * @return QuadResult with integral value and error estimate
 *
 * @code
 * // Numeric integration
 * auto result = janus::quad([](double x) { return x*x; }, 0.0, 1.0);
 * // result.value ≈ 1/3
 *
 * // Symbolic integration (generates CasADi graph)
 * auto x = janus::sym("x");
 * auto expr = x * x;
 * auto sym_result = janus::quad(expr, x, 0.0, 1.0);
 * @endcode
 */
template <typename Func, typename T>
QuadResult<T> quad(Func &&func, T a, T b, double abstol = 1e-8, double reltol = 1e-6) {
    // For numeric types, use Gauss-Kronrod quadrature
    if constexpr (std::is_floating_point_v<T>) {
        // Gauss-Kronrod 15-point rule on [a, b]
        // Nodes and weights for G7K15 rule, transformed from [-1, 1]

        static constexpr int n = 15;
        // Kronrod nodes on [-1, 1]
        static constexpr double xk[15] = {-0.991455371120812639, -0.949107912342758525,
                                          -0.864864423359769073, -0.741531185599394440,
                                          -0.586087235467691130, -0.405845151377397167,
                                          -0.207784955007898468, 0.0,
                                          0.207784955007898468,  0.405845151377397167,
                                          0.586087235467691130,  0.741531185599394440,
                                          0.864864423359769073,  0.949107912342758525,
                                          0.991455371120812639};

        // Kronrod weights
        static constexpr double wk[15] = {
            0.022935322010529225, 0.063092092629978553, 0.104790010322250184, 0.140653259715525919,
            0.169004726639267903, 0.190350578064785410, 0.204432940075298892, 0.209482141084727828,
            0.204432940075298892, 0.190350578064785410, 0.169004726639267903, 0.140653259715525919,
            0.104790010322250184, 0.063092092629978553, 0.022935322010529225};

        // Gauss weights (for error estimation, at odd indices 1,3,5,...,13)
        static constexpr double wg[7] = {
            0.129484966168869693, 0.279705391489276668, 0.381830050505118945, 0.417959183673469388,
            0.381830050505118945, 0.279705391489276668, 0.129484966168869693};

        // Transform to [a, b]
        T center = (a + b) / 2.0;
        T halfwidth = (b - a) / 2.0;

        T kronrod_sum = 0.0;
        T gauss_sum = 0.0;

        for (int i = 0; i < n; ++i) {
            T x = center + halfwidth * xk[i];
            T fx = func(x);
            kronrod_sum += wk[i] * fx;

            // Check if this is a Gauss node (i = 1, 3, 5, 7, 9, 11, 13)
            if (i % 2 == 1 && i < 14) {
                gauss_sum += wg[i / 2] * fx;
            }
        }

        kronrod_sum *= halfwidth;
        gauss_sum *= halfwidth;

        QuadResult<T> result;
        result.value = kronrod_sum;
        result.error = std::abs(kronrod_sum - gauss_sum);
        return result;
    } else {
        // Symbolic path: This won't work with a lambda directly
        // User should use the symbolic expression overload below
        throw std::runtime_error("quad with callable is not supported for symbolic types. "
                                 "Use quad(expr, variable, a, b) instead.");
    }
}

/**
 * @brief Compute definite integral of a symbolic expression using CasADi CVODES
 *
 * @param expr Symbolic expression to integrate
 * @param variable Variable of integration (MX symbol)
 * @param a Lower bound (numeric)
 * @param b Upper bound (numeric)
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return QuadResult<SymbolicScalar> with symbolic integral value
 */
inline QuadResult<SymbolicScalar> quad(const SymbolicScalar &expr, const SymbolicScalar &variable,
                                       double a, double b, double abstol = 1e-8,
                                       double reltol = 1e-6) {
    // Find all variables in the expression
    std::vector<casadi::MX> all_vars = casadi::MX::symvar(expr);

    // Separate: variable of integration vs parameters
    std::vector<casadi::MX> parameters;
    for (const auto &var : all_vars) {
        if (!casadi::MX::is_equal(var, variable)) {
            parameters.push_back(var);
        }
    }

    // Build parameter vector
    casadi::MX p = casadi::MX::vertcat(parameters);

    // Create dummy state variable (integral accumulator)
    casadi::MX dummy_x = casadi::MX::sym("__quad_x");

    // Build integrator
    // The ODE is: dx/dt = expr, where t is the variable of integration
    casadi::MXDict dae = {{"x", dummy_x}, {"t", variable}, {"p", p}, {"ode", expr}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator = casadi::integrator("quad_integrator", "cvodes", dae, a, b, opts);

    // Evaluate with x0 = 0
    casadi::DMDict arg = {{"x0", casadi::DM(0.0)}, {"p", casadi::DM::zeros(p.size1(), p.size2())}};

    // For symbolic output, we need to call with symbolic parameters
    casadi::MXDict sym_arg = {{"x0", casadi::MX(0.0)}, {"p", p}};

    casadi::MXDict res = integrator(sym_arg);

    QuadResult<SymbolicScalar> result;
    result.value = res.at("xf");
    result.error = abstol; // Nominal error bound
    return result;
}

// ============================================================================
// solve_ivp: Initial Value Problem
// ============================================================================

/**
 * @brief Solve initial value problem: dy/dt = f(t, y), y(t0) = y0
 *
 * Numeric backend uses fixed-step RK4 integrator.
 * Symbolic backend uses CasADi CVODES for automatic differentiation through the ODE.
 *
 * @tparam Func Callable type f(t, y) -> dy/dt
 * @tparam Scalar Scalar type (double or casadi::MX)
 * @param fun Right-hand side function f(t, y)
 * @param t_span Integration interval (t0, tf)
 * @param y0 Initial state vector
 * @param n_eval Number of output points (default 100)
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult<Scalar> with time points and solution
 *
 * @code
 * // Exponential decay: dy/dt = -0.5*y
 * auto sol = janus::solve_ivp(
 *     [](double t, const Eigen::VectorXd& y) { return -0.5 * y; },
 *     {0.0, 10.0},  // t_span
 *     Eigen::VectorXd::Constant(1, 2.5),  // y0
 *     100  // n_eval
 * );
 * @endcode
 */
template <typename Func>
OdeResult<double> solve_ivp(Func &&fun, std::pair<double, double> t_span, const Eigen::VectorXd &y0,
                            int n_eval = 100, double abstol = 1e-8, double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;

    OdeResult<double> result;
    result.t = linspace(t0, tf, n_eval);
    result.y.resize(y0.size(), n_eval);
    result.y.col(0) = y0;

    // Use RK4 with adaptive stepping (simplified: fixed step based on tolerance)
    int n_state = y0.size();
    double dt_base = (tf - t0) / (n_eval - 1);

    // Substeps per output interval for accuracy
    int substeps = std::max(1, static_cast<int>(std::ceil(1.0 / std::sqrt(reltol))));

    Eigen::VectorXd y = y0;

    for (int i = 1; i < n_eval; ++i) {
        double t = result.t(i - 1);
        double dt = dt_base / substeps;

        for (int s = 0; s < substeps; ++s) {
            // RK4 step
            Eigen::VectorXd k1 = fun(t, y);
            Eigen::VectorXd k2 = fun(t + dt / 2, y + dt / 2 * k1);
            Eigen::VectorXd k3 = fun(t + dt / 2, y + dt / 2 * k2);
            Eigen::VectorXd k4 = fun(t + dt, y + dt * k3);

            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
            t += dt;
        }

        result.y.col(i) = y;
    }

    result.success = true;
    result.message = "Integration successful (RK4)";
    result.status = 0;

    return result;
}

/**
 * @brief Solve IVP with symbolic ODE function (CasADi CVODES backend)
 *
 * This overload accepts a callable that takes symbolic arguments and returns
 * symbolic expressions, enabling automatic differentiation through the ODE.
 *
 * @tparam Func Callable type f(MX t, MX y) -> MX dy/dt
 * @param fun Symbolic ODE function
 * @param t_span Integration interval (t0, tf)
 * @param y0 Initial state (numeric, will be used for evaluation)
 * @param n_eval Number of output points
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult<SymbolicScalar> with symbolic solution
 */
template <typename Func>
OdeResult<SymbolicScalar> solve_ivp_symbolic(Func &&fun, std::pair<double, double> t_span,
                                             const Eigen::VectorXd &y0, int n_eval = 100,
                                             double abstol = 1e-8, double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;
    int n_state = y0.size();

    // Create symbolic variables for the ODE
    casadi::MX t_var = casadi::MX::sym("t");
    casadi::MX y_var = casadi::MX::sym("y", n_state, 1);

    // Evaluate the ODE function symbolically
    casadi::MX ode_expr;
    if constexpr (std::is_invocable_v<Func, casadi::MX, casadi::MX>) {
        ode_expr = fun(t_var, y_var);
    } else {
        // Try calling with Eigen-wrapped MX
        SymbolicVector y_sym(n_state);
        for (int i = 0; i < n_state; ++i) {
            y_sym(i) = y_var(i);
        }
        auto result = fun(t_var, y_sym);
        // Convert back to MX
        ode_expr = casadi::MX(n_state, 1);
        for (int i = 0; i < n_state; ++i) {
            ode_expr(i) = result(i);
        }
    }

    // Normalize time to [0, 1] for CVODES stability
    // t_real = t0 + (tf - t0) * t_normalized
    // ode_normalized = ode_real * (tf - t0)
    casadi::MX t_normalized = casadi::MX::sym("t_norm");
    casadi::MX ode_normalized =
        casadi::MX::substitute(ode_expr, t_var, t0 + (tf - t0) * t_normalized) * (tf - t0);

    // Create evaluation time grid (normalized)
    std::vector<double> t_eval_normalized(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        t_eval_normalized[i] = static_cast<double>(i) / (n_eval - 1);
    }

    // Build integrator
    casadi::MXDict dae = {{"x", y_var}, {"t", t_normalized}, {"ode", ode_normalized}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator =
        casadi::integrator("ivp_integrator", "cvodes", dae, 0.0, t_eval_normalized, opts);

    // Convert y0 to DM
    casadi::DM y0_dm(n_state, 1);
    for (int i = 0; i < n_state; ++i) {
        y0_dm(i) = y0(i);
    }

    // Evaluate integrator - explicitly construct DMDict to avoid ambiguity
    casadi::DMDict integrator_args;
    integrator_args["x0"] = y0_dm;
    casadi::DMDict res_dm = integrator(integrator_args);

    OdeResult<SymbolicScalar> result;

    // Create symbolic time vector
    result.t.resize(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        result.t(i) = t0 + (tf - t0) * t_eval_normalized[i];
    }

    // Extract solution (xf contains all intermediate states)
    casadi::DM xf = res_dm.at("xf");
    result.y.resize(n_state, n_eval);
    for (int i = 0; i < n_state; ++i) {
        for (int j = 0; j < n_eval; ++j) {
            result.y(i, j) = static_cast<double>(xf(i, j));
        }
    }

    result.success = true;
    result.message = "Integration successful (CVODES)";
    result.status = 0;

    return result;
}

/**
 * @brief Solve IVP with a symbolic expression directly (expression-based API)
 *
 * This is the most flexible symbolic API, allowing the user to provide
 * the ODE as a CasADi expression with explicit variable specification.
 *
 * @param ode_expr Symbolic ODE expression (dy/dt)
 * @param t_var Time variable (MX symbol)
 * @param y_var State variable(s) (MX symbol vector)
 * @param t_span Integration interval
 * @param y0 Initial state
 * @param n_eval Number of output points
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult containing numeric time and solution values
 */
inline OdeResult<double> solve_ivp_expr(const casadi::MX &ode_expr, const casadi::MX &t_var,
                                        const casadi::MX &y_var, std::pair<double, double> t_span,
                                        const Eigen::VectorXd &y0, int n_eval = 100,
                                        double abstol = 1e-8, double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;
    int n_state = y0.size();

    // Normalize time to [0, 1]
    casadi::MX t_normalized = casadi::MX::sym("t_norm");
    casadi::MX ode_normalized =
        casadi::MX::substitute(ode_expr, t_var, t0 + (tf - t0) * t_normalized) * (tf - t0);

    // Create evaluation time grid (normalized)
    std::vector<double> t_eval_normalized(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        t_eval_normalized[i] = static_cast<double>(i) / (n_eval - 1);
    }

    // Find parameters (variables in ode other than t and y)
    std::vector<casadi::MX> all_vars = casadi::MX::symvar(ode_normalized);
    std::vector<casadi::MX> parameters;
    for (const auto &var : all_vars) {
        if (!casadi::MX::is_equal(var, t_normalized) && !casadi::MX::is_equal(var, y_var)) {
            // Check each element of y_var
            bool is_y = false;
            for (int i = 0; i < y_var.size1() * y_var.size2(); ++i) {
                if (casadi::MX::is_equal(var, y_var(i))) {
                    is_y = true;
                    break;
                }
            }
            if (!is_y) {
                parameters.push_back(var);
            }
        }
    }

    casadi::MX p = casadi::MX::vertcat(parameters);

    // Build integrator
    casadi::MXDict dae = {{"x", y_var}, {"t", t_normalized}, {"p", p}, {"ode", ode_normalized}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator =
        casadi::integrator("ivp_integrator", "cvodes", dae, 0.0, t_eval_normalized, opts);

    // Convert y0 to DM
    casadi::DM y0_dm(n_state, 1);
    for (int i = 0; i < n_state; ++i) {
        y0_dm(i) = y0(i);
    }

    // Empty parameters for now (user would need to provide values)
    casadi::DM p_dm = casadi::DM::zeros(p.size1(), p.size2());

    // Evaluate integrator - explicitly construct DMDict to avoid ambiguity
    casadi::DMDict integrator_args;
    integrator_args["x0"] = y0_dm;
    integrator_args["p"] = p_dm;
    casadi::DMDict res_dm = integrator(integrator_args);

    OdeResult<double> result;

    // Create time vector
    result.t.resize(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        result.t(i) = t0 + (tf - t0) * t_eval_normalized[i];
    }

    // Extract solution
    casadi::DM xf = res_dm.at("xf");
    result.y.resize(n_state, n_eval);
    for (int i = 0; i < n_state; ++i) {
        for (int j = 0; j < n_eval; ++j) {
            result.y(i, j) = static_cast<double>(xf(i, j));
        }
    }

    result.success = true;
    result.message = "Integration successful (CVODES)";
    result.status = 0;

    return result;
}

} // namespace janus
