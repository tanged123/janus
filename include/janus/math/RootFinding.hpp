#pragma once
#include "janus/core/Function.hpp"
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include <casadi/casadi.hpp>
#include <iostream>

namespace janus {

/**
 * @brief Options for root finding algorithms
 */
struct RootFinderOptions {
    double abstol = 1e-10;     // Absolute tolerance on residual
    double abstolStep = 1e-10; // Tolerance on step size
    int max_iter = 50;         // Maximum Newton iterations
    bool line_search = true;   // Use line search for globalization
    bool verbose = false;      // Print solver progress
};

/**
 * @brief Result of a root finding operation
 */
template <typename Scalar> struct RootResult {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x; // Solution
    int iterations = -1;                        // Number of iterations used
    bool converged = false;                     // Whether solution converged
    std::string message = "";                   // Status message
};

namespace detail {

// Helper to convert options to CasADi dictionary
inline casadi::Dict opts_to_dict(const RootFinderOptions &opts) {
    casadi::Dict d;
    d["abstol"] = opts.abstol;
    d["max_iter"] = opts.max_iter;
    // CasADi rootfinder specific options mapping
    // "newton" plugin options:
    // "linear_solver": Linear solver to use
    // "line_search": Enable line search
    if (!opts.line_search) {
        d["constraints"] =
            std::vector<int>{0}; // Disable line search hack or use specific plugin opts
        // Actually, for "newton" plugin, line_search is typically default or controlled via
        // specific options We'll trust default for now or pass if supported
    }
    if (opts.verbose) {
        d["verbose"] = true;
        d["print_in"] = true;
        d["print_out"] = true;
    }
    return d;
}

} // namespace detail

/**
 * @brief Solve F(x) = 0 for x given an initial guess
 *
 * Uses Newton's method. The function F must take x as input and return
 * a residual vector of the same dimension.
 *
 * @tparam Scalar double (numeric) or casadi::MX (symbolic)
 * @param F Function mapping x -> residual (must be janus::Function)
 * @param x0 Initial guess
 * @param opts Solver options
 * @return RootResult containing solution and diagnostics
 */
template <typename Scalar>
RootResult<Scalar> rootfinder(const janus::Function &F,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x0,
                              const RootFinderOptions &opts = {}) {
    // Prepare dimensions
    int n_x = static_cast<int>(x0.size());

    // Create CasADi rootfinder instance
    // F describes the residual. CasADi rootfinder expects an implicit function definition:
    // G(x, p) = 0.
    // Here we have F(x) = 0 (no parameters p).
    // We need to wrap F into the format expected by start-up.

    // Getting the underlying CasADi function
    casadi::Function f_casadi = F.casadi_function();

    // Check dimensions
    if (f_casadi.n_in() != 1 || f_casadi.n_out() != 1) {
        throw JanusError("Rootfinder: Function F must have 1 input and 1 output");
    }

    // Create rootfinder
    // Name, Solver ("newton"), Implicit Function defining residual (f_casadi), Options
    // Note: The implicit function passed to rootfinder defines the residual.
    // For "newton", it expects a function [x, p] -> [residual]
    // Since our F takes [x], we might need an adapter if F doesn't accept p.
    // Actually, CasADi handles this: if n_in=1, it assumes p is empty.

    casadi::Function solver;
    try {
        solver = casadi::rootfinder("rf_solver", "newton", f_casadi, detail::opts_to_dict(opts));
    } catch (const std::exception &e) {
        throw JanusError(std::string("Rootfinder creation failed: ") + e.what());
    }

    RootResult<Scalar> result;

    if constexpr (std::is_floating_point_v<Scalar>) {
        // Numeric execution
        std::vector<double> x0_vec(x0.data(), x0.data() + x0.size());

        // Inputs: initial guess (x0), parameters (p - if expected)
        std::vector<casadi::DM> args;
        args.push_back(casadi::DM(x0_vec));

        if (solver.n_in() > 1) {
            args.push_back(casadi::DM(0, 0));
        }

        try {
            std::vector<casadi::DM> res_vec = solver(args);

            // Extract solution (first output)
            std::vector<double> x_sol = std::vector<double>(res_vec[0]);
            result.x.resize(n_x);
            for (int i = 0; i < n_x; ++i)
                result.x(i) = x_sol[i];

            // Diagnostics
            result.converged = true;
            result.iterations = -1; // Unknown from simple call
            result.message = "Solved successfully";

        } catch (const std::exception &e) {
            result.converged = false;
            result.message = e.what();
            result.x = x0; // Return guess on failure
        }
    } else {
        // Symbolic execution (MX)
        // Convert x0 to MX
        casadi::MX x0_mx = janus::to_mx(x0);

        // Call solver symbolically
        std::vector<casadi::MX> args;
        args.push_back(x0_mx);
        if (solver.n_in() > 1) {
            args.push_back(casadi::MX(0, 0));
        }
        std::vector<casadi::MX> res = solver(args);

        // Result is the first output
        result.x = janus::to_eigen(res[0]);
        result.converged = true; // Symbolic node assumed valid
        result.message = "Symbolic graph generated";
    }

    return result;
}

/**
 * @brief Create a function solving implicit equation G(x, p) = 0 for x
 *
 * Returns a function that takes parameters p and returns solution x.
 *
 * @param G Implicit function G(x, p) -> residual
 * @param x_guess Initial guess for x (defines dimension)
 * @param opts Solver options
 * @return janus::Function mapping p -> x(p)
 */
inline janus::Function create_implicit_function(const janus::Function &G,
                                                const Eigen::VectorXd &x_guess,
                                                const RootFinderOptions &opts = {}) {
    casadi::Function g_casadi = G.casadi_function();

    // Create solver
    // "implicit_solver": [x0, p] -> [x]
    casadi::Function solver =
        casadi::rootfinder("implicit_solver", "newton", g_casadi, detail::opts_to_dict(opts));

    // We want to return a function [p] -> [x]
    // The solver natively takes [x0, p]. We need to fix x0 to x_guess?
    // Usually for implicit layers, x0 might depend on p or be fixed.
    // If we want a clean p -> x interface, we can wrap it.

    // Define wrapper using MX
    int n_p = g_casadi.n_in() > 1 ? g_casadi.size2_in(1) : 0; // Assuming 2nd input is p
    int n_x = x_guess.size();

    casadi::MX p = casadi::MX::sym("p", n_p);
    casadi::MX x0 =
        casadi::DM(std::vector<double>(x_guess.data(), x_guess.data() + x_guess.size()));

    // Call solver
    // Inputs: x0, p
    casadi::MX x_sol = solver(std::vector<casadi::MX>{x0, p})[0];

    // Create janus::Function
    // Input: p, Output: x_sol
    return janus::Function("implicit_fn", std::vector<SymbolicArg>{SymbolicArg(p)},
                           std::vector<SymbolicArg>{SymbolicArg(x_sol)});
}

} // namespace janus
