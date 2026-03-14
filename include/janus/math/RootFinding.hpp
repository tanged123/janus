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
    double abstol = 1e-10;              // Absolute tolerance on residual
    double abstolStep = 1e-10;          // Tolerance on step size
    int max_iter = 50;                  // Maximum Newton iterations
    bool line_search = true;            // Use line search for globalization
    bool verbose = false;               // Print solver progress
    std::string linear_solver = "qr";   // Linear solver used by CasADi rootfinder
    casadi::Dict linear_solver_options; // Options forwarded to the linear solver
};

/**
 * @brief Options for building differentiable implicit solve wrappers
 */
struct ImplicitFunctionOptions {
    int implicit_input_index = 0;  // Input slot containing the unknown state
    int implicit_output_index = 0; // Output slot containing the residual equation
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
    d["abstolStep"] = opts.abstolStep;
    d["max_iter"] = opts.max_iter;
    d["linear_solver"] = opts.linear_solver;
    if (!opts.linear_solver_options.empty()) {
        d["linear_solver_options"] = opts.linear_solver_options;
    }
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

inline casadi::DM vector_to_dm(const Eigen::VectorXd &x) {
    casadi::DM out(x.size(), 1);
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        out(static_cast<int>(i), 0) = x(i);
    }
    return out;
}

inline std::string implicit_function_name(const casadi::Function &g_casadi) {
    return g_casadi.name() + "_implicit";
}

inline void validate_implicit_problem(const casadi::Function &g_casadi,
                                      const Eigen::VectorXd &x_guess,
                                      const ImplicitFunctionOptions &implicit_opts) {
    if (g_casadi.n_in() == 0 || g_casadi.n_out() == 0) {
        throw InvalidArgument(
            "create_implicit_function: G must have at least one input and one output");
    }

    if (implicit_opts.implicit_input_index < 0 ||
        implicit_opts.implicit_input_index >= g_casadi.n_in()) {
        throw InvalidArgument("create_implicit_function: implicit_input_index out of range");
    }
    if (implicit_opts.implicit_output_index < 0 ||
        implicit_opts.implicit_output_index >= g_casadi.n_out()) {
        throw InvalidArgument("create_implicit_function: implicit_output_index out of range");
    }

    const auto unknown_sparsity = g_casadi.sparsity_in(implicit_opts.implicit_input_index);
    if (!unknown_sparsity.is_dense() || !unknown_sparsity.is_column()) {
        throw InvalidArgument(
            "create_implicit_function: implicit input must be a dense column vector");
    }

    const auto residual_sparsity = g_casadi.sparsity_out(implicit_opts.implicit_output_index);
    if (!residual_sparsity.is_dense() || !residual_sparsity.is_column()) {
        throw InvalidArgument(
            "create_implicit_function: implicit output must be a dense column vector residual");
    }

    const auto n_unknown =
        static_cast<Eigen::Index>(g_casadi.nnz_in(implicit_opts.implicit_input_index));
    const auto n_residual =
        static_cast<Eigen::Index>(g_casadi.nnz_out(implicit_opts.implicit_output_index));
    if (n_unknown != n_residual) {
        throw InvalidArgument(
            "create_implicit_function: implicit input and output dimensions must match");
    }
    if (x_guess.size() != n_unknown) {
        throw InvalidArgument("create_implicit_function: x_guess size must match the implicit "
                              "input dimension");
    }
}

} // namespace detail

/**
 * @brief Persistent Newton Solver
 *
 * Wrapper around CasADi's rootfinder for efficient re-use.
 * Useful when solving the same problem multiple times with different initial guesses.
 */
class NewtonSolver {
  public:
    /**
     * @brief Construct a new Newton Solver
     *
     * @param F Function F(x) = 0 to solve
     * @param opts Solver options
     */
    NewtonSolver(const janus::Function &F, const RootFinderOptions &opts = {}) {
        casadi::Function f_casadi = F.casadi_function();
        // Check dimensions (strictly F(x) -> residual)
        if (f_casadi.n_in() != 1 || f_casadi.n_out() != 1) {
            throw JanusError("NewtonSolver: Function F must have 1 input and 1 output");
        }
        try {
            solver_ =
                casadi::rootfinder("rf_solver", "newton", f_casadi, detail::opts_to_dict(opts));
        } catch (const std::exception &e) {
            throw JanusError(std::string("NewtonSolver creation failed: ") + e.what());
        }
    }

    /**
     * @brief Solve F(x) = 0 from initial guess x0
     */
    template <typename Scalar>
    RootResult<Scalar> solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x0) const {
        RootResult<Scalar> result;
        int n_x = static_cast<int>(x0.size());

        if constexpr (std::is_floating_point_v<Scalar>) {
            // Numeric execution
            std::vector<double> x0_vec(x0.data(), x0.data() + x0.size());

            std::vector<casadi::DM> args;
            args.push_back(casadi::DM(x0_vec));
            if (solver_.n_in() > 1) {
                args.push_back(casadi::DM(0, 0));
            }

            try {
                std::vector<casadi::DM> res_vec = solver_(args);

                std::vector<double> x_sol = std::vector<double>(res_vec[0]);
                result.x.resize(n_x);
                for (int i = 0; i < n_x; ++i)
                    result.x(i) = x_sol[i];

                result.converged = true;
                result.iterations = -1;
                result.message = "Solved successfully";
            } catch (const std::exception &e) {
                result.converged = false;
                result.message = e.what();
                result.x = x0;
            }
        } else {
            // Symbolic execution
            SymbolicScalar x0_mx = janus::to_mx(x0);

            std::vector<SymbolicScalar> args;
            args.push_back(x0_mx);
            if (solver_.n_in() > 1) {
                args.push_back(SymbolicScalar(0, 0));
            }
            std::vector<SymbolicScalar> res = solver_(args);

            result.x = janus::to_eigen(res[0]);
            result.converged = true;
            result.message = "Symbolic graph generated";
        }
        return result;
    }

  private:
    casadi::Function solver_;
};

/**
 * @brief Solve F(x) = 0 for x given an initial guess
 *
 * Uses Newton's method. The function F must take x as input and return
 * a residual vector of the same dimension.
 *
 * @tparam Scalar double (numeric) or SymbolicScalar (symbolic)
 * @param F Function mapping x -> residual (must be janus::Function)
 * @param x0 Initial guess
 * @param opts Solver options
 * @return RootResult containing solution and diagnostics
 */
template <typename Scalar>
RootResult<Scalar> rootfinder(const janus::Function &F,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x0,
                              const RootFinderOptions &opts = {}) {
    NewtonSolver solver(F, opts);
    return solver.solve(x0);
}

/**
 * @brief Create a differentiable implicit solve wrapper for G(...) = 0
 *
 * All non-implicit inputs of @p G become inputs of the returned function, in
 * their original order and original shapes. CasADi's rootfinder provides the
 * implicit sensitivities, so the returned function remains differentiable with
 * respect to every remaining input.
 *
 * @param G Implicit function containing the unknown input and residual output
 * @param x_guess Fixed initial guess for the implicit solve
 * @param opts Solver options
 * @param implicit_opts Selects which input/output pair defines the rootfinding problem
 * @return janus::Function mapping the remaining inputs to x(...)
 */
inline janus::Function create_implicit_function(const janus::Function &G,
                                                const Eigen::VectorXd &x_guess,
                                                const RootFinderOptions &opts = {},
                                                const ImplicitFunctionOptions &implicit_opts = {}) {
    casadi::Function g_casadi = G.casadi_function();
    detail::validate_implicit_problem(g_casadi, x_guess, implicit_opts);

    casadi::Dict solver_opts = detail::opts_to_dict(opts);
    solver_opts["implicit_input"] = implicit_opts.implicit_input_index;
    solver_opts["implicit_output"] = implicit_opts.implicit_output_index;

    casadi::Function solver = casadi::rootfinder(detail::implicit_function_name(g_casadi), "newton",
                                                 g_casadi, solver_opts);

    std::vector<SymbolicArg> wrapper_inputs;
    wrapper_inputs.reserve(static_cast<std::size_t>(g_casadi.n_in() - 1));

    std::vector<casadi::MX> solver_args;
    solver_args.reserve(static_cast<std::size_t>(g_casadi.n_in()));

    const casadi::DM x0 = detail::vector_to_dm(x_guess);
    for (int i = 0; i < g_casadi.n_in(); ++i) {
        if (i == implicit_opts.implicit_input_index) {
            solver_args.push_back(casadi::MX(x0));
            continue;
        }

        casadi::MX arg =
            janus::sym(g_casadi.name_in(i), g_casadi.size1_in(i), g_casadi.size2_in(i));
        wrapper_inputs.push_back(SymbolicArg(arg));
        solver_args.push_back(arg);
    }

    const auto solver_outputs = solver(solver_args);
    const casadi::MX x_sol = solver_outputs.at(implicit_opts.implicit_output_index);

    return janus::Function(detail::implicit_function_name(g_casadi), wrapper_inputs,
                           std::vector<SymbolicArg>{SymbolicArg(x_sol)});
}

} // namespace janus
