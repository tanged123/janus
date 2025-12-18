#pragma once

#include <casadi/casadi.hpp>
#include <string>

namespace janus {

/**
 * @brief Available NLP solvers
 *
 * Solvers are provided via CasADi's nlpsol interface.
 * IPOPT is always available. Others require separate installation.
 */
enum class Solver {
    IPOPT,  ///< Interior Point OPTimizer (default, always available)
    SNOPT,  ///< Sparse Nonlinear OPTimizer (requires license)
    QPOASES ///< QP solver for QP subproblems
};

// Forward declaration (implemented after solver_name)
inline bool solver_available(Solver solver);

/**
 * @brief Get the CasADi solver name string
 *
 * @param solver Solver enum value
 * @return Solver name as expected by CasADi nlpsol
 */
inline const char *solver_name(Solver solver) {
    switch (solver) {
    case Solver::SNOPT:
        return "snopt";
    case Solver::QPOASES:
        return "qpoases";
    case Solver::IPOPT:
    default:
        return "ipopt";
    }
}

/**
 * @brief Check if a solver is available in the current CasADi build
 *
 * Uses CasADi's nlpsol plugin system to check availability.
 *
 * @param solver Solver to check
 * @return true if solver plugin is loaded and usable
 */
inline bool solver_available(Solver solver) { return casadi::has_nlpsol(solver_name(solver)); }

/**
 * @brief SNOPT-specific solver options
 *
 * These map to SNOPT's solver parameters.
 * See SNOPT documentation for detailed descriptions.
 */
struct SNOPTOptions {
    int major_iterations_limit = 1000;         ///< Max major (outer) iterations
    int minor_iterations_limit = 500;          ///< Max minor (QP) iterations per major
    double major_optimality_tolerance = 1e-6;  ///< Optimality tolerance
    double major_feasibility_tolerance = 1e-6; ///< Constraint feasibility tolerance
    int print_level = 0;                       ///< 0=silent, 1=summary, 2+=detailed

    // Builder pattern setters
    SNOPTOptions &set_major_iterations_limit(int v) {
        major_iterations_limit = v;
        return *this;
    }
    SNOPTOptions &set_minor_iterations_limit(int v) {
        minor_iterations_limit = v;
        return *this;
    }
    SNOPTOptions &set_major_optimality_tolerance(double v) {
        major_optimality_tolerance = v;
        return *this;
    }
    SNOPTOptions &set_major_feasibility_tolerance(double v) {
        major_feasibility_tolerance = v;
        return *this;
    }
    SNOPTOptions &set_print_level(int v) {
        print_level = v;
        return *this;
    }
};

/**
 * @brief Options for solving optimization problems
 *
 * Configures solver behavior. Most users can use defaults.
 *
 * Usage (designated initializers - must follow declaration order):
 *   opti.solve({.max_iter = 500, .verbose = false});
 *
 * Usage (builder pattern - any order):
 *   opti.solve(OptiOptions{}.set_verbose(false).set_max_iter(500));
 *
 * Usage (alternative solver):
 *   opti.solve({.solver = janus::Solver::SNOPT});
 */
struct OptiOptions {
    // Solver selection
    Solver solver = Solver::IPOPT; ///< NLP solver to use
    SNOPTOptions snopt_opts;       ///< SNOPT-specific options (if solver=SNOPT)

    // General solver options (used by all solvers where applicable)
    int max_iter = 1000;                  ///< Maximum iterations
    double max_cpu_time = 1e20;           ///< Maximum solve time [seconds]
    double tol = 1e-8;                    ///< Convergence tolerance
    bool verbose = true;                  ///< Print solver progress
    bool jit = false;                     ///< JIT compile expressions (experimental)
    bool detect_simple_bounds = false;    ///< Detect simple variable bounds
    std::string mu_strategy = "adaptive"; ///< Barrier parameter strategy (IPOPT)

    // Builder-style setters (any order, chainable) - prefixed with set_
    OptiOptions &set_max_iter(int v) {
        max_iter = v;
        return *this;
    }
    OptiOptions &set_max_cpu_time(double v) {
        max_cpu_time = v;
        return *this;
    }
    OptiOptions &set_tol(double v) {
        tol = v;
        return *this;
    }
    OptiOptions &set_verbose(bool v) {
        verbose = v;
        return *this;
    }
    OptiOptions &set_jit(bool v) {
        jit = v;
        return *this;
    }
    OptiOptions &set_detect_simple_bounds(bool v) {
        detect_simple_bounds = v;
        return *this;
    }
    OptiOptions &set_mu_strategy(const std::string &v) {
        mu_strategy = v;
        return *this;
    }
    OptiOptions &set_solver(Solver v) {
        solver = v;
        return *this;
    }
    OptiOptions &set_snopt_opts(const SNOPTOptions &v) {
        snopt_opts = v;
        return *this;
    }
};

} // namespace janus
