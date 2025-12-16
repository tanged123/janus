#pragma once

#include <string>

namespace janus {

/**
 * @brief Options for solving optimization problems
 *
 * Configures IPOPT solver behavior. Most users can use defaults.
 *
 * Usage (designated initializers - must follow declaration order):
 *   opti.solve({.max_iter = 500, .verbose = false});
 *
 * Usage (builder pattern - any order):
 *   opti.solve(OptiOptions{}.set_verbose(false).set_max_iter(500));
 */
struct OptiOptions {
    // Public fields (for designated initializers)
    int max_iter = 1000;                  ///< Maximum iterations
    double max_cpu_time = 1e20;           ///< Maximum solve time [seconds]
    double tol = 1e-8;                    ///< Convergence tolerance
    bool verbose = true;                  ///< Print IPOPT progress
    bool jit = false;                     ///< JIT compile expressions (experimental)
    bool detect_simple_bounds = false;    ///< Detect simple variable bounds
    std::string mu_strategy = "adaptive"; ///< Barrier parameter strategy

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
};

} // namespace janus
