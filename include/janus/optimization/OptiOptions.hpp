#pragma once

#include <string>

namespace janus {

/**
 * @brief Options for solving optimization problems
 *
 * Configures IPOPT solver behavior. Most users can use defaults.
 */
struct OptiOptions {
    int max_iter = 1000;               ///< Maximum iterations
    double max_cpu_time = 1e20;        ///< Maximum solve time [seconds]
    double tol = 1e-8;                 ///< Convergence tolerance
    bool verbose = true;               ///< Print IPOPT progress
    bool jit = false;                  ///< JIT compile expressions (experimental)
    bool detect_simple_bounds = false; ///< Detect simple variable bounds

    // IPOPT-specific options (advanced)
    std::string mu_strategy = "adaptive"; ///< Barrier parameter strategy
};

} // namespace janus
