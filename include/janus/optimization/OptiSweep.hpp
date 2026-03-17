#pragma once

#include "OptiSol.hpp"
#include "janus/core/JanusError.hpp"
#include <vector>

namespace janus {

/**
 * @brief Result of a parametric sweep
 *
 * Stores solutions obtained by solving an optimization problem
 * across a range of parameter values.
 *
 * @code
 * auto result = opti.solve_sweep(param, {0.1, 0.2, 0.3, 0.4, 0.5});
 * for (int i = 0; i < result.size(); ++i) {
 *     std::cout << "param=" << result.param_values[i]
 *               << " obj=" << result.objective(i) << "\n";
 * }
 * @endcode
 */
struct SweepResult {
    /// Parameter values that were swept
    std::vector<double> param_values;

    /// Solutions at each parameter value (only for converged points)
    std::vector<OptiSol> solutions;

    /// Number of iterations for each solve
    std::vector<int> iterations;

    /// Whether all solves converged successfully
    bool all_converged = true;

    /// Per-point convergence status (true = converged)
    std::vector<bool> converged;

    /// Error messages for failed points (empty string if converged)
    std::vector<std::string> errors;

    /// Number of sweep points
    size_t size() const { return param_values.size(); }

    /// Get objective value at sweep index.
    /// Requires the objective expression that was passed to minimize/maximize.
    double objective(size_t i, const SymbolicScalar &objective_expr) const {
        if (i >= solutions.size()) {
            throw InvalidArgument("SweepResult::objective: index out of range");
        }
        return solutions[i].value(objective_expr);
    }

    /// Extract values of a variable across all sweep points
    NumericVector values(const SymbolicScalar &var) const {
        NumericVector result(static_cast<int>(solutions.size()));
        for (size_t i = 0; i < solutions.size(); ++i) {
            result(static_cast<int>(i)) = solutions[i].value(var);
        }
        return result;
    }

    /// Extract values of a vector variable across all sweep points
    NumericMatrix values(const SymbolicVector &var) const {
        if (solutions.empty()) {
            return NumericMatrix(0, 0);
        }
        auto first = solutions[0].value(var);
        NumericMatrix result(first.size(), static_cast<int>(solutions.size()));
        result.col(0) = first;
        for (size_t i = 1; i < solutions.size(); ++i) {
            result.col(static_cast<int>(i)) = solutions[i].value(var);
        }
        return result;
    }
};

} // namespace janus
