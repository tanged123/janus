#pragma once

#include "janus/core/JanusTypes.hpp"
#include "janus/utils/JsonUtils.hpp"
#include <casadi/casadi.hpp>

namespace janus {

/**
 * @brief Solution wrapper for optimization results
 *
 * Provides type-safe extraction of optimized values from a solved
 * optimization problem.
 *
 * Example:
 *   auto sol = opti.solve();
 *   double x_opt = sol.value(x);
 *   NumericVector trajectory = sol.value(position);
 */
class OptiSol {
  public:
    /**
     * @brief Construct from CasADi solution
     * @param cas_sol CasADi OptiSol object
     */
    explicit OptiSol(casadi::OptiSol cas_sol) : cas_sol_(std::move(cas_sol)) {}

    /**
     * @brief Extract scalar value at optimum
     * @param var Symbolic scalar from opti.variable()
     * @return Optimized numeric value
     */
    double value(const SymbolicScalar &var) const {
        casadi::DM result = cas_sol_.value(var);
        return static_cast<double>(result);
    }

    /**
     * @brief Extract vector value at optimum
     * @param var Symbolic vector from opti.variable(n)
     * @return Optimized numeric vector
     */
    NumericVector value(const SymbolicVector &var) const {
        // Convert SymbolicVector (Eigen<MX>) to MX for CasADi
        casadi::MX mx_var = janus::to_mx(var);
        casadi::DM result = cas_sol_.value(mx_var);

        // Convert DM to NumericVector
        std::vector<double> elements = static_cast<std::vector<double>>(result);
        NumericVector vec(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) {
            vec(static_cast<Eigen::Index>(i)) = elements[i];
        }
        return vec;
    }

    /**
     * @brief Extract matrix value at optimum
     * @param var Symbolic matrix from opti operations
     * @return Optimized numeric matrix
     */
    NumericMatrix value(const SymbolicMatrix &var) const {
        casadi::MX mx_var = janus::to_mx(var);
        casadi::DM result = cas_sol_.value(mx_var);

        std::vector<double> elements = static_cast<std::vector<double>>(result);
        NumericMatrix mat(result.size1(), result.size2());

        // CasADi uses column-major (Fortran) order
        for (casadi_int j = 0; j < result.size2(); ++j) {
            for (casadi_int i = 0; i < result.size1(); ++i) {
                mat(i, j) = elements[j * result.size1() + i];
            }
        }
        return mat;
    }

    /**
     * @brief Get solver statistics
     * @return Dictionary of solver stats (iterations, timing, etc.)
     */
    casadi::Dict stats() const { return cas_sol_.stats(); }

    /**
     * @brief Get number of function evaluations
     */
    int num_function_evals() const {
        auto s = stats();
        if (s.count("n_call_nlp_f")) {
            return static_cast<int>(s.at("n_call_nlp_f"));
        }
        return -1;
    }

    /**
     * @brief Get number of iterations
     */
    int num_iterations() const {
        auto s = stats();
        if (s.count("iter_count")) {
            return static_cast<int>(s.at("iter_count"));
        }
        return -1;
    }

    /**
     * @brief Access underlying CasADi solution
     */
    const casadi::OptiSol &casadi_sol() const { return cas_sol_; }

    /**
     * @brief Save solution to JSON file
     *
     * @param filename Output filename (e.g. "sol.json")
     * @param named_vars Map of variable names to symbolic variables to save
     */
    void save(const std::string &filename,
              const std::map<std::string, SymbolicScalar> &named_vars) const {
        std::map<std::string, std::vector<double>> data;

        for (const auto &[name, var] : named_vars) {
            // Evaluate variable using helper
            double val = value(var);
            data[name] = {val};
        }

        janus::utils::write_json(filename, data);
    }

    /**
     * @brief Save solution map of vectors to JSON file
     */
    void save(const std::string &filename,
              const std::map<std::string, SymbolicVector> &named_vars) const {
        std::map<std::string, std::vector<double>> data;

        for (const auto &[name, var] : named_vars) {
            // Evaluate variable using helper
            NumericVector result = value(var);

            // Convert Eigen Vector to std::vector
            std::vector<double> elements(result.data(), result.data() + result.size());
            data[name] = elements;
        }

        janus::utils::write_json(filename, data);
    }

  private:
    casadi::OptiSol cas_sol_;
};

} // namespace janus
