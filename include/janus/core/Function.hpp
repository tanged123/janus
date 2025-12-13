#pragma once

#include "JanusTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <string>
#include <vector>

#include <atomic>

namespace janus {

/**
 * @brief Wrapper around casadi::Function to provide Janus-native types (Eigen)
 */
class Function {
  public:
    Function(const std::string& name, 
             const std::vector<SymbolicArg>& inputs, 
             const std::vector<SymbolicArg>& outputs)
        : fn_(name, convert_args(inputs), convert_args(outputs)) {}

    /**
     * @brief Constructor with auto-generated name
     */
    Function(const std::vector<SymbolicArg>& inputs, 
             const std::vector<SymbolicArg>& outputs)
        : fn_(generate_unique_name(), convert_args(inputs), convert_args(outputs)) {}

private:
    static std::string generate_unique_name() {
        static std::atomic<uint64_t> counter{0};
        return "janus_fn_" + std::to_string(counter.fetch_add(1));
    }

    static std::vector<SymbolicScalar> convert_args(const std::vector<SymbolicArg>& args) {
        std::vector<SymbolicScalar> ret;
        ret.reserve(args.size());
        for (const auto& arg : args) {
            ret.push_back(arg.get()); // or implicit cast
        }
        return ret;
    }

public:
    /**
     * @brief Evaluate function with arbitrary arguments (scalars or Eigen matrices)
     */
    template <typename... Args>
    std::vector<Eigen::MatrixXd> operator()(Args&&... args) const {
        std::vector<casadi::DM> dm_args;
        dm_args.reserve(sizeof...(args));
        (dm_args.push_back(to_dm(std::forward<Args>(args))), ...);
        
        auto res_dm = fn_(dm_args);
        return to_eigen_vector(res_dm);
    }

private:
    // Helper: Convert scalar (double/int) to DM
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>>>
    casadi::DM to_dm(T val) const {
        return casadi::DM(static_cast<double>(val));
    }

    // Helper: Convert Eigen Matrix to DM
    template <typename Derived>
    casadi::DM to_dm(const Eigen::MatrixBase<Derived>& val) const {
        casadi::DM m(val.rows(), val.cols());
        for(Eigen::Index i=0; i<val.rows(); ++i) {
            for(Eigen::Index j=0; j<val.cols(); ++j) {
                m(i, j) = val(i, j);
            }
        }
        return m;
    }

  public:
    /**
     * @brief Evaluate function with vector of arguments (non-const lvalue)
     * Resolves ambiguity with variadic template
     */
    std::vector<Eigen::MatrixXd> operator()(std::vector<double>& args) const {
        return operator()(const_cast<const std::vector<double>&>(args));
    }

    /**
     * @brief Evaluate function with vector of arguments (const lvalue)
     */
    std::vector<Eigen::MatrixXd> operator()(const std::vector<double> &args) const {
        std::vector<casadi::DM> dm_args;
        dm_args.reserve(args.size());
        for (double val : args) {
            dm_args.push_back(casadi::DM(val));
        }

        auto res_dm = fn_(dm_args);
        return to_eigen_vector(res_dm);
    }

    // Allow evaluation with Eigen matrices if needed (phase 2 extension)

  private:
    casadi::Function fn_;

    std::vector<Eigen::MatrixXd> to_eigen_vector(const std::vector<casadi::DM> &dms) const {
        std::vector<Eigen::MatrixXd> ret;
        ret.reserve(dms.size());
        for (const auto &dm : dms) {
            Eigen::MatrixXd mat(dm.size1(), dm.size2());
            // Safe dense copy
            // CasADi is column-major, Eigen is column-major by default.
            std::vector<double> elements = static_cast<std::vector<double>>(dm);

            // Map directly if possible, or memcpy
            // elements gives column-major data
            for (Eigen::Index j = 0; j < mat.cols(); ++j) {
                for (Eigen::Index i = 0; i < mat.rows(); ++i) {
                    mat(i, j) = elements[j * mat.rows() + i];
                }
            }
            ret.push_back(mat);
        }
        return ret;
    }
};

} // namespace janus
