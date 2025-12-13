#pragma once

#include "JanusTypes.hpp"
#include <vector>
#include <string>
#include <vector>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include <atomic>

namespace janus {

/**
 * @brief Wrapper around casadi::Function to provide Janus-native types (Eigen)
 */
class Function {
public:
    Function(const std::string& name, 
             const std::vector<SymbolicScalar>& inputs, 
             const std::vector<SymbolicScalar>& outputs)
        : fn_(name, inputs, outputs) {}

    /**
     * @brief Constructor with auto-generated name
     */
    Function(const std::vector<SymbolicScalar>& inputs, 
             const std::vector<SymbolicScalar>& outputs)
        : fn_(generate_unique_name(), inputs, outputs) {}

private:
    static std::string generate_unique_name() {
        static std::atomic<uint64_t> counter{0};
        return "janus_fn_" + std::to_string(counter.fetch_add(1));
    }

public:
    /**
     * @brief Evaluate function with scalar (double) arguments
     */
    template <typename... Args>
    std::vector<Eigen::MatrixXd> operator()(Args... args) const {
        std::vector<casadi::DM> dm_args;
        (dm_args.push_back(casadi::DM(args)), ...);
        
        auto res_dm = fn_(dm_args);
        return to_eigen_vector(res_dm);
    }

    /**
     * @brief Evaluate function with vector of arguments
     */
    std::vector<Eigen::MatrixXd> operator()(const std::vector<double>& args) const {
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

    std::vector<Eigen::MatrixXd> to_eigen_vector(const std::vector<casadi::DM>& dms) const {
        std::vector<Eigen::MatrixXd> ret;
        ret.reserve(dms.size());
        for (const auto& dm : dms) {
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
