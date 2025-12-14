#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <limits>

/**
 * @file JanusIO.hpp
 * @brief IO Utilities and Traits for Janus
 *
 * Provides:
 * 1. Helper functions for printing/displaying matrices with wrappers.
 * 2. Evaluation utilities (eval).
 *
 * Note: Eigen::NumTraits definitions are in JanusTypes.hpp usually.
 */

namespace janus {

/**
 * @brief Print a Matrix (Numeric or Symbolic) to stdout with a label
 */
template <typename Derived>
void print(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    std::cout << label << ":\n" << mat << "\n" << std::endl;
}

/**
 * @brief Deprecated alias for print
 */
template <typename Derived>
void disp(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    print(label, mat);
}

/**
 * @brief Evaluation helper for symbolic matrices
 * Evaluates a symbolic matrix to a numeric Eigen matrix.
 * Assumes the matrix contains no free variables (i.e. is constant).
 * Throws if evaluation fails.
 */
template <typename Derived> auto eval(const Eigen::MatrixBase<Derived> &mat) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        // Flatten to MX, evaluate, map back
        casadi::MX flat = casadi::MX::zeros(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                flat(i, j) = mat(i, j);
            }
        }

        try {
            casadi::Function f("f", {}, {flat});
            auto res = f(std::vector<casadi::DM>{});
            casadi::DM res_dm = res[0];

            Eigen::MatrixXd res_eigen(mat.rows(), mat.cols());
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    res_eigen(i, j) = static_cast<double>(res_dm(i, j));
                }
            }
            return res_eigen;
        } catch (const std::exception &e) {
            // If it fails (variables present), return the symbolic expression?
            // Or throw? The user likely expects a number.
            throw std::runtime_error("janus::eval failed (likely contains free variables): " +
                                     std::string(e.what()));
        }
    } else {
        return mat.eval();
    }
}

// Explicit overload for casadi::MX scalar
inline double eval(const casadi::MX &val) {
    try {
        casadi::Function f("f", {}, {val});
        auto res = f(std::vector<casadi::DM>{});
        casadi::DM res_dm = res[0];
        return static_cast<double>(res_dm);
    } catch (const std::exception &e) {
        throw std::runtime_error("janus::eval scalar failed: " + std::string(e.what()));
    }
}

// Overload for numeric types (passthrough)
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0> T eval(const T &val) {
    return val;
}

} // namespace janus
