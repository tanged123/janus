#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <janus/math/Linalg.hpp>
#include <vector>

// Helper to evaluate 0-argument CasADi MX to double
inline double eval_scalar(const casadi::MX &x) {
    try {
        casadi::Function f("f", std::vector<casadi::MX>{}, std::vector<casadi::MX>{x});
        auto res = f(std::vector<casadi::DM>{});
        return static_cast<double>(res[0]);
    } catch (const std::exception &e) {
        std::cerr << "CasADi eval failed: " << e.what() << std::endl;
        throw;
    }
}

// Helper to evaluate CasADi MX to Eigen Matrix
inline Eigen::MatrixXd eval_matrix(const casadi::MX &x) {
    casadi::Function f("f", std::vector<casadi::MX>{}, std::vector<casadi::MX>{x});
    auto res = f(std::vector<casadi::DM>{});
    casadi::DM res_dm = res[0];

    Eigen::MatrixXd mat(res_dm.size1(), res_dm.size2());
    // Copy data
    for (int i = 0; i < res_dm.size1(); ++i) {
        for (int j = 0; j < res_dm.size2(); ++j) {
            mat(i, j) = static_cast<double>(res_dm(i, j));
        }
    }
    return mat;
}

// Overload for Eigen matrix of MX
template <typename Derived>
inline Eigen::MatrixXd eval_matrix(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return eval_matrix(janus::to_mx(x));
    } else if constexpr (std::is_arithmetic_v<Scalar>) {
        return x.template cast<double>();
    } else {
        // Fallback or error
        return eval_matrix(janus::to_mx(x));
    }
}

// Overload for scalar MX in eval_scalar just in case
inline double eval_scalar(const double &x) { return x; }
