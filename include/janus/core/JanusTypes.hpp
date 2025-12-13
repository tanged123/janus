#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace janus {

// --- Matrix Types ---
/**
 * Universal Matrix Template
 *
 * Provides a unified matrix type for both numeric and symbolic backends.
 * Uses dynamic sizing for maximum flexibility.
 */
template <typename Scalar>
using JanusMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Numeric Backend
using NumericScalar = double;
using NumericMatrix = JanusMatrix<NumericScalar>; // Equivalent to Eigen::MatrixXd

// Symbolic Backend
using SymbolicScalar = casadi::MX;
using SymbolicMatrix = JanusMatrix<SymbolicScalar>;

// --- Symbolic Variable Creation ---
inline SymbolicScalar sym(const std::string &name) { return casadi::MX::sym(name); }

inline SymbolicScalar sym(const std::string &name, int rows, int cols = 1) {
    return casadi::MX::sym(name, rows, cols);
}

} // namespace janus
