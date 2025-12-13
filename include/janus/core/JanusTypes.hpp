#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace janus {

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

} // namespace janus
