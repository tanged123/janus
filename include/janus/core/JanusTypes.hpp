#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace janus {
// Numeric Backend
using NumericScalar = double;
using NumericMatrix = Eigen::MatrixXd;

// Symbolic Backend
using SymbolicScalar = casadi::MX;
// Note: Eigen wrapping of CasADi requires careful handling (Phase 2),
// for Phase 1 start with scalar types.
} // namespace janus
