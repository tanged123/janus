#pragma once

/**
 * @file using.hpp
 * @brief Convenience header bringing common Janus symbols into scope
 *
 * Include this header for cleaner code without explicit janus:: prefixes.
 *
 * @code
 * #include <janus/using.hpp>
 *
 * auto x = sym("x");
 * auto y = sin(x) + pow(x, 2);
 * auto z = where(x > 0, x, -x);
 * @endcode
 *
 * @warning This header pollutes the global namespace. Use with care in
 * library code - prefer explicit janus:: prefixes there.
 */

#include "janus.hpp"

// Bring common Janus symbols into the global namespace
using janus::sym;
using janus::sym_vec;
using janus::sym_vec_pair;
using janus::sym_vector;

// Type aliases
using janus::DenseLinearSolver;
using janus::IterativeKrylovSolver;
using janus::IterativePreconditioner;
using janus::LinearSolveBackend;
using janus::LinearSolvePolicy;
using janus::NumericMatrix;
using janus::NumericScalar;
using janus::NumericVector;
using janus::PolynomialChaosBasis;
using janus::PolynomialChaosBasisOptions;
using janus::PolynomialChaosDimension;
using janus::PolynomialChaosFamily;
using janus::PolynomialChaosTerm;
using janus::PolynomialChaosTruncation;
using janus::SmolyakQuadratureOptions;
using janus::SparseDirectLinearSolver;
using janus::StochasticQuadratureGrid;
using janus::StochasticQuadratureRule;
using janus::StructuralDiagnosticsOptions;
using janus::StructuralDiagnosticsReport;
using janus::StructuralSensitivityOptions;
using janus::StructuralSensitivityReport;
using janus::SymbolicMatrix;
using janus::SymbolicScalar;
using janus::SymbolicVector;
using janus::UnivariateQuadratureRule;

// Conversion helpers
using janus::as_mx;
using janus::as_vector;
using janus::to_eigen;
using janus::to_mx;

// Structural diagnostics
using janus::analyze_structural_diagnostics;
using janus::analyze_structural_identifiability;
using janus::analyze_structural_observability;

// Math functions
using janus::abs;
using janus::acos;
using janus::asin;
using janus::atan;
using janus::atan2;
using janus::ceil;
using janus::cos;
using janus::cosh;
using janus::exp;
using janus::fabs;
using janus::floor;
using janus::fmax;
using janus::fmin;
using janus::log;
using janus::log10;
using janus::pow;
using janus::sin;
using janus::sinh;
using janus::sqrt;
using janus::tan;
using janus::tanh;

// Control flow
using janus::where;

// Calculus
using janus::gradient;
using janus::hermite_dimension;
using janus::hessian;
using janus::hessian_vector_product;
using janus::jacobi_dimension;
using janus::jacobian;
using janus::lagrangian_hessian_vector_product;
using janus::laguerre_dimension;
using janus::legendre_dimension;
using janus::pce_mean;
using janus::pce_polynomial;
using janus::pce_projection_coefficients;
using janus::pce_regression_coefficients;
using janus::pce_squared_norm;
using janus::pce_variance;
using janus::smolyak_sparse_grid;
using janus::stochastic_quadrature_level;
using janus::stochastic_quadrature_rule;
using janus::tensor_product_quadrature;

// Spacing
using janus::linspace;
