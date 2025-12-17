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
using janus::NumericMatrix;
using janus::NumericScalar;
using janus::NumericVector;
using janus::SymbolicMatrix;
using janus::SymbolicScalar;
using janus::SymbolicVector;

// Conversion helpers
using janus::as_mx;
using janus::as_vector;
using janus::to_eigen;
using janus::to_mx;

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
using janus::hessian;
using janus::jacobian;

// Spacing
using janus::linspace;
