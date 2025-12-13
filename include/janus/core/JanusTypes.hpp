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

// --- Symbolic Helper Types ---

/**
 * @brief Universal Symbolic Argument Wrapper
 * 
 * Allows automatic flattening of Eigen matrices (SymbolicMatrix) and Scalars (SymbolicScalar)
 * into a single casadi::MX type for function definitions and Jacobians.
 * Enables mixed initializer lists: {scalar_sym, matrix_sym}.
 */
class SymbolicArg {
public:
    // From Scalar (MX)
    SymbolicArg(const SymbolicScalar& s) : mx_(s) {}

    // From Matrix (Eigen<MX>)
    template <typename Derived>
    SymbolicArg(const Eigen::MatrixBase<Derived>& e) {
        if (e.size() == 0) {
            mx_ = casadi::MX(e.rows(), e.cols());
            return;
        }
        mx_ = casadi::MX(e.rows(), e.cols());
        for(Eigen::Index i=0; i<e.rows(); ++i) {
            for(Eigen::Index j=0; j<e.cols(); ++j) {
                mx_(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
            }
        }
    }

    // Implicit conversion to MX
    operator SymbolicScalar() const { return mx_; }
    SymbolicScalar get() const { return mx_; }

private:
    SymbolicScalar mx_;
};

} // namespace janus
