#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace janus {

// --- Conversion Helpers ---

/**
 * @brief Convert Eigen matrix of MX to CasADi MX
 * @tparam Derived Eigen matrix type
 * @param e Input Eigen matrix
 * @return CasADi MX (dense)
 */
template <typename Derived> casadi::MX to_mx(const Eigen::MatrixBase<Derived> &e) {
    if (e.size() == 0)
        return casadi::MX(e.rows(), e.cols());

    // Create an MX of correct shape
    casadi::MX m(e.rows(), e.cols());
    // Fill it element-wise
    for (Eigen::Index i = 0; i < e.rows(); ++i) {
        for (Eigen::Index j = 0; j < e.cols(); ++j) {
            m(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
        }
    }
    return m;
}

/**
 * @brief Convert CasADi MX to Eigen matrix of MX
 * @param m Input CasADi MX
 * @return Eigen matrix (dynamic size)
 */
inline Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> to_eigen(const casadi::MX &m) {
    Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> e(m.size1(), m.size2());
    for (int i = 0; i < m.size1(); ++i) {
        for (int j = 0; j < m.size2(); ++j) {
            e(i, j) = m(i, j);
        }
    }
    return e;
}

// --- solve(A, b) ---
/**
 * @brief Solves linear system Ax = b
 * Uses QR decomposition for numeric types, and symbolic solve for CasADi types.
 *
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
template <typename DerivedA, typename DerivedB>
auto solve(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;

    if constexpr (std::is_floating_point_v<Scalar>) {
        // Numeric: Use reliable QR solver
        return A.colPivHouseholderQr().solve(b).eval();
    } else {
        // Symbolic: Use casadi::solve
        casadi::MX A_mx = to_mx(A);
        casadi::MX b_mx = to_mx(b);
        // casadi::solve(A, b) returns MX
        casadi::MX x_mx = casadi::MX::solve(A_mx, b_mx);
        return to_eigen(x_mx);
    }
}

// --- norm(x) ---
/**
 * @brief Computes L2 norm of a vector
 * @param x Input vector
 * @return L2 norm
 */
template <typename Derived> auto norm(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return x.norm();
    } else {
        return norm_2(to_mx(x));
    }
}

// --- outer(x, y) ---
/**
 * @brief Computes outer product x * y^T
 * @param x First vector
 * @param y Second vector
 * @return Outer product matrix
 */
template <typename DerivedX, typename DerivedY>
auto outer(const Eigen::MatrixBase<DerivedX> &x, const Eigen::MatrixBase<DerivedY> &y) {
    // Eigen's outer product works efficiently for both numeric and symbolic scalars
    // because MX * MX (scalar mult) creates a standard multiplication node.
    return x * y.transpose();
}

// --- Dot Product ---
template <typename DerivedA, typename DerivedB>
auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    return a.dot(b);
}

// --- Cross Product ---
template <typename DerivedA, typename DerivedB>
auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    // Manual implementation to support Dynamic vectors (Eigen::cross requires fixed size 3)
    using Scalar = typename DerivedA::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res(3);
    res(0) = a(1) * b(2) - a(2) * b(1);
    res(1) = a(2) * b(0) - a(0) * b(2);
    res(2) = a(0) * b(1) - a(1) * b(0);
    return res;
}

// --- Inverse ---
/**
 * @brief Computes matrix inverse
 * @param A Input matrix
 * @return Inverse of A
 */
template <typename Derived> auto inv(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.inverse().eval();
    } else {
        casadi::MX A_mx = to_mx(A);
        casadi::MX inv_mx = inv(A_mx);
        return to_eigen(inv_mx);
    }
}

// --- Determinant ---
/**
 * @brief Computes matrix determinant
 * @param A Input matrix
 * @return Determinant of A
 */
template <typename Derived> auto det(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.determinant();
    } else {
        casadi::MX A_mx = to_mx(A);
        return det(A_mx);
    }
}

} // namespace janus
