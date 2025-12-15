#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
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

/**
 * @brief Convert CasADi MX vector to SymbolicVector (Eigen container of MX)
 *
 * This unpacks a CasADi MX (which internally represents a vector) into an
 * Eigen container where each element is an individual MX scalar.
 * Useful for passing symbolic vectors to templated functions that expect
 * JanusVector<Scalar> types.
 *
 * @param m Input CasADi MX (column vector)
 * @return SymbolicVector (Eigen::Matrix<casadi::MX, Dynamic, 1>)
 */
inline SymbolicVector as_vector(const casadi::MX &m) {
    SymbolicVector v(m.size1());
    for (int i = 0; i < m.size1(); ++i) {
        v(i) = m(i);
    }
    return v;
}

// Backwards compatibility alias
inline SymbolicVector to_eigen_vec(const casadi::MX &m) { return as_vector(m); }

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
/**
 * @brief Computes 3D cross product
 * @param a First 3-element vector
 * @param b Second 3-element vector
 * @return Cross product vector
 * @throws InvalidArgument if vectors are not 3 elements
 */
template <typename DerivedA, typename DerivedB>
auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    if (a.size() != 3 || b.size() != 3) {
        throw InvalidArgument("cross: both vectors must have exactly 3 elements");
    }
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

// --- Inner Product ---
/**
 * @brief Computes inner product of two vectors (dot product)
 * @param x First vector
 * @param y Second vector
 * @return Inner product
 */
template <typename DerivedA, typename DerivedB>
auto inner(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    return janus::dot(a, b);
}

// --- Pseudo-Inverse ---
/**
 * @brief Computes Moore-Penrose pseudo-inverse
 * @param A Input matrix
 * @return Pseudo-inverse of A
 */
template <typename Derived> auto pinv(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.completeOrthogonalDecomposition().pseudoInverse();
    } else {
        casadi::MX A_mx = to_mx(A);
        casadi::MX pinv_mx = casadi::MX::pinv(A_mx);
        return to_eigen(pinv_mx);
    }
}

// --- Norm Extensions ---

enum class NormType { L1, L2, Inf, Frobenius };

/**
 * @brief Computes vector/matrix norm
 * @param x Input vector/matrix
 * @param type Norm type (L1, L2, Inf, Frobenius)
 * @return Norm value
 */
template <typename Derived>
auto norm(const Eigen::MatrixBase<Derived> &x, NormType type = NormType::L2) {
    using Scalar = typename Derived::Scalar;

    if constexpr (std::is_floating_point_v<Scalar>) {
        switch (type) {
        case NormType::L1:
            return x.template lpNorm<1>();
        case NormType::L2:
            return x.norm();
        case NormType::Inf:
            return x.template lpNorm<Eigen::Infinity>();
        case NormType::Frobenius:
            return x.norm(); // Frobenius equals L2 for vectors
        default:
            return x.norm();
        }
    } else {
        casadi::MX x_mx = to_mx(x);
        switch (type) {
        case NormType::L1:
            return casadi::MX::norm_1(x_mx);
        case NormType::L2:
            return casadi::MX::norm_2(x_mx);
        case NormType::Inf:
            return casadi::MX::norm_inf(x_mx);
        case NormType::Frobenius:
            return casadi::MX::norm_fro(x_mx);
        default:
            return casadi::MX::norm_2(x_mx);
        }
    }
}

// Backwards compatibility overload for just L2 norm (default handled above really, but if called
// without args, it works)

// --- Explicit 3x3 Symmetric Inverse (AeroSandbox Helper) ---
/**
 * @brief Explicit inverse of symmetric 3x3 matrix.
 * Returns tuple of elements: a11, a22, a33, a12, a23, a13
 */
template <typename T>
std::tuple<T, T, T, T, T, T> inv_symmetric_3x3_explicit(const T &m11, const T &m22, const T &m33,
                                                        const T &m12, const T &m23, const T &m13) {

    T det = m11 * (m33 * m22 - m23 * m23) - m12 * (m33 * m12 - m23 * m13) +
            m13 * (m23 * m12 - m22 * m13);

    T inv_det = 1.0 / det;

    T a11 = (m33 * m22 - m23 * m23) * inv_det;
    T a12 = (m13 * m23 - m33 * m12) * inv_det;
    T a13 = (m12 * m23 - m13 * m22) * inv_det;
    T a22 = (m33 * m11 - m13 * m13) * inv_det;
    T a23 = (m12 * m13 - m11 * m23) * inv_det;
    T a33 = (m11 * m22 - m12 * m12) * inv_det;

    return {a11, a22, a33, a12, a23, a13};
}

} // namespace janus
