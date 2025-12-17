#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace janus {

// --- Conversion Helpers ---
// (Moved to JanusTypes.hpp: to_mx, to_eigen, as_vector)

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
        // Symbolic: Use SymbolicScalar::solve
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar b_mx = to_mx(b);
        // SymbolicScalar::solve(A, b) returns SymbolicScalar
        SymbolicScalar x_mx = SymbolicScalar::solve(A_mx, b_mx);
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
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar inv_mx = inv(A_mx);
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
        SymbolicScalar A_mx = to_mx(A);
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
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar pinv_mx = SymbolicScalar::pinv(A_mx);
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
        SymbolicScalar x_mx = to_mx(x);
        switch (type) {
        case NormType::L1:
            return SymbolicScalar::norm_1(x_mx);
        case NormType::L2:
            return SymbolicScalar::norm_2(x_mx);
        case NormType::Inf:
            return SymbolicScalar::norm_inf(x_mx);
        case NormType::Frobenius:
            return SymbolicScalar::norm_fro(x_mx);
        default:
            return SymbolicScalar::norm_2(x_mx);
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

// --- Sparse Matrix Utilities ---
// NOTE: Sparse matrices are for NUMERIC data only. CasADi MX handles sparsity internally.
// For symbolic sparsity analysis, use janus::SparsityPattern.

/**
 * @brief Compile-time check for numeric scalar types
 *
 * This prevents confusing errors when someone tries to use sparse matrices
 * with symbolic types in templated code.
 */
template <typename Scalar>
constexpr bool is_numeric_scalar_v = std::is_floating_point_v<Scalar> || std::is_integral_v<Scalar>;

/**
 * @brief Create sparse matrix from triplets
 *
 * Triplets specify (row, col, value) entries. Duplicate entries are summed.
 *
 * @code
 * std::vector<janus::SparseTriplet> triplets;
 * triplets.emplace_back(0, 0, 1.0);
 * triplets.emplace_back(1, 1, 2.0);
 * auto sp = janus::sparse_from_triplets(2, 2, triplets);
 * @endcode
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param triplets Vector of (row, col, value) triplets
 * @return Compressed sparse matrix
 */
inline SparseMatrix sparse_from_triplets(int rows, int cols,
                                         const std::vector<SparseTriplet> &triplets) {
    SparseMatrix m(rows, cols);
    m.setFromTriplets(triplets.begin(), triplets.end());
    return m;
}

/**
 * @brief Convert dense matrix to sparse
 *
 * Elements with absolute value <= tol are treated as zero.
 *
 * @param dense Dense numeric matrix
 * @param tol Tolerance for zero (default 0.0 = exact zeros only)
 * @return Sparse matrix
 */
inline SparseMatrix to_sparse(const NumericMatrix &dense, double tol = 0.0) {
    std::vector<SparseTriplet> triplets;
    triplets.reserve(static_cast<size_t>(dense.size()) / 4); // Heuristic

    for (int j = 0; j < dense.cols(); ++j) {
        for (int i = 0; i < dense.rows(); ++i) {
            double val = dense(i, j);
            if (std::abs(val) > tol) {
                triplets.emplace_back(i, j, val);
            }
        }
    }

    return sparse_from_triplets(static_cast<int>(dense.rows()), static_cast<int>(dense.cols()),
                                triplets);
}

/**
 * @brief Convert sparse matrix to dense
 *
 * @param sparse Sparse matrix
 * @return Dense numeric matrix
 */
inline NumericMatrix to_dense(const SparseMatrix &sparse) { return NumericMatrix(sparse); }

/**
 * @brief Create identity sparse matrix
 *
 * @param n Size (n x n)
 * @return Sparse identity matrix
 */
inline SparseMatrix sparse_identity(int n) {
    SparseMatrix I(n, n);
    I.setIdentity();
    return I;
}

} // namespace janus
