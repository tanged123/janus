#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Arithmetic.hpp"
#include "janus/math/Logic.hpp"
#include "janus/math/Trig.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <casadi/casadi.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <vector>

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

template <typename Scalar> struct EigenDecomposition {
    JanusVector<Scalar> eigenvalues;
    JanusMatrix<Scalar> eigenvectors;
};

namespace detail {

template <typename Scalar> JanusVector<Scalar> normalize_vector(const JanusVector<Scalar> &v) {
    const auto v_norm = janus::norm(v);
    if constexpr (std::is_floating_point_v<Scalar>) {
        if (v_norm <= std::numeric_limits<Scalar>::epsilon()) {
            throw InvalidArgument("eigendecomposition: eigenvector construction failed");
        }
    }
    return v / v_norm;
}

template <typename Scalar>
JanusVector<Scalar> best_eigenvector_candidate(const std::array<JanusVector<Scalar>, 3> &cands) {
    const auto n01 = janus::dot(cands[0], cands[0]);
    const auto n02 = janus::dot(cands[1], cands[1]);
    const auto n12 = janus::dot(cands[2], cands[2]);

    if constexpr (std::is_floating_point_v<Scalar>) {
        const auto *best = &cands[0];
        Scalar best_norm = n01;
        if (n02 > best_norm) {
            best = &cands[1];
            best_norm = n02;
        }
        if (n12 > best_norm) {
            best = &cands[2];
        }
        return normalize_vector(*best);
    } else {
        auto best = janus::logic_detail::select(n02 > n01, cands[1], cands[0]);
        auto best_norm = janus::where(n02 > n01, n02, n01);
        best = janus::logic_detail::select(n12 > best_norm, cands[2], best);
        return normalize_vector(best);
    }
}

template <typename Scalar>
JanusVector<Scalar> symmetric_eigenvector_2x2(const JanusMatrix<Scalar> &A, const Scalar &lambda) {
    JanusVector<Scalar> first(2);
    first << A(0, 1), lambda - A(0, 0);

    JanusVector<Scalar> second(2);
    second << lambda - A(1, 1), A(1, 0);

    if constexpr (std::is_floating_point_v<Scalar>) {
        return normalize_vector(janus::dot(first, first) >= janus::dot(second, second) ? first
                                                                                       : second);
    } else {
        return normalize_vector(janus::logic_detail::select(
            janus::dot(first, first) >= janus::dot(second, second), first, second));
    }
}

template <typename Scalar>
JanusVector<Scalar> symmetric_eigenvector_3x3(const JanusMatrix<Scalar> &A, const Scalar &lambda) {
    JanusMatrix<Scalar> shifted = A;
    shifted(0, 0) = shifted(0, 0) - lambda;
    shifted(1, 1) = shifted(1, 1) - lambda;
    shifted(2, 2) = shifted(2, 2) - lambda;

    const JanusVector<Scalar> r0 = shifted.row(0).transpose();
    const JanusVector<Scalar> r1 = shifted.row(1).transpose();
    const JanusVector<Scalar> r2 = shifted.row(2).transpose();

    return best_eigenvector_candidate<Scalar>(
        {janus::cross(r0, r1), janus::cross(r0, r2), janus::cross(r1, r2)});
}

template <typename Scalar> Scalar determinant_3x3(const JanusMatrix<Scalar> &A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}

template <typename Scalar>
void sort_eigenpairs(JanusVector<Scalar> &eigenvalues, JanusMatrix<Scalar> &eigenvectors) {
    std::vector<Eigen::Index> order(static_cast<size_t>(eigenvalues.size()));
    std::iota(order.begin(), order.end(), Eigen::Index{0});
    std::sort(order.begin(), order.end(), [&](Eigen::Index lhs, Eigen::Index rhs) {
        return eigenvalues(lhs) < eigenvalues(rhs);
    });

    JanusVector<Scalar> sorted_values(eigenvalues.size());
    JanusMatrix<Scalar> sorted_vectors(eigenvectors.rows(), eigenvectors.cols());
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        sorted_values(i) = eigenvalues(order[static_cast<size_t>(i)]);
        sorted_vectors.col(i) = eigenvectors.col(order[static_cast<size_t>(i)]);
    }

    eigenvalues = std::move(sorted_values);
    eigenvectors = std::move(sorted_vectors);
}

template <typename Scalar>
EigenDecomposition<Scalar> eig_symmetric_symbolic(const JanusMatrix<Scalar> &A) {
    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig_symmetric: input must be square");
    }

    if (A.rows() == 1) {
        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(1);
        result.eigenvalues(0) = A(0, 0);
        result.eigenvectors = JanusMatrix<Scalar>::Identity(1, 1);
        return result;
    }

    if (A.rows() == 2) {
        const Scalar trace = A(0, 0) + A(1, 1);
        const Scalar disc = janus::sqrt((A(0, 0) - A(1, 1)) * (A(0, 0) - A(1, 1)) +
                                        Scalar(4.0) * A(0, 1) * A(0, 1));

        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(2);
        result.eigenvalues(0) = Scalar(0.5) * (trace - disc);
        result.eigenvalues(1) = Scalar(0.5) * (trace + disc);

        result.eigenvectors.resize(2, 2);
        result.eigenvectors.col(0) = symmetric_eigenvector_2x2(A, result.eigenvalues(0));
        result.eigenvectors.col(1) = symmetric_eigenvector_2x2(A, result.eigenvalues(1));
        return result;
    }

    if (A.rows() == 3) {
        const Scalar q = (A(0, 0) + A(1, 1) + A(2, 2)) / Scalar(3.0);
        const Scalar p1 = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
        const Scalar a00 = A(0, 0) - q;
        const Scalar a11 = A(1, 1) - q;
        const Scalar a22 = A(2, 2) - q;
        const Scalar p2 = a00 * a00 + a11 * a11 + a22 * a22 + Scalar(2.0) * p1;

        if constexpr (std::is_floating_point_v<Scalar>) {
            if (p2 <= std::numeric_limits<Scalar>::epsilon()) {
                EigenDecomposition<Scalar> result;
                result.eigenvalues = JanusVector<Scalar>::Constant(3, q);
                result.eigenvectors = JanusMatrix<Scalar>::Identity(3, 3);
                return result;
            }
        }

        const auto has_spread = p2 > Scalar(0.0);
        const Scalar p = janus::where(has_spread, janus::sqrt(p2 / Scalar(6.0)), Scalar(1.0));

        JanusMatrix<Scalar> centered = A;
        centered(0, 0) = centered(0, 0) - q;
        centered(1, 1) = centered(1, 1) - q;
        centered(2, 2) = centered(2, 2) - q;
        const JanusMatrix<Scalar> B = centered / p;

        const Scalar r = janus::clamp(determinant_3x3(B) / Scalar(2.0), -1.0, 1.0);
        const Scalar phi = janus::where(has_spread, janus::acos(r) / Scalar(3.0), Scalar(0.0));

        constexpr double kTwoPiOverThree = 2.0943951023931954923;
        Scalar largest = q + Scalar(2.0) * p * janus::cos(phi);
        Scalar smallest = q + Scalar(2.0) * p * janus::cos(phi + Scalar(kTwoPiOverThree));
        Scalar middle = Scalar(3.0) * q - largest - smallest;

        largest = janus::where(has_spread, largest, q);
        middle = janus::where(has_spread, middle, q);
        smallest = janus::where(has_spread, smallest, q);

        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(3);
        result.eigenvalues << smallest, middle, largest;

        JanusMatrix<Scalar> vectors(3, 3);
        vectors.col(0) = symmetric_eigenvector_3x3(A, smallest);
        vectors.col(1) = symmetric_eigenvector_3x3(A, middle);
        vectors.col(2) = symmetric_eigenvector_3x3(A, largest);

        const JanusMatrix<Scalar> identity = JanusMatrix<Scalar>::Identity(3, 3);
        result.eigenvectors = janus::logic_detail::select(has_spread, vectors, identity);
        return result;
    }

    throw InvalidArgument(
        "eig_symmetric: symbolic support is limited to 1x1, 2x2, and 3x3 matrices");
}

} // namespace detail

// --- Eigendecomposition ---
/**
 * @brief Computes the eigendecomposition of a square matrix with a real spectrum.
 *
 * Returns eigenvalues in ascending order and eigenvectors as columns of the returned matrix.
 * Numeric matrices use Eigen's general eigensolver. Symbolic MX matrices are not supported in the
 * general case because CasADi does not expose a compatible MX eigendecomposition.
 */
template <typename Derived> auto eig(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;

    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig: input must be square");
    }

    if constexpr (std::is_floating_point_v<Scalar>) {
        using Matrix = JanusMatrix<Scalar>;
        using Complex = std::complex<Scalar>;
        using ComplexMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
        using ComplexVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

        if (A.rows() == 1) {
            EigenDecomposition<Scalar> result;
            result.eigenvalues.resize(1);
            result.eigenvalues(0) = A(0, 0);
            result.eigenvectors = Matrix::Identity(1, 1);
            return result;
        }

        Eigen::EigenSolver<Matrix> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("eig: EigenSolver failed");
        }

        const ComplexVector values_complex = solver.eigenvalues();
        const ComplexMatrix vectors_complex = solver.eigenvectors();
        constexpr Scalar kImagTol = Scalar(1e-10);

        for (Eigen::Index i = 0; i < values_complex.size(); ++i) {
            if (std::abs(values_complex(i).imag()) > kImagTol) {
                throw InvalidArgument("eig: complex eigenvalues are not supported");
            }
        }

        for (Eigen::Index i = 0; i < vectors_complex.rows(); ++i) {
            for (Eigen::Index j = 0; j < vectors_complex.cols(); ++j) {
                if (std::abs(vectors_complex(i, j).imag()) > kImagTol) {
                    throw InvalidArgument("eig: complex eigenvectors are not supported");
                }
            }
        }

        EigenDecomposition<Scalar> result;
        result.eigenvalues = values_complex.real();
        result.eigenvectors = vectors_complex.real();
        detail::sort_eigenpairs(result.eigenvalues, result.eigenvectors);
        return result;
    } else {
        throw InvalidArgument(
            "eig: symbolic general eigendecomposition is not supported for CasADi MX; use "
            "eig_symmetric() for 1x1, 2x2, or 3x3 symmetric matrices");
    }
}

/**
 * @brief Computes the eigendecomposition of a symmetric matrix.
 *
 * Returns eigenvalues in ascending order and orthonormal eigenvectors as columns.
 * Numeric matrices delegate to Eigen's SelfAdjointEigenSolver. Symbolic MX matrices support only
 * 1x1, 2x2, and 3x3 symmetric inputs.
 */
template <typename Derived> auto eig_symmetric(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;

    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig_symmetric: input must be square");
    }

    if constexpr (std::is_floating_point_v<Scalar>) {
        if (!A.isApprox(A.transpose(), 1e-12)) {
            throw InvalidArgument("eig_symmetric: numeric input must be symmetric");
        }

        using Matrix = JanusMatrix<Scalar>;
        Eigen::SelfAdjointEigenSolver<Matrix> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("eig_symmetric: SelfAdjointEigenSolver failed");
        }

        EigenDecomposition<Scalar> result;
        result.eigenvalues = solver.eigenvalues();
        result.eigenvectors = solver.eigenvectors();
        detail::sort_eigenpairs(result.eigenvalues, result.eigenvectors);
        return result;
    } else {
        return detail::eig_symmetric_symbolic(A.eval());
    }
}

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
