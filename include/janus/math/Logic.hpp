#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Arithmetic.hpp"
#include <Eigen/Dense>

namespace janus {

// --- Trait to deduce the boolean type for a scalar ---
template <typename T> struct BooleanType {
    using type = bool;
};

template <> struct BooleanType<casadi::MX> {
    using type = casadi::MX;
};

template <typename T> using BooleanType_t = typename BooleanType<T>::type;

// --- where (Scalar) ---
/**
 * @brief Select values based on condition (ternary operator)
 * Returns: cond ? if_true : if_false
 * Supports mixed types.
 *
 * @param cond Condition
 * @param if_true Value if true
 * @param if_false Value if false
 * @return Selected value
 */
// Relaxed to allow mixed types (e.g. MX and double)
template <typename Cond, JanusScalar T1, JanusScalar T2>
auto where(const Cond &cond, const T1 &if_true, const T2 &if_false) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return cond ? if_true : if_false;
    } else {
        // Assume non-floating point (e.g. CasADi) handles mixed ops
        return if_else(cond, if_true, if_false);
    }
}

// --- where (Vector/Matrix) ---
/**
 * @brief Element-wise select
 *
 * @param cond Condition matrix/array
 * @param if_true Matrix of values if true
 * @param if_false Matrix of values if false
 * @return Result matrix
 */
template <typename DerivedCond, typename DerivedTrue, typename DerivedFalse>
auto where(const Eigen::ArrayBase<DerivedCond> &cond, const Eigen::MatrixBase<DerivedTrue> &if_true,
           const Eigen::MatrixBase<DerivedFalse> &if_false) {
    using Scalar = typename DerivedTrue::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        // Manual element-wise select for generic types/CasADi
        // Assuming if_true and if_false have same dimensions as cond
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(if_true.rows(), if_true.cols());
        for (Eigen::Index i = 0; i < if_true.rows(); ++i) {
            for (Eigen::Index j = 0; j < if_true.cols(); ++j) {
                // cond(i,j) might be an expression, evaluate it.
                res(i, j) = janus::where(cond.derived().coeff(i, j), if_true(i, j), if_false(i, j));
            }
        }
        return res;
    } else {
        return cond.select(if_true, if_false);
    }
}

// --- Min ---
/**
 * @brief Computes minimum of two values
 * @param a First value
 * @param b Second value
 * @return Minimum value
 */
// Relaxed for mixed types
template <JanusScalar T1, JanusScalar T2> auto min(const T1 &a, const T2 &b) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return std::min(a, b);
    } else {
        // use fmin for mixed (fmin(double, MX) works in CasADi)
        // std::min(double, MX) does NOT work usually?
        // Actually, CasADi overloads fmin.
        return fmin(a, b);
    }
}

/**
 * @brief Computes minimum element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Matrix of minimums
 */
template <typename Derived>
auto min(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> res(a.rows(),
                                                                                          a.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            for (Eigen::Index j = 0; j < a.cols(); ++j) {
                res(i, j) = janus::min(a(i, j), b(i, j));
            }
        }
        return res;
    } else {
        return a.cwiseMin(b);
    }
}

// --- Max ---
/**
 * @brief Computes maximum of two values
 * @param a First value
 * @param b Second value
 * @return Maximum value
 */
// Relaxed for mixed types
template <JanusScalar T1, JanusScalar T2> auto max(const T1 &a, const T2 &b) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return std::max(a, b);
    } else {
        return fmax(a, b);
    }
}

/**
 * @brief Computes maximum element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Matrix of maximums
 */
template <typename Derived>
auto max(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> res(a.rows(),
                                                                                          a.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            for (Eigen::Index j = 0; j < a.cols(); ++j) {
                res(i, j) = janus::max(a(i, j), b(i, j));
            }
        }
        return res;
    } else {
        return a.cwiseMax(b);
    }
}

// --- Clamp ---
/**
 * @brief Clamps value between low and high
 * @param val Input value
 * @param low Lower bound
 * @param high Upper bound
 * @return Clamped value
 */
// Relaxed for mixed types
template <JanusScalar T, JanusScalar TLow, JanusScalar THigh>
auto clamp(const T &val, const TLow &low, const THigh &high) {
    return janus::min(janus::max(val, low), high);
}

/**
 * @brief Clamps values element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Bounds type
 * @param val Input matrix
 * @param low Lower bound
 * @param high Upper bound
 * @return Matrix of clamped values
 */
template <typename Derived, typename Scalar>
auto clamp(const Eigen::MatrixBase<Derived> &val, const Scalar &low, const Scalar &high) {
    return val.cwiseMax(low).cwiseMin(high);
}

// --- Less Than (lt) ---
/**
 * @brief Element-wise less than comparison
 * @return Boolean expression or mask
 */
// --- Less Than (lt) ---
/**
 * @brief Element-wise less than comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a < b
 */
// Returns expressions suitable for 'where' condition
template <typename DerivedA, typename DerivedB>
auto lt(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x < y; });
    } else {
        return (a.array() < b.array());
    }
}

// --- Greater Than (gt) ---
/**
 * @brief Element-wise greater than comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a > b
 */
template <typename DerivedA, typename DerivedB>
auto gt(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x > y; });
    } else {
        return (a.array() > b.array());
    }
}

// --- Less Than or Equal (le) ---
/**
 * @brief Element-wise less than or equal comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a <= b
 */
template <typename DerivedA, typename DerivedB>
auto le(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x <= y; });
    } else {
        return (a.array() <= b.array());
    }
}

// --- Greater Than or Equal (ge) ---
/**
 * @brief Element-wise greater than or equal comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a >= b
 */
template <typename DerivedA, typename DerivedB>
auto ge(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x >= y; });
    } else {
        return (a.array() >= b.array());
    }
}

// --- Equal (eq) ---
/**
 * @brief Element-wise equality comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a == b
 */
template <typename DerivedA, typename DerivedB>
auto eq(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x == y; });
    } else {
        return (a.array() == b.array());
    }
}

// --- Not Equal (neq) ---
/**
 * @brief Element-wise inequality comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a != b
 */
template <typename DerivedA, typename DerivedB>
auto neq(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x != y; });
    } else {
        return (a.array() != b.array());
    }
}

// --- sigmoid_blend ---
/**
 * @brief Smoothly blends between val_low and val_high using a sigmoid function
 * blend = val_low + (val_high - val_low) * (1 / (1 + exp(-sharpness * x)))
 *
 * @param x Control variable (0 centers the sigmoid)
 * @param val_low Value when x is negative large
 * @param val_high Value when x is positive large
 * @param sharpness Steepness of the transition
 * @return Blended value
 */
// Relaxed for mixed types
template <JanusScalar T, JanusScalar TLow, JanusScalar THigh, JanusScalar Sharpness = double>
auto sigmoid_blend(const T &x, const TLow &val_low, const THigh &val_high,
                   const Sharpness &sharpness = 1.0) {
    // using janus::exp from Arithmetic.hpp
    auto alpha = 1.0 / (1.0 + janus::exp(-sharpness * x));
    return val_low + alpha * (val_high - val_low);
}

// Vectorized sigmoid_blend could be added here if needed,
// strictly relying on .array() operations in implementation code might be enough
// if we make a vectorized wrapper like in Arithmetic.hpp

/**
 * @brief Smoothly blends element-wise for a matrix using a sigmoid function
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Scalar type
 * @param x Control variable matrix
 * @param val_low Low value
 * @param val_high High value
 * @param sharpness Steepness
 * @return Matrix of blended values
 */
template <typename Derived, typename Scalar>
auto sigmoid_blend(const Eigen::MatrixBase<Derived> &x, const Scalar &val_low,
                   const Scalar &val_high, const Scalar &sharpness = 1.0) {
    auto alpha = (1.0 + (-sharpness * x.array()).exp()).inverse();
    return (val_low + alpha * (val_high - val_low)).matrix();
}

} // namespace janus
