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
// Returns: cond ? if_true : if_false
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
// For Eigen types: uses .select()
// For CasADi: handled by scalar overload (as MX is natively a matrix)
// --- where (Vector/Matrix) ---
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
// Relaxed for mixed types
template <JanusScalar T1, JanusScalar T2> auto max(const T1 &a, const T2 &b) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return std::max(a, b);
    } else {
        return fmax(a, b);
    }
}

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
// Relaxed for mixed types
template <JanusScalar T, JanusScalar TLow, JanusScalar THigh>
auto clamp(const T &val, const TLow &low, const THigh &high) {
    return janus::min(janus::max(val, low), high);
}

template <typename Derived, typename Scalar>
auto clamp(const Eigen::MatrixBase<Derived> &val, const Scalar &low, const Scalar &high) {
    return val.cwiseMax(low).cwiseMin(high);
}

// --- Less Than (lt) ---
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
// Smoothly blends between val_low and val_high based on x
// blend = val_low + (val_high - val_low) * (1 / (1 + exp(-sharpness * x)))
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

template <typename Derived, typename Scalar>
auto sigmoid_blend(const Eigen::MatrixBase<Derived> &x, const Scalar &val_low,
                   const Scalar &val_high, const Scalar &sharpness = 1.0) {
    auto alpha = (1.0 + (-sharpness * x.array()).exp()).inverse();
    return (val_low + alpha * (val_high - val_low)).matrix();
}

} // namespace janus
