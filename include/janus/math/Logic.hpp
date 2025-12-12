#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Arithmetic.hpp"
#include <Eigen/Dense>

namespace janus {

// --- Trait to deduce the boolean type for a scalar ---
template<typename T>
struct BooleanType {
    using type = bool;
};

template<>
struct BooleanType<casadi::MX> {
    using type = casadi::MX;
};

template<typename T>
using BooleanType_t = typename BooleanType<T>::type;


// --- where (Scalar) ---
// Returns: cond ? if_true : if_false
template <JanusScalar T>
T where(const BooleanType_t<T>& cond, const T& if_true, const T& if_false) {
    if constexpr (std::is_floating_point_v<T>) {
        return cond ? if_true : if_false;
    } else {
        return if_else(cond, if_true, if_false);
    }
}

// --- where (Vector/Matrix) ---
// For Eigen types: uses .select()
// For CasADi: handled by scalar overload (as MX is natively a matrix)
template <typename Derived>
auto where(const Eigen::ArrayBase<Derived>& cond, 
           const Eigen::MatrixBase<Derived>& if_true, 
           const Eigen::MatrixBase<Derived>& if_false) {
    return cond.select(if_true, if_false);
}


// --- sigmoid_blend ---
// Smoothly blends between val_low and val_high based on x
// blend = val_low + (val_high - val_low) * (1 / (1 + exp(-sharpness * x)))
template <JanusScalar T>
T sigmoid_blend(const T& x, const T& val_low, const T& val_high, const T& sharpness = static_cast<T>(1.0)) {
    // using janus::exp from Arithmetic.hpp
    T alpha = static_cast<T>(1.0) / (static_cast<T>(1.0) + janus::exp(-sharpness * x));
    return val_low + alpha * (val_high - val_low);
}

// Vectorized sigmoid_blend could be added here if needed, 
// strictly relying on .array() operations in implementation code might be enough 
// if we make a vectorized wrapper like in Arithmetic.hpp

template <typename Derived, typename Scalar>
auto sigmoid_blend(const Eigen::MatrixBase<Derived>& x, 
                   const Scalar& val_low, 
                   const Scalar& val_high, 
                   const Scalar& sharpness = 1.0) {
    auto alpha = (1.0 + (-sharpness * x.array()).exp()).inverse();
    return (val_low + alpha * (val_high - val_low)).matrix();
}

} // namespace janus
