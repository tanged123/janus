#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace janus {

// --- Absolute Value ---
template <JanusScalar T>
T abs(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        return fabs(x);
    }
}

template <typename Derived>
auto abs(const Eigen::MatrixBase<Derived>& x) {
    return x.array().abs().matrix();
}

// --- Square Root ---
template <JanusScalar T>
T sqrt(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(x);
    } else {
        return sqrt(x);
    }
}

template <typename Derived>
auto sqrt(const Eigen::MatrixBase<Derived>& x) {
    return x.array().sqrt().matrix();
}

// --- Power ---
template <JanusScalar T>
T pow(const T& base, const T& exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return pow(base, exponent);
    }
}

template <typename Derived, typename Scalar>
auto pow(const Eigen::MatrixBase<Derived>& base, const Scalar& exponent) {
    return base.array().pow(exponent).matrix();
}

// --- Exp ---
template <JanusScalar T>
T exp(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp(x);
    } else {
        return exp(x);
    }
}

template <typename Derived>
auto exp(const Eigen::MatrixBase<Derived>& x) {
    return x.array().exp().matrix();
}

// --- Log ---
template <JanusScalar T>
T log(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log(x);
    } else {
        return log(x);
    }
}

template <typename Derived>
auto log(const Eigen::MatrixBase<Derived>& x) {
    return x.array().log().matrix();
}

} // namespace janus
