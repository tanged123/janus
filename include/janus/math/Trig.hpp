#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace janus {

// --- Sin ---
template <JanusScalar T>
T sin(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sin(x);
    } else {
        return sin(x);
    }
}

template <typename Derived>
auto sin(const Eigen::MatrixBase<Derived>& x) {
    return x.array().sin().matrix();
}

// --- Cos ---
template <JanusScalar T>
T cos(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cos(x);
    } else {
        return cos(x);
    }
}

template <typename Derived>
auto cos(const Eigen::MatrixBase<Derived>& x) {
    return x.array().cos().matrix();
}

// --- Tan ---
template <JanusScalar T>
T tan(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tan(x);
    } else {
        return tan(x);
    }
}

template <typename Derived>
auto tan(const Eigen::MatrixBase<Derived>& x) {
    return x.array().tan().matrix();
}

// --- Arcsin ---
template <JanusScalar T>
T asin(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::asin(x);
    } else {
        return asin(x);
    }
}

template <typename Derived>
auto asin(const Eigen::MatrixBase<Derived>& x) {
    return x.array().asin().matrix();
}

// --- Arccos ---
template <JanusScalar T>
T acos(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::acos(x);
    } else {
        return acos(x);
    }
}

template <typename Derived>
auto acos(const Eigen::MatrixBase<Derived>& x) {
    return x.array().acos().matrix();
}

// --- Arctan ---
template <JanusScalar T>
T atan(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::atan(x);
    } else {
        return atan(x);
    }
}

template <typename Derived>
auto atan(const Eigen::MatrixBase<Derived>& x) {
    return x.array().atan().matrix();
}

// --- Atan2 ---
template <JanusScalar T>
T atan2(const T& y, const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::atan2(y, x);
    } else {
        return atan2(y, x);
    }
}

} // namespace janus
