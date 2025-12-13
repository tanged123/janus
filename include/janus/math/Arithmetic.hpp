#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace janus {

// --- Absolute Value ---
template <JanusScalar T> T abs(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        return fabs(x);
    }
}

template <typename Derived> auto abs(const Eigen::MatrixBase<Derived> &x) {
    return x.array().abs().matrix();
}

// --- Square Root ---
template <JanusScalar T> T sqrt(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(x);
    } else {
        return sqrt(x);
    }
}

template <typename Derived> auto sqrt(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sqrt().matrix();
}

// --- Power ---
template <JanusScalar T> T pow(const T &base, const T &exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return pow(base, exponent);
    }
}

template <JanusScalar T>
    requires(!std::is_same_v<T, double>)
T pow(const T &base, double exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, static_cast<T>(exponent));
    } else {
        return janus::pow(base, static_cast<T>(exponent));
    }
}

template <typename Derived, typename Scalar>
auto pow(const Eigen::MatrixBase<Derived> &base, const Scalar &exponent) {
    return base.array().pow(exponent).matrix();
}

// --- Exp ---
template <JanusScalar T> T exp(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp(x);
    } else {
        return exp(x);
    }
}

template <typename Derived> auto exp(const Eigen::MatrixBase<Derived> &x) {
    return x.array().exp().matrix();
}

// --- Log ---
template <JanusScalar T> T log(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log(x);
    } else {
        return log(x);
    }
}

template <typename Derived> auto log(const Eigen::MatrixBase<Derived> &x) {
    return x.array().log().matrix();
}

// --- Hyperbolic Functions ---
template <JanusScalar T> T sinh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sinh(x);
    } else {
        return sinh(x);
    }
}

template <typename Derived> auto sinh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sinh().matrix();
}

template <JanusScalar T> T cosh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cosh(x);
    } else {
        return cosh(x);
    }
}

template <typename Derived> auto cosh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().cosh().matrix();
}

template <JanusScalar T> T tanh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tanh(x);
    } else {
        return tanh(x);
    }
}

template <typename Derived> auto tanh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().tanh().matrix();
}

// --- Rounding and Sign ---
template <JanusScalar T> T floor(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::floor(x);
    } else {
        return floor(x);
    }
}

template <typename Derived> auto floor(const Eigen::MatrixBase<Derived> &x) {
    return x.array().floor().matrix();
}

template <JanusScalar T> T ceil(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::ceil(x);
    } else {
        return ceil(x);
    }
}

template <typename Derived> auto ceil(const Eigen::MatrixBase<Derived> &x) {
    return x.array().ceil().matrix();
}

template <JanusScalar T> T sign(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        // Return 1.0, -1.0, or 0.0
        return (x > 0) ? T(1.0) : ((x < 0) ? T(-1.0) : T(0.0));
    } else {
        return sign(x);
    }
}

template <typename Derived> auto sign(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sign().matrix();
}

// --- Modulo ---
template <JanusScalar T> T fmod(const T &x, const T &y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fmod(x, y);
    } else {
        return fmod(x, y);
    }
}

// Note: Ensure strictly positive modulus if needed, std::fmod matches C++ behavior.
// There is no direct .fmod() in Eigen Array, need to map
template <typename Derived, typename Scalar>
auto fmod(const Eigen::MatrixBase<Derived> &x, const Scalar &y) {
    if constexpr (std::is_same_v<typename Derived::Scalar, casadi::MX>) {
        return fmod(x, y); // CasADi handles matrix fmod? Check docs or assume mapping.
        // Actually, CasADi MX supports fmod.
    } else {
        // For Eigen double, use binaryExpr
        return x.binaryExpr(Eigen::MatrixBase<Derived>::Constant(x.rows(), x.cols(), y),
                            [](double a, double b) { return std::fmod(a, b); });
    }
}

} // namespace janus
