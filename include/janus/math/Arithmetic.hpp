#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace janus {

// --- Absolute Value ---
/**
 * @brief Computes the absolute value of a scalar
 * @tparam T Scalar type (numeric or symbolic)
 * @param x Input value
 * @return Absolute value of x
 */
template <JanusScalar T> T abs(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        return fabs(x);
    }
}

/**
 * @brief Computes absolute value element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of absolute values
 */
template <typename Derived> auto abs(const Eigen::MatrixBase<Derived> &x) {
    return x.array().abs().matrix();
}

// --- Square Root ---
/**
 * @brief Computes the square root of a scalar
 * @tparam T Scalar type
 * @param x Input value
 * @return Square root of x
 */
template <JanusScalar T> T sqrt(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(x);
    } else {
        return sqrt(x);
    }
}

/**
 * @brief Computes square root element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of square roots
 */
template <typename Derived> auto sqrt(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sqrt().matrix();
}

// --- Power ---
/**
 * @brief Computes the power function: base^exponent
 * @tparam T Scalar type
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to the power of exponent
 */
template <JanusScalar T> T pow(const T &base, const T &exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return pow(base, exponent);
    }
}

/**
 * @brief Computes power function base^exponent for scalars (mixed types)
 * @tparam T Scalar type
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to exponent
 */
template <JanusScalar T>
    requires(!std::is_same_v<T, double>)
T pow(const T &base, double exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, static_cast<T>(exponent));
    } else {
        return janus::pow(base, static_cast<T>(exponent));
    }
}

/**
 * @brief Computes power function base^exponent for scalars (mixed types: double base)
 * @tparam T Scalar type
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to exponent
 */
template <JanusScalar T>
    requires(!std::is_same_v<T, double>)
T pow(double base, const T &exponent) {
    // Cast base to T to match homogeneous overload
    return janus::pow(static_cast<T>(base), exponent);
}

/**
 * @brief Computes power function element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Exponent type
 * @param base Base matrix
 * @param exponent Exponent
 * @return Matrix of powers
 */
template <typename Derived, typename Scalar>
auto pow(const Eigen::MatrixBase<Derived> &base, const Scalar &exponent) {
    return base.array().pow(exponent).matrix();
}

// --- Exp ---
/**
 * @brief Computes the exponential function e^x
 * @tparam T Scalar type
 * @param x Input value
 * @return e raised to the power of x
 */
template <JanusScalar T> T exp(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp(x);
    } else {
        return exp(x);
    }
}

/**
 * @brief Computes exponential function element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of exponentials
 */
template <typename Derived> auto exp(const Eigen::MatrixBase<Derived> &x) {
    return x.array().exp().matrix();
}

// --- Log ---
/**
 * @brief Computes the natural logarithm of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Natural logarithm of x
 */
template <JanusScalar T> T log(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log(x);
    } else {
        return log(x);
    }
}

/**
 * @brief Computes natural logarithm element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of logarithms
 */
template <typename Derived> auto log(const Eigen::MatrixBase<Derived> &x) {
    return x.array().log().matrix();
}

/**
 * @brief Computes the base-10 logarithm of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Base-10 logarithm of x
 */
template <JanusScalar T> T log10(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log10(x);
    } else {
        return log10(x);
    }
}

/**
 * @brief Computes base-10 logarithm element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of base-10 logarithms
 */
template <typename Derived> auto log10(const Eigen::MatrixBase<Derived> &x) {
    return x.array().log10().matrix();
}

// --- Hyperbolic Functions ---
/**
 * @brief Computes hyperbolic sine
 * @tparam T Scalar type
 * @param x Input value
 * @return Hyperbolic sine of x
 */
template <JanusScalar T> T sinh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sinh(x);
    } else {
        return sinh(x);
    }
}

/**
 * @brief Computes hyperbolic sine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic sines
 */
template <typename Derived> auto sinh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sinh().matrix();
}

/**
 * @brief Computes hyperbolic cosine of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Hyperbolic cosine of x
 */
template <JanusScalar T> T cosh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cosh(x);
    } else {
        return cosh(x);
    }
}

/**
 * @brief Computes hyperbolic cosine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic cosines
 */
template <typename Derived> auto cosh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().cosh().matrix();
}

/**
 * @brief Computes hyperbolic tangent of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Hyperbolic tangent of x
 */
template <JanusScalar T> T tanh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tanh(x);
    } else {
        return tanh(x);
    }
}

/**
 * @brief Computes hyperbolic tangent element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic tangents
 */
template <typename Derived> auto tanh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().tanh().matrix();
}

// --- Rounding and Sign ---
/**
 * @brief Computes floor of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Floor of x
 */
template <JanusScalar T> T floor(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::floor(x);
    } else {
        return floor(x);
    }
}

/**
 * @brief Computes floor element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of floors
 */
template <typename Derived> auto floor(const Eigen::MatrixBase<Derived> &x) {
    return x.array().floor().matrix();
}

/**
 * @brief Computes ceiling of x
 * @tparam T Scalar type
 * @param x Input value
 * @return Ceiling of x
 */
template <JanusScalar T> T ceil(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::ceil(x);
    } else {
        return ceil(x);
    }
}

/**
 * @brief Computes ceiling element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of ceilings
 */
template <typename Derived> auto ceil(const Eigen::MatrixBase<Derived> &x) {
    return x.array().ceil().matrix();
}

/**
 * @brief Computes sign of x
 * @tparam T Scalar type
 * @param x Input value
 * @return 1.0 if x > 0, -1.0 if x < 0, 0.0 otherwise
 */
template <JanusScalar T> T sign(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        // Return 1.0, -1.0, or 0.0
        return (x > 0) ? T(1.0) : ((x < 0) ? T(-1.0) : T(0.0));
    } else {
        return sign(x);
    }
}

/**
 * @brief Computes sign element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of signs
 */
template <typename Derived> auto sign(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sign().matrix();
}

// --- Modulo ---
/**
 * @brief Computes floating-point remainder of x/y
 * @tparam T Scalar type
 * @param x Numerator
 * @param y Denominator
 * @return Remainder of x/y
 */
template <JanusScalar T> T fmod(const T &x, const T &y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fmod(x, y);
    } else {
        return fmod(x, y);
    }
}

// Note: Ensure strictly positive modulus if needed, std::fmod matches C++ behavior.
// There is no direct .fmod() in Eigen Array, need to map
/**
 * @brief Computes floating-point remainder element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Scalar type
 * @param x Numerator matrix
 * @param y Denominator scalar
 * @return Matrix of remainders
 */
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
