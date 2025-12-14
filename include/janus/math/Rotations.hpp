#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Trig.hpp"
#include <Eigen/Dense>

namespace janus {

// --- 2D Rotation Matrix ---
// --- 2D Rotation Matrix ---
/**
 * @brief Creates a 2x2 rotation matrix
 * Returns [cos(theta), -sin(theta); sin(theta), cos(theta)]
 *
 * @param theta Rotation angle (radians)
 * @return 2x2 Rotation matrix
 */
template <typename T> Eigen::Matrix<T, 2, 2> rotation_matrix_2d(const T &theta) {
    T c = janus::cos(theta);
    T s = janus::sin(theta);

    Eigen::Matrix<T, 2, 2> R;
    R(0, 0) = c;
    R(0, 1) = -s;
    R(1, 0) = s;
    R(1, 1) = c;
    return R;
}

// --- 3D Rotation Matrix (Principal Axes) ---
// --- 3D Rotation Matrix (Principal Axes) ---
/**
 * @brief Creates a 3x3 rotation matrix about a principal axis
 *
 * @param theta Rotation angle (radians)
 * @param axis Axis index (0=X, 1=Y, 2=Z)
 * @return 3x3 Rotation matrix
 */
template <typename T> Eigen::Matrix<T, 3, 3> rotation_matrix_3d(const T &theta, int axis) {
    T c = janus::cos(theta);
    T s = janus::sin(theta);
    T one = static_cast<T>(1.0);
    T zero = static_cast<T>(0.0);

    Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();

    switch (axis) {
    case 0: // X-axis
        R(1, 1) = c;
        R(1, 2) = -s;
        R(2, 1) = s;
        R(2, 2) = c;
        break;
    case 1: // Y-axis
        R(0, 0) = c;
        R(0, 2) = s;
        R(2, 0) = -s;
        R(2, 2) = c;
        break;
    case 2: // Z-axis
        R(0, 0) = c;
        R(0, 1) = -s;
        R(1, 0) = s;
        R(1, 1) = c;
        break;
    default:
        // Simple fallback or error. For now, Identity.
        // In critical code, throw or assert.
        break;
    }
    return R;
}

} // namespace janus
