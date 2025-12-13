#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Trig.hpp"
#include <Eigen/Dense>

namespace janus {

// --- 2D Rotation Matrix ---
// Returns [cos(theta), -sin(theta); sin(theta), cos(theta)]
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
// axis: 0 for X, 1 for Y, 2 for Z
// Throws if axis is invalid (only in Numeric mode ideally, but we can just assert or fallback)
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
