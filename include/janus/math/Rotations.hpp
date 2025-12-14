#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Arithmetic.hpp"
#include "janus/math/Linalg.hpp"
#include "janus/math/Trig.hpp"
#include <Eigen/Dense>

namespace janus {

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

// --- Euler Angles ---
/**
 * @brief Creates a 3x3 rotation matrix from Euler angles (Yaw-Pitch-Roll sequence)
 * R = R_z(yaw) * R_y(pitch) * R_x(roll)
 *
 * @param roll Roll angle (X-axis)
 * @param pitch Pitch angle (Y-axis)
 * @param yaw Yaw angle (Z-axis)
 * @return 3x3 Rotation matrix
 */
template <typename T>
Eigen::Matrix<T, 3, 3> rotation_matrix_from_euler_angles(const T &roll, const T &pitch,
                                                         const T &yaw) {
    T sa = janus::sin(yaw);
    T ca = janus::cos(yaw);
    T sb = janus::sin(pitch);
    T cb = janus::cos(pitch);
    T sc = janus::sin(roll);
    T cc = janus::cos(roll);

    Eigen::Matrix<T, 3, 3> R;
    // Row 0
    R(0, 0) = ca * cb;
    R(0, 1) = ca * sb * sc - sa * cc;
    R(0, 2) = ca * sb * cc + sa * sc;
    // Row 1
    R(1, 0) = sa * cb;
    R(1, 1) = sa * sb * sc + ca * cc;
    R(1, 2) = sa * sb * cc - ca * sc;
    // Row 2
    R(2, 0) = -sb;
    R(2, 1) = cb * sc;
    R(2, 2) = cb * cc;

    return R;
}

// --- Validation ---
/**
 * @brief Checks if matrix is a valid rotation matrix
 * Checks determinant approx 1 and orthogonality (A^T * A approx I)
 *
 * @param a Input matrix
 * @param tol Tolerance
 * @return True (or symbolic condition) if valid
 */
template <typename Derived>
auto is_valid_rotation_matrix(const Eigen::MatrixBase<Derived> &a, double tol = 1e-9) {
    using Scalar = typename Derived::Scalar;

    // Explicit determinant for small matrices to avoid CasADi Determinant node issues in evaluation
    Scalar det_a;
    if (a.rows() == 3 && a.cols() == 3) {
        det_a = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
                a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
                a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    } else if (a.rows() == 2 && a.cols() == 2) {
        det_a = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
    } else {
        det_a = janus::det(a);
    }

    // Identity check: a.T * a approx I
    auto eye_approx = a.transpose() * a;
    auto eye = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(a.rows(), a.cols());

    if constexpr (std::is_floating_point_v<Scalar>) {
        bool det_ok = std::abs(det_a - 1.0) < tol;
        // Frobenius norm of error
        bool ortho_ok = (eye_approx - eye).norm() < tol;
        return det_ok && ortho_ok;
    } else {
        // Symbolic
        auto diff_det = det_a - 1.0;
        auto diff_eye = eye_approx - eye;

        // Use Frobenius norm for matrix error
        auto err_ortho = janus::norm(diff_eye, NormType::Frobenius);

        // Conditions
        auto det_cond = (janus::abs(diff_det) < tol);
        auto ortho_cond = (err_ortho < tol);

        return det_cond && ortho_cond;
    }
}

} // namespace janus
