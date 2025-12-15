#pragma once

#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Arithmetic.hpp"
#include "janus/math/Linalg.hpp"
#include "janus/math/Logic.hpp"
#include "janus/math/Trig.hpp"

namespace janus {

/**
 * @brief Quaternion class for rotation representation
 *
 * Stores quaternion in (w, x, y, z) convention where w is the scalar part.
 * All operations support both numeric and symbolic types.
 */
template <typename Scalar> class Quaternion {
  public:
    Scalar w, x, y, z;

    // --- Constructors ---

    /// Default constructor: Identity quaternion (1, 0, 0, 0)
    Quaternion()
        : w(static_cast<Scalar>(1.0)), x(static_cast<Scalar>(0.0)), y(static_cast<Scalar>(0.0)),
          z(static_cast<Scalar>(0.0)) {}

    /// Component constructor
    Quaternion(Scalar w, Scalar x, Scalar y, Scalar z) : w(w), x(x), y(y), z(z) {}

    /// Construct from Vec4 [w, x, y, z]
    explicit Quaternion(const Vec4<Scalar> &v) : w(v(0)), x(v(1)), y(v(2)), z(v(3)) {}

    // --- Algebraic Operations ---

    /// Hamilton product
    Quaternion operator*(const Quaternion &other) const {
        return Quaternion(w * other.w - x * other.x - y * other.y - z * other.z,
                          w * other.x + x * other.w + y * other.z - z * other.y,
                          w * other.y - x * other.z + y * other.w + z * other.x,
                          w * other.z + x * other.y - y * other.x + z * other.w);
    }

    /// Scalar multiplication
    Quaternion operator*(const Scalar &s) const { return Quaternion(w * s, x * s, y * s, z * s); }

    /// Quaternion addition
    Quaternion operator+(const Quaternion &other) const {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    /// Conjugate (w, -x, -y, -z)
    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }

    /// Inverse (conjugate / norm_sq)
    Quaternion inverse() const { return conjugate() * (static_cast<Scalar>(1.0) / squared_norm()); }

    /// Squared Norm
    Scalar squared_norm() const { // Changed name to avoid conflict with standard library norm
        return w * w + x * x + y * y + z * z;
    }

    /// Norm
    Scalar norm() const { return janus::sqrt(squared_norm()); }

    /// Normalized version
    Quaternion normalized() const {
        Scalar n = norm();
        // Avoid division by zero check for symbolic if possible,
        // strictly speaking we should probably use a safe variant or assume it's not zero.
        // For now, standard division.
        return Quaternion(w / n, x / n, y / n, z / n);
    }

    /// Vector rotation: v_rot = q * v * q_conj
    Vec3<Scalar> rotate(const Vec3<Scalar> &v) const {
        // Optimization: q * (0, v) * q_conj
        // Or specific formula: v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)

        Scalar q0 = w;
        Scalar q1 = x;
        Scalar q2 = y;
        Scalar q3 = z;

        // Extract vector part of quaternion
        Vec3<Scalar> q_vec;
        q_vec << q1, q2, q3;

        Vec3<Scalar> t = static_cast<Scalar>(2.0) * janus::cross(q_vec, v);
        return v + (q0 * t) + janus::cross(q_vec, t);
    }

    // --- Conversions ---

    /// Convert to 3x3 Rotation Matrix
    Mat3<Scalar> to_rotation_matrix() const {
        Mat3<Scalar> R;
        Scalar one = static_cast<Scalar>(1.0);
        Scalar two = static_cast<Scalar>(2.0);

        Scalar xx = x * x;
        Scalar yy = y * y;
        Scalar zz = z * z;
        Scalar xy = x * y;
        Scalar xz = x * z;
        Scalar yz = y * z;
        Scalar wx = w * x;
        Scalar wy = w * y;
        Scalar wz = w * z;

        R(0, 0) = one - two * (yy + zz);
        R(0, 1) = two * (xy - wz);
        R(0, 2) = two * (xz + wy);

        R(1, 0) = two * (xy + wz);
        R(1, 1) = one - two * (xx + zz);
        R(1, 2) = two * (yz - wx);

        R(2, 0) = two * (xz - wy);
        R(2, 1) = two * (yz + wx);
        R(2, 2) = one - two * (xx + yy);

        return R;
    }

    /// Export as vector [w, x, y, z]
    Vec4<Scalar> coeffs() const {
        Vec4<Scalar> res;
        res << w, x, y, z;
        return res;
    }

    // --- Static Factories ---

    /**
     * @brief Create from Euler Angles (Yaw-Pitch-Roll / Z-Y-X sequence)
     * Matches rotation_matrix_from_euler_angles
     */
    static Quaternion from_euler(Scalar roll, Scalar pitch, Scalar yaw) {
        Scalar half = static_cast<Scalar>(0.5);
        Scalar cr = janus::cos(roll * half);
        Scalar sr = janus::sin(roll * half);
        Scalar cp = janus::cos(pitch * half);
        Scalar sp = janus::sin(pitch * half);
        Scalar cy = janus::cos(yaw * half);
        Scalar sy = janus::sin(yaw * half);

        return Quaternion(cr * cp * cy + sr * sp * sy, // w
                          sr * cp * cy - cr * sp * sy, // x
                          cr * sp * cy + sr * cp * sy, // y
                          cr * cp * sy - sr * sp * cy  // z
        );
    }

    static Quaternion from_axis_angle(const Vec3<Scalar> &axis, Scalar angle) {
        Scalar half = static_cast<Scalar>(0.5);
        Scalar s = janus::sin(angle * half);
        Scalar c = janus::cos(angle * half);

        // Assume axis is normalized? Usually safer to normalize.
        // If symbolic, normalization adds complexity, but for correctness it's good.
        // Let's assume user passes normalized axis or we normalize it.
        // Standard library implementations usually assume normalized or normalize.
        // We will normalize to be safe.
        auto n_axis = axis / janus::norm(axis);

        return Quaternion(c, n_axis(0) * s, n_axis(1) * s, n_axis(2) * s);
    }

    static Quaternion from_rotation_vector(const Vec3<Scalar> &rot_vec) {
        Scalar angle = janus::norm(rot_vec);
        // Avoid division by zero if angle is small -> identity
        // But symbolic might not like branching.
        // For now, assume angle > 0 or handle logic outside.
        // Or if angle is 0, axis is undefined but result should be identity.
        // axis * angle / angle = axis.

        // Simple safe normalization:
        Scalar safe_angle = angle + static_cast<Scalar>(1e-16); // Tiny epsilon
        return from_axis_angle(rot_vec / safe_angle, angle);
    }

    static Quaternion from_rotation_matrix(const Mat3<Scalar> &mat) {
        // Implementation based on standard robust algorithms (e.g., Eigen's or Shepperd's)
        // Here we use a simplified version for brevity but covering standard cases.
        // For symbolic compatibility, we need to be careful with branching.
        // It is notoriously hard to do robust rotation matrix -> quaternion symbolicly because of
        // the 4-way branching. If we must be symbolic, we might pick one branch (e.g. max trace)
        // and hope, or use 'where'.

        // For now, let's implement a standard numeric-friendly trace check.
        // If Scalar is symbolic (casadi::MX), regular if/else won't work on values.

        Scalar trace = mat.trace();
        Scalar q_w, q_x, q_y, q_z;
        Scalar one = static_cast<Scalar>(1.0);
        Scalar half = static_cast<Scalar>(0.5);
        Scalar two = static_cast<Scalar>(2.0);

        if constexpr (std::is_floating_point_v<Scalar>) {
            // Numeric implementation (efficient branching)
            if (trace > 0) {
                Scalar s = static_cast<Scalar>(0.5) / janus::sqrt(trace + 1.0);
                q_w = 0.25 / s;
                q_x = (mat(2, 1) - mat(1, 2)) * s;
                q_y = (mat(0, 2) - mat(2, 0)) * s;
                q_z = (mat(1, 0) - mat(0, 1)) * s;
            } else {
                if (mat(0, 0) > mat(1, 1) && mat(0, 0) > mat(2, 2)) {
                    Scalar s = 2.0 * janus::sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2));
                    q_w = (mat(2, 1) - mat(1, 2)) / s;
                    q_x = 0.25 * s;
                    q_y = (mat(0, 1) + mat(1, 0)) / s;
                    q_z = (mat(0, 2) + mat(2, 0)) / s;
                } else if (mat(1, 1) > mat(2, 2)) {
                    Scalar s = 2.0 * janus::sqrt(1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2));
                    q_w = (mat(0, 2) - mat(2, 0)) / s;
                    q_x = (mat(0, 1) + mat(1, 0)) / s;
                    q_y = 0.25 * s;
                    q_z = (mat(1, 2) + mat(2, 1)) / s;
                } else {
                    Scalar s = 2.0 * janus::sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1));
                    q_w = (mat(1, 0) - mat(0, 1)) / s;
                    q_x = (mat(0, 2) + mat(2, 0)) / s;
                    q_y = (mat(1, 2) + mat(2, 1)) / s;
                    q_z = 0.25 * s;
                }
            }
        } else {
            // Symbolic implementation: Assume trace > 0 (safe for small deviations)
            // Or use logic::where if available.
            // Implementing full symbolic 4-way branch is heavy.
            // For Beta, we'll assume the primary branch (trace > 0) works, which is true for small
            // rotations. WARNING: This will fail for large rotations (180 deg).

            // TODO: Use janus::where for robust symbolic implementation.
            // For now, basic trace method.
            Scalar s = half / janus::sqrt(trace + one);
            q_w = static_cast<Scalar>(0.25) / s;
            q_x = (mat(2, 1) - mat(1, 2)) * s;
            q_y = (mat(0, 2) - mat(2, 0)) * s;
            q_z = (mat(1, 0) - mat(0, 1)) * s;
        }
        return Quaternion(q_w, q_x, q_y, q_z);
    }

    // Euler angles (Roll-Pitch-Yaw / XYZ) extraction
    Vec3<Scalar> to_euler() const {
        // Roll (x-axis rotation)
        Scalar sinr_cosp = static_cast<Scalar>(2.0) * (w * x + y * z);
        Scalar cosr_cosp = static_cast<Scalar>(1.0) - static_cast<Scalar>(2.0) * (x * x + y * y);
        Scalar roll = janus::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        Scalar sinp = static_cast<Scalar>(2.0) * (w * y - z * x);
        Scalar pitch;
        // Check for gimbal lock
        // Logic::where ideally
        if constexpr (std::is_floating_point_v<Scalar>) {
            if (std::abs(sinp) >= 1)
                pitch = std::copysign(std::numbers::pi_v<double> / 2,
                                      sinp); // use 90 degrees if out of range
            else
                pitch = std::asin(sinp);
        } else {
            // Symbolic: assume no gimbal lock or underlying library handles asin domain
            pitch = janus::asin(sinp);
        }

        // Yaw (z-axis rotation)
        Scalar siny_cosp = static_cast<Scalar>(2.0) * (w * z + x * y);
        Scalar cosy_cosp = static_cast<Scalar>(1.0) - static_cast<Scalar>(2.0) * (y * y + z * z);
        Scalar yaw = janus::atan2(siny_cosp, cosy_cosp);

        return Vec3<Scalar>(roll, pitch, yaw);
    }
};

// --- Free Functions ---

/**
 * @brief Spherical Linear Interpolation (full fidelity)
 *
 * Features:
 * - Shortest path: If dot < 0, negates q1 to interpolate via the shorter arc
 * - Numerical stability: Falls back to normalized linear interpolation for small angles
 *
 * Works for both numeric and symbolic (CasADi) types via janus::where.
 */
template <typename Scalar>
Quaternion<Scalar> slerp(const Quaternion<Scalar> &q0, const Quaternion<Scalar> &q1, Scalar t) {
    Scalar one = static_cast<Scalar>(1.0);
    Scalar zero = static_cast<Scalar>(0.0);
    Scalar dot_threshold = static_cast<Scalar>(0.9995); // Threshold for linear fallback

    // Compute dot product
    Scalar dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;

    // --- Shortest path fix ---
    // If dot < 0, negate q1 to take shorter arc.
    // We compute a sign factor: sign = where(dot < 0, -1, 1)
    Scalar sign = janus::where(dot < zero, -one, one);

    // Effective q1 and dot (flipped if needed)
    Quaternion<Scalar> q1_eff(q1.w * sign, q1.x * sign, q1.y * sign, q1.z * sign);
    Scalar dot_eff = dot * sign; // Now dot_eff >= 0

    // --- Numerical stability: handle near-identity case ---
    // If dot_eff is very close to 1, theta ≈ 0 and sin(theta) ≈ 0 (division issues).
    // Fall back to normalized linear interpolation (nlerp).

    Scalar theta = janus::acos(dot_eff);
    Scalar sin_theta = janus::sin(theta);

    // Compute slerp weights
    Scalar wa_slerp = janus::sin((one - t) * theta) / sin_theta;
    Scalar wb_slerp = janus::sin(t * theta) / sin_theta;

    // Compute nlerp weights (simple linear blend, then normalize result)
    Scalar wa_nlerp = one - t;
    Scalar wb_nlerp = t;

    // Use janus::where to select between slerp and nlerp based on dot_eff
    Scalar use_slerp = dot_eff < dot_threshold; // True if slerp is safe

    Scalar wa = janus::where(use_slerp, wa_slerp, wa_nlerp);
    Scalar wb = janus::where(use_slerp, wb_slerp, wb_nlerp);

    // Compute result
    Quaternion<Scalar> result = q0 * wa + q1_eff * wb;

    // Normalize for nlerp case (harmless for slerp case, just ensures unit quaternion)
    return result.normalized();
}

} // namespace janus
