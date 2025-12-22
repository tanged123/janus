#pragma once
/**
 * @file IntegratorStep.hpp
 * @brief Single-step explicit ODE integrators for Janus framework
 *
 * Provides `euler_step`, `rk2_step`, `rk4_step`, and `rk45_step` for explicit
 * fixed-step integration. All functions are templated on Scalar for dual-mode
 * (numeric/symbolic) compatibility.
 *
 * These are the fundamental building blocks used by `solve_ivp` and can be
 * called directly for custom simulation loops.
 */

#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusTypes.hpp"
#include <Eigen/Dense>

namespace janus {

// ============================================================================
// Forward Euler (1st order)
// ============================================================================

/**
 * @brief Forward Euler integration step
 *
 * Computes x_{n+1} = x_n + dt * f(t_n, x_n)
 *
 * @tparam Scalar Numeric or symbolic scalar type
 * @tparam Func Callable type f(t, x) -> dx/dt
 * @param f Right-hand side function
 * @param x Current state vector
 * @param t Current time
 * @param dt Time step
 * @return State at t + dt
 *
 * @code
 * // Exponential decay: dy/dt = -y
 * auto y_next = janus::euler_step(
 *     [](double t, const janus::NumericVector& y) {
 *         janus::NumericVector dydt = -y;  // Explicit return type
 *         return dydt;
 *     },
 *     y, t, 0.01
 * );
 * @endcode
 */
template <typename Scalar, typename Func>
JanusVector<Scalar> euler_step(Func &&f, const JanusVector<Scalar> &x, Scalar t, Scalar dt) {
    JanusVector<Scalar> k1 = f(t, x);
    return (x + dt * k1).eval();
}

// ============================================================================
// Heun's Method / RK2 (2nd order)
// ============================================================================

/**
 * @brief Heun's method (RK2) integration step
 *
 * Computes x_{n+1} using the average of Euler and corrector slopes:
 * k1 = f(t, x)
 * k2 = f(t + dt, x + dt * k1)
 * x_{n+1} = x + dt/2 * (k1 + k2)
 *
 * @tparam Scalar Numeric or symbolic scalar type
 * @tparam Func Callable type f(t, x) -> dx/dt
 * @param f Right-hand side function
 * @param x Current state vector
 * @param t Current time
 * @param dt Time step
 * @return State at t + dt
 */
template <typename Scalar, typename Func>
JanusVector<Scalar> rk2_step(Func &&f, const JanusVector<Scalar> &x, Scalar t, Scalar dt) {
    JanusVector<Scalar> k1 = f(t, x);
    JanusVector<Scalar> x1 = (x + dt * k1).eval();
    JanusVector<Scalar> k2 = f(t + dt, x1);
    return (x + (dt / 2.0) * (k1 + k2)).eval();
}

// ============================================================================
// Classic Runge-Kutta (4th order)
// ============================================================================

/**
 * @brief Classic 4th-order Runge-Kutta integration step
 *
 * The workhorse of explicit integration. Computes:
 * k1 = f(t, x)
 * k2 = f(t + dt/2, x + dt/2 * k1)
 * k3 = f(t + dt/2, x + dt/2 * k2)
 * k4 = f(t + dt, x + dt * k3)
 * x_{n+1} = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * @tparam Scalar Numeric or symbolic scalar type
 * @tparam Func Callable type f(t, x) -> dx/dt
 * @param f Right-hand side function
 * @param x Current state vector
 * @param t Current time
 * @param dt Time step
 * @return State at t + dt
 *
 * @code
 * // Harmonic oscillator: y'' = -ω²y => [y, v]' = [v, -ω²y]
 * double omega = 2.0;
 * auto state_next = janus::rk4_step(
 *     [omega](double t, const janus::NumericVector& s) {
 *         janus::NumericVector ds(2);
 *         ds << s(1), -omega * omega * s(0);
 *         return ds;
 *     },
 *     state, t, 0.01
 * );
 * @endcode
 */
template <typename Scalar, typename Func>
JanusVector<Scalar> rk4_step(Func &&f, const JanusVector<Scalar> &x, Scalar t, Scalar dt) {
    JanusVector<Scalar> k1 = f(t, x);
    JanusVector<Scalar> x1 = (x + dt * 0.5 * k1).eval();
    JanusVector<Scalar> k2 = f(t + dt * 0.5, x1);
    JanusVector<Scalar> x2 = (x + dt * 0.5 * k2).eval();
    JanusVector<Scalar> k3 = f(t + dt * 0.5, x2);
    JanusVector<Scalar> x3 = (x + dt * k3).eval();
    JanusVector<Scalar> k4 = f(t + dt, x3);

    return (x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).eval();
}

// ============================================================================
// Dormand-Prince RK45 (Adaptive step with error estimate)
// ============================================================================

/**
 * @brief Result of RK45 step with error estimate
 *
 * @tparam Scalar Numeric or symbolic scalar type
 */
template <typename Scalar> struct RK45Result {
    /// 5th-order solution (recommended for propagation)
    JanusVector<Scalar> y5;

    /// 4th-order solution (used for error estimation)
    JanusVector<Scalar> y4;

    /// Estimated local truncation error: ||y5 - y4||
    Scalar error;
};

/**
 * @brief Dormand-Prince RK45 integration step with embedded error estimate
 *
 * Computes both 4th and 5th order solutions using the Dormand-Prince
 * coefficients. The difference provides a local error estimate for
 * adaptive step size control.
 *
 * @tparam Scalar Numeric or symbolic scalar type
 * @tparam Func Callable type f(t, x) -> dx/dt
 * @param f Right-hand side function
 * @param x Current state vector
 * @param t Current time
 * @param dt Time step
 * @return RK45Result with y5, y4, and error estimate
 *
 * @note For adaptive stepping, compare `result.error` against tolerance
 *       and adjust dt accordingly.
 */
template <typename Scalar, typename Func>
RK45Result<Scalar> rk45_step(Func &&f, const JanusVector<Scalar> &x, Scalar t, Scalar dt) {
    // Dormand-Prince coefficients
    // a (time) coefficients
    constexpr double a2 = 1.0 / 5.0;
    constexpr double a3 = 3.0 / 10.0;
    constexpr double a4 = 4.0 / 5.0;
    constexpr double a5 = 8.0 / 9.0;
    // a6 = 1.0, a7 = 1.0

    // b (state) coefficients - rows of Butcher tableau
    constexpr double b21 = 1.0 / 5.0;

    constexpr double b31 = 3.0 / 40.0;
    constexpr double b32 = 9.0 / 40.0;

    constexpr double b41 = 44.0 / 45.0;
    constexpr double b42 = -56.0 / 15.0;
    constexpr double b43 = 32.0 / 9.0;

    constexpr double b51 = 19372.0 / 6561.0;
    constexpr double b52 = -25360.0 / 2187.0;
    constexpr double b53 = 64448.0 / 6561.0;
    constexpr double b54 = -212.0 / 729.0;

    constexpr double b61 = 9017.0 / 3168.0;
    constexpr double b62 = -355.0 / 33.0;
    constexpr double b63 = 46732.0 / 5247.0;
    constexpr double b64 = 49.0 / 176.0;
    constexpr double b65 = -5103.0 / 18656.0;

    constexpr double b71 = 35.0 / 384.0;
    // b72 = 0
    constexpr double b73 = 500.0 / 1113.0;
    constexpr double b74 = 125.0 / 192.0;
    constexpr double b75 = -2187.0 / 6784.0;
    constexpr double b76 = 11.0 / 84.0;

    // 5th order weights (for y5)
    constexpr double c1 = 35.0 / 384.0;
    // c2 = 0
    constexpr double c3 = 500.0 / 1113.0;
    constexpr double c4 = 125.0 / 192.0;
    constexpr double c5 = -2187.0 / 6784.0;
    constexpr double c6 = 11.0 / 84.0;
    // c7 = 0

    // 4th order weights (for y4, error estimation)
    constexpr double d1 = 5179.0 / 57600.0;
    // d2 = 0
    constexpr double d3 = 7571.0 / 16695.0;
    constexpr double d4 = 393.0 / 640.0;
    constexpr double d5 = -92097.0 / 339200.0;
    constexpr double d6 = 187.0 / 2100.0;
    constexpr double d7 = 1.0 / 40.0;

    // Compute stages (explicitly evaluate intermediate states)
    JanusVector<Scalar> k1 = f(t, x);
    JanusVector<Scalar> x2 = (x + dt * b21 * k1).eval();
    JanusVector<Scalar> k2 = f(t + dt * a2, x2);
    JanusVector<Scalar> x3 = (x + dt * (b31 * k1 + b32 * k2)).eval();
    JanusVector<Scalar> k3 = f(t + dt * a3, x3);
    JanusVector<Scalar> x4 = (x + dt * (b41 * k1 + b42 * k2 + b43 * k3)).eval();
    JanusVector<Scalar> k4 = f(t + dt * a4, x4);
    JanusVector<Scalar> x5 = (x + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)).eval();
    JanusVector<Scalar> k5 = f(t + dt * a5, x5);
    JanusVector<Scalar> x6 =
        (x + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)).eval();
    JanusVector<Scalar> k6 = f(t + dt, x6);
    JanusVector<Scalar> x7 =
        (x + dt * (b71 * k1 + b73 * k3 + b74 * k4 + b75 * k5 + b76 * k6)).eval();
    JanusVector<Scalar> k7 = f(t + dt, x7);

    // 5th order solution
    JanusVector<Scalar> y5 = (x + dt * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)).eval();

    // 4th order solution
    JanusVector<Scalar> y4 =
        (x + dt * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7)).eval();

    // Error estimate (norm of difference)
    JanusVector<Scalar> diff = (y5 - y4).eval();
    Scalar error = diff.norm();

    return RK45Result<Scalar>{y5, y4, error};
}

} // namespace janus
