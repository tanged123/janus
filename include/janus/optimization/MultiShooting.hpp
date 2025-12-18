/**
 * @file MultiShooting.hpp
 * @brief Multiple Shooting for Trajectory Optimization
 *
 * Provides MultipleShooting class for transcribing optimal control problems
 * using high-accuracy numerical integration (CVODES) for continuity constraints.
 */

#pragma once

#include "Opti.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Spacing.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>

namespace janus {

/**
 * @brief Options for MultipleShooting
 */
struct MultiShootingOptions {
    int n_intervals = 20;              ///< Number of shooting intervals
    std::string integrator = "cvodes"; ///< Integrator plugin ("cvodes", "rk", "idas")
    double tol = 1e-8;                 ///< Integrator required tolerance
    bool normalize_time = true;        ///< If true, integrates on [0,1] and scales ODE by dt
};

/**
 * @brief Multiple Shooting transcription
 *
 * Enforces continuity X[k+1] = Integ(X[k], U[k], dt) using CasADi integrators.
 *
 * Features:
 * - High-accuracy integration (CVODES by default)
 * - Supports variable time optimization
 * - Handles stiff systems better than direct collocation
 */
class MultipleShooting {
  public:
    explicit MultipleShooting(Opti &opti) : opti_(opti) {}

    /**
     * @brief Setup decision variables
     *
     * @return Tuple (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const MultiShootingOptions &opts = {}) {
        return setup_impl(n_states, n_controls, t0, tf, false, opts);
    }

    /**
     * @brief Setup with variable final time
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const MultiShootingOptions &opts = {}) {
        tf_symbolic_ = tf;
        return setup_impl(n_states, n_controls, t0, 1.0, true, opts); // tf dummy
    }

    template <typename Func> void set_dynamics(Func &&dynamics) {
        if (!setup_complete_) {
            throw std::runtime_error("MultipleShooting: call setup() first");
        }

        // 1. Integrator variables must be symbolic primitives
        SymbolicScalar x_sym = sym("x", n_states_);
        SymbolicScalar p_sym = sym("p", n_controls_ + 1); // p = [u; dt]
        SymbolicScalar t = sym("t");

        // 2. Extract u and dt from p_sym
        // u occupies p[0] ... p[n_controls-1]
        // dt occupies p[n_controls]
        SymbolicScalar u_sym;
        if (n_controls_ > 0) {
            u_sym = p_sym(casadi::Slice(0, n_controls_));
        } else {
            u_sym = casadi::MX(0, 1);
        }
        SymbolicScalar dt_sym = p_sym(n_controls_);

        // 3. Convert to Eigen vectors for user dynamics
        SymbolicVector x_vec = as_vector(x_sym);
        SymbolicVector u_vec = as_vector(u_sym);

        // 4. Evaluate dynamics
        SymbolicVector dxdt = dynamics(x_vec, u_vec, t);
        SymbolicVector ode_scaled = dxdt * dt_sym;

        casadi::MXDict dae = {
            {"x", x_sym},              // Primitive
            {"p", p_sym},              // Primitive
            {"ode", to_mx(ode_scaled)} // Expression
        };

        casadi::Dict opts;
        // tf defaults to 1.0 if not specified, which matches our time-scaled dynamics.

        // Pass standard scalar options if using cvodes
        if (opts_.integrator == "cvodes") {
            opts["abstol"] = opts_.tol;
            opts["reltol"] = opts_.tol;
        }

        integrator_ = casadi::integrator("shooting_intg", opts_.integrator, dae, opts);
        dynamics_set_ = true;
    }

    void add_continuity_constraints() {
        if (!dynamics_set_)
            throw std::runtime_error("Dynamics not set");

        SymbolicScalar total_T;
        if (tf_is_variable_) {
            total_T = tf_symbolic_ - t0_;
        } else {
            total_T = tf_fixed_ - t0_;
        }

        // Time step per interval
        SymbolicScalar dt = total_T / static_cast<double>(n_intervals_);

        for (int k = 0; k < n_intervals_; ++k) {
            SymbolicVector x_k = states_.row(k).transpose();
            SymbolicVector u_k = controls_.row(k).transpose();
            SymbolicVector x_kp1 = states_.row(k + 1).transpose();

            // Integrator parameters: [u_k; dt]
            SymbolicScalar p = SymbolicScalar::vertcat({to_mx(u_k), dt});

            // Call integrator
            // Input: 'x0' -> state, 'p' -> params
            // Output: 'xf' -> final state
            // Call integrator using named arguments to avoid positional errors
            // Inputs: x0 (initial state), p (parameters)
            casadi::MXDict args = {{"x0", to_mx(x_k)}, {"p", p}};

            casadi::MXDict res = integrator_(args);

            SymbolicScalar x_integrated = res.at("xf"); // 'xf' is final state output

            // Continuity constraint: X[k+1] == Integrated(X[k])
            opti_.subject_to(to_mx(x_kp1) == x_integrated);
        }
    }

    // -- Standard Setters/Getters --

    void set_initial_state(const NumericVector &x0) {
        for (int i = 0; i < n_states_; ++i)
            opti_.subject_to(states_(0, i) == x0(i));
    }

    void set_final_state(const NumericVector &xf) {
        for (int i = 0; i < n_states_; ++i)
            opti_.subject_to(states_(n_intervals_, i) == xf(i));
    }

    void set_initial_state(int idx, double val) { opti_.subject_to(states_(0, idx) == val); }
    void set_final_state(int idx, double val) {
        opti_.subject_to(states_(n_intervals_, idx) == val);
    }

    const SymbolicMatrix &states() const { return states_; }
    const SymbolicMatrix &controls() const { return controls_; }
    const NumericVector &time_grid() const { return tau_; }
    int n_nodes() const { return n_intervals_ + 1; } // Intervals + 1

  private:
    Opti &opti_;
    SymbolicMatrix states_;   // [N+1 x nx]
    SymbolicMatrix controls_; // [N x nu] (controls piecewise constant over interval)
    NumericVector tau_;

    MultiShootingOptions opts_;
    casadi::Function integrator_;

    int n_states_ = 0;
    int n_controls_ = 0;
    int n_intervals_ = 0;

    double t0_ = 0.0;
    double tf_fixed_ = 1.0;
    SymbolicScalar tf_symbolic_;
    bool tf_is_variable_ = false;

    bool setup_complete_ = false;
    bool dynamics_set_ = false;

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup_impl(int n_states, int n_controls, double t0, double tf_val, bool variable_tf,
               const MultiShootingOptions &opts) {
        n_states_ = n_states;
        n_controls_ = n_controls;
        opts_ = opts;
        n_intervals_ = opts.n_intervals;
        t0_ = t0;

        if (variable_tf) {
            tf_is_variable_ = true;
        } else {
            tf_fixed_ = tf_val;
            tf_is_variable_ = false;
        }

        // States at N+1 nodes
        states_ = SymbolicMatrix(n_intervals_ + 1, n_states);
        for (int k = 0; k <= n_intervals_; ++k) {
            for (int i = 0; i < n_states; ++i)
                states_(k, i) = opti_.variable(0.0);
        }

        // Controls at N intervals (piecewise constant)
        controls_ = SymbolicMatrix(n_intervals_, n_controls);
        for (int k = 0; k < n_intervals_; ++k) {
            for (int i = 0; i < n_controls; ++i)
                controls_(k, i) = opti_.variable(0.0);
        }

        tau_ = linspace(0.0, 1.0, n_intervals_ + 1);
        setup_complete_ = true;

        return {states_, controls_, tau_};
    }
};

} // namespace janus
