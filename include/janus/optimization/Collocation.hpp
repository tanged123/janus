/**
 * @file Collocation.hpp
 * @brief Direct Collocation for Trajectory Optimization
 *
 * Provides DirectCollocation class for transcribing continuous-time
 * optimal control problems into nonlinear programs (NLP).
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
 * @brief Collocation scheme for dynamics discretization
 */
enum class CollocationScheme {
    Trapezoidal,   ///< 2nd order, implicit midpoint rule
    HermiteSimpson ///< 4th order, uses midpoint interpolation
};

/**
 * @brief Options for DirectCollocation setup
 */
struct CollocationOptions {
    CollocationScheme scheme = CollocationScheme::Trapezoidal;
    int n_nodes = 21; ///< Number of discretization nodes (including endpoints)
};

/**
 * @brief Direct Collocation for trajectory optimization
 *
 * Transcribes a continuous-time OCP into an NLP by discretizing state
 * and control trajectories, then enforcing dynamics via defect constraints.
 *
 * Workflow:
 * @code
 * janus::Opti opti;
 * janus::DirectCollocation dc(opti);
 *
 * auto [x, u, t] = dc.setup(n_states, n_controls, t0, tf);
 *
 * dc.set_dynamics([](const auto& state, const auto& control, auto time) {
 *     // Return dx/dt
 *     return dynamics_ode(state, control, time);
 * });
 *
 * dc.add_defect_constraints();
 * dc.set_initial_state(x0);
 * dc.set_final_state(xf);
 *
 * opti.minimize(some_objective);
 * auto sol = opti.solve();
 * @endcode
 */
class DirectCollocation {
  public:
    /**
     * @brief Construct DirectCollocation attached to an Opti instance
     * @param opti The optimization environment to use
     */
    explicit DirectCollocation(Opti &opti) : opti_(opti) {}

    /**
     * @brief Setup decision variables for trajectory optimization
     *
     * Creates decision variables for states and controls at each node.
     *
     * @param n_states Number of state variables
     * @param n_controls Number of control variables
     * @param t0 Initial time
     * @param tf Final time (can be a decision variable if passed as symbolic)
     * @param opts Collocation options
     * @return Tuple of (states, controls, time_grid)
     *         - states: Matrix [n_nodes x n_states]
     *         - controls: Matrix [n_nodes x n_controls]
     *         - time_grid: Normalized time vector [0, 1]
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf, const CollocationOptions &opts = {}) {
        n_states_ = n_states;
        n_controls_ = n_controls;
        n_nodes_ = opts.n_nodes;
        scheme_ = opts.scheme;
        t0_ = t0;
        tf_fixed_ = tf;
        tf_is_variable_ = false;

        // Create time grid (normalized tau in [0, 1])
        tau_ = linspace(0.0, 1.0, n_nodes_);

        // Create state decision variables: each row is a node
        states_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                states_(k, i) = opti_.variable(0.0);
            }
        }

        // Create control decision variables
        controls_ = SymbolicMatrix(n_nodes_, n_controls_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_controls_; ++i) {
                controls_(k, i) = opti_.variable(0.0);
            }
        }

        setup_complete_ = true;
        return {states_, controls_, tau_};
    }

    /**
     * @brief Setup with variable final time
     *
     * Use this when optimizing the trajectory duration.
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const CollocationOptions &opts = {}) {
        auto result = setup(n_states, n_controls, t0, 1.0, opts);
        tf_symbolic_ = tf;
        tf_is_variable_ = true;
        return result;
    }

    /**
     * @brief Set the dynamics function (ODE)
     *
     * The dynamics function takes (state_vector, control_vector, time) and
     * returns the state derivative (dx/dt).
     *
     * @tparam Func Callable type
     * @param dynamics Function f(x, u, t) -> dx/dt
     */
    template <typename Func> void set_dynamics(Func &&dynamics) {
        if (!setup_complete_) {
            throw std::runtime_error("DirectCollocation: call setup() before set_dynamics()");
        }

        // Store dynamics as a type-erased wrapper
        dynamics_ = [fn = std::forward<Func>(dynamics)](
                        const SymbolicVector &x, const SymbolicVector &u,
                        const SymbolicScalar &t) -> SymbolicVector { return fn(x, u, t); };

        dynamics_set_ = true;
    }

    /**
     * @brief Add dynamics constraints (unified API)
     *
     * Applies the selected collocation scheme to enforce dynamics consistency
     * between adjacent nodes. This is the preferred unified method name.
     */
    void add_dynamics_constraints() {
        if (!dynamics_set_) {
            throw std::runtime_error(
                "DirectCollocation: call set_dynamics() before add_dynamics_constraints()");
        }

        SymbolicScalar dt = get_duration() / static_cast<double>(n_nodes_ - 1);

        for (int k = 0; k < n_nodes_ - 1; ++k) {
            // Get state and control at node k and k+1
            SymbolicVector x_k = get_state_at_node(k);
            SymbolicVector x_kp1 = get_state_at_node(k + 1);
            SymbolicVector u_k = get_control_at_node(k);
            SymbolicVector u_kp1 = get_control_at_node(k + 1);

            // Time at each node
            SymbolicScalar t_k = get_time_at_node(k);
            SymbolicScalar t_kp1 = get_time_at_node(k + 1);

            // Evaluate dynamics at nodes
            SymbolicVector f_k = dynamics_(x_k, u_k, t_k);
            SymbolicVector f_kp1 = dynamics_(x_kp1, u_kp1, t_kp1);

            switch (scheme_) {
            case CollocationScheme::Trapezoidal:
                add_trapezoidal_constraints(x_k, x_kp1, f_k, f_kp1, dt);
                break;

            case CollocationScheme::HermiteSimpson:
                add_hermite_simpson_constraints(x_k, x_kp1, u_k, u_kp1, f_k, f_kp1, t_k, t_kp1, dt);
                break;
            }
        }
    }

    /**
     * @brief Add defect constraints enforcing dynamics (legacy alias)
     *
     * @deprecated Use add_dynamics_constraints() for unified API
     */
    void add_defect_constraints() { add_dynamics_constraints(); }

    /**
     * @brief Add continuity constraints (alias for unified API)
     *
     * Provided for API consistency with MultipleShooting.
     */
    void add_continuity_constraints() { add_dynamics_constraints(); }

    /**
     * @brief Constrain the initial state
     * @param x0 Initial state values
     */
    void set_initial_state(const NumericVector &x0) {
        if (x0.size() != n_states_) {
            throw std::invalid_argument("DirectCollocation: x0 size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(0, i) == x0(i));
        }
    }

    /**
     * @brief Constrain the final state
     * @param xf Final state values
     */
    void set_final_state(const NumericVector &xf) {
        if (xf.size() != n_states_) {
            throw std::invalid_argument("DirectCollocation: xf size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(n_nodes_ - 1, i) == xf(i));
        }
    }

    /**
     * @brief Constrain a specific state at the final time
     */
    void set_final_state(int state_idx, double value) {
        opti_.subject_to(states_(n_nodes_ - 1, state_idx) == value);
    }

    /**
     * @brief Constrain a specific state at the initial time
     */
    void set_initial_state(int state_idx, double value) {
        opti_.subject_to(states_(0, state_idx) == value);
    }

    /**
     * @brief Get the state trajectory matrix [n_nodes x n_states]
     */
    const SymbolicMatrix &states() const { return states_; }

    /**
     * @brief Get the control trajectory matrix [n_nodes x n_controls]
     */
    const SymbolicMatrix &controls() const { return controls_; }

    /**
     * @brief Get the normalized time grid [0, 1]
     */
    const NumericVector &time_grid() const { return tau_; }

    /**
     * @brief Get number of nodes
     */
    int n_nodes() const { return n_nodes_; }

    /**
     * @brief Get number of intervals (nodes - 1)
     *
     * Provided for API consistency with MultipleShooting.
     */
    int n_intervals() const { return n_nodes_ - 1; }

  private:
    Opti &opti_;

    int n_states_ = 0;
    int n_controls_ = 0;
    int n_nodes_ = 0;
    CollocationScheme scheme_ = CollocationScheme::Trapezoidal;

    double t0_ = 0.0;
    double tf_fixed_ = 1.0;
    SymbolicScalar tf_symbolic_;
    bool tf_is_variable_ = false;

    NumericVector tau_;       ///< Normalized time [0, 1]
    SymbolicMatrix states_;   ///< [n_nodes x n_states]
    SymbolicMatrix controls_; ///< [n_nodes x n_controls]

    bool setup_complete_ = false;
    bool dynamics_set_ = false;

    std::function<SymbolicVector(const SymbolicVector &, const SymbolicVector &,
                                 const SymbolicScalar &)>
        dynamics_;

    SymbolicScalar get_duration() const {
        if (tf_is_variable_) {
            return tf_symbolic_ - t0_;
        } else {
            return SymbolicScalar(tf_fixed_ - t0_);
        }
    }

    SymbolicScalar get_time_at_node(int k) const {
        double tau_k = tau_(k);
        return t0_ + tau_k * get_duration();
    }

    SymbolicVector get_state_at_node(int k) const {
        SymbolicVector x(n_states_);
        for (int i = 0; i < n_states_; ++i) {
            x(i) = states_(k, i);
        }
        return x;
    }

    SymbolicVector get_control_at_node(int k) const {
        SymbolicVector u(n_controls_);
        for (int i = 0; i < n_controls_; ++i) {
            u(i) = controls_(k, i);
        }
        return u;
    }

    /**
     * @brief Trapezoidal collocation: x[k+1] - x[k] = 0.5 * h * (f[k] + f[k+1])
     */
    void add_trapezoidal_constraints(const SymbolicVector &x_k, const SymbolicVector &x_kp1,
                                     const SymbolicVector &f_k, const SymbolicVector &f_kp1,
                                     const SymbolicScalar &dt) {
        for (int i = 0; i < n_states_; ++i) {
            SymbolicScalar defect = x_kp1(i) - x_k(i) - 0.5 * dt * (f_k(i) + f_kp1(i));
            opti_.subject_to(defect == 0.0);
        }
    }

    /**
     * @brief Hermite-Simpson collocation (4th order)
     *
     * Uses cubic Hermite interpolation with Simpson's rule:
     *   x_mid = 0.5*(x[k] + x[k+1]) + h/8*(f[k] - f[k+1])
     *   f_mid = f(x_mid, u_mid, t_mid)
     *   x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
     */
    void add_hermite_simpson_constraints(const SymbolicVector &x_k, const SymbolicVector &x_kp1,
                                         const SymbolicVector &u_k, const SymbolicVector &u_kp1,
                                         const SymbolicVector &f_k, const SymbolicVector &f_kp1,
                                         const SymbolicScalar &t_k, const SymbolicScalar &t_kp1,
                                         const SymbolicScalar &dt) {
        // Midpoint interpolation
        SymbolicVector x_mid(n_states_);
        for (int i = 0; i < n_states_; ++i) {
            x_mid(i) = 0.5 * (x_k(i) + x_kp1(i)) + dt / 8.0 * (f_k(i) - f_kp1(i));
        }

        // Control midpoint (linear interpolation)
        SymbolicVector u_mid(n_controls_);
        for (int i = 0; i < n_controls_; ++i) {
            u_mid(i) = 0.5 * (u_k(i) + u_kp1(i));
        }

        // Time midpoint
        SymbolicScalar t_mid = 0.5 * (t_k + t_kp1);

        // Evaluate dynamics at midpoint
        SymbolicVector f_mid = dynamics_(x_mid, u_mid, t_mid);

        // Simpson's rule defect constraints
        for (int i = 0; i < n_states_; ++i) {
            SymbolicScalar defect =
                x_kp1(i) - x_k(i) - dt / 6.0 * (f_k(i) + 4.0 * f_mid(i) + f_kp1(i));
            opti_.subject_to(defect == 0.0);
        }
    }
};

} // namespace janus
