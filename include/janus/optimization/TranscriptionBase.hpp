#pragma once

#include "Opti.hpp"
#include "janus/core/JanusTypes.hpp"
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace janus {

/**
 * @brief Shared CRTP base for transcription methods.
 */
template <typename Derived> class TranscriptionBase {
  public:
    explicit TranscriptionBase(Opti &opti) : opti_(opti) {}

    void set_initial_state(const NumericVector &x0) {
        if (!setup_complete_) {
            throw std::runtime_error("TranscriptionBase: call setup() before set_initial_state()");
        }
        if (x0.size() != n_states_) {
            throw std::invalid_argument("TranscriptionBase: x0 size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(0, i) == x0(i));
        }
    }

    void set_initial_state(int idx, double value) {
        if (!setup_complete_) {
            throw std::runtime_error("TranscriptionBase: call setup() before set_initial_state()");
        }
        if (idx < 0 || idx >= n_states_) {
            throw std::out_of_range("TranscriptionBase: initial state index out of range");
        }
        opti_.subject_to(states_(0, idx) == value);
    }

    void set_final_state(const NumericVector &xf) {
        if (!setup_complete_) {
            throw std::runtime_error("TranscriptionBase: call setup() before set_final_state()");
        }
        if (xf.size() != n_states_) {
            throw std::invalid_argument("TranscriptionBase: xf size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(n_nodes_ - 1, i) == xf(i));
        }
    }

    void set_final_state(int idx, double value) {
        if (!setup_complete_) {
            throw std::runtime_error("TranscriptionBase: call setup() before set_final_state()");
        }
        if (idx < 0 || idx >= n_states_) {
            throw std::out_of_range("TranscriptionBase: final state index out of range");
        }
        opti_.subject_to(states_(n_nodes_ - 1, idx) == value);
    }

    template <typename Func> void set_dynamics(Func &&dynamics) {
        if (!setup_complete_) {
            throw std::runtime_error("TranscriptionBase: call setup() before set_dynamics()");
        }

        dynamics_ = [fn = std::forward<Func>(dynamics)](
                        const SymbolicVector &x, const SymbolicVector &u,
                        const SymbolicScalar &t) -> SymbolicVector { return fn(x, u, t); };
        dynamics_set_ = true;
    }

    const SymbolicMatrix &states() const { return states_; }
    const SymbolicMatrix &controls() const { return controls_; }
    const NumericVector &time_grid() const { return tau_; }
    int n_nodes() const { return n_nodes_; }
    int n_states() const { return n_states_; }
    int n_controls() const { return n_controls_; }

    void add_dynamics_constraints() {
        if (!setup_complete_) {
            throw std::runtime_error(
                "TranscriptionBase: call setup() before add_dynamics_constraints()");
        }
        static_cast<Derived *>(this)->add_dynamics_constraints_impl();
    }

    void add_defect_constraints() { add_dynamics_constraints(); }
    void add_continuity_constraints() { add_dynamics_constraints(); }

  protected:
    Opti &opti_;
    int n_states_ = 0;
    int n_controls_ = 0;
    int n_nodes_ = 0;

    double t0_ = 0.0;
    double tf_fixed_ = 1.0;
    SymbolicScalar tf_symbolic_;
    bool tf_is_variable_ = false;

    bool setup_complete_ = false;
    bool dynamics_set_ = false;

    NumericVector tau_;
    SymbolicMatrix states_;
    SymbolicMatrix controls_;

    std::function<SymbolicVector(const SymbolicVector &, const SymbolicVector &,
                                 const SymbolicScalar &)>
        dynamics_;

    SymbolicScalar get_duration() const {
        if (tf_is_variable_) {
            return tf_symbolic_ - t0_;
        }
        return SymbolicScalar(tf_fixed_ - t0_);
    }

    SymbolicScalar get_time_at_node(int k) const {
        if (k < 0 || k >= tau_.size()) {
            throw std::out_of_range("TranscriptionBase: node index out of range");
        }
        return t0_ + tau_(k) * get_duration();
    }

    SymbolicVector get_state_at_node(int k) const {
        if (k < 0 || k >= states_.rows()) {
            throw std::out_of_range("TranscriptionBase: state node index out of range");
        }
        SymbolicVector x(n_states_);
        for (int i = 0; i < n_states_; ++i) {
            x(i) = states_(k, i);
        }
        return x;
    }

    SymbolicVector get_control_at_node(int k) const {
        if (k < 0 || k >= controls_.rows()) {
            throw std::out_of_range("TranscriptionBase: control node index out of range");
        }
        SymbolicVector u(n_controls_);
        for (int i = 0; i < n_controls_; ++i) {
            u(i) = controls_(k, i);
        }
        return u;
    }
};

} // namespace janus
