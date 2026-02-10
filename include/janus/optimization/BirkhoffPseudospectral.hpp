/**
 * @file BirkhoffPseudospectral.hpp
 * @brief Birkhoff pseudospectral transcription for trajectory optimization
 */

#pragma once

#include "TranscriptionBase.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/math/OrthogonalPolynomials.hpp"
#include <tuple>
#include <vector>

namespace janus {

enum class BirkhoffScheme {
    LGL, ///< Legendre-Gauss-Lobatto nodes
    CGL  ///< Chebyshev-Gauss-Lobatto nodes
};

struct BirkhoffOptions {
    BirkhoffScheme scheme = BirkhoffScheme::LGL;
    int n_nodes = 21; ///< Number of collocation nodes (including endpoints)
};

class BirkhoffPseudospectral : public TranscriptionBase<BirkhoffPseudospectral> {
    friend class TranscriptionBase<BirkhoffPseudospectral>;

  public:
    explicit BirkhoffPseudospectral(Opti &opti) : TranscriptionBase<BirkhoffPseudospectral>(opti) {}

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf, const BirkhoffOptions &opts = {}) {
        if (opts.n_nodes < 2) {
            throw InvalidArgument("BirkhoffPseudospectral: n_nodes must be >= 2");
        }

        n_states_ = n_states;
        n_controls_ = n_controls;
        n_nodes_ = opts.n_nodes;
        scheme_ = opts.scheme;
        t0_ = t0;
        tf_fixed_ = tf;
        tf_is_variable_ = false;

        NumericVector nodes;
        switch (scheme_) {
        case BirkhoffScheme::LGL:
            nodes = lgl_nodes(n_nodes_);
            break;
        case BirkhoffScheme::CGL:
            nodes = cgl_nodes(n_nodes_);
            break;
        }

        tau_ = (nodes.array() + 1.0) * 0.5;
        B_ = birkhoff_integration_matrix(nodes);
        bk_weights_ = B_.row(n_nodes_ - 1).transpose();

        states_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                states_(k, i) = opti_.variable(0.0);
            }
        }

        controls_ = SymbolicMatrix(n_nodes_, n_controls_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_controls_; ++i) {
                controls_(k, i) = opti_.variable(0.0);
            }
        }

        V_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                V_(k, i) = opti_.variable(0.0);
            }
        }

        setup_complete_ = true;
        dynamics_set_ = false;
        return {states_, controls_, tau_};
    }

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const BirkhoffOptions &opts = {}) {
        auto result = setup(n_states, n_controls, t0, 1.0, opts);
        tf_symbolic_ = tf;
        tf_is_variable_ = true;
        return result;
    }

    const NumericMatrix &integration_matrix() const { return B_; }
    const NumericVector &quadrature_weights() const { return bk_weights_; }
    const SymbolicMatrix &virtual_vars() const { return V_; }

    SymbolicScalar quadrature(const SymbolicVector &integrand) const {
        if (!setup_complete_) {
            throw RuntimeError("BirkhoffPseudospectral: call setup() before quadrature()");
        }
        if (integrand.size() != n_nodes_) {
            throw InvalidArgument("BirkhoffPseudospectral: integrand size mismatch");
        }

        SymbolicScalar weighted_sum = SymbolicScalar(0.0);
        for (int k = 0; k < n_nodes_; ++k) {
            weighted_sum = weighted_sum + bk_weights_(k) * integrand(k);
        }
        return get_duration() / 2.0 * weighted_sum;
    }

  private:
    BirkhoffScheme scheme_ = BirkhoffScheme::LGL;
    NumericMatrix B_;
    NumericVector bk_weights_;
    SymbolicMatrix V_;

    void add_dynamics_constraints_impl() {
        if (!dynamics_set_) {
            throw RuntimeError(
                "BirkhoffPseudospectral: call set_dynamics() before add_dynamics_constraints()");
        }

        const SymbolicScalar half_dt = get_duration() / 2.0;

        std::vector<SymbolicVector> f(static_cast<std::size_t>(n_nodes_));
        for (int k = 0; k < n_nodes_; ++k) {
            f[static_cast<std::size_t>(k)] =
                dynamics_(get_state_at_node(k), get_control_at_node(k), get_time_at_node(k));
        }

        for (int s = 0; s < n_states_; ++s) {
            // Pointwise nonlinear dynamics: V_i = (dt/2) * f_i(X_i, U_i, t_i)
            for (int i = 0; i < n_nodes_; ++i) {
                opti_.subject_to(V_(i, s) == half_dt * f[static_cast<std::size_t>(i)](s));
            }

            // Linear state recovery for interior nodes.
            for (int i = 1; i < n_nodes_ - 1; ++i) {
                SymbolicScalar Bv_i = SymbolicScalar(0.0);
                for (int j = 0; j < n_nodes_; ++j) {
                    Bv_i = Bv_i + B_(i, j) * V_(j, s);
                }
                opti_.subject_to(states_(i, s) == states_(0, s) + Bv_i);
            }

            // Right boundary grid-equivalency.
            SymbolicScalar wv = SymbolicScalar(0.0);
            for (int j = 0; j < n_nodes_; ++j) {
                wv = wv + bk_weights_(j) * V_(j, s);
            }
            opti_.subject_to(states_(n_nodes_ - 1, s) == states_(0, s) + wv);
        }
    }
};

} // namespace janus
