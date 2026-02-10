/**
 * @file Pseudospectral.hpp
 * @brief Pseudospectral transcription for trajectory optimization
 */

#pragma once

#include "TranscriptionBase.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/math/OrthogonalPolynomials.hpp"
#include <tuple>
#include <vector>

namespace janus {

enum class PseudospectralScheme {
    LGL, ///< Legendre-Gauss-Lobatto
    CGL  ///< Chebyshev-Gauss-Lobatto
};

struct PseudospectralOptions {
    PseudospectralScheme scheme = PseudospectralScheme::LGL;
    int n_nodes = 21; ///< Number of collocation nodes (including endpoints)
};

class Pseudospectral : public TranscriptionBase<Pseudospectral> {
    friend class TranscriptionBase<Pseudospectral>;

  public:
    explicit Pseudospectral(Opti &opti) : TranscriptionBase<Pseudospectral>(opti) {}

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const PseudospectralOptions &opts = {}) {
        if (opts.n_nodes < 2) {
            throw InvalidArgument("Pseudospectral: n_nodes must be >= 2");
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
        case PseudospectralScheme::LGL:
            nodes = lgl_nodes(n_nodes_);
            weights_ = lgl_weights(n_nodes_, nodes);
            break;
        case PseudospectralScheme::CGL:
            nodes = cgl_nodes(n_nodes_);
            weights_ = cgl_weights(n_nodes_, nodes);
            break;
        }

        tau_ = (nodes.array() + 1.0) * 0.5;
        D_ = spectral_diff_matrix(nodes);

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

        setup_complete_ = true;
        dynamics_set_ = false;
        return {states_, controls_, tau_};
    }

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const PseudospectralOptions &opts = {}) {
        auto result = setup(n_states, n_controls, t0, 1.0, opts);
        tf_symbolic_ = tf;
        tf_is_variable_ = true;
        return result;
    }

    const NumericMatrix &diff_matrix() const { return D_; }
    const NumericVector &quadrature_weights() const { return weights_; }

    SymbolicScalar quadrature(const SymbolicVector &integrand) const {
        if (!setup_complete_) {
            throw RuntimeError("Pseudospectral: call setup() before quadrature()");
        }
        if (integrand.size() != n_nodes_) {
            throw InvalidArgument("Pseudospectral: integrand size mismatch");
        }

        SymbolicScalar weighted_sum = SymbolicScalar(0.0);
        for (int k = 0; k < n_nodes_; ++k) {
            weighted_sum = weighted_sum + weights_(k) * integrand(k);
        }
        return get_duration() / 2.0 * weighted_sum;
    }

  private:
    PseudospectralScheme scheme_ = PseudospectralScheme::LGL;
    NumericMatrix D_;
    NumericVector weights_;

    void add_dynamics_constraints_impl() {
        if (!dynamics_set_) {
            throw RuntimeError(
                "Pseudospectral: call set_dynamics() before add_dynamics_constraints()");
        }

        const SymbolicScalar half_dt = get_duration() / 2.0;

        std::vector<SymbolicVector> f(static_cast<std::size_t>(n_nodes_));
        for (int k = 0; k < n_nodes_; ++k) {
            SymbolicVector x_k = get_state_at_node(k);
            SymbolicVector u_k = get_control_at_node(k);
            SymbolicScalar t_k = get_time_at_node(k);
            f[static_cast<std::size_t>(k)] = dynamics_(x_k, u_k, t_k);
        }

        for (int s = 0; s < n_states_; ++s) {
            for (int i = 0; i < n_nodes_; ++i) {
                SymbolicScalar Dx_i = SymbolicScalar(0.0);
                for (int j = 0; j < n_nodes_; ++j) {
                    Dx_i = Dx_i + D_(i, j) * states_(j, s);
                }
                opti_.subject_to(Dx_i == half_dt * f[static_cast<std::size_t>(i)](s));
            }
        }
    }
};

} // namespace janus
