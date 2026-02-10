# Pseudospectral in Janus

`janus::Pseudospectral` implements global polynomial optimal-control transcription using spectral differentiation matrices.

## Overview

Direct collocation enforces local defects interval-by-interval. Pseudospectral transcription enforces dynamics globally:

```
D * X = (dt / 2) * F(X, U, t)
```

- `D`: differentiation matrix on Lobatto nodes in `[-1, 1]`
- `X`: state values at all nodes
- `F`: dynamics evaluated at all nodes

For smooth problems this gives spectral convergence, so high accuracy is often possible with fewer nodes.

## Supported Schemes

```cpp
enum class PseudospectralScheme {
    LGL, // Legendre-Gauss-Lobatto
    CGL  // Chebyshev-Gauss-Lobatto
};
```

Both include endpoints, so boundary-condition setup matches `DirectCollocation`.

## Basic Usage

```cpp
#include <janus/janus.hpp>

janus::Opti opti;
janus::Pseudospectral ps(opti);

janus::PseudospectralOptions opts;
opts.scheme = janus::PseudospectralScheme::LGL;
opts.n_nodes = 31;

auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0); // free final time
auto [x, u, tau] = ps.setup(3, 1, 0.0, T, opts);

ps.set_dynamics(my_ode);
ps.add_dynamics_constraints();

ps.set_initial_state(x0);
ps.set_final_state(0, xf_x);
ps.set_final_state(1, xf_y);
```

## Quadrature for Running Costs

Use `quadrature()` for Lagrange objectives:

```cpp
SymbolicVector integrand(ps.n_nodes());
for (int k = 0; k < ps.n_nodes(); ++k) {
    integrand(k) = u(k, 0) * u(k, 0);
}
opti.minimize(ps.quadrature(integrand));
```

This computes:

```
J ≈ (dt / 2) * sum_i w_i * L_i
```

where `w_i` are the scheme quadrature weights.

## Accessing Matrices

Advanced workflows can inspect the internals:

```cpp
const auto &D = ps.diff_matrix();
const auto &w = ps.quadrature_weights();
```

## Notes

- `time_grid()` returns normalized `[0, 1]` values for API consistency with `DirectCollocation`.
- Internally, nodes and `D` are built on `[-1, 1]` and scaled correctly in constraints/objectives.
- Pseudospectral methods are strongest for smooth trajectories; sharp bang-bang controls can need mesh refinement or alternative transcription.

## Example

See `examples/optimization/pseudospectral_demo.cpp` for a full brachistochrone setup with free final time.
