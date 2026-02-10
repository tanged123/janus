# Birkhoff Pseudospectral in Janus

`janus::BirkhoffPseudospectral` is a Birkhoff-form pseudospectral transcription where state-derivative variables are collocated directly and states are recovered through a linear integration matrix.

## Core Formulation

For nodes on `[-1, 1]`:

- Dynamics collocation (pointwise):
```
V_i = (dt / 2) * f(X_i, U_i, t_i)
```
- State recovery (linear):
```
X = x_a * 1 + B * V
```

`B` is the Birkhoff integration matrix with entries:
```
B_ij = integral_{tau_0}^{tau_i} ell_j(s) ds
```

## Basic Usage

```cpp
#include <janus/janus.hpp>

janus::Opti opti;
janus::BirkhoffPseudospectral bk(opti);

auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);
auto [x, u, tau] = bk.setup(
    3, 1, 0.0, T,
    {.scheme = janus::BirkhoffScheme::LGL, .n_nodes = 31}
);

bk.set_dynamics(my_ode);
bk.add_dynamics_constraints();

bk.set_initial_state(x0);
bk.set_final_state(0, xf_x);
bk.set_final_state(1, xf_y);
```

## Useful Accessors

```cpp
const auto &B = bk.integration_matrix();
const auto &w = bk.quadrature_weights();
const auto &V = bk.virtual_vars();
```

## Quadrature

```cpp
SymbolicVector integrand(bk.n_nodes());
for (int k = 0; k < bk.n_nodes(); ++k) {
    integrand(k) = u(k, 0) * u(k, 0);
}
opti.minimize(bk.quadrature(integrand));
```

## Unified Comparison Example

See:

- `examples/optimization/transcription_comparison_demo.cpp`

This compares Direct Collocation, Multiple Shooting, classical Pseudospectral, and Birkhoff Pseudospectral on the same problem with performance and convergence output.
