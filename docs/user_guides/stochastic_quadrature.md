# Stochastic Quadrature

Janus includes a dedicated `Quadrature.hpp` layer for probability-measure quadrature rules that feed directly into polynomial-chaos workflows. One-dimensional rules return nodes and weights on the probability measure, tensor-product grids preserve the standard Janus sample layout, and Smolyak sparse grids reduce point count for higher-dimensional problems. This is separate from `quad(...)` in `Integrate.hpp`, which is a definite-integral API; `Quadrature.hpp` is a structured collocation/projection API. Works in numeric mode.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Build a 3rd-order Gauss rule for a uniform variable
auto rule = janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);

// Integrate f(x) = x^2 against the uniform measure on [-1, 1]
double integral = 0.0;
for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
    integral += rule.weights(i) * rule.nodes(i) * rule.nodes(i);
}
// integral ~ 1/3 (second moment of uniform on [-1,1])
```

## Core API

*   **`janus::stochastic_quadrature_rule(dim, order)`**: Build a fixed-order quadrature rule for a given polynomial family.
*   **`janus::stochastic_quadrature_level(dim, level, rule_type)`**: Build a refinement-level rule (supports nesting for bounded-support families).
*   **`janus::tensor_product_quadrature(rules)`**: Combine 1D rules into an N-D tensor-product grid.
*   **`janus::smolyak_sparse_grid(dims, level, opts)`**: Build a Smolyak sparse grid for multi-dimensional integration.
*   **`janus::StochasticQuadratureRule`**: Rule type selector (`Gauss`, `ClenshawCurtis`, `GaussKronrod15`, `AutoNested`).
*   **`janus::SmolyakQuadratureOptions`**: Options for merge tolerance and zero-weight tolerance.

## Usage Patterns

### Rule Types

The public selector is `janus::StochasticQuadratureRule`:

- `Gauss`: family-appropriate Gauss rule
- `ClenshawCurtis`: nested Clenshaw-Curtis-like rule on bounded support
- `GaussKronrod15`: embedded 7/15-point Legendre rule reused from `quad(...)`
- `AutoNested`: use Clenshaw-Curtis for bounded Legendre/Jacobi dimensions and Gauss otherwise

All returned weights sum to `1` because they integrate against the underlying probability measure rather than against raw `dx`.

### One-Dimensional Rules

Build a fixed-order rule:

```cpp
auto uniform_rule =
    janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);

auto gaussian_rule =
    janus::stochastic_quadrature_rule(janus::hermite_dimension(), 4);
```

Build a refinement-level rule:

```cpp
auto coarse =
    janus::stochastic_quadrature_level(
        janus::legendre_dimension(),
        2,
        janus::StochasticQuadratureRule::ClenshawCurtis);

auto fine =
    janus::stochastic_quadrature_level(
        janus::legendre_dimension(),
        4,
        janus::StochasticQuadratureRule::ClenshawCurtis);
```

`fine.nodes` contains every node from `coarse.nodes`, which makes incremental refinement practical for bounded-support dimensions.

### Tensor-Product Grids

Combine one-dimensional rules:

```cpp
auto x_rule = janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);
auto y_rule = janus::stochastic_quadrature_rule(janus::hermite_dimension(), 3);

auto grid = janus::tensor_product_quadrature({x_rule, y_rule});
```

The resulting grid uses:

- `grid.samples`: `N x d` sample matrix
- `grid.weights`: `N` probability weights

It plugs straight into Janus PCE projection:

```cpp
janus::PolynomialChaosBasis basis(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    2);

janus::NumericVector coeffs =
    janus::pce_projection_coefficients(basis, grid, values);
```

No manual `samples.col(0) = nodes` reshaping is required.

### Smolyak Sparse Grids

For higher-dimensional problems:

```cpp
auto grid = janus::smolyak_sparse_grid(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    3);
```

By default, `AutoNested` means:

- Legendre / Jacobi axes use nested Clenshaw-Curtis refinement
- Hermite / Laguerre axes use family-appropriate Gauss rules

This keeps bounded dimensions nested while still supporting unbounded Gaussian / gamma-like variables.

You can tune merging behavior with `SmolyakQuadratureOptions`:

```cpp
auto grid = janus::smolyak_sparse_grid(
    dims,
    4,
    janus::SmolyakQuadratureOptions{
        .rule = janus::StochasticQuadratureRule::AutoNested,
        .merge_tolerance = 1e-12,
        .zero_weight_tolerance = 1e-14
    });
```

### Legendre Embedded 7/15 Rule

`StochasticQuadratureRule::GaussKronrod15` exposes the same embedded Legendre 7/15-point rule that `quad(...)` uses internally for adaptive definite integration.

That gives you a lightweight nested refinement path for one-dimensional uniform variables:

```cpp
auto coarse =
    janus::stochastic_quadrature_level(
        janus::legendre_dimension(),
        1,
        janus::StochasticQuadratureRule::GaussKronrod15); // 7 points

auto fine =
    janus::stochastic_quadrature_level(
        janus::legendre_dimension(),
        2,
        janus::StochasticQuadratureRule::GaussKronrod15); // 15 points
```

This is intentionally a small embedded rule, not a full Gauss-Patterson ladder.

## See Also

- [Polynomial Chaos Guide](polynomial_chaos.md) - PCE basis construction and coefficient recovery
- [Integration Guide](integration.md) - Definite-integral `quad(...)` API
- [`examples/math/polynomial_chaos_demo.cpp`](../../examples/math/polynomial_chaos_demo.cpp) - Sparse-grid and projection demo
- [`include/janus/math/Quadrature.hpp`](../../include/janus/math/Quadrature.hpp) - Quadrature rule API implementation
