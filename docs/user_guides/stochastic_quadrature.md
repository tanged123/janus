# Stochastic Quadrature in Janus

Janus now includes a dedicated `Quadrature.hpp` layer for probability-measure quadrature rules that feed directly into polynomial-chaos workflows.

The core idea is simple:

- one-dimensional rules return nodes and weights on the **probability measure**
- tensor-product grids preserve the standard Janus sample layout
- Smolyak sparse grids reduce point count for higher-dimensional problems

This is separate from `quad(...)` in `Integrate.hpp`. `quad(...)` is a definite-integral API. `Quadrature.hpp` is a structured collocation / projection API.

## Rule Types

The public selector is `janus::StochasticQuadratureRule`:

- `Gauss`: family-appropriate Gauss rule
- `ClenshawCurtis`: nested Clenshaw-Curtis-like rule on bounded support
- `GaussKronrod15`: embedded 7/15-point Legendre rule reused from `quad(...)`
- `AutoNested`: use Clenshaw-Curtis for bounded Legendre/Jacobi dimensions and Gauss otherwise

All returned weights sum to `1` because they integrate against the underlying probability measure rather than against raw `dx`.

## One-Dimensional Rules

Build a fixed-order rule with:

```cpp
auto uniform_rule =
    janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);

auto gaussian_rule =
    janus::stochastic_quadrature_rule(janus::hermite_dimension(), 4);
```

Build a refinement-level rule with:

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

## Tensor-Product Grids

Combine one-dimensional rules with:

```cpp
auto x_rule = janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);
auto y_rule = janus::stochastic_quadrature_rule(janus::hermite_dimension(), 3);

auto grid = janus::tensor_product_quadrature({x_rule, y_rule});
```

The resulting grid uses:

- `grid.samples`: `N x d` sample matrix
- `grid.weights`: `N` probability weights

That means it plugs straight into Janus PCE projection:

```cpp
janus::PolynomialChaosBasis basis(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    2);

janus::NumericVector coeffs =
    janus::pce_projection_coefficients(basis, grid, values);
```

No manual `samples.col(0) = nodes` reshaping is required.

## Smolyak Sparse Grids

For higher-dimensional problems, use:

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

## Legendre Embedded 7/15 Rule

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

## Example

See `examples/math/polynomial_chaos_demo.cpp` for:

- one-dimensional Legendre projection through `stochastic_quadrature_rule(...)`
- a mixed Legendre/Hermite Smolyak sparse-grid expectation
- symbolic moment propagation through fitted PCE coefficients
