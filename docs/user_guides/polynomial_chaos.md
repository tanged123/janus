# Polynomial Chaos

Janus includes a PCE (Polynomial Chaos Expansion) layer in `PolynomialChaos.hpp` for spectral uncertainty quantification. It provides Askey-scheme basis polynomials, multidimensional basis construction, structured stochastic quadrature rules and sparse grids, coefficient recovery from projection or regression samples, and symbolic mean and variance propagation. The key benefit is that fitted PCE coefficients can remain symbolic `casadi::MX` expressions, so statistical moments can be differentiated with respect to design variables without nesting Monte Carlo inside an optimizer. This works in both numeric and symbolic modes.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Build a 1D Legendre basis of degree <= 2
janus::PolynomialChaosBasis basis({janus::legendre_dimension()}, 2);

// Get quadrature rule for projection
auto rule = janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);

// Evaluate your model at quadrature nodes
janus::NumericVector values(rule.nodes.size());
for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
    values(i) = std::sin(rule.nodes(i));  // Example: sin(xi)
}

// Recover PCE coefficients via projection
janus::NumericVector coeffs = janus::pce_projection_coefficients(basis, rule, values);

// Extract mean and variance
double mean = janus::pce_mean(coeffs);
double variance = janus::pce_variance(basis, coeffs);
```

## Core API

*   **`janus::hermite_dimension()`**: Hermite basis for standard normal variables.
*   **`janus::legendre_dimension()`**: Legendre basis for uniform variables on `[-1, 1]`.
*   **`janus::jacobi_dimension(alpha, beta)`**: Jacobi basis for beta-family variables on `[-1, 1]`.
*   **`janus::laguerre_dimension()`**: Laguerre basis for exponential/gamma-family variables on `[0, inf)`.
*   **`janus::pce_polynomial(dim, order, x)`**: Evaluate a univariate basis function (normalized by default).
*   **`janus::pce_squared_norm(dim, order, normalized)`**: Get the squared norm of a basis function.
*   **`janus::PolynomialChaosBasis(dims, degree, opts)`**: Build a multidimensional basis with total-order or tensor-product truncation.
*   **`janus::pce_projection_coefficients(basis, rule, values)`**: Recover coefficients via weighted projection.
*   **`janus::pce_regression_coefficients(basis, samples, values, ridge)`**: Recover coefficients via regression.
*   **`janus::pce_mean(coeffs)`**: Extract the mean from PCE coefficients.
*   **`janus::pce_variance(basis, coeffs)`**: Extract the variance from PCE coefficients.

## Usage Patterns

### Supported Families

Use one `PolynomialChaosDimension` per uncertain input:

```cpp
auto gaussian = janus::hermite_dimension();
auto uniform = janus::legendre_dimension();
auto beta_like = janus::jacobi_dimension(1.0, 2.0);
auto exponential = janus::laguerre_dimension();  // alpha = 0
```

The canonical variable domains are:

- Hermite: standard normal variable
- Legendre: uniform variable on `[-1, 1]`
- Jacobi: mapped beta-family variable on `[-1, 1]`
- Laguerre: exponential / gamma-family variable on `[0, inf)`

Evaluate one univariate basis function with:

```cpp
double psi2 = janus::pce_polynomial(uniform, 2, 0.25);
double raw_norm = janus::pce_squared_norm(uniform, 2, false);
```

By default `pce_polynomial(...)` returns the **normalized** basis function. Pass `normalized = false` if you want the classical raw polynomial instead.

### Building a Multidimensional Basis

`PolynomialChaosBasis` builds the multi-index set and caches each term's squared norm:

```cpp
janus::PolynomialChaosBasis basis(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    2);
```

This constructs a **total-order** basis of degree `<= 2`. To switch to tensor-product truncation:

```cpp
janus::PolynomialChaosBasis basis(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    2,
    janus::PolynomialChaosBasisOptions{
        .truncation = janus::PolynomialChaosTruncation::TensorProduct,
        .normalized = true
    });
```

Inspect the generated terms and evaluate:

```cpp
for (const auto &term : basis.terms()) {
    // term.multi_index, term.squared_norm
}

janus::NumericVector xi(2);
xi << 0.25, -0.4;
janus::NumericVector psi = basis.evaluate(xi);
```

### Sample Matrix Layout

For projection and regression, Janus expects the sample matrix as:

- rows: sample points
- columns: stochastic dimensions

```cpp
// For one uncertain variable:
janus::NumericVector nodes = janus::lgl_nodes(5);
janus::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

// For d uncertain variables and N sample points, samples should be N x d.
```

### Structured Quadrature Rules

For probability-measure quadrature, prefer `Quadrature.hpp` over manually assembling nodes and weights:

```cpp
auto rule =
    janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);
```

The returned weights already integrate against the probability measure, so for Legendre/uniform variables they sum to `1` directly. You do not need the extra `0.5` scaling that raw `lgl_weights(...)` require.

For bounded-support families, nested refinement is available:

```cpp
auto coarse =
    janus::stochastic_quadrature_level(
        janus::legendre_dimension(),
        2,
        janus::StochasticQuadratureRule::ClenshawCurtis);
```

For higher-dimensional problems:

```cpp
auto sparse_grid = janus::smolyak_sparse_grid(
    {janus::legendre_dimension(), janus::hermite_dimension()},
    3);
```

See [Stochastic Quadrature Guide](stochastic_quadrature.md) for the full rule set and sparse-grid details.

### Projection Coefficients

Use weighted projection when you already have quadrature-like nodes and weights:

```cpp
janus::PolynomialChaosBasis basis({janus::legendre_dimension()}, 2);

auto rule =
    janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);

janus::NumericVector values(rule.nodes.size());
for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
    values(i) = 1.2 * janus::pce_polynomial(janus::legendre_dimension(), 0, rule.nodes(i))
              + 0.5 * janus::pce_polynomial(janus::legendre_dimension(), 1, rule.nodes(i))
              - 0.7 * janus::pce_polynomial(janus::legendre_dimension(), 2, rule.nodes(i));
}

janus::NumericVector coeffs =
    janus::pce_projection_coefficients(basis, rule, values);
```

If you already have your own `samples` matrix and `weights` vector, the original overloads still work:

```cpp
janus::NumericVector coeffs =
    janus::pce_projection_coefficients(basis, samples, weights, values);
```

### Regression Coefficients

Use regression when you have collocation samples without matching quadrature weights:

```cpp
// continued from "Projection Coefficients" above (reuses basis, values)

janus::NumericVector nodes(6);
nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;

janus::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

janus::NumericVector coeffs =
    janus::pce_regression_coefficients(basis, samples, values, 0.0);
```

The final `ridge` argument is an optional Tikhonov regularization parameter. When `ridge = 0`, Janus rejects rank-deficient design matrices instead of silently returning a bad fit.

### Symbolic Moments

The main Janus-specific payoff is that the sampled response can stay symbolic:

```cpp
// continued from "Regression Coefficients" above (reuses basis, nodes, samples)

auto a = janus::sym("a");
janus::SymbolicVector sample_values(nodes.size());

for (Eigen::Index i = 0; i < nodes.size(); ++i) {
    sample_values(i) = (1.0 + a) * janus::pce_polynomial(janus::legendre_dimension(), 0, nodes(i))
                     + 0.5 * janus::pce_polynomial(janus::legendre_dimension(), 1, nodes(i))
                     + (0.25 * a) * janus::pce_polynomial(janus::legendre_dimension(), 2, nodes(i));
}

auto coeffs = janus::pce_regression_coefficients(basis, samples, sample_values, 0.0);
auto mean = janus::pce_mean(coeffs);
auto variance = janus::pce_variance(basis, coeffs);
```

Because `coeffs`, `mean`, and `variance` are symbolic expressions, you can differentiate them directly:

```cpp
auto dmean_da = janus::jacobian(mean, a);
auto dvariance_da = janus::jacobian(variance, a);
```

With the default normalized basis:

- `pce_mean(coeffs)` returns the constant coefficient
- `pce_variance(basis, coeffs)` returns the sum of squared non-constant coefficients

If you build the basis with `.normalized = false`, Janus uses each term's stored squared norm instead.

## See Also

- [Stochastic Quadrature Guide](stochastic_quadrature.md) - Full rule set and sparse-grid details
- [Symbolic Computing Guide](symbolic_computing.md) - Symbolic expressions and differentiation
- [`examples/math/polynomial_chaos_demo.cpp`](../../examples/math/polynomial_chaos_demo.cpp) - Full PCE demo
- [`include/janus/math/PolynomialChaos.hpp`](../../include/janus/math/PolynomialChaos.hpp) - PCE API implementation
- [`include/janus/math/Quadrature.hpp`](../../include/janus/math/Quadrature.hpp) - Quadrature rule API
