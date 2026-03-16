# Polynomial Chaos in Janus

Janus now includes a first PCE layer in `PolynomialChaos.hpp`. The focus is on the building blocks you need for spectral uncertainty quantification:

- Askey-scheme basis polynomials
- multidimensional basis construction
- coefficient recovery from projection or regression samples
- symbolic mean and variance propagation

The practical benefit is that fitted PCE coefficients can remain symbolic `casadi::MX` expressions, so statistical moments can be differentiated with respect to design variables without nesting Monte Carlo inside an optimizer.

## Supported Families

Use one `PolynomialChaosDimension` per uncertain input:

```cpp
auto gaussian = janus::hermite_dimension();
auto uniform = janus::legendre_dimension();
auto beta_like = janus::jacobi_dimension(1.0, 2.0);
auto exponential = janus::laguerre_dimension(); // alpha = 0
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

## Building a Multidimensional Basis

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

Inspect the generated terms through:

```cpp
for (const auto &term : basis.terms()) {
    // term.multi_index, term.squared_norm
}
```

Evaluate the full basis at one stochastic point:

```cpp
janus::NumericVector xi(2);
xi << 0.25, -0.4;

janus::NumericVector psi = basis.evaluate(xi);
```

## Sample Matrix Layout

For projection and regression, Janus expects the sample matrix as:

- rows: sample points
- columns: stochastic dimensions

For one uncertain variable:

```cpp
janus::NumericVector nodes = janus::lgl_nodes(5);
janus::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;
```

For `d` uncertain variables and `N` sample points, `samples` should be `N x d`.

## Projection Coefficients

Use weighted projection when you already have quadrature-like nodes and weights:

```cpp
janus::PolynomialChaosBasis basis({janus::legendre_dimension()}, 2);

janus::NumericVector nodes = janus::lgl_nodes(5);
janus::NumericVector weights = 0.5 * janus::lgl_weights(5, nodes); // expectation weights

janus::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

janus::NumericVector values(nodes.size());
for (Eigen::Index i = 0; i < nodes.size(); ++i) {
    values(i) = 1.2 * janus::pce_polynomial(janus::legendre_dimension(), 0, nodes(i))
              + 0.5 * janus::pce_polynomial(janus::legendre_dimension(), 1, nodes(i))
              - 0.7 * janus::pce_polynomial(janus::legendre_dimension(), 2, nodes(i));
}

janus::NumericVector coeffs =
    janus::pce_projection_coefficients(basis, samples, weights, values);
```

For Legendre bases on `[-1, 1]`, remember that `lgl_weights(...)` integrate with respect to `dx`, so expectation weights need the extra `0.5` factor.

## Regression Coefficients

Use regression when you have collocation samples without matching quadrature weights:

```cpp
janus::NumericVector nodes(6);
nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;

janus::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

janus::NumericVector coeffs =
    janus::pce_regression_coefficients(basis, samples, values, 0.0);
```

The final `ridge` argument is an optional Tikhonov regularization parameter. When `ridge = 0`, Janus now rejects rank-deficient design matrices instead of silently returning a bad fit.

## Symbolic Moments

The main Janus-specific payoff is that the sampled response can stay symbolic:

```cpp
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

## Example

See `examples/math/polynomial_chaos_demo.cpp`, which demonstrates:

- total-order basis construction in two dimensions
- projection and regression coefficient recovery
- symbolic gradients of PCE mean and variance with respect to a design parameter
