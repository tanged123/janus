# Symbolic Computing Guide

Janus provides a powerful symbolic computing layer built on top of CasADi, but abstracted to feel like native C++ with Eigen integration. This allows you to compute derivatives, generate code, and optimize systems using the same code you write for simulation.

## 1. Concepts

*   **`janus::SymbolicScalar`**: Alias for `casadi::MX`. Represents a symbolic value or expression.
*   **`janus::SymbolicMatrix`**: Alias for `Eigen::Matrix<casadi::MX, -1, -1>`. Allows you to use familiar Eigen syntax (block operations, coeff access) on symbolic variables.

## 2. Creating Variables

Use the `janus::sym` helper to create symbolic variables cleanly.

```cpp
#include <janus/janus.hpp>

// Scalar variable "x"
auto x = janus::sym("x");

// Column vector "v" (3x1)
auto v = janus::sym("v", 3);

// Matrix "A" (2x2)
auto A = janus::sym("A", 2, 2);
```

## 3. Building Expressions

You can use standard arithmetic operators (`+`, `-`, `*`, `/`) and Janus math functions. These build a computational graph instead of executing immediately.

```cpp
auto y = janus::sin(x) + janus::pow(x, 2.0);
auto z = A * v; // Matrix multiplication
```

## 4. Defining Functions (`janus::Function`)

To evaluate expressions numerically, you must wrap them in a `janus::Function`. This wrapper handles type conversion between Eigen and CasADi.

```cpp
// Define f(x, v) -> y
janus::Function f({x, v}, {y});

// Evaluate with numeric input (doubles/Eigen)
// Returns std::vector<Eigen::MatrixXd>
auto result = f(1.5, Eigen::Vector3d::Zero()); 
std::cout << result[0] << std::endl;
```

**Features:**
*   **Automatic Naming**: You don't need to provide a string name; one is generated automatically.
*   **Eigen Integration**: Inputs and outputs are converted to/from Eigen types seamlessly.

## 5. Automatic Differentiation (`janus::jacobian`)

Compute derivatives effortlessly. Janus handles the tedious task of concatenating variables for CasADi.

```cpp
// Compute Jacobian J = dy/dx
auto J = janus::jacobian({y}, {x});

// Compute Jacobian w.r.t multiple variables: J = dy/d[x, v]
auto J_full = janus::jacobian({y}, {x, v});
```

## 6. Sensitivity Regime Switching

For compiled `janus::Function` objects, Janus can now choose between forward and adjoint
Jacobian construction automatically based on parameter count, output count, and optional
trajectory hints.

```cpp
janus::Function f("f", {x}, {y});

auto rec = janus::select_sensitivity_regime(
    f,
    0,      // output block
    0,      // input block
    400,    // horizon length hint
    true    // stiff trajectory hint
);

auto J_fun = janus::sensitivity_jacobian(f, 0, 0, 400, true);
auto J = J_fun.eval(x_val);
```

Use `rec.integrator_options()` when you want the same recommendation expressed as
CasADi/SUNDIALS options (`nfwd` or `nadj`, plus checkpoint settings for long-horizon adjoints).

## 7. Matrix-Free Second-Order Products

For large optimization problems, you often want `H * v` without ever forming the dense Hessian.
Janus now exposes matrix-free Hessian-vector products for both plain scalar expressions and
Lagrangians:

```cpp
auto x = janus::sym("x", 3);
auto v = janus::sym("v", 3);
auto lam = janus::sym("lam", 2);

casadi::MX x0 = x(0);
casadi::MX x1 = x(1);
casadi::MX x2 = x(2);

auto objective = x0 * x0 + x1 * x2 + janus::sin(x2);
auto constraints = casadi::MX::vertcat({x0 + x1, x1 * x2});

auto hvp = janus::hessian_vector_product(objective, x, v);
auto lag_hvp =
    janus::lagrangian_hessian_vector_product(objective, constraints, x, lam, v);
```

For compiled `janus::Function` objects, the wrappers return another `janus::Function`:

```cpp
janus::Function model("model", {x}, {objective});
janus::Function hvp_fun = janus::hessian_vector_product(model, 0, 0);

auto hv = hvp_fun.eval(x_val, v_val);   // original inputs..., then direction v
```

The Lagrangian variant appends the multiplier block first and the direction block last:

```cpp
janus::Function nlp_terms("nlp_terms", {x}, {objective, constraints});
janus::Function lag_hvp_fun =
    janus::lagrangian_hessian_vector_product(nlp_terms, 0, 1, 0);

auto hv = lag_hvp_fun.eval(x_val, lam_val, v_val);
```

These products use CasADi's forward-over-reverse AD internally, so the dense Hessian is never
constructed as an intermediate.

## 8. Full Workflow Example

```cpp
// 1. Symbols
auto x = janus::sym("x");

// 2. Expression
auto f = x * x;

// 3. Jacobian
auto J = janus::jacobian({f}, {x});

// 4. Compile into Function
janus::Function fn({x}, {f, J});

// 5. Evaluate
auto res = fn(3.0);
std::cout << "Values: " << res[0] << ", Gradient: " << res[1] << std::endl;
```
