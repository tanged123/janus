# Symbolic Computing

Janus provides a powerful symbolic computing layer built on top of CasADi, abstracted to feel like native C++ with Eigen integration. This allows you to compute derivatives, generate code, and optimize systems using the same code you write for simulation. Symbolic mode works by building a computational graph instead of executing immediately, enabling automatic differentiation, sensitivity analysis, and matrix-free second-order products.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Create a symbolic variable
auto x = janus::sym("x");

// Build an expression (creates a computation graph)
auto f = x * x;

// Compute the Jacobian symbolically
auto J = janus::jacobian({f}, {x});

// Compile into a callable function
janus::Function fn({x}, {f, J});

// Evaluate numerically
auto res = fn(3.0);
std::cout << "f(3) = " << res[0] << ", f'(3) = " << res[1] << std::endl;
```

## Core API

*   **`janus::SymbolicScalar`**: Alias for `casadi::MX`. Represents a symbolic value or expression.
*   **`janus::SymbolicMatrix`**: Alias for `Eigen::Matrix<casadi::MX, -1, -1>`. Allows you to use familiar Eigen syntax (block operations, coeff access) on symbolic variables.
*   **`janus::sym(name)`**: Create a scalar symbolic variable.
*   **`janus::sym(name, n)`**: Create a column vector symbolic variable (`n x 1`).
*   **`janus::sym(name, r, c)`**: Create a matrix symbolic variable (`r x c`).
*   **`janus::Function({inputs}, {outputs})`**: Compile symbolic expressions into a callable function.
*   **`janus::jacobian({outputs}, {inputs})`**: Compute the Jacobian of outputs with respect to inputs.

## Usage Patterns

### Creating Variables

Use the `janus::sym` helper to create symbolic variables cleanly.

```cpp
// Scalar variable "x"
auto x = janus::sym("x");

// Column vector "v" (3x1)
auto v = janus::sym("v", 3);

// Matrix "A" (2x2)
auto A = janus::sym("A", 2, 2);
```

### Building Expressions

You can use standard arithmetic operators (`+`, `-`, `*`, `/`) and Janus math functions. These build a computational graph instead of executing immediately.

```cpp
auto y = janus::sin(x) + janus::pow(x, 2.0);
auto z = A * v; // Matrix multiplication
```

### Defining Functions (`janus::Function`)

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

### Automatic Differentiation (`janus::jacobian`)

Compute derivatives effortlessly. Janus handles the tedious task of concatenating variables for CasADi.

```cpp
// Compute Jacobian J = dy/dx
auto J = janus::jacobian({y}, {x});

// Compute Jacobian w.r.t multiple variables: J = dy/d[x, v]
auto J_full = janus::jacobian({y}, {x, v});
```

## Advanced Usage

### Sensitivity Regime Switching

For compiled `janus::Function` objects, Janus can choose between forward and adjoint
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

### Matrix-Free Second-Order Products

For large optimization problems, you often want `H * v` without ever forming the dense Hessian.
Janus exposes matrix-free Hessian-vector products for both plain scalar expressions and
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

## See Also

- [Numeric Computing Guide](numeric_computing.md) - The numeric counterpart to symbolic mode
- [Math Functions Guide](math_functions.md) - All available `janus::` math functions
- [Optimization Guide](optimization.md) - Using symbolic expressions in optimization
- [`include/janus/core/Function.hpp`](../../include/janus/core/Function.hpp) - `janus::Function` implementation
- [`include/janus/math/AutoDiff.hpp`](../../include/janus/math/AutoDiff.hpp) - Jacobian and Hessian-vector product API
- [`examples/intro/energy_intro.cpp`](../../examples/intro/energy_intro.cpp) - Introductory symbolic example
