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

## 6. Full Workflow Example

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
