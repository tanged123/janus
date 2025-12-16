# Janus Optimization Guide

This guide walks you through using the Janus Optimization Framework (`janus::Opti`), demonstrating how to solve constrained nonlinear optimization problems (NLP) while reusing your simulation code.

## 1. Introduction to `janus::Opti`

`janus::Opti` is a high-level C++ interface to cell-bounded optimization solvers (currently wrapping CasADi/IPOPT). It allows you to:
- Define optimization variables (scalars or vectors).
- Write objectives and constraints using standard C++ syntax.
- Reuse your physicist-written simulation models directly in the optimization loop.

---

## 2. Example 1: The Rosenbrock Benchmark

The Rosenbrock function (or "banana function") is a classic test for optimization algorithms.
Code reference: [`examples/optimization/rosenbrock.cpp`](../../examples/optimization/rosenbrock.cpp)

### Step 1: Initialize the Solver
```cpp
#include <janus/janus.hpp>

janus::Opti opti;
```

### Step 2: Define Variables
Use `.variable()` to create decision variables. You can provide an initial guess.
```cpp
auto x = opti.variable(0.0); // Initial guess x=0
auto y = opti.variable(0.0); // Initial guess y=0
```

### Step 3: Define the Objective
The goal is to minimize the function $f(x, y) = (1-x)^2 + 100(y-x^2)^2$.
```cpp
auto f = janus::pow(1 - x, 2) + 100 * janus::pow(y - janus::pow(x, 2), 2);
opti.minimize(f);
```

### Step 4: Add Constraints
You can add equality (`==`) or inequality (`<=`, `>=`) constraints.
```cpp
// Example: constrain the solution to a specific region
opti.subject_to(janus::pow(x, 2) + janus::pow(y, 2) <= 2.0); // Unit circle
```

### Step 5: Solve and Retrieve Results
Call `.solve()` to execute IPOPT. You can pass options like `max_iter` or `verbose`.
```cpp
auto sol = opti.solve({.max_iter = 500, .verbose = false});

double x_star = sol.value(x);
double y_star = sol.value(y);

std::cout << "Optimal Solution: x=" << x_star << ", y=" << y_star << std::endl;
```

---

## 3. Example 2: "Write Once, Use Everywhere" (C++20 Style)

The true power of Janus is reusing your existing physics code. With C++20 **Abbreviated Function Templates** (`auto` parameters), this is seamless.

Code reference: [`examples/optimization/drag_optimization.cpp`](../../examples/optimization/drag_optimization.cpp)

### Step 1: The Shared Physics Function
Use `auto` for arguments whenever you want to support mixed types (e.g., multiplying a `double` constant by a `SymbolicScalar` variable).

```cpp
// Works for double, SymbolicScalar, or a mix of both!
auto compute_drag(auto rho, auto v, auto S, 
                  auto Cd0, auto k, auto Cl, auto Cl0) {
    // Dynamic pressure
    auto q = 0.5 * rho * janus::pow(v, 2.0);
    // Drag polar
    auto Cd = Cd0 + k * janus::pow(Cl - Cl0, 2.0);
    // Drag force
    return q * S * Cd;
}
```

### Step 2: Using it in Optimization
Just pass your variables and constants. The compiler deduces the return type automatically (`double * Symbolic -> Symbolic`).

```cpp
janus::Opti opti;

// 1. Decision Variables (Symbolic)
auto V = opti.variable(50.0); 
auto Cl = opti.variable(0.5);

// 2. Constants (Numeric)
const double rho = 1.225;
const double S = 16.0;

// 3. Call the SHARED function directly
// No casting, no explicit templates needed. Just like Python!
auto D = compute_drag(
    rho,            // double
    V,              // SymbolicScalar
    S,              // double
    0.02,           // double literal
    0.04,           // double literal
    Cl,             // SymbolicScalar
    0.1             // double literal
);

// 4. Optimize directly on the physics result
opti.minimize(D); // Minimize Drag
opti.subject_to_bounds(V, 20.0, 150.0);
```

### Why this matters
*   **Zero Boilerplate**: No `template <typename T>` syntax required for users.
*   **Type Safety**: Still statically checked at compile time.
*   **Performance**: Compiles down to optimized machine code (Numeric mode) or efficient graph construction (Symbolic mode).

---

## 4. Advanced: Bounds Handling

Janus provides explicit helpers for variable bounds, which is more efficient than general constraints for the solver.

```cpp
// Scalar variable bounds
auto x = opti.variable(0.5);
opti.subject_to_bounds(x, 0.0, 1.0);  // 0 <= x <= 1
opti.subject_to_lower(x, 0.0);        // x >= 0
opti.subject_to_upper(x, 1.0);        // x <= 1

// Vector variable bounds (applies to all elements)
auto vec = opti.variable(10, 0.0);
opti.subject_to_bounds(vec, -5.0, 5.0);
```

For more details on vector operations and trajectory optimization, see the [Brachistochrone Example](../../examples/optimization/brachistochrone_opti.cpp).
