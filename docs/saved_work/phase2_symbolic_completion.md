# Phase 2: Symbolic & Numerics Completion

**Date:** 2025-12-13
**Status:** Complete / Validation Passed (22/22 Tests)

## Summary of Work
This work expands the Janus framework to fully support symbolic computation and automatic differentiation with a cleaner, C++ nativelike API. It also completes the logical and linear algebra operator sets.

### 1. Functional Improvements
*   **Logic Operators**: Implemented `gt`, `le`, `ge`, `eq`, `neq` for Matrix types (both numeric and symbolic).
*   **Linear Algebra**: Added `dot`, `cross`, `inv`, `det` with dual-backend support.
*   **Symbolic Wrappers**:
    *   `janus::sym(name, [rows, cols])`: Helper to create symbolic variables without `casadi::MX` syntax.
    *   `janus::Function`: A robust wrapper around `casadi::Function` that accepts/returns Eigen types and handles naming automatically (using thread-safe atomic counters).
    *   `janus::jacobian({exprs}, {vars})`: Helper for AD that handles variable concatenation automatically.

### 2. Code Quality & Safety
*   **Thread-Safety**: Replaced `std::rand()` in function naming with `std::atomic<uint64_t>` to ensure robust, unique function names without race conditions.
*   **Testing**:
    *   Refactored monolithic `test_math` into granular test suites (`test_logic.cpp`, `test_function.cpp`, etc.).
    *   Added comprehensive coverage for new `Function` wrapper and logic operators.
    *   Verified "Jacobian" computation against analytic results in `drag_coefficient` example.

### 3. Files Created/Modified
*   `include/janus/math/Logic.hpp`: Logic operators.
*   `include/janus/math/Linalg.hpp`: Linalg operators.
*   `include/janus/math/DiffOps.hpp`: `jacobian` helper.
*   `include/janus/core/Function.hpp`: **NEW** `janus::Function` class.
*   `include/janus/core/JanusTypes.hpp`: `janus::sym` helpers.
*   `examples/drag_coefficient.cpp`: Updated to use new API.
*   `tests/core/test_function.cpp`: **NEW** Tests for Function wrapper.
*   `tests/math/test_*.cpp`: Modularized tests.

---

## symbolic_api_guide.md
*(Included here for reference, as requested)*

# Symbolic Computing in Janus Reference

Janus provides a unified, user-friendly interface for symbolic computation.

### 1. Creating Symbolics (`janus::sym`)
Instead of verbose CasADi calls, use the helper functions:

```cpp
auto x = janus::sym("x");        // Scalar
auto v = janus::sym("v", 3);     // Vector (3x1)
auto M = janus::sym("M", 2, 2);  // Matrix (2x2)
```

### 2. Defining Functions (`janus::Function`)
Wrap your symbolic expressions into callable functions. The wrapper handles type conversions (Eigen/CasADi) and unique naming automatically.

```cpp
// Define inputs and outputs: f(x, y) -> z
janus::Function f({x, y}, {z});

// Evaluate directly with numeric types
auto res = f(1.0, 2.0); // Returns std::vector<Eigen::MatrixXd>
std::cout << res[0] << std::endl;
```

### 3. Automatic Differentiation (`janus::jacobian`)
Compute Jacobians easily without manual vector stacking.

```cpp
// Compute J = dy/dx
auto J = janus::jacobian({y}, {x});

// Compute J = dy/d[v1, v2] (concatenates variables automatically)
auto J_total = janus::jacobian({y}, {v1, v2});

// Create a function for the Jacobian
janus::Function f_J({x}, {J});
```
