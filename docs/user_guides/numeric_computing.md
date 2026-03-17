# Numeric Computing

Janus is designed to be "Template-First": you write your physics and math logic using generic templates (`T`), and compile them into highly optimized machine code using the Numeric Backend (standard `double` and `Eigen`). Numeric mode provides zero-overhead math calls and Eigen's optimized expression templates with vectorization (AVX, SSE). This guide covers writing generic code, numeric execution, linear algebra, and performance tips for the numeric path.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Write a generic function using janus:: math
template <typename Scalar>
Scalar my_physics(const Scalar& x) {
    return janus::sin(x) + janus::pow(x, 2.0);
}

// Instantiate with double for numeric execution
double result = my_physics(10.0);
std::cout << "Result: " << result << std::endl;
```

## Core API

*   **`janus::NumericScalar`**: Alias for `double`.
*   **`janus::NumericMatrix`**: Alias for `Eigen::MatrixXd`.
*   **`janus::JanusScalar` Concept**: A C++20 concept that matches both `numeric` and `symbolic` scalar types. Use this for templating your functions.
*   `janus::solve(A, b)` -> Uses the default backend policy (`ColPivHouseholderQR` for dense numeric matrices, CasADi default for symbolic, `SparseLU` for sparse numeric input).
*   `janus::solve(A, b, policy)` -> Select dense direct, sparse direct, or iterative Krylov backends explicitly with `janus::LinearSolvePolicy`.
*   `janus::inv(A)` -> Uses `inverse()`.
*   `janus::det(A)` -> Uses `determinant()`.

## Usage Patterns

### Writing Generic Code

To support both numeric and symbolic modes, write your functions as templates:

```cpp
template <typename Scalar>
Scalar my_physics(const Scalar& x) {
    // Use janus:: math namespace, NOT std::
    return janus::sin(x) + janus::pow(x, 2.0);
}
```

### Numeric Execution

When you instantiate your template with `double` (or `janus::NumericScalar`), Janus compiles down to direct C++ math calls and Eigen operations.

*   **Zero Overhead**: `janus::sin` directly calls `std::sin`.
*   **Eigen Speed**: Matrix operations use Eigen's optimized expression templates and vectorization (AVX, SSE).

```cpp
// Instantiation
double result = my_physics(10.0);
```

### Linear Algebra

Janus provides wrappers for common linear algebra operations that work for both backends. In numeric mode, these delegate directly to efficient Eigen Decompositions.

```cpp
janus::NumericMatrix A = /* ... */;
janus::NumericVector b = /* ... */;

// Default solver (ColPivHouseholderQR for dense)
auto x = janus::solve(A, b);

// Explicit policy selection
auto x2 = janus::solve(A, b, janus::LinearSolvePolicy::SparseDirect);
```

## Advanced Usage

*   **Avoid `auto` types in return signatures** of public APIs if you can use `Scalar` or `JanusMatrix<Scalar>`.
*   **Use `janus::where` instead of `if/else`**: This ensures your code works in symbolic mode and compiles to efficient branchless select instructions (where possible) in numeric mode.
*   **Pass by Reference**: `const Scalar&` avoids copying expensive types in symbolic mode, and is negligible for doubles.

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - The symbolic counterpart to numeric mode
- [Math Functions Guide](math_functions.md) - All available `janus::` math functions
- [`examples/intro/numeric_intro.cpp`](../../examples/intro/numeric_intro.cpp) - Benchmark of numeric operations
- [`include/janus/core/JanusTypes.hpp`](../../include/janus/core/JanusTypes.hpp) - Type alias definitions
