# Numeric Computing Guide

Janus is designed to be "Template-First". This means you can write your physics and math logic using generic templates (`T`), and compile them into highly optimized machine code using the Numeric Backend (standard `double` and `Eigen`).

## 1. Concepts

*   **`janus::NumericScalar`**: Alias for `double`.
*   **`janus::NumericMatrix`**: Alias for `Eigen::MatrixXd`.
*   **`janus::JanusScalar` Concept**: A C++20 concept that matches both `numeric` and `symbolic` scalar types. Use this for templating your functions.

## 2. Writing Generic Code

To support both modes, write your functions as templates:

```cpp
template <typename Scalar>
Scalar my_physics(const Scalar& x) {
    // Use janus:: math namespace, NOT std::
    return janus::sin(x) + janus::pow(x, 2.0);
}
```

## 3. Numeric Execution

When you instantiate your template with `double` (or `janus::NumericScalar`), Janus compiles down to direct C++ math calls and Eigen operations.

*   **Zero Overhead**: `janus::sin` directly calls `std::sin`.
*   **Eigen Speed**: Matrix operations use Eigen's optimized expression templates and vectorization (AVX, SSE).

```cpp
// Instantiation
double result = my_physics(10.0);
```

## 4. Linear Algebra

Janus provides wrappers for common linear algebra operations that work for both backends. In numeric mode, these delegate directly to efficient Eigen Decompositions.

*   `janus::solve(A, b)` -> Uses `householderQr().solve()`.
*   `janus::inv(A)` -> Uses `inverse()`.
*   `janus::det(A)` -> Uses `determinant()`.

## 5. Performance Tips

*   **Avoid `auto` types in return signatures** of public APIs if you can use `Scalar` or `JanusMatrix<Scalar>`.
*   **Use `janus::where` instead of `if/else`**: This ensures your code works in symbolic mode and compiles to efficient branchless select instructions (where possible) in numeric mode.
*   **Pass by Reference**: `const Scalar&` avoids copying expensive types in symbolic mode, and is negligible for doubles.

## 6. Example

See `examples/numeric_intro.cpp` for a benchmark of numeric operations.
