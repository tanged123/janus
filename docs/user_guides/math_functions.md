# Math Functions

Janus provides a comprehensive set of math functions that work seamlessly with both numeric (`double`) and symbolic (`SymbolicScalar`) types. When called with a `janus::SymbolicScalar` argument, C++ automatically finds the correct overload via ADL (Argument-Dependent Lookup), so you can write `sin(x)` or `janus::sin(x)` interchangeably. These functions are the foundation for writing generic template code that compiles to both fast numeric execution and symbolic graph construction.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Works with both double and SymbolicScalar
template <typename Scalar>
Scalar my_function(const Scalar& x, const Scalar& y) {
    return janus::sin(x) + janus::pow(y, 2.0) + janus::exp(-x * y);
}

// Numeric usage
double result = my_function(1.0, 2.0);

// Symbolic usage
auto x = janus::sym("x");
auto y = janus::sym("y");
auto expr = my_function(x, y);
```

## Core API

### Trigonometric
- `janus::sin(x)`, `janus::cos(x)`, `janus::tan(x)`
- `janus::asin(x)`, `janus::acos(x)`, `janus::atan(x)`, `janus::atan2(y, x)`

### Exponential & Logarithmic
- `janus::exp(x)`, `janus::log(x)`, `janus::log10(x)`
- `janus::pow(base, exp)`, `janus::sqrt(x)`

### Hyperbolic
- `janus::sinh(x)`, `janus::cosh(x)`, `janus::tanh(x)`

### Utility
- `janus::abs(x)`, `janus::fabs(x)`
- `janus::floor(x)`, `janus::ceil(x)`
- `janus::fmin(a, b)`, `janus::fmax(a, b)`

### Control Flow
- `janus::where(condition, if_true, if_false)` - Branch-free conditional that works in both numeric and symbolic mode

## Usage Patterns

### ADL (Argument-Dependent Lookup)

When you call a math function with a `janus::SymbolicScalar` argument, C++ automatically finds the correct function via ADL:

```cpp
janus::SymbolicScalar x = janus::sym("x");
auto y = sin(x);  // Finds janus::sin via ADL
auto z = pow(x, 2);  // Finds janus::pow via ADL
```

For mixed types or explicit calls:
```cpp
auto y = janus::sin(x);  // Always works
auto z = janus::pow(x, 2.0);  // Mixed symbolic/numeric
```

### Branch-Free Conditionals with `janus::where`

Use `janus::where` instead of `if/else` to ensure your code works in symbolic mode:

```cpp
template <typename Scalar>
Scalar safe_sqrt(const Scalar& x) {
    return janus::where(x > 0.0, janus::sqrt(x), Scalar(0.0));
}
```

In numeric mode, this compiles to efficient branchless select instructions (where possible). In symbolic mode, it builds a valid conditional node in the computation graph.

### Convenience Header

For cleaner code, use the convenience header:

```cpp
#include <janus/using.hpp>

// Now sym, sin, cos, pow, where are in scope
auto x = sym("x");
auto y = sin(x) + pow(x, 2);
```

## See Also

- [Numeric Computing Guide](numeric_computing.md) - Writing generic template code
- [Symbolic Computing Guide](symbolic_computing.md) - Building symbolic expressions
- [`include/janus/math/Trig.hpp`](../../include/janus/math/Trig.hpp) - Trigonometric function implementations
- [`include/janus/math/Arithmetic.hpp`](../../include/janus/math/Arithmetic.hpp) - Arithmetic function implementations
- [`include/janus/math/Logic.hpp`](../../include/janus/math/Logic.hpp) - `where` and branching logic
- [`examples/math/branching_logic.cpp`](../../examples/math/branching_logic.cpp) - Branching logic examples
