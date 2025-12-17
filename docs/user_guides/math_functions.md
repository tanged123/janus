# Math Functions Guide

Janus provides a comprehensive set of math functions that work seamlessly with both numeric (`double`) and symbolic (`SymbolicScalar`) types.

## ADL (Argument-Dependent Lookup)

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

## Available Functions

### Trigonometric
- `sin(x)`, `cos(x)`, `tan(x)`
- `asin(x)`, `acos(x)`, `atan(x)`, `atan2(y, x)`

### Exponential & Logarithmic
- `exp(x)`, `log(x)`, `log10(x)`
- `pow(base, exp)`, `sqrt(x)`

### Hyperbolic
- `sinh(x)`, `cosh(x)`, `tanh(x)`

### Utility
- `abs(x)`, `fabs(x)`
- `floor(x)`, `ceil(x)`
- `fmin(a, b)`, `fmax(a, b)`

### Control Flow
- `where(condition, if_true, if_false)` - Branch-free conditional

## Best Practice

For cleaner code, use the convenience header:
```cpp
#include <janus/using.hpp>

// Now sym, sin, cos, pow, where are in scope
auto x = sym("x");
auto y = sin(x) + pow(x, 2);
```
