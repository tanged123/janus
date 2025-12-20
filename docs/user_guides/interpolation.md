# N-Dimensional Interpolation User Guide

This guide covers Janus's N-dimensional interpolation capabilities, one of the core features for working with gridded data in both numeric and symbolic modes.

## Overview

The `interpn` function provides N-dimensional interpolation on regular grids, supporting dimensions from 1D up to 7D and beyond. It's designed to work seamlessly with Janus's dual-backend architecture, allowing the same code to run in both fast numeric mode and symbolic trace mode for optimization.

## Quick Start

```cpp
#include <janus/math/Interpolate.hpp>

// Create a 2D grid
janus::NumericVector x_pts(3), y_pts(3);
x_pts << 0.0, 1.0, 2.0;
y_pts << 0.0, 1.0, 2.0;

std::vector<janus::NumericVector> points = {x_pts, y_pts};

// Grid values: z = x + y (Fortran order)
janus::NumericVector values(9);
values << 0, 1, 2, 1, 2, 3, 2, 3, 4;

// Query points
janus::NumericMatrix xi(2, 2);
xi << 0.5, 0.5,   // Point 1
      1.5, 1.0;   // Point 2

// Interpolate
auto result = janus::interpn<double>(points, values, xi);
// result(0) ≈ 1.0, result(1) ≈ 2.5
```

---

## Interpolation Methods

Janus supports four interpolation methods with different smoothness properties:

| Method | Enum Value | Continuity | Symbolic | Description |
|--------|------------|------------|----------|-------------|
| **Linear** | `InterpolationMethod::Linear` | C0 | ✅ Yes | Piecewise linear. Fast, simple, but has gradient discontinuities at grid points. |
| **Hermite** | `InterpolationMethod::Hermite` | C1 | ❌ Numeric only | Catmull-Rom cubic spline. Smooth gradients, good for animation and trajectories. |
| **BSpline** | `InterpolationMethod::BSpline` | C2 | ✅ Yes | Cubic B-spline. Smoothest option, ideal for optimization. Requires ≥4 points per dimension. |
| **Nearest** | `InterpolationMethod::Nearest` | None | ❌ Numeric only | Nearest neighbor. Fast lookup, non-differentiable. |

### Choosing a Method

```
Use Case                              Recommended Method
─────────────────────────────────────────────────────────
Fast lookup, low accuracy             Nearest
General purpose, good balance         Linear
Smooth trajectories, C1 gradients     Hermite
Optimization problems, C2 smooth      BSpline
Symbolic differentiation              Linear or BSpline
```

### Smoothness Comparison

```
                Grid Point
                    │
                    ▼
    ┌───────────────●───────────────┐
    │               │               │
C0  │    ╱──────────●──────────╲    │  Linear: Gradient jumps at knots
    │   ╱           │           ╲   │
    ├───────────────●───────────────┤
    │            ╱──●──╲            │
C1  │          ╱    │    ╲          │  Hermite: Smooth gradient, curvature jump
    │         ╱     │     ╲         │
    ├───────────────●───────────────┤
    │           ╭───●───╮           │
C2  │          ╭    │    ╮          │  BSpline: Smooth gradient AND curvature
    │         ╭     │     ╮         │
    └───────────────────────────────┘
```

---

## API Reference

### `interpn<Scalar>()`

```cpp
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
interpn(const std::vector<Eigen::VectorXd>& points,
        const Eigen::VectorXd& values_flat,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& xi,
        InterpolationMethod method = InterpolationMethod::Linear,
        std::optional<Scalar> fill_value = std::nullopt);
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `std::vector<Eigen::VectorXd>` | Grid coordinates for each dimension. Each vector must be sorted. |
| `values_flat` | `Eigen::VectorXd` | Flattened grid values in **Fortran order** (column-major). |
| `xi` | `Eigen::Matrix<Scalar, ...>` | Query points, shape `(n_points, n_dims)` or `(n_dims, n_points)`. |
| `method` | `InterpolationMethod` | Interpolation algorithm (default: `Linear`). |
| `fill_value` | `std::optional<Scalar>` | Value for out-of-bounds queries. If `nullopt`, clamps to boundary. |

#### Returns

`Eigen::Matrix<Scalar, Dynamic, 1>` - Vector of interpolated values at each query point.

#### Type Aliases (Recommended)

Use Janus types for better readability:

```cpp
// Numeric
janus::NumericVector  // = Eigen::VectorXd
janus::NumericMatrix  // = Eigen::MatrixXd
janus::NumericScalar  // = double

// Symbolic
janus::SymbolicVector // = Eigen::Matrix<casadi::MX, Dynamic, 1>
janus::SymbolicMatrix // = Eigen::Matrix<casadi::MX, Dynamic, Dynamic>
janus::SymbolicScalar // = casadi::MX
```

---

## Data Layout: Fortran Order

Grid values must be flattened in **Fortran order** (column-major), where the first dimension varies fastest.

### 2D Example

For a 2×3 grid with coordinates `x = [0, 1]` and `y = [0, 1, 2]`:

```
Grid Layout (visual):          Flattening Order:
    y=0   y=1   y=2
x=0  a     b     c            [a, d, b, e, c, f]
x=1  d     e     f             ↑  ↑
                               └──┴── x varies fastest
```

### Code Example

```cpp
// 2D grid: x = [0, 1], y = [0, 1, 2]
janus::NumericVector x(2), y(3);
x << 0, 1;
y << 0, 1, 2;

// Values: z(x,y) = x + y
// Fortran order: iterate x first for each y
janus::NumericVector values(6);
values << 0,   // (0,0) = 0+0
          1,   // (1,0) = 1+0
          1,   // (0,1) = 0+1
          2,   // (1,1) = 1+1
          2,   // (0,2) = 0+2
          3;   // (1,2) = 1+2
```

---

## Boundary Handling

### Default: Clamping

When `fill_value` is not provided, out-of-bounds queries are **clamped** to the grid boundary:

```cpp
// Query outside grid bounds
xi << -1.0, 0.5;  // x = -1 is outside [0, 1]

auto result = janus::interpn<double>(points, values, xi);
// Effectively queries at (0.0, 0.5) - clamped to boundary
```

### Fill Value

Use `fill_value` to return a specific value for out-of-bounds queries:

```cpp
auto result = janus::interpn<double>(
    points, values, xi,
    janus::InterpolationMethod::Linear,
    std::optional<double>(-999.0)  // Fill value
);
// Returns -999.0 for any query outside grid bounds
```

### Linear Extrapolation (1D Only)

For optimization problems, clamping produces **zero gradients** outside bounds, which can stall solvers. The `ExtrapolationConfig` class provides linear extrapolation with optional safety bounds:

```cpp
#include <janus/math/Interpolate.hpp>

janus::NumericVector x(4), y(4);
x << 0, 1, 2, 3;
y << 0, 10, 40, 90;  // y = 10*x^2 sampled

// Linear extrapolation with output bounds [0, 200]
janus::Interpolator interp(x, y,
    janus::InterpolationMethod::BSpline,
    janus::ExtrapolationConfig::linear(0.0, 200.0));

// Query outside bounds
double val = interp(4.0);  // Extrapolates linearly from boundary slope
                            // then clamps result to [0, 200]
```

#### ExtrapolationConfig Factory Methods

| Factory Method | Behavior |
|----------------|----------|
| `ExtrapolationConfig::clamp()` | Clamp queries to grid bounds (default, safe) |
| `ExtrapolationConfig::linear()` | Linear extrapolation, unbounded |
| `ExtrapolationConfig::linear(lower, upper)` | Linear extrapolation with output bounds |

#### How Linear Extrapolation Works

```
                Left extrapolation          Right extrapolation
                     │                           │
      ╲              │              ╱            │              ╱
       ╲             │  ●──────────●            │  ●──────────●
        ╲            │ ╱            ╲           │ ╱            ╲
─────────●──────────●──────────────●───────────●──────────────●─────────
       x_min                     x_max
         │                         │
         └─ slope = (y[1]-y[0])    └─ slope = (y[n-1]-y[n-2])
               ───────────             ───────────────────
              (x[1]-x[0])              (x[n-1]-x[n-2])
```

#### Symbolic Support

Linear extrapolation works in symbolic mode with full AD support:

```cpp
auto x_sym = janus::sym("x");
auto y_sym = interp(x_sym);

// Compute gradient (non-zero even outside bounds!)
auto dy_dx = janus::jacobian(y_sym, x_sym);
```

> [!IMPORTANT]
> Linear extrapolation is currently **1D only**. For N-D interpolators, use clamping or add physical constraints to your optimization.

---

## Implementation Details

### CasADi Backend

Both numeric and symbolic interpolation use **CasADi** as the underlying engine:

```
┌─────────────────────────────────────────────────────┐
│                   janus::interpn                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Scalar = double        Scalar = casadi::MX          │
│  ────────────────       ─────────────────────        │
│        │                       │                     │
│        ▼                       ▼                     │
│  ┌───────────────────────────────────────────┐      │
│  │         casadi::interpolant()              │      │
│  │  Creates CasADi Function for grid lookup   │      │
│  └───────────────────────────────────────────┘      │
│        │                       │                     │
│        ▼                       ▼                     │
│  casadi::DM call         casadi::MX call             │
│  (numeric result)        (symbolic expression)       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

#### Why CasADi?

1. **Single Implementation**: Same code path for numeric and symbolic
2. **Automatic Differentiation**: Symbolic mode generates AD-compatible graphs
3. **Optimization Ready**: Interpolated values can be optimization variables
4. **Performance**: CasADi's C++ core is highly optimized

### Hermite (C1) Implementation

The Hermite method uses a **custom Catmull-Rom implementation** because CasADi doesn't provide a native C1 interpolant:

```cpp
// Catmull-Rom slope estimation at grid point i:
// m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

// Cubic Hermite basis functions:
// h00(t) = 2t³ - 3t² + 1
// h10(t) = t³ - 2t² + t
// h01(t) = -2t³ + 3t²
// h11(t) = t³ - t²

// Interpolated value:
// p(t) = h00*y0 + h10*m0*h + h01*y1 + h11*m1*h
```

This provides C1 continuity (continuous first derivative) at all grid points.

---

## High-Dimensional Interpolation

Janus supports interpolation in arbitrary dimensions using **tensor product** extension:

```cpp
// 5D interpolation example
janus::NumericVector pts(3);
pts << 0.0, 1.0, 2.0;

std::vector<janus::NumericVector> points(5, pts);  // 5 dimensions

// Values: 3^5 = 243 grid points
janus::NumericVector values(243);
// ... fill values ...

janus::NumericMatrix xi(1, 5);
xi << 0.5, 0.5, 0.5, 0.5, 0.5;

auto result = janus::interpn<double>(points, values, xi);
```

### Performance Considerations

| Dimensions | Grid Size | Points | Memory |
|------------|-----------|--------|--------|
| 2D | 10×10 | 100 | ~1 KB |
| 3D | 10×10×10 | 1,000 | ~8 KB |
| 4D | 10×10×10×10 | 10,000 | ~80 KB |
| 5D | 10×10×10×10×10 | 100,000 | ~800 KB |

> **Note**: High-dimensional grids grow exponentially. Consider sparse grids or surrogate models for >5D.

---

## Symbolic Mode

For optimization problems, use symbolic interpolation to embed lookups in your objective:

```cpp
// Create symbolic query variables
janus::SymbolicMatrix xi(1, 2);
xi(0, 0) = janus::sym("x");
xi(0, 1) = janus::sym("y");

// Symbolic interpolation
auto z_sym = janus::interpn<janus::SymbolicScalar>(
    points, values, xi,
    janus::InterpolationMethod::BSpline  // C2 smooth for optimization
);

// Use in optimization objective
auto f = janus::Function("lookup", {xi(0,0), xi(0,1)}, {z_sym(0)});
```

### Method Compatibility

| Method | Symbolic Mode | Reason |
|--------|---------------|--------|
| Linear | ✅ | Uses CasADi `interpolant` |
| Hermite | ❌ | Requires interval finding (branching) |
| BSpline | ✅ | Uses CasADi `interpolant` |
| Nearest | ❌ | Requires rounding (non-smooth) |

---

## Common Patterns

### Loading Data from File

```cpp
// Example: Load aerodynamic coefficients from CSV
Eigen::MatrixXd data = load_csv("aero_coeffs.csv");

// Extract grid axes
janus::NumericVector mach = data.col(0).head(n_mach);
janus::NumericVector alpha = data.col(1).head(n_alpha);
janus::NumericVector CL = data.col(2);

std::vector<janus::NumericVector> points = {mach, alpha};

// Query lift coefficient at any (Mach, alpha)
auto cl = janus::interpn<double>(points, CL, query_pts);
```

### Surrogate Model in Optimization

```cpp
template <typename Scalar>
Scalar drag_model(const janus::JanusVector<Scalar>& state) {
    // Use BSpline for C2 smooth optimization
    janus::JanusMatrix<Scalar> query(1, 2);
    query(0, 0) = state(0);  // Mach
    query(0, 1) = state(1);  // Angle of attack
    
    auto cd = janus::interpn<Scalar>(
        aero_grid, cd_values, query,
        janus::InterpolationMethod::BSpline
    );
    return cd(0);
}
```

---

## Error Handling

The interpolation functions throw `janus::InterpolationError` for invalid inputs:

```cpp
try {
    auto result = janus::interpn<double>(points, values, xi);
} catch (const janus::InterpolationError& e) {
    std::cerr << "Interpolation failed: " << e.what() << std::endl;
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "points cannot be empty" | Empty grid | Provide at least 2 points per dimension |
| "points must be sorted" | Unsorted grid axis | Sort grid coordinates ascending |
| "values_flat size mismatch" | Wrong value count | Ensure `prod(dims)` values in Fortran order |
| "Hermite not supported for symbolic" | Using Hermite with MX | Use Linear or BSpline for symbolic |
| "BSpline: Need more data points" | Grid < 4 points | Use ≥4 points per dimension for BSpline |

---

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - Working with CasADi symbolic types
- [Numeric Computing Guide](numeric_computing.md) - Janus type system overview
- `include/janus/math/Interpolate.hpp` - Full API implementation
