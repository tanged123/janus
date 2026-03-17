# Direct Collocation

Direct collocation transforms continuous-time optimal control problems into large sparse NLPs by discretizing time into nodes and enforcing dynamics at each segment using defect constraints. Janus provides the `janus::DirectCollocation` class in `<janus/optimization/Collocation.hpp>`. It supports trapezoidal (2nd order) and Hermite-Simpson (4th order) schemes and works in **symbolic mode** via the `janus::Opti` interface.

## Quick Start

```cpp
#include <janus/janus.hpp>

janus::Opti opti;
janus::DirectCollocation dc(opti);

auto [x, u, tau] = dc.setup(
    2, 1, 0.0, 2.0,
    {.scheme = janus::CollocationScheme::HermiteSimpson, .n_nodes = 31}
);

dc.set_dynamics([](const auto& x, const auto& u, const auto& t) {
    janus::SymbolicVector dxdt(2);
    dxdt(0) = x(1);
    dxdt(1) = u(0);
    return dxdt;
});

dc.add_defect_constraints();
dc.set_initial_state(janus::NumericVector{{0.0, 0.0}});
dc.set_final_state(janus::NumericVector{{1.0, 0.0}});

opti.minimize(/* objective */);
auto sol = opti.solve();
```

## Core API

| Method | Description |
|--------|-------------|
| `DirectCollocation(opti)` | Construct with a `janus::Opti` instance |
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function: `(x, u, t) -> dxdt` |
| `add_defect_constraints()` | Apply collocation defect constraints |
| `add_dynamics_constraints()` | Unified alias for `add_defect_constraints()` |
| `set_initial_state(x0)` | Set initial boundary condition (full vector or per-index) |
| `set_final_state(xf)` | Set final boundary condition (full vector or per-index) |
| `n_nodes()` | Number of collocation nodes |
| `time_grid()` | Normalized time grid `[0, 1]` |

**Collocation schemes:**

| Scheme | Order | Description |
|--------|-------|-------------|
| `CollocationScheme::Trapezoidal` | 2nd | Uses average of endpoint derivatives |
| `CollocationScheme::HermiteSimpson` | 4th | Uses cubic interpolation with midpoint |

**Trapezoidal:**
```
x[k+1] - x[k] = 0.5 * h * (f[k] + f[k+1])
```

**Hermite-Simpson:**
```
x_mid = 0.5*(x[k] + x[k+1]) + h/8*(f[k] - f[k+1])
x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
```

## Usage Patterns

### Brachistochrone Example

The brachistochrone problem finds the fastest path for a bead sliding under gravity.

**Dynamics:**
```cpp
janus::SymbolicVector brachistochrone_dynamics(
    const janus::SymbolicVector &state,    // [x, y, v]
    const janus::SymbolicVector &control,  // [theta]
    const janus::SymbolicScalar &t)
{
    janus::SymbolicScalar v = state(2);
    janus::SymbolicScalar theta = control(0);

    janus::SymbolicVector dxdt(3);
    dxdt(0) = v * janus::sin(theta);    // x' = v*sin(theta)
    dxdt(1) = -v * janus::cos(theta);   // y' = -v*cos(theta)
    dxdt(2) = 9.81 * janus::cos(theta); // v' = g*cos(theta)
    return dxdt;
}
```

**Setup:**
```cpp
janus::Opti opti;
janus::DirectCollocation dc(opti);

auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

auto [x, u, tau] = dc.setup(3, 1, 0.0, T,
    {.scheme = janus::CollocationScheme::HermiteSimpson, .n_nodes = 31});

dc.set_dynamics(brachistochrone_dynamics);
dc.add_defect_constraints();

dc.set_initial_state(janus::NumericVector{{0.0, 10.0, 0.001}});
dc.set_final_state(0, 10.0);  // Final x
dc.set_final_state(1, 5.0);   // Final y

opti.minimize(T);  // Minimize time
auto sol = opti.solve();
```

**Result:** T* = 1.8016s (matches Dymos reference: 1.8019s, error < 0.02%)

### Free vs Fixed Final Time

**Fixed time:** Pass `double` for `tf`
```cpp
dc.setup(n_states, n_controls, 0.0, 2.0, opts);
```

**Free time:** Pass `SymbolicScalar` for `tf`
```cpp
auto T = opti.variable(2.0);  // Decision variable
dc.setup(n_states, n_controls, 0.0, T, opts);
opti.minimize(T);  // Minimize time
```

### Manual vs DirectCollocation Comparison

**Manual collocation** (50+ lines):
```cpp
for (int i = 0; i < N - 1; ++i) {
    janus::SymbolicVector state_i(3), state_ip1(3);
    state_i << x(i), y(i), v(i);
    state_ip1 << x(i+1), y(i+1), v(i+1);
    auto f_i = ode(state_i, theta(i));
    auto f_ip1 = ode(state_ip1, theta(i+1));
    opti.subject_to(x(i+1) - x(i) == 0.5 * dt * (f_i(0) + f_ip1(0)));
    // ... repeat for each state ...
}
```

**DirectCollocation** (~10 lines):
```cpp
dc.set_dynamics(ode);
dc.add_defect_constraints();
dc.set_initial_state(x0);
dc.set_final_state(xf);
```

### Unified Comparison Example

The file `examples/optimization/transcription_comparison_demo.cpp` solves the same brachistochrone with Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral, printing single-grid performance stats and a multi-grid convergence table.

### When to Use

| Problem | Use Collocation? |
|---------|-----------------|
| Trajectory optimization | Yes |
| Minimum-time problems | Yes (free tf) |
| Path constraints | Yes |
| Stiff systems | Yes (implicit) |
| Bang-bang control | Consider multiple shooting |

## See Also

- [Transcription Methods Guide](transcription_methods.md) -- Comparison of all four transcription methods
- [Multiple Shooting Guide](multiple_shooting.md) -- Alternative transcription via numerical integration
- [Pseudospectral Guide](pseudospectral.md) -- Global polynomial transcription
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md) -- Birkhoff-form transcription
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [Collocation.hpp](../../include/janus/optimization/Collocation.hpp) -- API reference
