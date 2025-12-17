# Direct Collocation in Janus

Direct collocation transforms continuous-time optimal control problems into large sparse nonlinear programs (NLPs). This guide explains the technique and how to use the `DirectCollocation` class.

## What is Direct Collocation?

In trajectory optimization, you want to find state trajectories $x(t)$ and control inputs $u(t)$ that minimize some objective while satisfying dynamics constraints:

$$\dot{x} = f(x, u, t)$$

Direct collocation **discretizes** time into nodes and enforces dynamics at each segment using **defect constraints**.

### Collocation Schemes

| Scheme | Order | Description |
|--------|-------|-------------|
| Trapezoidal | 2nd | Uses average of endpoint derivatives |
| Hermite-Simpson | 4th | Uses cubic interpolation with midpoint |

**Trapezoidal** (simpler):
```
x[k+1] - x[k] = 0.5 * h * (f[k] + f[k+1])
```

**Hermite-Simpson** (more accurate):
```
x_mid = 0.5*(x[k] + x[k+1]) + h/8*(f[k] - f[k+1])
x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
```

---

## The DirectCollocation Class

Located in `<janus/optimization/Collocation.hpp>`.

### Basic Usage

```cpp
#include <janus/janus.hpp>

// 1. Create Opti and DirectCollocation
janus::Opti opti;
janus::DirectCollocation dc(opti);

// 2. Setup decision variables
auto [x, u, tau] = dc.setup(
    n_states,    // Number of state variables
    n_controls,  // Number of control variables
    t0, tf,      // Time bounds
    {.scheme = CollocationScheme::HermiteSimpson, .n_nodes = 31}
);

// 3. Set dynamics (ODE function)
dc.set_dynamics([](const auto& x, const auto& u, const auto& t) {
    SymbolicVector dxdt(2);
    dxdt(0) = x(1);      // dx/dt = velocity
    dxdt(1) = u(0);      // dv/dt = control
    return dxdt;
});

// 4. Apply collocation constraints
dc.add_defect_constraints();

// 5. Set boundary conditions
dc.set_initial_state(x0);
dc.set_final_state(xf);

// 6. Set objective and solve
opti.minimize(objective);
auto sol = opti.solve();
```

---

## Example: Brachistochrone

The brachistochrone problem finds the fastest path for a bead sliding under gravity.

**Dynamics**:
```cpp
SymbolicVector brachistochrone_dynamics(
    const SymbolicVector &state,    // [x, y, v]
    const SymbolicVector &control,  // [theta]
    const SymbolicScalar &t) 
{
    SymbolicScalar v = state(2);
    SymbolicScalar theta = control(0);

    SymbolicVector dxdt(3);
    dxdt(0) = v * janus::sin(theta);   // x' = v*sin(θ)
    dxdt(1) = -v * janus::cos(theta);  // y' = -v*cos(θ)
    dxdt(2) = 9.81 * janus::cos(theta); // v' = g*cos(θ)
    return dxdt;
}
```

**Setup**:
```cpp
Opti opti;
DirectCollocation dc(opti);

// Final time is a decision variable
auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

auto [x, u, tau] = dc.setup(3, 1, 0.0, T, 
    {.scheme = CollocationScheme::HermiteSimpson, .n_nodes = 31});

dc.set_dynamics(brachistochrone_dynamics);
dc.add_defect_constraints();

dc.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
dc.set_final_state(0, 10.0);  // Final x
dc.set_final_state(1, 5.0);   // Final y

opti.minimize(T);  // Minimize time
auto sol = opti.solve();
```

**Result**: T* = 1.8016s (matches Dymos reference: 1.8019s, error < 0.02%)

See [collocation_demo.cpp](file:///home/tanged/sources/janus/examples/optimization/collocation_demo.cpp) for the full example.

---

## Free vs Fixed Final Time

**Fixed time**: Pass `double` for `tf`
```cpp
dc.setup(n_states, n_controls, 0.0, 2.0, opts);
```

**Free time**: Pass `SymbolicScalar` for `tf`
```cpp
auto T = opti.variable(2.0);  // Decision variable
dc.setup(n_states, n_controls, 0.0, T, opts);
opti.minimize(T);  // Minimize time
```

---

## Comparison: Manual vs DirectCollocation

**Manual collocation** (50+ lines):
```cpp
for (int i = 0; i < N - 1; ++i) {
    SymbolicVector state_i(3), state_ip1(3);
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

---

## When to Use

| Problem | Use Collocation? |
|---------|-----------------|
| Trajectory optimization | ✓ Yes |
| Minimum-time problems | ✓ Yes (free tf) |
| Path constraints | ✓ Yes |
| Stiff systems | ✓ Yes (implicit) |
| Bang-bang control | Consider multiple shooting |
