# ODE Integration

Janus provides explicit ODE integrators for simulation with dual-mode (numeric/symbolic) support.

## Quick Start

```cpp
#include <janus/janus.hpp>

// Define dynamics: dy/dt = f(t, y)
auto dynamics = [](double t, const janus::NumericVector& y) {
    return -0.5 * y;  // Exponential decay
};

janus::NumericVector y(1);
y(0) = 1.0;
double t = 0.0, dt = 0.1;

// Single step
y = janus::rk4_step(dynamics, y, t, dt);
```

## Step Integrators

| Function | Order | Evaluations | Use Case |
|----------|-------|-------------|----------|
| `euler_step` | 1st | 1 | Fast, low accuracy |
| `rk2_step` | 2nd | 2 | Moderate accuracy |
| `rk4_step` | 4th | 4 | General purpose |
| `rk45_step` | 4th/5th | 7 | Adaptive stepping |

### RK4 Step (Recommended)

```cpp
// Harmonic oscillator: y'' = -ω²y
// State: [y, v] where dy/dt=v, dv/dt=-ω²y
double omega = 2.0;

auto dynamics = [omega](double t, const janus::NumericVector& s) {
    janus::NumericVector ds(2);
    ds << s(1), -omega * omega * s(0);
    return ds;
};

janus::NumericVector state(2);
state << 1.0, 0.0;  // y=1, v=0

for (int i = 0; i < 100; ++i) {
    state = janus::rk4_step(dynamics, state, t, 0.01);
    t += 0.01;
}
```

### Adaptive Stepping with RK45

```cpp
double dt = 0.1;
double tol = 1e-6;

while (t < t_final) {
    auto result = janus::rk45_step(dynamics, y, t, dt);
    
    if (result.error < tol) {
        y = result.y5;  // Accept step
        t += dt;
        dt *= 1.5;      // Grow step
    } else {
        dt *= 0.5;      // Shrink step and retry
    }
}
```

## Trajectory Solver

For full trajectory integration, use `solve_ivp`:

```cpp
auto sol = janus::solve_ivp(
    dynamics,
    {0.0, 10.0},  // t_span
    {1.0},        // y0 (initializer list)
    100           // n_eval points
);

// sol.t - time points
// sol.y - solution matrix (rows=states, cols=time)
```

## Symbolic Mode

All step functions work with symbolic types for optimization:

```cpp
auto x = janus::sym_vec("x", 2);
auto t = janus::sym("t");
auto dt = janus::sym("dt");

auto x_next = janus::rk4_step(
    [](auto t, const auto& x) {
        janus::SymbolicVector dx(2);
        dx << x(1), -4.0 * x(0);
        return dx;
    },
    x, t, dt
);

// x_next is symbolic - can be used in optimization
casadi::Function step_fn("step", {janus::to_mx(x), t, dt}, {janus::to_mx(x_next)});
```

## Component-Based Integration (Icarus Pattern)

For simulation frameworks with component models:

```cpp
template <typename Scalar>
class IntegrableState {
public:
    virtual JanusVector<Scalar> get_state() const = 0;
    virtual void set_state(const JanusVector<Scalar>& s) = 0;
    virtual JanusVector<Scalar> compute_derivative(Scalar t) const = 0;
};

// Your component inherits IntegrableState
class RigidBody : public IntegrableState<double> {
    Vec3<double> position, velocity, acceleration;
    
    JanusVector<double> get_state() const override { /* ... */ }
    void set_state(const JanusVector<double>& s) override { /* ... */ }
    JanusVector<double> compute_derivative(double t) const override {
        NumericVector dydt(6);
        dydt << velocity, acceleration;
        return dydt;
    }
};
```

See [integration_demo.cpp](file:///home/tanged/sources/janus/examples/integration_demo.cpp) for complete example.
