# Multiple Shooting User Guide

`janus::MultipleShooting` provides a transcription method for optimal control problems that enforces continuity via high-accuracy numerical integration (using CasADi's integrator interface, e.g., CVODES or IDAS).

## Overview

Multiple Shooting divides the time horizon into $N$ intervals. The controls are typically piecewise constant. The system dynamics are integrated over each interval $[t_k, t_{k+1}]$ starting from an initial state guess $x_k$. Continuity constraints ensure that the end of one interval matches the start of the next:
$$ x_{k+1} = \int_{t_k}^{t_{k+1}} f(x(\tau), u_k) d\tau + x_k $$

### Advantages over Direct Collocation
- **High Accuracy**: Uses variable-step/variable-order integrators (like CVODES) instead of fixed-order polynomials.
- **Stiffness Handling**: Better suited for stiff systems where explicit or low-order implicit schemes fail.
- **Sparse Structure**: Retains the block-sparse structure of the NLP.

### Disadvantages
- **Cost**: Evaluating derivatives (sensitivities) of the integrator can be computationally expensive compared to evaluating a polynomial defect.
- **Initialization**: Can be harder to initialize if the dynamics are unstable.

## Basic Usage

### 1. Include Header
```cpp
#include <janus/optimization/MultiShooting.hpp>
```

### 2. Setup
```cpp
janus::Opti opti;
janus::MultipleShooting ms(opti);

// Options
janus::MultiShootingOptions opts;
opts.n_intervals = 20;
opts.integrator = "cvodes"; // or "rk", "idas"
opts.tol = 1e-6; // Integrator tolerance

// Setup variables (t0=0, tf=2.0)
auto T = opti.variable(2.0); // Variable final time
auto [x, u, tau] = ms.setup(n_states, n_controls, 0.0, T, opts);
```

### 3. Define Dynamics & Constraints
```cpp
// Define ODE
// must return dxdt
ms.set_dynamics([](const SymbolicVector& x, const SymbolicVector& u, const SymbolicScalar& t) {
    return ...; 
});

// Add continuity constraints (calling the integrator)
ms.add_continuity_constraints();

// Boundary conditions
ms.set_initial_state(x0);
ms.set_final_state(xf);
```

### 4. Solve
```cpp
opti.minimize(T);
auto sol = opti.solve();
```

## Performance Comparison

Comparing Multiple Shooting (CVODES) vs Direct Collocation (Hermite-Simpson) on the Brachistochrone problem:

| Method | Nodes/Intervals | Accuracy Error | Relative Accuracy |
| :--- | :--- | :--- | :--- |
| Direct Collocation (Hermite-Simpson) | 30 | ~3.5e-4 s | 1x |
| Multiple Shooting (CVODES) | 20 | ~2.2e-4 s | **1.6x** |

Multiple Shooting achieved higher accuracy with fewer decision variables, leveraging the high-order integrator.
