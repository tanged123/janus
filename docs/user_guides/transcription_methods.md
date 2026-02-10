# Transcription Methods: Collocation vs Shooting vs Pseudospectral vs Birkhoff

This guide compares **Direct Collocation**, **Multiple Shooting**, **Pseudospectral**, and **Birkhoff Pseudospectral** and provides guidance on when to use each.

## Overview

All methods transcribe continuous-time optimal control problems into discrete NLPs, but they differ in how dynamics and trajectory coupling are enforced:

| Aspect | Direct Collocation | Multiple Shooting | Pseudospectral | Birkhoff Pseudospectral |
|--------|-------------------|-------------------|----------------|-------------------------|
| **Dynamics Enforcement** | Local defect constraints | Numerical integration | Global differentiation matrix | Pointwise derivative collocation |
| **State Coupling** | Local between neighbors | Local across intervals | Dense nonlinear coupling via `D*X` | Linear coupling via integration matrix `B` |
| **Accuracy Source** | Fixed-order scheme (2nd/4th) | Adaptive-step integrator | Spectral convergence (smooth) | Spectral-like with better conditioning |
| **Control Representation** | Values at each node | Piecewise constant per interval | Values at each node | Values at each node |
| **Computational Cost** | Lower per iteration | Higher (integrator calls) | Moderate-high | Moderate |

---

## Direct Collocation

Uses polynomial interpolation to approximate state trajectories and enforces dynamics via **defect constraints**.

### How It Works

States and controls are decision variables at discrete nodes. Dynamics are enforced by requiring the polynomial interpolant to match the ODE:

```
Trapezoidal:      x[k+1] - x[k] = h/2 * (f[k] + f[k+1])
Hermite-Simpson:  x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
```

### Strengths

- **Fast iterations**: Constraint Jacobians are cheap to evaluate
- **Easy initialization**: States at each node are independent guesses
- **Dense trajectory output**: Solution available at all collocation nodes
- **Robust convergence**: Good for poorly-initialized problems

### Weaknesses

- **Fixed accuracy**: Limited by polynomial order (2nd or 4th)
- **May struggle with stiff systems**: Polynomial approximation can fail
- **Needs more nodes for accuracy**: Refinement requires adding nodes

### Best For

| Use Case | Why |
|----------|-----|
| Smooth trajectories | Polynomial assumptions hold |
| Real-time MPC | Fast iteration times |
| Poor initial guesses | Independent node states aid convergence |
| Teaching/prototyping | Simple, intuitive formulation |

---

## Multiple Shooting

Divides time into intervals and integrates dynamics numerically. Continuity constraints connect intervals.

### How It Works

For each interval $[t_k, t_{k+1}]$:
1. Integrate the ODE from state $x_k$ using control $u_k$
2. Constrain: $x_{k+1} = \text{Integrate}(x_k, u_k, \Delta t)$

```cpp
ms.set_dynamics(ode);
ms.add_dynamics_constraints();  // Creates integrator-based constraints
```

### Strengths

- **High accuracy**: Leverages CVODES/IDAS adaptive integrators
- **Handles stiff systems**: BDF methods in CVODES excel here
- **Fewer intervals needed**: Integrator accuracy compensates
- **Respects physics**: True ODE solution within each interval

### Weaknesses

- **Expensive derivatives**: Integrator sensitivities cost more
- **Initialization sensitivity**: Unstable dynamics can cause divergence
- **Coarser trajectory**: Solution only at interval boundaries

### Best For

| Use Case | Why |
|----------|-----|
| High-fidelity simulation | Integrator accuracy |
| Stiff ODEs | Adaptive implicit methods |
| Matching simulation results | Same integration scheme |
| Fewer decision variables | Accuracy without more nodes |

---

## Pseudospectral

Uses global polynomial interpolation with Lobatto nodes and a spectral differentiation matrix:

```
D * X = (dt / 2) * F(X, U, t)
```

### Strengths

- **High accuracy per node** for smooth dynamics
- **Built-in high-order quadrature** for running costs
- **Good low-node performance** for minimum-time/fuel style problems

### Weaknesses

- **Dense coupling across nodes** can increase linear-solve cost
- **Less robust for discontinuous/bang-bang controls** without refinement

### Best For

| Use Case | Why |
|----------|-----|
| Smooth OCPs with tight accuracy targets | Spectral convergence |
| Low node budgets | More accuracy per node |
| Accurate running-cost integration | Native quadrature weights |

---

## Birkhoff Pseudospectral

Uses an integration matrix formulation:

```
X = x_a * 1 + B * V
V = (dt / 2) * F(X, U, t)
```

This keeps dense coupling mostly in linear constraints while dynamics are pointwise.

### Strengths

- **Improved numerical conditioning** versus classical `D*X` form at higher node counts
- **Pointwise nonlinear dynamics constraints** (`V_i` depends on node `i` only)
- **Natural quadrature from Birkhoff weights**

### Weaknesses

- More decision variables (`X` and `V` both present)
- Still sensitive on highly non-smooth controls

### Best For

| Use Case | Why |
|----------|-----|
| Smooth OCPs with larger node counts | Better conditioning behavior |
| Problems where classical PS becomes iteration-heavy | Pointwise nonlinear structure |
| Experiments with high-order pseudospectral formulations | Flexible integration form |

---

## Side-by-Side Comparison

### Performance on Brachistochrone

| Method | Nodes/Intervals | Optimal Time | Error vs Reference |
|--------|-----------------|--------------|-------------------|
| Collocation (Hermite-Simpson) | 31 | 1.8016 s | 0.02% |
| Multiple Shooting (CVODES) | 20 | 1.8019 s | < 0.01% |

Multiple shooting achieved better accuracy with fewer decision variables.

### Problem Structure

```
Collocation (31 nodes, 3 states, 1 control):
  Decision variables: 31×3 + 31×1 = 124
  Constraints: 30×3 (defects) + BCs

Multiple Shooting (20 intervals, 3 states, 1 control):
  Decision variables: 21×3 + 20×1 = 83
  Constraints: 20×3 (continuity) + BCs
```

---

## Decision Flowchart

```
┌─────────────────────────────────────┐
│ Is the system stiff or very         │
│ sensitive to integration accuracy?  │
├──────────────┬──────────────────────┤
│      YES     │         NO           │
│      ↓       │         ↓            │
│  Multiple    │   Is fast iteration  │
│  Shooting    │   time critical?     │
│              ├──────────┬───────────┤
│              │   YES    │    NO     │
│              │    ↓     │     ↓     │
│              │ Colloc.  │  Either   │
│              │          │  works    │
└──────────────┴──────────┴───────────┘
```

---

## Unified API

Both classes now share a unified API for interchangeability:

```cpp
// Works with EITHER DirectCollocation OR MultipleShooting
transcription.set_dynamics(my_ode);
transcription.add_dynamics_constraints();  // Unified method
transcription.set_initial_state(x0);
transcription.set_final_state(xf);
```

| Unified Method | DirectCollocation Alias | MultipleShooting Alias |
|----------------|------------------------|------------------------|
| `add_dynamics_constraints()` | `add_defect_constraints()` | `add_continuity_constraints()` |
| `n_nodes()` | — | — |
| `n_intervals()` | — | — |

---

## Quick Reference

**Choose Direct Collocation when:**
- You need fast solver iterations (MPC, real-time)
- Initial guess is poor or unknown
- Problem is smooth and non-stiff
- You want dense trajectory output

**Choose Multiple Shooting when:**
- High integration accuracy is required
- System is stiff (chemical kinetics, some robotics)
- You're matching high-fidelity simulation
- Fewer decision variables are preferred

**Choose Pseudospectral when:**
- Dynamics and controls are smooth
- You need high accuracy with relatively few nodes
- Running-cost integration accuracy is important

**Choose Birkhoff Pseudospectral when:**
- You want pseudospectral accuracy but better conditioning behavior
- Node counts are moderate-to-high
- You want pointwise nonlinear dynamics with linear global recovery

---

## See Also

- [Direct Collocation Guide](collocation.md)
- [Multiple Shooting Guide](multiple_shooting.md)
- [Pseudospectral Guide](pseudospectral.md)
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md)
- [Example: Brachistochrone Comparison](file:///home/tanged/sources/janus/examples/optimization/transcription_comparison_demo.cpp)
