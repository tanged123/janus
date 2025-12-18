# Transcription Methods: Collocation vs Multiple Shooting

This guide compares **Direct Collocation** and **Multiple Shooting**—the two trajectory optimization methods available in Janus—and provides guidance on when to use each.

## Overview

Both methods transcribe continuous-time optimal control problems into discrete NLPs, but they differ fundamentally in how they enforce dynamics:

| Aspect | Direct Collocation | Multiple Shooting |
|--------|-------------------|-------------------|
| **Dynamics Enforcement** | Polynomial defect constraints | Numerical integration |
| **Accuracy Source** | Fixed-order scheme (2nd or 4th) | Adaptive-step integrator |
| **Control Representation** | Values at each node | Piecewise constant per interval |
| **Computational Cost** | Lower per iteration | Higher (integrator calls) |
| **Sparsity** | Block-banded Jacobian | Block-sparse Jacobian |

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

---

## See Also

- [Direct Collocation Guide](collocation.md)
- [Multiple Shooting Guide](multiple_shooting.md)
- [Example: Brachistochrone Comparison](file:///home/tanged/sources/janus/examples/optimization/multishoot_comparison.cpp)
