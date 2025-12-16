# Janus Design Overview

## 1. Project Mission

Janus is a high-performance C++ numerical framework designed to implement the Code Transformations paradigm. It serves as a drop-in replacement for standard math libraries, enabling engineers to write physics models once and execute them in two distinct modes:

* **Fast Numeric Mode**: For simulation, debugging, and real-time control (using standard double and Eigen).
* **Symbolic Trace Mode**: For generating static computational graphs to enable automatic differentiation and gradient-based optimization (using CasADi).

## 2. Core Architectural Principles

### A. The "Template-First" Traceability Paradigm

The core innovation is Traceability: the ability to inspect and transform numerical code.

* **Constraint**: User physics models must be templated on a generic Scalar type.
* **Implementation**: Use C++ Templates (compile-time polymorphism) rather than std::variant (runtime polymorphism).
* **Goal**: Zero-cost abstraction. In Numeric Mode, the compiler generates assembly identical to raw Eigen/C++.

### B. The Dual-Backend Type System

The framework defines a unified type alias system that routes to specific backends via template specialization.

| Feature | Numeric Backend (Fast) | Symbolic Backend (Graph) |
|---|---|---|
| **Scalar** | `double` (or float) | `casadi::MX` |
| **Matrix** | `Eigen::MatrixXd` | `Eigen::Matrix<casadi::MX>` |
| **Mutation** | Direct Memory | CasADi Register Machine |

### C. The Dispatch Layer (The "Agnostic" Math Stack)

To maintain traceability, the framework must intercept all mathematical operators.

* **Mechanism**: A custom namespace (`janus::`) shadows `std::`.
* **Implementation**: Use C++20 Concepts to dispatch logic (e.g., `std::sin` vs `casadi::sin`) at compile time.

## 3. Type Handling & Control Flow Policies

To preserve the static computational graph structure, strict policies are enforced on C++ types.

### A. The "Red Line": Structural vs. Value Logic

* **Structural Logic (Allowed)**: Integers, Booleans, and logic determined at compile/trace time (e.g., `num_segments`, `use_viscous_model`). These define the shape of the graph.
* **Value Logic (Modified)**: Floating-point values dependent on optimization variables. These define the data flow through the graph.

### B. Branching Logic (`janus::where`)

Standard C++ `if/else` cannot branch on symbolic types because symbols do not evaluate to true/false during graph construction.

* **Solution**: Implement `janus::where(condition, true_val, false_val)`.
  * **Numeric Mode**: Compiles to `cond ? a : b`.
  * **Symbolic Mode**: Compiles to a switch node `casadi::if_else(cond, a, b)`.

### C. Loop Handling

* **Standard Loops**: Standard `for` loops are supported. The symbolic backend unrolls these loops into a chain of graph operations.
* **Variable Bounds**: Loops with variable iteration counts (e.g., `i < optimization_var`) are banned as they break the static graph topology.

## 4. Implementation Plan

- [x] **Phase 1**: Scalar Concept & Math Primitives
- [x] **Phase 2**: The Linear Algebra Shim (Eigen Integration)
- [x] **Phase 2.5**: Data-Driven Primitives (Interpolation)
- [x] **Phase 3**: AeroSandbox Numerical Parity (Logic, Calculus, Surrogate Models, Discrete Integration)
- [x] **Phase 4**: Beta 1.0 Release (ODE Integration, Extended Spacing, Quaternions, Graph Visualization, Full Coverage)
- [x] **Phase 5**: Advanced Math Utilities (N-Dimensional Interpolation, Root Finding, B-Splines)
- [x] **Phase 6**: Optimization Framework (Opti Interface, Trajectory Optimization, JIT Compilation, IPOPT Integration)
- [ ] **Phase 7**: Advanced Control & Estimation (Kalman Filters, MPC)

## 5. API Reference Example

```cpp
// User Physics Model
template <typename Scalar>
Scalar compute_drag(const Scalar& velocity, const Scalar& rho) {
    // 1. Procedural declarations allowed
    Scalar drag = 0.0;
    
    // 2. Logic using janus::where (not if/else)
    auto is_supersonic = (velocity > 343.0);
    Scalar cd = janus::where(is_supersonic, 0.5, 0.02);
    
    // 3. Janus Math Dispatch (not std::pow)
    drag = 0.5 * rho * janus::pow(velocity, 2) * cd;
    
    return drag;
}
```
