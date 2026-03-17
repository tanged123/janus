# Janus v2.0.0 Usage Guide

> **Purpose**: Map of the Janus library for AI agents and developers. Each section gives a brief overview and links to the detailed user guide. For the full API, see the [Doxygen docs](https://tanged123.github.io/janus/index.html).

---

## Table of Contents

1. [Best Practices](#best-practices)
2. [Type System](#type-system)
3. [Module Reference](#module-reference)
   - [Core Layer](#core-layer)
   - [Math Layer](#math-layer)
   - [Optimization Layer](#optimization-layer)
4. [User Guide Index](#user-guide-index)

---

## Best Practices

### 1. Template-First Design (MANDATORY)

All physics/math functions **must** be templated on `Scalar`:

```cpp
// Correct
template <typename Scalar>
Scalar my_function(const Scalar& x) { ... }

// Wrong -- breaks symbolic mode
double my_function(double x) { ... }
```

### 2. Math Dispatch -- Use `janus::` Namespace

**Always** use Janus math functions instead of `std::`:

```cpp
janus::sin(x);  janus::pow(x, 2);  janus::sqrt(x);  janus::exp(x);
// Never: std::sin(x), std::pow(x, 2), etc.
```

### 3. Branching -- Use `janus::where()`, Never `if/else`

```cpp
Scalar result = janus::where(x > 0, x, -x);

// Multi-way branching
Scalar cd = janus::select(
    {mach < 0.3, mach < 0.8, mach < 1.2},
    {Scalar(0.02), Scalar(0.025), Scalar(0.05)},
    Scalar(0.03));  // default
```

### 4. Loops -- Structural Bounds Only

```cpp
// Correct -- structural bound (known at trace time)
for (int i = 0; i < N; ++i) { ... }

// Wrong -- dynamic bound breaks symbolic mode
while (error > tolerance) { ... }
```

### 5. Type Aliases -- Use Janus Native Types

```cpp
#include <janus/core/JanusTypes.hpp>

janus::Vec3<Scalar>   // 3D vector
janus::Mat3<Scalar>   // 3x3 matrix
janus::VecX<Scalar>   // Dynamic vector
janus::MatX<Scalar>   // Dynamic matrix
```

### 6. Include Convention

```cpp
#include <janus/janus.hpp>    // Everything (recommended for applications)
#include <janus/using.hpp>    // Convenience header -- brings common symbols into scope
```

---

## Type System

### Backend Types

| Type | Numeric Mode | Symbolic Mode |
|------|--------------|---------------|
| **Scalar** | `double` | `casadi::MX` |
| **Matrix** | `Eigen::MatrixXd` | `Eigen::Matrix<casadi::MX>` |
| **Vector** | `Eigen::VectorXd` | `Eigen::Matrix<casadi::MX, Dynamic, 1>` |

### Janus Type Aliases (`janus/core/JanusTypes.hpp`)

```cpp
// Symbolic types (for graph building)
janus::SymbolicScalar   // casadi::MX
janus::SymbolicMatrix   // Eigen::Matrix<casadi::MX, Dynamic, Dynamic>
janus::SymbolicVector   // Eigen::Matrix<casadi::MX, Dynamic, 1>

// Numeric types (for evaluation)
janus::NumericMatrix    // Eigen::MatrixXd
janus::NumericVector    // Eigen::VectorXd

// Fixed-size templated types
janus::Vec2<T>, janus::Vec3<T>, janus::Vec4<T>
janus::Mat2<T>, janus::Mat3<T>, janus::Mat4<T>
janus::VecX<T>, janus::MatX<T>, janus::RowVecX<T>

// Sparse types (numeric only)
janus::SparseMatrix     // Eigen::SparseMatrix<double>
janus::SparseTriplet    // Eigen::Triplet<double>
```

### Symbolic Variable Creation

```cpp
auto x = janus::sym("x");                          // Scalar
auto M = janus::sym("M", rows, cols);              // Matrix (MX)
auto v = janus::sym_vector("v", size);             // SymbolicVector (Eigen)
auto [vec, mx] = janus::sym_vec_pair("state", 3);  // Both representations
```

### Conversion Utilities

```cpp
janus::to_mx(eigen_matrix)     // Eigen -> CasADi MX
janus::to_eigen(casadi_mx)     // CasADi MX -> Eigen
janus::as_mx(symbolic_vector)  // SymbolicVector -> single MX
janus::as_vector(casadi_mx)    // MX -> SymbolicVector
```

---

## Module Reference

### Core Layer

| File | Description |
|------|-------------|
| `JanusTypes.hpp` | Type system, aliases, `janus::sym()`, `janus::to_mx()`, `janus::to_eigen()` |
| `JanusConcepts.hpp` | C++20 concepts: `ScalarType`, `NumericScalar`, `SymbolicScalar` |
| `JanusError.hpp` | Exception hierarchy: `JanusError`, `InvalidArgument`, `RuntimeError`, `IntegrationError`, `InterpolationError` |
| `JanusIO.hpp` | `janus::eval()`, `janus::print()`, `janus::to_dot()`, `janus::graphviz()` |
| `Function.hpp` | `janus::Function` -- compiled symbolic function wrapper |
| `Sparsity.hpp` | Sparsity patterns, graph coloring, `janus::sparse_jacobian()`, `janus::sparse_hessian()` |
| `Diagnostics.hpp` | Structural observability/identifiability: `janus::analyze_structural_observability()`, `janus::analyze_structural_identifiability()` |
| `StructuralTransforms.hpp` | `janus::alias_eliminate()`, `janus::block_triangularize()`, `janus::structural_analyze()` |

See `docs/user_guides/sparsity.md`, `docs/user_guides/structural_diagnostics.md`, `docs/user_guides/structural_transforms.md`.

---

### Math Layer

#### Arithmetic & Trigonometry

Standard math dispatch (`janus::sin`, `janus::pow`, `janus::exp`, etc.) with scalar and matrix overloads. See `docs/user_guides/math_functions.md` for the full function table.

#### Logic & Branching

`janus::where()`, `janus::select()`, `janus::min()`, `janus::max()`, `janus::clamp()`, element-wise comparisons, smooth blending (`janus::sigmoid_blend`, `janus::blend`). See `docs/user_guides/math_functions.md`.

#### Calculus & Autodiff

Gradient, Jacobian, Hessian, Hessian-vector products, Lagrangian second-order adjoints, sensitivity regime selection. See `docs/user_guides/symbolic_computing.md`.

```cpp
auto J = janus::jacobian(f, x);
auto H = janus::hessian(f, x);
auto Hv = janus::hessian_vector_product(f, x, direction);
```

#### Linear Algebra

`janus::solve(A, b)` with optional `LinearSolvePolicy` for backend selection:

```cpp
janus::LinearSolvePolicy policy;
policy.backend = janus::LinearSolveBackend::SparseDirect;
policy.sparse_direct_solver = janus::SparseDirectLinearSolver::SparseLU;
auto x = janus::solve(A, b, policy);
```

Available backends: `Dense` (ColPivHouseholderQR, PartialPivLU, FullPivLU, LLT, LDLT), `SparseDirect` (SparseLU, SparseQR, SimplicialLLT, SimplicialLDLT), `IterativeKrylov` (BiCGSTAB, GMRES with preconditioner hooks). Also includes `janus::dot`, `janus::cross`, `janus::norm`, `janus::inv`, `janus::det`, `janus::eye`, `janus::zeros`, `janus::ones`, `janus::block_diag`, and more.

See `docs/user_guides/math_functions.md`.

#### Interpolation

1D and N-dimensional interpolation with `"linear"`, `"cubic"`, and `"monotonic"` methods. See `docs/user_guides/interpolation.md`.

```cpp
auto y = janus::interp1(x, xp, fp);
auto y = janus::interp_nd(point, table);
```

#### Polynomial Chaos Expansions (PCE)

Askey-scheme orthogonal polynomial bases (Hermite, Legendre, Jacobi, Laguerre), total-order and tensor-product truncation, coefficient fitting via projection or regression, symbolic mean/variance extraction.

```cpp
janus::PolynomialChaosBasis basis(dimensions, order, options);
auto coeffs = janus::pce_projection_coefficients(basis, grid, values);
auto mu = janus::pce_mean(coeffs);
auto var = janus::pce_variance(basis, coeffs);
```

See `docs/user_guides/polynomial_chaos.md`.

#### Stochastic Quadrature

Probability-measure quadrature rules, tensor-product grids, and Smolyak sparse grids for high-dimensional integration and PCE projection.

```cpp
auto rule = janus::stochastic_quadrature_rule(dim, order, family);
auto grid = janus::tensor_product_quadrature(rules);
auto sparse = janus::smolyak_sparse_grid(dimensions, level, options);
```

See `docs/user_guides/stochastic_quadrature.md`.

#### Root Finding

Nonlinear solve for `F(x) = 0` with a numeric globalization stack (trust-region Newton, line-search Newton, Broyden, pseudo-transient continuation) and differentiable implicit function wrappers for embedding solves inside symbolic graphs.

```cpp
auto result = janus::rootfinder(function, x0, opts);

// Differentiable implicit solve for use inside optimization
auto implicit_fn = janus::create_implicit_function(function, x_guess, opts, implicit_opts);
```

See `docs/user_guides/root_finding.md`.

#### ODE Integration

IVP solvers with multiple steppers (`RK4`, `CVODES`, `BDF1`, `RosenbrockEuler`), definite integration via Gauss-Kronrod quadrature. See `docs/user_guides/integration.md`.

```cpp
auto result = janus::solve_ivp(dynamics, x0, t_span);
double I = janus::quad(f, a, b);
```

#### Second-Order Integrators

Dedicated solvers for systems of the form `q'' = a(t, q)`:

```cpp
auto result = janus::solve_second_order_ivp(accel, q0, v0, t_span);
// Single steps:
janus::stormer_verlet_step(accel, q, v, t, dt);  // Symplectic
janus::rkn4_step(accel, q, v, t, dt);            // 4th-order RKN
```

#### Mass-Matrix Integrators

Native support for stiff systems `M(t,y) y' = f(t,y)`:

```cpp
auto result = janus::solve_ivp_mass_matrix(rhs, M, x0, t_span);         // Numeric
auto result = janus::solve_ivp_mass_matrix_expr(rhs, M, t, y, x0, t_span); // Symbolic (IDAS)
```

#### Sparsity Pipelines

NaN-propagation sparsity detection, graph coloring, and compiled sparse derivative kernels that avoid materializing dense Jacobian/Hessian matrices.

```cpp
auto J = janus::sparse_jacobian(result, x);
auto H = janus::sparse_hessian(objective, vars);
auto nz = J.values(x_val);  // Evaluate only nonzero entries
```

See `docs/user_guides/sparsity.md`.

#### Structural Diagnostics

Preflight checks for structural observability and identifiability before committing to an optimization solve.

```cpp
auto obs = janus::analyze_structural_observability(measurement_fn, 0);
auto id  = janus::analyze_structural_identifiability(measurement_fn, 1);
auto all = janus::analyze_structural_diagnostics(system_fn, options);
```

See `docs/user_guides/structural_diagnostics.md`.

#### Structural Transforms

Alias elimination, BLT (block lower-triangular) decomposition, and tearing recommendations for large-scale equation systems.

```cpp
auto alias   = janus::alias_eliminate(residual_fn);
auto blt     = janus::block_triangularize(residual_fn);
auto analysis = janus::structural_analyze(residual_fn);
```

See `docs/user_guides/structural_transforms.md`.

#### Other Math Modules

- **Spacing**: `janus::linspace`, `janus::cosspace`, `janus::sinspace`, `janus::logspace`, `janus::geomspace`
- **Discrete integration**: rectangular, trapezoidal, Simpson, cubic, squared-curvature methods
- **Finite differences**: `janus::finite_difference_coefficients()`
- **Rotations**: `janus::rotation_x/y/z()`, `janus::rotation_2d()`
- **Quaternions**: `janus::Quaternion<Scalar>` with full algebra, conversions, `janus::slerp()`
- **Surrogate models**: `janus::softmax`, `janus::softmin`, `janus::softabs`, `janus::sigmoid`, `janus::tanh_blend`

---

### Optimization Layer

#### Opti Interface (`Opti.hpp`)

```cpp
janus::Opti opti;

// Variables
auto x = opti.variable(1.0);                    // Scalar
auto v = opti.variable(3, 0.0);                 // Vector
auto x = opti.variable(1.0, {.category = "Wing", .freeze = true});

// Parameters (fixed between solves)
auto p = opti.parameter(5.0);

// Objective
opti.minimize(cost_function);
opti.minimize(cost_function, 1e6);   // Explicit objective scaling
opti.maximize(profit_function);

// Constraints
opti.subject_to(x >= 0);
opti.subject_to(g == 0, 1e3);        // Explicit constraint scaling
opti.subject_to(x * x + y * y <= 1);
opti.subject_to_bounds(x, lower, upper);  // Box constraints
```

See `docs/user_guides/optimization.md`.

#### Solving and Options (`OptiOptions.hpp`)

Options are passed to `solve()` via `OptiOptions`:

```cpp
auto sol = opti.solve();                                           // Defaults
auto sol = opti.solve({.max_iter = 500, .verbose = false});        // Designated initializers
auto sol = opti.solve(janus::OptiOptions{}.set_tol(1e-10));       // Builder pattern

// Solver selection
auto sol = opti.solve({.solver = janus::Solver::Ipopt});           // Default
auto sol = opti.solve({.solver = janus::Solver::Snopt});           // Requires SNOPT license

if (janus::solver_available(janus::Solver::Snopt)) { ... }
```

#### Solution Extraction (`OptiSol.hpp`)

```cpp
auto sol = opti.solve();

double x_opt = sol.value(x);                  // Scalar
janus::NumericVector v_opt = sol.value(v);    // Vector
janus::NumericMatrix M_opt = sol.value(M);    // Matrix
auto stats = sol.stats();                     // Solver statistics

// Save / load
sol.save("result.json", {{"x", x}, {"y", y}});
auto data = janus::OptiSol::load("result.json");
```

#### Scaling Diagnostics

Preflight analysis of variable, constraint, and objective scaling before solving:

```cpp
auto report = opti.analyze_scaling();
// Report contains ScalingIssue entries with severity, suggested scales, etc.
```

#### Parametric Sweep (`OptiSweep.hpp`)

```cpp
janus::OptiSweep sweep(opti);
auto results = sweep.run(parameter, values);
```

#### Trajectory Optimization

Four transcription methods are available, all sharing a common `TranscriptionBase` interface. See `docs/user_guides/transcription_methods.md` for comparison.

- **Direct collocation** -- `docs/user_guides/collocation.md`
- **Multiple shooting** -- `docs/user_guides/multiple_shooting.md`
- **Pseudospectral** (LGL/CGL) -- `docs/user_guides/pseudospectral.md`
- **Birkhoff pseudospectral** (LGL/CGL) -- `docs/user_guides/birkhoff_pseudospectral.md`

```cpp
janus::Opti opti;
janus::DirectCollocation colloc(opti);
auto [X, U, tau] = colloc.setup(n_states, n_controls, t0, tf);

colloc.set_dynamics([](const auto& x, const auto& u, const auto& t) {
    return dynamics(x, u, t);
});
colloc.add_dynamics_constraints();
colloc.set_initial_state(x0);
colloc.set_final_state(xf);

opti.minimize(objective);
auto sol = opti.solve();
```

---

## User Guide Index

| Guide | File | Topics |
|-------|------|--------|
| Numeric Computing | `docs/user_guides/numeric_computing.md` | Numeric mode, evaluation |
| Symbolic Computing | `docs/user_guides/symbolic_computing.md` | Symbolic mode, graph building |
| Math Functions | `docs/user_guides/math_functions.md` | Full janus:: math dispatch table |
| Interpolation | `docs/user_guides/interpolation.md` | 1D/ND interpolation |
| Integration | `docs/user_guides/integration.md` | ODE solvers, quadrature |
| Root Finding | `docs/user_guides/root_finding.md` | Nonlinear solves, implicit functions |
| Polynomial Chaos | `docs/user_guides/polynomial_chaos.md` | PCE bases, fitting, moments |
| Stochastic Quadrature | `docs/user_guides/stochastic_quadrature.md` | Quadrature rules, sparse grids |
| Sparsity | `docs/user_guides/sparsity.md` | Sparsity, coloring, sparse derivatives |
| Structural Diagnostics | `docs/user_guides/structural_diagnostics.md` | Observability, identifiability |
| Structural Transforms | `docs/user_guides/structural_transforms.md` | Alias elimination, BLT, tearing |
| Graph Visualization | `docs/user_guides/graph_visualization.md` | DOT export, Graphviz |
| Optimization | `docs/user_guides/optimization.md` | Opti interface, constraints |
| Collocation | `docs/user_guides/collocation.md` | Direct collocation |
| Multiple Shooting | `docs/user_guides/multiple_shooting.md` | Multiple shooting |
| Pseudospectral | `docs/user_guides/pseudospectral.md` | LGL/CGL pseudospectral |
| Birkhoff Pseudospectral | `docs/user_guides/birkhoff_pseudospectral.md` | LGL/CGL Birkhoff |
| Transcription Methods | `docs/user_guides/transcription_methods.md` | Comparison of methods |

---

## Summary for Agents

### DO NOT Reimplement

The following functionality already exists in Janus -- check the relevant user guide before building anything new:

- All basic math (`sin`, `cos`, `pow`, `exp`, `log`, `sqrt`, etc.)
- Linear algebra (`dot`, `cross`, `norm`, `inv`, `det`, `solve` with policies)
- Quaternion algebra and rotations
- Interpolation (1D, ND, multiple methods)
- Root finding (globalization stack, implicit function wrappers)
- ODE integration (`solve_ivp`, `quad`, second-order, mass-matrix)
- Polynomial chaos and stochastic quadrature
- Sparsity analysis and sparse derivative kernels
- Structural diagnostics and transforms
- Discrete integration (trapz, Simpson, etc.)
- Branching logic (`where`, `select`, `clamp`, `min`, `max`)
- Function compilation
- Optimization (`Opti`, IPOPT, SNOPT)
- Trajectory optimization (collocation, multiple shooting, pseudospectral, Birkhoff)
- Scaling diagnostics

### When Building on Janus

1. **Import via** `#include <janus/janus.hpp>` (includes everything)
2. **Use Janus types** (`janus::Vec3<Scalar>`, `janus::SymbolicScalar`)
3. **Use Janus math** (`janus::sin`, not `std::sin`)
4. **Use Janus branching** (`janus::where`, not `if/else`)
5. **Template everything** on `Scalar`
6. **Test both modes** (numeric AND symbolic)
