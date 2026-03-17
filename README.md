# Janus

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tanged123.github.io/janus/) [![Janus CI](https://github.com/tanged123/janus/actions/workflows/ci.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/ci.yml) [![Clang-Format Check](https://github.com/tanged123/janus/actions/workflows/format.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/format.yml) [![codecov](https://codecov.io/github/tanged123/janus/graph/badge.svg?token=0DSF7KK8W7)](https://codecov.io/github/tanged123/janus)

Janus is a C++20 header-only numerical framework that implements a dual-mode code-transformations paradigm inspired by [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox). You write physics models once using generic templates; the same code compiles to optimized native arithmetic (Eigen) for simulation **and** builds a symbolic computational graph (CasADi) for automatic differentiation and gradient-based optimization. No code duplication, no runtime dispatch overhead.

## Building & Development

### Prerequisites

**Recommended:** [Nix](https://nixos.org/) with flakes enabled -- the `flake.nix` pins every dependency.

**Manual:** CMake 3.20+, a C++20 compiler (Clang recommended), Eigen 3.4+, CasADi, GoogleTest, Ninja.

### Dev environment

```bash
# Enter the Nix dev shell (sets up compiler, Eigen, CasADi, gtest, ccache, doxygen, lcov, etc.)
nix develop

# Or use the convenience wrapper
./scripts/dev.sh
```

### Build, test, run

```bash
./scripts/build.sh              # CMake + Ninja debug build
./scripts/build.sh --release    # Release build
./scripts/test.sh               # Build (if needed) + ctest
./scripts/coverage.sh           # Build with coverage, run tests, generate lcov report
./scripts/run_examples.sh       # Build and run all 30 example programs
./scripts/verify.sh             # Full pre-push check: build + test + examples
```

Formatting is enforced via `treefmt` (clang-format, cmake-format, nixfmt):

```bash
nix fmt                         # Format everything
./scripts/install-hooks.sh      # Install pre-commit formatting hook
```

## Project Structure

```
janus/
  include/janus/
    core/                  # Type system, concepts, diagnostics, sparsity, structural analysis, IO
    math/                  # Trig, calculus, autodiff, interpolation, integration, root finding,
                           #   quadrature, PCE, linalg, quaternions, rotations, surrogates
    optimization/          # Opti interface, scaling, collocation, multiple shooting,
                           #   pseudospectral, Birkhoff pseudospectral
    janus.hpp              # Umbrella header
  examples/
    intro/                 # Getting started (numeric, energy, sparsity, printing)
    math/                  # Branching, graphs, loops, sensitivity, PCE, structural analysis
    interpolation/         # N-D tables, scattered interpolation, root finding, file-based tables
    simulation/            # ODE integration, drag, brachistochrone, hybrid, attitudes
    optimization/          # Rosenbrock, drag, beam, brachistochrone, sweeps, transcription comparison
    integration_demo.cpp   # RK4, RK45, Stormer-Verlet, mass-matrix integrators
  tests/                   # GoogleTest suites mirroring include/ layout (core/, math/, optimization/)
  docs/
    user_guides/           # 18 topic guides (see Documentation below)
    patterns/              # Reusable patterns (branching, loops, hybrid optimization)
    design_overview.md     # Architecture deep-dive
  scripts/                 # Build, test, CI, coverage, formatting, doc generation
  flake.nix                # Nix dev environment and package definition
```

## Architecture Overview

Janus is built on a **template-first traceability** paradigm. User models are templated on a generic scalar type; the framework provides two backends that satisfy the same concept constraints. The **numeric backend** maps to `double` and `Eigen::MatrixXd` -- the compiler generates assembly identical to hand-written C++. The **symbolic backend** maps to `casadi::MX` and `Eigen::Matrix<casadi::MX>` -- the same code constructs a static computational graph suitable for automatic differentiation, sparsity detection, and NLP solvers.

A **dispatch layer** in the `janus::` namespace shadows `std::` math functions and uses C++20 concepts to route calls (`janus::sin`, `janus::pow`, etc.) to the correct backend at compile time. The framework draws a strict line between **structural logic** (integers, booleans, loop bounds -- these shape the graph) and **value logic** (floating-point quantities that flow through the graph). Standard `if/else` cannot branch on symbolic values; `janus::where(condition, true_val, false_val)` compiles to a ternary in numeric mode and a `casadi::if_else` switch node in symbolic mode.

For the full design rationale, see [docs/design_overview.md](docs/design_overview.md).

## Documentation

### Organization

- **`docs/user_guides/`** -- 18 standalone guides, each covering one subsystem end-to-end.
- **`docs/patterns/`** -- Reusable coding patterns (branching, loops, hybrid optimization).
- **`docs/design_overview.md`** -- Architecture principles and type system.
- **Doxygen API docs** -- Generated from source comments.

### Generating docs

```bash
./scripts/generate_docs.sh      # Or: doxygen Doxyfile
```

Hosted on [GitHub Pages](https://tanged123.github.io/janus/).

### User guides

| Guide | Description |
|---|---|
| [numeric_computing](docs/user_guides/numeric_computing.md) | Numeric backend basics -- templates, Eigen, optimized machine code |
| [symbolic_computing](docs/user_guides/symbolic_computing.md) | Symbolic backend -- CasADi graph construction, derivatives, code generation |
| [math_functions](docs/user_guides/math_functions.md) | Dispatch layer and ADL for dual-mode math (`janus::sin`, `janus::pow`, etc.) |
| [interpolation](docs/user_guides/interpolation.md) | N-dimensional gridded interpolation in numeric and symbolic modes |
| [integration](docs/user_guides/integration.md) | ODE solvers -- RK4, RK45, Stormer-Verlet, mass-matrix, second-order systems |
| [root_finding](docs/user_guides/root_finding.md) | Newton-Raphson and bracketing solvers with globalization |
| [optimization](docs/user_guides/optimization.md) | `janus::Opti` interface for constrained NLP (IPOPT/SNOPT/QPOASES) |
| [collocation](docs/user_guides/collocation.md) | Direct collocation transcription for optimal control |
| [multiple_shooting](docs/user_guides/multiple_shooting.md) | Multiple shooting transcription via CasADi integrators (CVODES/IDAS) |
| [pseudospectral](docs/user_guides/pseudospectral.md) | Global polynomial pseudospectral transcription |
| [birkhoff_pseudospectral](docs/user_guides/birkhoff_pseudospectral.md) | Birkhoff-form pseudospectral with derivative collocation |
| [transcription_methods](docs/user_guides/transcription_methods.md) | Comparison and selection guide for all transcription methods |
| [graph_visualization](docs/user_guides/graph_visualization.md) | Visualizing computational graphs for debugging |
| [polynomial_chaos](docs/user_guides/polynomial_chaos.md) | Polynomial chaos expansion for uncertainty quantification |
| [stochastic_quadrature](docs/user_guides/stochastic_quadrature.md) | Probability-measure quadrature rules for PCE workflows |
| [sparsity](docs/user_guides/sparsity.md) | Jacobian/Hessian sparsity detection and exploitation |
| [structural_transforms](docs/user_guides/structural_transforms.md) | Alias elimination and BLT decomposition for residual systems |
| [structural_diagnostics](docs/user_guides/structural_diagnostics.md) | Structural observability analysis for state estimation |

## Key Features

**Core** -- Dual-mode type system, C++20 concept constraints, compile-time dispatch, runtime diagnostics, graph serialization and visualization.

**Math** -- Automatic differentiation (forward/reverse via CasADi), N-D interpolation (gridded, scattered, B-spline), root finding (Newton-Raphson, bracketing with globalization), discrete and continuous calculus, stochastic quadrature, polynomial chaos expansion, linear algebra (templated Eigen ops), quaternions and rotations, surrogate models (sigmoid, softmax smoothing).

**Integration** -- `solve_ivp` with RK4, RK45 (adaptive), Stormer-Verlet (symplectic), mass-matrix DAE integration, second-order ODE support, discrete integration utilities.

**Optimization** -- `janus::Opti` high-level NLP interface (IPOPT, SNOPT, QPOASES), variable/constraint scaling, direct collocation, multiple shooting, pseudospectral, Birkhoff pseudospectral transcriptions, parametric sweeps.

**Structural Analysis** -- Jacobian/Hessian sparsity detection, alias elimination, BLT (block lower triangular) decomposition, structural observability checking.

## Examples

The `examples/` directory contains 30 runnable demos organized by topic:

- **intro/** (4) -- Numeric basics, energy model, sparsity intro, printing
- **math/** (10) -- Branching logic, graph visualization, loops, sensitivity, PCE, linear solve policies, structural analysis
- **interpolation/** (4) -- N-D tables, scattered data, root finding, file-based table loading
- **simulation/** (5) -- Drag model, brachistochrone, hybrid systems, aircraft attitudes, smooth trajectories
- **optimization/** (6) -- Rosenbrock, drag optimization, beam deflection, brachistochrone, parametric sweeps, transcription comparison
- **integration** (1) -- RK4, RK45, Stormer-Verlet, mass-matrix integrators

Build and run all examples:

```bash
./scripts/run_examples.sh
```

## Quick Look

Write a model once, evaluate it numerically and symbolically:

```cpp
#include <janus/janus.hpp>

// Generic model -- works with double or CasADi symbolic types
auto drag(auto rho, auto v, auto S, auto Cd) {
    return 0.5 * rho * janus::pow(v, 2) * S * Cd;
}

int main() {
    // Numeric mode -- compiles to native arithmetic
    double D = drag(1.225, 50.0, 10.0, 0.02);

    // Symbolic mode -- same function builds a CasADi graph
    janus::Opti opti;
    auto v = opti.variable(50.0);
    auto D_sym = drag(1.225, v, 10.0, 0.02);

    opti.minimize(D_sym);
    opti.subject_to_bounds(v, 10, 100);
    auto sol = opti.solve();  // IPOPT under the hood
}
```

## Contributing

**Language and style:** C++20, header-only, heavily templated. Formatting is enforced by `nix fmt` (clang-format). Run `./scripts/install-hooks.sh` to auto-format on commit.

**Adding tests:** Test files live in `tests/` and mirror the `include/janus/` directory layout (`tests/core/`, `tests/math/`, `tests/optimization/`). Use GoogleTest. Run `./scripts/test.sh` to verify.

**Adding examples:** Drop a `.cpp` file in the appropriate `examples/` subdirectory and add it to `examples/CMakeLists.txt`. Run `./scripts/run_examples.sh` to verify.

**PR workflow:** Fork, branch, make changes, ensure `./scripts/verify.sh` passes (build + test + examples), open a PR against `main`.
