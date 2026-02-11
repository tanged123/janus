# Janus 👥

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tanged123.github.io/janus/) [![Janus CI](https://github.com/tanged123/janus/actions/workflows/ci.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/ci.yml) [![Clang-Format Check](https://github.com/tanged123/janus/actions/workflows/format.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/format.yml) [![codecov](https://codecov.io/github/tanged123/janus/graph/badge.svg?token=0DSF7KK8W7)](https://codecov.io/github/tanged123/janus)

**Janus** is a high-performance C++ numerical framework named after the Roman god of duality. True to its name, it allows a single physics model to face two directions: **Numeric Mode** for fast execution and **Symbolic Mode** for graph generation and optimization.

Built on C++20, Eigen, and CasADi, Janus implements the Code Transformations paradigm, enabling engineers to write physics models once and execute them in two distinct modes:

1. **Fast Numeric Mode**: For real-time simulation and control (standard C++/Eigen).
2. **Symbolic Trace Mode**: For gradient-based optimization and graph visualization (CasADi).

> For a deep dive into the architecture, see [Design Overview](docs/design_overview.md).

## Features

- 🎭 **Dual-Mode Physics**: Write once, run as Numeric (Fast C++) or Symbolic (CasADi Graph).
- 🔢 **Unified Math**: Std/CasADi agnostic math functions (`janus::sin`, `janus::pow`, `janus::where`).
- ⚡ **Linear Algebra**: Eigen-based matrix operations compatible with symbolic types.
- 📉 **Optimization**: High-level `Opti` interface for NLP solvers (IPOPT/SNOPT/QPOASES) with Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral transcriptions.
- 🔁 **Differentiation**: Automatic differentiation (Forward/Reverse) via CasADi.
- ⏱️ **Integration**: ODE solvers (`solve_ivp`), Runge-Kutta methods, discrete integration.
- 📈 **Interpolation**: 1D, 2D, Sparse, and N-D table lookups with B-spline support.
- 📐 **Geometry**: Quaternions, Rotation Matrices, Euler Angles.
- 🔍 **Root Finding**: Newton-Raphson and bracketing solvers.
- 🧠 **Surrogates**: Differentiable approximations (sigmoid, softmax) for discontinuous functions.
- 💾 **IO**: Graph visualization and serialization.

## Quick Start

### Prerequisites

- **Nix**: This project uses Nix Flakes to provide a reproducible development environment.

### Dev Shell

Enter the development environment:

```bash
./scripts/dev.sh
```

### Build & Test

We provide shorthand scripts to streamline the workflow:

1. **Build**: Configures and compiles the project.

    ```bash
    ./scripts/build.sh
    ```

2. **Test**: Rebuilds (if necessary) and runs the test suite.

    ```bash
    ./scripts/test.sh
    ```

3. **Clean**: Cleans out the build folder.

    ```bash
    ./scripts/clean.sh
    ```

4. **CI / Clean Verification**: Runs the full build and test pipeline inside the reproducible Nix environment (what CI does).

    ```bash
    ./scripts/ci.sh
    ```

5. **Examples**: Runs all example simulations.

    ```bash
    ./scripts/run_examples.sh
    ```

6. **Full Verification**: Runs everything (Build + Test + Examples). This is the recommended pre-push check.

    ```bash
    ./scripts/verify.sh
    ```

    Logs are saved to `logs/ci.log`, `logs/tests.log`, `logs/examples.log`, and `logs/verify.log`.

### Formatting

We use **treefmt** (via `nix fmt`) to enforce code style for C++, CMake, and Nix files.

**Manual formatting:**

```bash
nix fmt
```

**Auto-format on commit (recommended):**

```bash
./scripts/install-hooks.sh
```

This installs a pre-commit hook that automatically formats your code before each commit, so you never forget!

## Usage Example

Write physics models once using `janus` math functions, and execute them in both numeric and symbolic modes.

```cpp
#include <janus/janus.hpp>
#include <iostream>

// 1. Define Physics (Write Once)
// Use C++20 'auto' to support both double and Symbolic types automatically.
// This single function drives both simulation and optimization!
auto compute_drag(auto rho, auto v, auto S, auto Cd) {
    return 0.5 * rho * janus::pow(v, 2) * S * Cd;
}

int main() {
    // 2. Numeric Mode (Fast Simulation)
    // Compiles to standard optimized C++ machine code
    double D = compute_drag(1.225, 50.0, 10.0, 0.02); 
    std::cout << "Drag: " << D << " N" << std::endl;

    // 3. Symbolic Mode (Optimization)
    // Same function builds a computational graph for the solver
    janus::Opti opti;
    auto v = opti.variable(50.0);       // Decision Variable
    
    // Mix doubles and symbols seamlessly (no casting required!)
    auto D_sym = compute_drag(1.225, v, 10.0, 0.02); 
    
    opti.minimize(D_sym);               // Minimize Drag
    opti.subject_to_bounds(v, 10, 100); // Speed limit constraint
    
    auto sol = opti.solve();            // Solve with IPOPT
    std::cout << "Optimal V: " << sol.value(v) << " m/s" << std::endl;
}
```

For more details:

- **Numeric Simulation**: [Numeric Computing Guide](docs/user_guides/numeric_computing.md) and `examples/intro/numeric_intro.cpp`.
- **Symbolic Generation**: [Symbolic Computing Guide](docs/user_guides/symbolic_computing.md) and `examples/simulation/drag_coefficient.cpp`.
- **Optimization**: [Optimization Guide](docs/user_guides/optimization.md) and `examples/optimization/drag_optimization.cpp`.

## Project Structure

```plaintext
janus/
├── docs/                    # Documentation
├── examples/                # Example implementations
│   ├── intro/               # Basic Janus usage examples
│   ├── interpolation/       # Interpolation and table lookup demos
│   ├── math/                # Math/symbolic graph demos
│   ├── optimization/        # Optimization/transcription demos
│   └── simulation/          # ODE and physics simulation demos
├── include/
│   └── janus/
│       ├── core/                # Core symbolic/numeric interfaces and IO
│       ├── math/                # Math, interpolation, integration, autodiff helpers
│       ├── optimization/        # Opti + transcription methods
│       ├── utils/               # Utility helpers (e.g., JSON)
│       ├── janus.hpp            # Main umbrella include
│       └── using.hpp            # Convenience aliases/imports
├── scripts/                 # Build, test, CI, and verification scripts
├── tests/                   # GoogleTest suites (core/math/optimization)
├── CMakeLists.txt
└── flake.nix                # Nix environment definition
```

## Inspiration & Credits

Janus is heavily inspired by **AeroSandbox**, Peter Sharpe's Python-based design optimization framework. Janus serves as a C++ implementation and extension of the "Code Transformations" paradigm pioneered by Sharpe.

- **Primary Inspiration**: [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) by Peter Sharpe.
- **Theoretical Foundation**: Sharpe, Peter D. *AeroSandbox: A Differentiable Framework for Aircraft Design Optimization*. PhD Thesis, MIT, 2024. [Read Thesis](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/sharpe-pds-phd-AeroAstro-2024-thesis.pdf)

Janus is built upon the shoulders of giants:

- **[Eigen](https://eigen.tuxfamily.org/)**: For high-performance linear algebra and numeric storage.
- **[CasADi](https://web.casadi.org/)**: For symbolic graph generation, automatic differentiation, and optimization interfaces.
