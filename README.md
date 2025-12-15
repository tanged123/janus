# Janus

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tanged123.github.io/janus/) [![Janus CI](https://github.com/tanged123/janus/actions/workflows/ci.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/ci.yml) [![Clang-Format Check](https://github.com/tanged123/janus/actions/workflows/format.yml/badge.svg)](https://github.com/tanged123/janus/actions/workflows/format.yml) [![codecov](https://codecov.io/github/tanged123/janus/graph/badge.svg?token=0DSF7KK8W7)](https://codecov.io/github/tanged123/janus)

**Janus** is a high-performance C++ numerical framework named after the Roman god of duality. True to its name, it allows a single physics model to face two directions: **Numeric Mode** for fast execution and **Symbolic Mode** for graph generation and optimization.

Built on C++20, Eigen, and CasADi, Janus implements the Code Transformations paradigm, enabling engineers to write physics models once and execute them in two distinct modes:

1. **Fast Numeric Mode**: For real-time simulation and control (standard C++/Eigen).
2. **Symbolic Trace Mode**: For gradient-based optimization and graph visualization (CasADi).

> For a deep dive into the architecture, see [Design Overview](docs/design_overview.md).

## Quick Start

### Prerequisites

* **Nix**: This project uses Nix Flakes to provide a reproducible development environment.

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

// 1. Define a generic physics model
template <typename Scalar>
Scalar compute_energy(const Scalar& v, const Scalar& m) {
    // Use janus:: math functions for dual-backend support
    return 0.5 * m * janus::pow(v, 2.0);
}

int main() {
    // 2. Numeric Mode (Fast Standard Execution)
    double v = 10.0, m = 2.0;
    double E = compute_energy(v, m); // Returns 100.0

    // 3. Symbolic Mode (Graph Generation & Derivatives)
    auto v_sym = janus::sym("v");
    auto m_sym = janus::sym("m");
    auto E_sym = compute_energy(v_sym, m_sym);

    // Automatic Differentiation (Compute dE/dv)
    auto dE_dv = janus::jacobian({E_sym}, {v_sym}); 
    
    // Create Callable Function (wraps CasADi)
    janus::Function f_grad({v_sym, m_sym}, {dE_dv});
    
    // Evaluate derivatives numerically
    auto dE_dv_result = f_grad.eval(10.0, 2.0); // Returns 1x1 matrix with dE/dv = m*v = 20
}
```

For more details, see the [Symbolic Computing Guide](docs/user_guides/symbolic_computing.md) or check `examples/drag_coefficient.cpp`.

For high-performance numeric simulations, see the [Numeric Computing Guide](docs/user_guides/numeric_computing.md) and `examples/numeric_intro.cpp`.

## Project Structure

```plaintext
janus/
├── docs/               # Documentation
├── examples/           # Example Implementations (numeric & symbolic)
├── include/
│   └── janus/
│       ├── core/           # Concepts, Types & Function Wrapper
│       │   ├── Function.hpp      # CasADi Function wrapper
│       │   ├── JanusConcepts.hpp # Type concepts & constraints
│       │   ├── JanusError.hpp    # Exception handling types
│       │   ├── JanusIO.hpp       # Graph visualization & IO
│       │   └── JanusTypes.hpp    # Matrix/Vector types
│       ├── janus.hpp           # Main entry point
│       └── math/           # Math & Numerics Layer
│           ├── Arithmetic.hpp       # Core arithmetic
│           ├── AutoDiff.hpp         # Automatic differentiation
│           ├── Calculus.hpp         # Numerical calculus (gradient, trapz)
│           ├── DiffOps.hpp          # Differential operators
│           ├── FiniteDifference.hpp # Finite difference coefficients
│           ├── Integrate.hpp        # ODE integration (quad, solve_ivp)
│           ├── IntegrateDiscrete.hpp # Discrete integration schemes
│           ├── Interpolate.hpp      # Interpolation utilities
│           ├── JanusMath.hpp        # Math aggregation header
│           ├── Linalg.hpp           # Linear algebra extensions
│           ├── Logic.hpp            # Logical ops & branching (where)
│           ├── Quaternion.hpp       # Quaternion algebra
│           ├── Rotations.hpp        # 2D/3D rotations
│           ├── Spacing.hpp          # Grid generation
│           ├── SurrogateModel.hpp   # Smooth surrogates (sigmoid, etc)
│           └── Trig.hpp             # Trigonometry
├── scripts/            # Build, Test & Verify Scripts
├── tests/              # GoogleTest Suite
└── flake.nix           # Nix Environment Definition
```

