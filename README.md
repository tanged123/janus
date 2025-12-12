# Janus

**Janus** is a high-performance C++ numerical framework designed to implement the Code Transformations paradigm. It allows engineers to write physics models once and execute them in two distinct modes:

1. **Fast Numeric Mode**: For real-time simulation and control (standard C++/Eigen).
2. **Symbolic Trace Mode**: For gradient-based optimization (CasADi).

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

3. **CI / Clean Verification**: Runs the full build and test pipeline inside the reproducible Nix environment (what CI does).

    ```bash
    ./scripts/ci.sh
    ```
    
    Logs are saved to `logs/ci.log` and `logs/tests.log`.
### Formatting

We use **treefmt** to enforce code style for C++, CMake, and Nix files.

```bash
nix fmt
```

## Project Structure

```plaintext
janus/
├── docs/               # Documentation
├── include/janus/      # Core Library Headers
│   ├── core/           # Concepts & Types (JanusScalar, etc.)
│   ├── math/           # Math & Numerics Layer
│   │   ├── Arithmetic.hpp   # Core arithmetic (pow, exp, log...)
│   │   ├── Trig.hpp         # Trigonometry (sin, cos, atan2...)
│   │   ├── Logic.hpp        # Branching (where) & sigmoids
│   │   ├── Linalg.hpp       # Linear Algebra (solve, norm)
│   │   ├── DiffOps.hpp      # Calculus (gradient, trapz)
│   │   └── Interpolate.hpp  # Interpolation utilities
│   └── linalg/         # Matrix extensions (future)
├── scripts/            # Build & Test Scripts (ci.sh, etc.)
├── tests/              # GoogleTest Suite
└── flake.nix           # Nix Environment Definition
```
