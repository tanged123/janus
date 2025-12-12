# Janus

**Janus** is a high-performance C++ numerical framework designed to implement the Code Transformations paradigm. It allows engineers to write physics models once and execute them in two distinct modes:

1. **Fast Numeric Mode**: For real-time simulation and control (standard C++/Eigen).
2. **Symbolic Trace Mode**: For gradient-based optimization (CasADi).

> For a deep dive into the architecture, see [Design Overview](docs/design_overview.md).

## Quick Start

### Prerequisites

* **Nix**: This project uses Nix Flakes to provide a reproducible development environment.
* **Direnv** (Optional but recommended): For automatic shell environment loading.

### Dev Shell

Enter the development environment:

```bash
./scripts/dev.sh
```

Or if you use **direnv**:

```bash
direnv allow
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
│   ├── core/           # Concepts & Types
│   ├── math/           # Math Dispatcher
│   └── linalg/         # Linear Algebra
├── tests/              # GoogleTest Suite
└── flake.nix           # Nix Environment Definition
```
