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
nix develop
# OR if using direnv
direnv allow
```

### Build & Test

From the repo root:

1. **Configure**:

    ```bash
    cmake -B build -G Ninja
    ```

2. **Build**:

    ```bash
    ninja -C build
    ```

3. **Run Tests**:

    ```bash
    ctest --test-dir build --output-on-failure
    ```

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
