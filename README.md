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
    
4. **Examples**: Runs the example simulations (`drag_coefficient`, `energy_intro`, `numeric_intro`).

    ```bash
    ./scripts/examples.sh
    ```

5. **Full Verification**: Runs everything (Build + Test + Examples). This is the recommended pre-push check.

    ```bash
    ./scripts/verify.sh
    ```
    
    Logs are saved to `logs/ci.log`, `logs/tests.log`, `logs/examples.log`, and `logs/verify.log`.
### Formatting

We use **treefmt** to enforce code style for C++, CMake, and Nix files.

```bash
nix fmt
```

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
    
    auto result = f_grad(10.0, 2.0); // Evaluate derivatives numerically
}
```

For more details, see the [Symbolic Computing Guide](docs/user_guides/symbolic_computing.md) or check `examples/drag_coefficient.cpp`.

For high-performance numeric simulations, see the [Numeric Computing Guide](docs/user_guides/numeric_computing.md) and `examples/numeric_intro.cpp`.

## Project Structure

```plaintext
janus/
├── docs/               # Documentation
├── examples/           # Example Implementations (numeric & symbolic)
├── include/janus/      # Core Library Headers
│   ├── core/           # Concepts, Types & Function Wrapper
│   ├── math/           # Math & Numerics Layer
│   │   ├── Arithmetic.hpp   # Core arithmetic (pow, exp, log...)
│   │   ├── Trig.hpp         # Trigonometry (sin, cos, atan2...)
│   │   ├── Logic.hpp        # Branching (where) & sigmoids
│   │   ├── Linalg.hpp       # Linear Algebra (solve, norm)
│   │   ├── DiffOps.hpp      # Calculus (gradient, trapz)
│   │   ├── Interpolate.hpp  # Interpolation utilities
│   │   ├── Spacing.hpp      # Grid generation (linspace)
│   │   └── Rotations.hpp    # 2D Rotations (DCM)
│   └── linalg/         # Matrix extensions (future)
├── scripts/            # Build, Test & Verify Scripts
├── tests/              # GoogleTest Suite
└── flake.nix           # Nix Environment Definition
```
