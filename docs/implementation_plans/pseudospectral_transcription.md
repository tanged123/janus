# Pseudospectral Transcription for Trajectory Optimization

**Goal**: Implement pseudospectral (p-method) optimal control transcription alongside the existing direct collocation (h-method), with a unified transcription interface that cleanly supports both families.

**Status**: Planning Draft
**Created**: 2026-02-10
**Branch**: `pseudospectral`

---

## Executive Summary

Direct collocation (trapezoidal, Hermite-Simpson) enforces dynamics via local defect constraints between adjacent node pairs. Pseudospectral methods take a fundamentally different approach: they represent the state as a **global polynomial** and enforce dynamics via a **differentiation matrix** applied simultaneously across all nodes. This yields **spectral convergence** -- exponentially fast error decay for smooth problems -- meaning 20-50 nodes can match accuracy that would require hundreds or thousands of h-method nodes.

This plan introduces:
1. **`Pseudospectral` class** -- LGL and CGL transcription with differentiation-matrix dynamics
2. **Orthogonal polynomial math** -- Legendre polynomial evaluation, LGL/CGL node computation, differentiation matrices, quadrature weights
3. **Unified transcription interface** -- Common base/concept shared by `DirectCollocation`, `MultipleShooting`, and `Pseudospectral`
4. **Quadrature-based cost integration** -- Gauss quadrature for Lagrange (integral) objectives

### When to Use Pseudospectral vs Collocation

| Scenario | Recommended Method |
|----------|--------------------|
| Smooth dynamics, high accuracy needed | Pseudospectral |
| Minimum-time / minimum-fuel (smooth) | Pseudospectral |
| Low node budget (real-time / embedded) | Pseudospectral |
| Bang-bang or discontinuous control | Direct collocation |
| Very large state dimension | Direct collocation |
| Poor initial guess / robustness needed | Direct collocation |

---

## Part 1: Orthogonal Polynomial Infrastructure

New file: `include/janus/math/OrthogonalPolynomials.hpp`

This is the mathematical foundation. All functions are pure numeric (no symbolic types) and operate on `NumericVector` / `NumericMatrix`.

### 1.1 Legendre Polynomial Evaluation

```cpp
namespace janus {

/// Evaluate Legendre polynomial P_n(x) and its derivative P'_n(x)
/// Uses the stable three-term recurrence:
///   P_0(x) = 1,  P_1(x) = x
///   (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
///
/// Returns {P_n(x), P'_n(x)}
std::pair<double, double> legendre_poly(int n, double x);

/// Evaluate P_n(x) at every element of a vector
NumericVector legendre_poly_vec(int n, const NumericVector &x);

} // namespace janus
```

**Implementation notes:**
- The recurrence is numerically stable for all `n` we'll use (typically < 100).
- The derivative satisfies: `P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)` for `|x| != 1`, with special-case formulas at the endpoints: `P'_n(1) = n(n+1)/2`, `P'_n(-1) = (-1)^{n+1} * n(n+1)/2`.

### 1.2 LGL Node Computation

```cpp
/// Compute Legendre-Gauss-Lobatto nodes on [-1, 1]
///
/// The N LGL nodes are: tau_1 = -1, tau_N = +1, and the N-2 interior
/// nodes are roots of P'_{N-1}(tau). Found via Newton iteration using
/// Chebyshev nodes as initial guesses.
///
/// @param N Number of nodes (>= 2)
/// @return Sorted vector of N nodes in [-1, 1]
NumericVector lgl_nodes(int N);
```

**Algorithm:**
1. Endpoints: `tau[0] = -1`, `tau[N-1] = +1`
2. Initial guesses for interior nodes: `tau_j^(0) = -cos(pi * j / (N-1))` (Chebyshev approximation)
3. Newton iteration: `tau^{(k+1)} = tau^{(k)} - P'_{N-1}(tau^{(k)}) / P''_{N-1}(tau^{(k)})` where `P''_{N-1}` is obtained from the identity `(1 - x^2) P''_n(x) = -2x P'_n(x) + n(n+1) P_n(x)`.
4. Converge to machine epsilon (typically 4-6 iterations).

### 1.3 CGL Node Computation

```cpp
/// Compute Chebyshev-Gauss-Lobatto nodes on [-1, 1]
///
/// Explicit closed-form: tau_j = -cos(j * pi / (N-1)),  j = 0, ..., N-1
///
/// @param N Number of nodes (>= 2)
/// @return Sorted vector of N nodes in [-1, 1]
NumericVector cgl_nodes(int N);
```

No iteration needed -- purely algebraic.

### 1.4 Quadrature Weights

```cpp
/// LGL quadrature weights
///   w_i = 2 / (N*(N-1) * [P_{N-1}(tau_i)]^2)
///
/// Exact for polynomials of degree <= 2N-3
NumericVector lgl_weights(int N, const NumericVector &nodes);

/// CGL (Clenshaw-Curtis) quadrature weights
/// Exact for polynomials of degree <= N-1
NumericVector cgl_weights(int N, const NumericVector &nodes);
```

### 1.5 Differentiation Matrix

```cpp
/// Compute the spectral differentiation matrix D on the given nodes.
///
/// D is N x N, where D_{ij} = L'_j(tau_i) with L_j the Lagrange
/// basis polynomial through the nodes. The matrix maps function values
/// at nodes to derivative values at nodes:  f' = D * f
///
/// Supports both LGL and CGL nodes. Uses the barycentric formula for
/// off-diagonal entries and the negative-sum trick for diagonal stability.
///
/// @param nodes  The collocation nodes (LGL or CGL), length N
/// @return N x N differentiation matrix
NumericMatrix spectral_diff_matrix(const NumericVector &nodes);
```

**Algorithm (barycentric form):**

1. Compute barycentric weights: `lambda_j = 1 / prod_{k != j} (tau_j - tau_k)`
2. Off-diagonal: `D_{ij} = (lambda_j / lambda_i) / (tau_i - tau_j)` for `i != j`
3. Diagonal (negative-sum trick): `D_{ii} = -sum_{j != i} D_{ij}`

The negative-sum trick ensures `D * [1, 1, ..., 1]^T = 0` exactly, which is critical for numerical stability.

**For LGL nodes specifically**, there is also an explicit Legendre-based formula:
- `D_{ij} = P_{N-1}(tau_i) / (P_{N-1}(tau_j) * (tau_i - tau_j))` for `i != j`
- `D_{00} = -N(N-1)/4`, `D_{N-1,N-1} = +N(N-1)/4`
- `D_{ii} = 0` for interior nodes

We'll implement the barycentric form since it works for both LGL and CGL nodes uniformly.

### 1.6 File Layout

```
include/janus/math/OrthogonalPolynomials.hpp
    legendre_poly()
    lgl_nodes()
    cgl_nodes()
    lgl_weights()
    cgl_weights()
    spectral_diff_matrix()
```

All functions are `inline` or `constexpr` where possible, consistent with the header-only library.

---

## Part 2: Unified Transcription Interface

Currently, `DirectCollocation` and `MultipleShooting` share a near-identical API by convention but have no formal relationship. Before adding a third transcription method, we should formalize the interface.

### 2.1 The Problem

All three transcription classes share:
- `setup(n_states, n_controls, t0, tf, opts)` returning `{states, controls, time_grid}`
- `set_dynamics(func)`
- `add_dynamics_constraints()`
- `set_initial_state()` / `set_final_state()`
- `states()`, `controls()`, `time_grid()`, `n_nodes()`

But they also have **method-specific options** (CollocationOptions, MultiShootingOptions, and now PseudospectralOptions), making a single virtual interface awkward.

### 2.2 Approach: CRTP Mixin for Shared State + Free Functions

Rather than forcing runtime polymorphism, we use a **CRTP base class** that factors out the common data members and boundary-condition logic, while leaving the transcription-specific parts (setup internals, constraint generation) to the derived class.

```cpp
// include/janus/optimization/TranscriptionBase.hpp

namespace janus {

/// Common state shared by all transcription methods
template <typename Derived>
class TranscriptionBase {
public:
    explicit TranscriptionBase(Opti &opti) : opti_(opti) {}

    // --- Boundary conditions (identical across all methods) ---
    void set_initial_state(const NumericVector &x0);
    void set_initial_state(int idx, double value);
    void set_final_state(const NumericVector &xf);
    void set_final_state(int idx, double value);

    // --- Dynamics (type-erased storage, identical across all) ---
    template <typename Func> void set_dynamics(Func &&dynamics);

    // --- Accessors ---
    const SymbolicMatrix &states() const { return states_; }
    const SymbolicMatrix &controls() const { return controls_; }
    const NumericVector &time_grid() const { return tau_; }
    int n_nodes() const { return n_nodes_; }
    int n_states() const { return n_states_; }
    int n_controls() const { return n_controls_; }

    // --- Unified constraint entry point (delegates to derived) ---
    void add_dynamics_constraints() {
        static_cast<Derived*>(this)->add_dynamics_constraints_impl();
    }

    // Legacy aliases
    void add_defect_constraints() { add_dynamics_constraints(); }
    void add_continuity_constraints() { add_dynamics_constraints(); }

protected:
    Opti &opti_;
    int n_states_ = 0, n_controls_ = 0, n_nodes_ = 0;
    double t0_ = 0.0, tf_fixed_ = 1.0;
    SymbolicScalar tf_symbolic_;
    bool tf_is_variable_ = false;
    bool setup_complete_ = false, dynamics_set_ = false;

    NumericVector tau_;
    SymbolicMatrix states_, controls_;

    std::function<SymbolicVector(const SymbolicVector &,
                                  const SymbolicVector &,
                                  const SymbolicScalar &)> dynamics_;

    // Helpers
    SymbolicScalar get_duration() const;
    SymbolicScalar get_time_at_node(int k) const;
    SymbolicVector get_state_at_node(int k) const;
    SymbolicVector get_control_at_node(int k) const;
};

} // namespace janus
```

### 2.3 Migration Path

This is a **non-breaking refactor**. The existing classes gain a base without changing their public API:

```cpp
// Before:
class DirectCollocation { ... };

// After:
class DirectCollocation : public TranscriptionBase<DirectCollocation> {
    friend class TranscriptionBase<DirectCollocation>;
    void add_dynamics_constraints_impl();  // The actual trapezoidal/HS logic
    // setup() stays as-is, just populates base members
};
```

`MultipleShooting` gets the same treatment. All existing user code compiles unchanged because the public method signatures are identical.

### 2.4 Phasing

| Step | What | Breaking? |
|------|------|-----------|
| 1 | Create `TranscriptionBase.hpp` | No |
| 2 | Refactor `DirectCollocation` to inherit from it | No (API identical) |
| 3 | Refactor `MultipleShooting` to inherit from it | No (API identical) |
| 4 | Build `Pseudospectral` on the same base | No (new class) |

---

## Part 3: Pseudospectral Class

New file: `include/janus/optimization/Pseudospectral.hpp`

### 3.1 Options

```cpp
enum class PseudospectralScheme {
    LGL,  ///< Legendre-Gauss-Lobatto (both endpoints are nodes)
    CGL   ///< Chebyshev-Gauss-Lobatto (both endpoints, closed-form nodes)
};

struct PseudospectralOptions {
    PseudospectralScheme scheme = PseudospectralScheme::LGL;
    int n_nodes = 21;  ///< Number of collocation nodes (including endpoints)
};
```

**Why LGL + CGL first (not LGR)?**
- Both include endpoints as nodes, so boundary conditions work identically to `DirectCollocation` -- users get a familiar workflow.
- CGL nodes are trivial to compute (good for initial validation).
- LGL is the classic pseudospectral method (DIDO, Matlab `bvp5c`).
- LGR (used in GPOPS-II) is a natural follow-on but requires rectangular differentiation matrices and non-collocated terminal state handling -- meaningfully more complex. We note this as a future extension in Part 5.

### 3.2 Class Skeleton

```cpp
class Pseudospectral : public TranscriptionBase<Pseudospectral> {
    friend class TranscriptionBase<Pseudospectral>;
public:
    explicit Pseudospectral(Opti &opti)
        : TranscriptionBase<Pseudospectral>(opti) {}

    /// Setup decision variables
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const PseudospectralOptions &opts = {});

    /// Setup with free final time
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const PseudospectralOptions &opts = {});

    /// Access the differentiation matrix (useful for advanced users)
    const NumericMatrix &diff_matrix() const { return D_; }

    /// Access the quadrature weights (useful for cost integration)
    const NumericVector &quadrature_weights() const { return weights_; }

    /// Convenience: compute Gauss quadrature of an integrand at nodes
    ///   integral ≈ (dt/2) * sum_i w_i * f_i
    SymbolicScalar quadrature(const SymbolicVector &integrand) const;

private:
    PseudospectralScheme scheme_;
    NumericMatrix D_;         ///< N x N differentiation matrix
    NumericVector weights_;   ///< N quadrature weights

    void add_dynamics_constraints_impl();
};
```

### 3.3 Setup Implementation

```cpp
auto Pseudospectral::setup(int n_states, int n_controls,
                           double t0, double tf,
                           const PseudospectralOptions &opts)
    -> std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
{
    n_states_ = n_states;
    n_controls_ = n_controls;
    n_nodes_ = opts.n_nodes;
    scheme_ = opts.scheme;
    t0_ = t0;
    tf_fixed_ = tf;
    tf_is_variable_ = false;

    // 1. Compute collocation nodes on [-1, 1]
    NumericVector nodes;
    switch (scheme_) {
    case PseudospectralScheme::LGL:
        nodes = lgl_nodes(n_nodes_);
        weights_ = lgl_weights(n_nodes_, nodes);
        break;
    case PseudospectralScheme::CGL:
        nodes = cgl_nodes(n_nodes_);
        weights_ = cgl_weights(n_nodes_, nodes);
        break;
    }

    // 2. Map nodes from [-1, 1] to [0, 1] for tau_ (consistent with
    //    DirectCollocation's convention)
    tau_ = (nodes.array() + 1.0) * 0.5;  // tau in [0, 1]

    // 3. Compute differentiation matrix on original [-1, 1] nodes
    D_ = spectral_diff_matrix(nodes);

    // 4. Create decision variables (same pattern as DirectCollocation)
    states_ = SymbolicMatrix(n_nodes_, n_states_);
    for (int k = 0; k < n_nodes_; ++k)
        for (int i = 0; i < n_states_; ++i)
            states_(k, i) = opti_.variable(0.0);

    controls_ = SymbolicMatrix(n_nodes_, n_controls_);
    for (int k = 0; k < n_nodes_; ++k)
        for (int i = 0; i < n_controls_; ++i)
            controls_(k, i) = opti_.variable(0.0);

    setup_complete_ = true;
    return {states_, controls_, tau_};
}
```

**Key design decision -- node domain convention:** Internally we compute nodes on `[-1, 1]` (the natural domain for Legendre/Chebyshev polynomials) and store `D_` in that domain. The user-facing `tau_` is mapped to `[0, 1]` for consistency with `DirectCollocation`. The time scaling in `add_dynamics_constraints_impl` accounts for both mappings.

### 3.4 Dynamics Constraints -- The Core Difference

In direct collocation, dynamics are enforced as **local defect constraints** between adjacent pairs of nodes. In pseudospectral methods, dynamics are enforced as a **global matrix equation**:

```
D * X_col = (dt / 2) * F(X, U, t)
```

where `D` is the differentiation matrix on `[-1, 1]`, `X_col` is the column of state values at all nodes for a single state component, and the `dt/2` factor comes from the chain rule `dx/dt = (2/dt) * dx/dtau`.

```cpp
void Pseudospectral::add_dynamics_constraints_impl() {
    if (!dynamics_set_)
        throw std::runtime_error(
            "Pseudospectral: call set_dynamics() before add_dynamics_constraints()");

    // Duration: symbolic if free, numeric if fixed
    SymbolicScalar duration = get_duration();

    // Time scaling factor: D operates on [-1,1], so dx/dt = (2/dt) * D*x
    // Rearranged: D*x = (dt/2) * f(x, u, t)
    SymbolicScalar half_dt = duration / 2.0;

    // Evaluate dynamics at every node
    std::vector<SymbolicVector> f(n_nodes_);
    for (int k = 0; k < n_nodes_; ++k) {
        SymbolicVector x_k = get_state_at_node(k);
        SymbolicVector u_k = get_control_at_node(k);
        SymbolicScalar t_k = get_time_at_node(k);
        f[k] = dynamics_(x_k, u_k, t_k);
    }

    // For each state component, enforce D * x_col = (dt/2) * f_col
    for (int s = 0; s < n_states_; ++s) {
        for (int i = 0; i < n_nodes_; ++i) {
            // (D * x)_i = sum_j D_{ij} * x_j(s)
            SymbolicScalar Dx_i = SymbolicScalar(0.0);
            for (int j = 0; j < n_nodes_; ++j) {
                Dx_i = Dx_i + D_(i, j) * states_(j, s);
            }

            // Constraint: (D*x)_i == (dt/2) * f_i(s)
            opti_.subject_to(Dx_i == half_dt * f[i](s));
        }
    }
}
```

This produces `N * n_states` equality constraints -- the same count as direct collocation with N nodes, but each constraint couples **all** nodes (through `D`), producing a dense constraint Jacobian block. This is the expected structure for pseudospectral methods and is handled efficiently by IPOPT for typical node counts (N < 100).

### 3.5 Quadrature for Integral Objectives

A key advantage of pseudospectral methods is that the quadrature weights provide highly accurate integral approximations for free. This is essential for Lagrange-type (running cost) objectives:

```
J = integral_{t0}^{tf} L(x, u, t) dt
  ≈ (dt/2) * sum_{i=1}^{N} w_i * L(x_i, u_i, t_i)
```

```cpp
SymbolicScalar Pseudospectral::quadrature(
    const SymbolicVector &integrand) const
{
    // integrand(k) = L(x_k, u_k, t_k) evaluated at each node
    SymbolicScalar sum = SymbolicScalar(0.0);
    for (int k = 0; k < n_nodes_; ++k) {
        sum = sum + weights_(k) * integrand(k);
    }
    // Scale by dt/2 (the [-1,1] to [t0,tf] Jacobian)
    return get_duration() / 2.0 * sum;
}
```

**Usage in a problem:**
```cpp
Pseudospectral ps(opti);
auto [x, u, tau] = ps.setup(2, 1, 0.0, T, opts);
ps.set_dynamics(my_ode);
ps.add_dynamics_constraints();

// Minimize integral of u^2 (fuel-optimal-like objective)
SymbolicVector integrand(ps.n_nodes());
for (int k = 0; k < ps.n_nodes(); ++k)
    integrand(k) = u(k, 0) * u(k, 0);
opti.minimize(ps.quadrature(integrand));
```

### 3.6 Also Add Quadrature to DirectCollocation

For API parity, we should add a `quadrature()` method to `DirectCollocation` as well (using trapezoidal or Simpson rule consistent with the selected scheme). This can live in the base class with a virtual dispatch or as a simple method on each class. Since the weights differ by scheme, each class implements its own.

---

## Part 4: Implementation Order & Testing

### Phase 4.1: Math Foundation

**Files:** `include/janus/math/OrthogonalPolynomials.hpp`
**Tests:** `tests/math/test_orthogonal_polynomials.cpp`

| Task | Test |
|------|------|
| `legendre_poly(n, x)` | Verify P_0..P_5 against known values. Check recurrence at edges. |
| `lgl_nodes(N)` | Verify N=2 gives {-1, 1}. Check symmetry. Compare N=5,10,20 against tabulated values. Verify P'_{N-1} vanishes at interior nodes. |
| `cgl_nodes(N)` | Verify against explicit cosine formula. Check endpoints. |
| `lgl_weights(N)` | Sum of weights == 2. Integrate x^k exactly for k <= 2N-3. |
| `cgl_weights(N)` | Sum of weights == 2. Integrate x^k exactly for k <= N-1. |
| `spectral_diff_matrix(nodes)` | D * [1,1,...,1]^T == 0 (differentiates constants to zero). D * tau == [1,1,...,1]^T (differentiates identity). D applied to sin(tau) matches cos(tau) at nodes. |

**Estimated scope:** ~200 lines of implementation, ~150 lines of tests.

### Phase 4.2: TranscriptionBase Refactor

**Files:**
- New: `include/janus/optimization/TranscriptionBase.hpp`
- Modified: `include/janus/optimization/Collocation.hpp`
- Modified: `include/janus/optimization/MultiShooting.hpp`

**Validation:** All existing tests in `tests/optimization/test_collocation.cpp` and `tests/optimization/test_multishoot.cpp` pass unchanged. This is a pure refactor with zero behavioral change.

| Task | Validation |
|------|------------|
| Extract common members into `TranscriptionBase` | Compiles, all tests pass |
| `DirectCollocation` inherits from base | All collocation tests pass |
| `MultipleShooting` inherits from base | All shooting tests pass |

**Estimated scope:** ~120 lines for the base, net-negative lines in derived classes.

### Phase 4.3: Pseudospectral Class

**Files:**
- New: `include/janus/optimization/Pseudospectral.hpp`
- Modified: `include/janus/janus.hpp` (add include)
- New: `tests/optimization/test_pseudospectral.cpp`
- New: `examples/optimization/pseudospectral_demo.cpp`

**Test cases (progressive difficulty):**

| Test | Description | Expected |
|------|-------------|----------|
| Double integrator (LGL) | x'' = u, min integral u^2, x(0)=0, x(1)=1 | Matches collocation to 6+ digits |
| Double integrator (CGL) | Same, Chebyshev nodes | Same accuracy |
| Harmonic oscillator | x'' + x = 0, verify energy conservation | Energy drift < 1e-10 |
| Brachistochrone (LGL) | Minimize time, compare with DC result (1.8016s) | Matches to < 0.1% |
| Free final time | Same brachistochrone with T as decision variable | T converges correctly |
| Convergence study | Run N = 5, 9, 13, 17, 21 on double integrator, verify exponential error decay | Error plot shows spectral convergence |
| Quadrature accuracy | Integrate known functions, compare with analytical | Machine precision for polynomials up to expected degree |

### Phase 4.4: Example & Documentation

**Files:**
- New: `examples/optimization/pseudospectral_demo.cpp`
- New: `docs/user_guides/pseudospectral.md`

The demo mirrors `collocation_demo.cpp` (brachistochrone) so users can directly compare the two approaches. The user guide follows the structure of `collocation.md`.

---

## Part 5: Future Extensions (Not in This PR)

These are natural follow-ons that inform the current design but are **not** part of this implementation:

### 5.1 Legendre-Gauss-Radau (LGR) Scheme

LGR nodes include only the initial endpoint (tau = -1). The differentiation matrix is **(N) x (N+1) rectangular** (N collocation nodes, N+1 state nodes including the non-collocated terminal point). This is what GPOPS-II uses and provides:
- Full-rank differentiation matrix (no singularity issues)
- Superior costate convergence
- Better accuracy per node

**Design note:** The current `TranscriptionBase` assumes `states_` is `[n_nodes x n_states]` where nodes and state evaluation points coincide. LGR breaks this assumption because the terminal state is an extra decision variable beyond the collocation nodes. The base class should be designed to not preclude this (store `n_collocation_nodes` and `n_state_nodes` separately, or handle it in the derived class). For this first implementation we document this consideration but don't over-engineer for it.

### 5.2 hp-Adaptive Mesh Refinement

Divide the trajectory into multiple intervals, each with its own set of pseudospectral nodes. Adapt both the number of intervals (h-refinement) and polynomial degree per interval (p-refinement). This recovers sparsity (block-diagonal Jacobian) and handles non-smooth solutions.

### 5.3 Costate Estimation

The KKT multipliers of the pseudospectral NLP approximate the continuous-time costates (Pontryagin's minimum principle). Extracting and returning these is valuable for verification and guidance law design.

---

## Part 6: File Summary

### New Files
| File | Contents |
|------|----------|
| `include/janus/math/OrthogonalPolynomials.hpp` | Legendre polynomials, LGL/CGL nodes, weights, differentiation matrix |
| `include/janus/optimization/TranscriptionBase.hpp` | CRTP base for shared transcription logic |
| `include/janus/optimization/Pseudospectral.hpp` | Pseudospectral transcription class |
| `tests/math/test_orthogonal_polynomials.cpp` | Unit tests for polynomial infrastructure |
| `tests/optimization/test_pseudospectral.cpp` | Integration tests for PS transcription |
| `examples/optimization/pseudospectral_demo.cpp` | Brachistochrone via pseudospectral |
| `docs/user_guides/pseudospectral.md` | User guide |

### Modified Files
| File | Change |
|------|--------|
| `include/janus/optimization/Collocation.hpp` | Inherit from `TranscriptionBase`, move shared members to base |
| `include/janus/optimization/MultiShooting.hpp` | Inherit from `TranscriptionBase`, move shared members to base |
| `include/janus/janus.hpp` | Add `#include` for new headers |
| `tests/CMakeLists.txt` | Add new test targets |
| `examples/CMakeLists.txt` | Add new example target |

---

## Appendix A: Mathematical Reference

### A.1 Time Domain Mapping

Physical time `t in [t0, tf]` maps to computational time `tau in [-1, 1]`:

```
tau = 2*(t - t0)/(tf - t0) - 1
t   = t0 + (tf - t0)*(tau + 1)/2
dt/dtau = (tf - t0)/2
```

The differentiation matrix `D` computes `dx/dtau`. To get `dx/dt`:

```
dx/dt = (dx/dtau) / (dt/dtau) = (2 / (tf - t0)) * D * x
```

So the dynamics constraint `dx/dt = f(x, u, t)` becomes:

```
D * x = ((tf - t0) / 2) * f(x, u, t)
```

### A.2 LGL Quadrature Weights

```
w_i = 2 / (N * (N-1) * [P_{N-1}(tau_i)]^2)
```

### A.3 CGL (Clenshaw-Curtis) Weights

Computed via the DCT-based algorithm or the explicit formula:

```
w_0 = w_{N-1} = 1 / (N*(N-2))    [endpoint weights, N odd]

For interior nodes, computed from:
  w_j = (2/N) * (1 - sum_{k=1}^{floor((N-1)/2)} b_k * cos(2*k*j*pi/(N-1)))
  where b_k = 2/(4k^2 - 1), with b_{(N-1)/2} halved if N is odd
```

### A.4 NLP Size Comparison (N nodes, n_x states, n_u controls)

| | Direct Collocation | Pseudospectral |
|--|-------------------|----------------|
| Decision variables | N*(n_x + n_u) | N*(n_x + n_u) |
| Dynamics constraints | (N-1)*n_x | N*n_x |
| Jacobian sparsity | Banded (each constraint couples 2 nodes) | Dense (each constraint couples all N nodes) |
| Typical N for 1e-6 accuracy | 100-500 | 15-40 |
