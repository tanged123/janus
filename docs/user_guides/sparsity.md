# Sparsity in Janus

Understanding the sparsity pattern of your Jacobian and Hessian matrices is crucial for high-performance optimization. Dense solvers scale poorly ($O(N^3)$), while sparse solvers (like MUMPS, MA57) scale much better, provided the problem structure is exploited.

Janus provides tools to:
- inspect sparsity patterns directly from symbolic graphs
- visualize those patterns as ASCII, PDF, or HTML spy plots
- compile sparse Jacobian and Hessian value evaluators that return only structural nonzeros
- surface CasADi graph coloring so you can see how many directional sweeps the sparse derivative pipeline needs

## The `SparsityPattern` Class

The core of this feature is `janus::SparsityPattern` (in `<janus/core/Sparsity.hpp>`). It wraps CasADi's sparsity logic but integrates seamlessly with Janus native types.

### Key Features
- **Query**: Check `nnz()` (number of non-zeros), `density()`, dimensions, etc.
- **Visualize**: Generate ASCII "spy plots" for console or high-quality PDFs for reports.
- **Analyze**: Extract patterns from symbolic expressions or Janus Functions.

## Extracting Sparsity

You can extract sparsity in three main ways:

### 1. From Symbolic Expressions
If you have a symbolic expression $f(x)$, you can ask for the sparsity of its derivatives:

```cpp
auto x = janus::sym("x", 10);
auto f = ...; // some expression depending on x

// Jacobian sparsity (df/dx)
janus::SparsityPattern J_sp = janus::sparsity_of_jacobian(f, x);

// Hessian sparsity (d²f/dx²)
janus::SparsityPattern H_sp = janus::sparsity_of_hessian(f, x);
```

### 2. From `janus::Function`
If you have already compiled a function:
```cpp
janus::Function func(inputs, outputs);
auto sp = janus::get_jacobian_sparsity(func);
```

For multi-input or multi-output functions, use explicit block selection:

```cpp
janus::Function func(inputs, outputs);

auto J_sp = janus::get_jacobian_sparsity(func, output_idx, input_idx);
auto H_sp = janus::get_hessian_sparsity(func, scalar_output_idx, input_idx);
```

### 3. From CasADi Types directly
```cpp
casadi::Sparsity raw = ...;
janus::SparsityPattern sp(raw);

janus::SymbolicScalar expr = ...;
janus::SparsityPattern expr_sp(expr); // Extract from MX sparsity
```

## Sparse Derivative Value Kernels

Pattern inspection is useful, but high-performance workflows usually need more: a reusable derivative evaluator that only returns structural nonzeros.

Janus now exposes that pipeline directly:

```cpp
auto x = janus::sym("x", 10);
auto f = ...; // vector expression
auto phi = ...; // scalar expression

auto J = janus::sparse_jacobian(f, x);
auto H = janus::sparse_hessian(phi, x);
```

Both evaluators cache:
- the fixed sparsity pattern
- graph coloring metadata
- a compiled `janus::Function` that emits only structural nonzero values

### Jacobian Pipeline

```cpp
auto x = janus::sym("x", 6);
auto f = janus::SymbolicScalar::vertcat({
    x(1) - x(0),
    x(2) - janus::sin(x(1)),
    x(3) - x(2)
});

auto J = janus::sparse_jacobian(f, x);

std::cout << "nnz = " << J.nnz() << "\n";
std::cout << "forward colors = " << J.forward_coloring().n_colors() << "\n";
std::cout << "reverse colors = " << J.reverse_coloring().n_colors() << "\n";
std::cout << "preferred mode = "
          << (J.preferred_mode() == janus::SparseJacobianMode::Forward ? "forward" : "reverse")
          << "\n";

janus::NumericMatrix x_val(6, 1);
x_val << 0.0, 0.2, 0.5, 0.9, 0.0, 0.0;

janus::NumericMatrix jac_nz = J.values(x_val);
```

`jac_nz` is a column vector of derivative values in the same CCS ordering reported by:
- `J.sparsity().get_triplet()`
- `J.sparsity().get_ccs()`

That ordering is fixed, so the sparsity structure can be reused across many evaluations.

### Hessian Pipeline

```cpp
auto x = janus::sym("x", 5);
auto phi = 0;
for (int k = 0; k < 4; ++k) {
    auto diff = x(k + 1) - x(k);
    phi = phi + diff * diff;
}

auto H = janus::sparse_hessian(phi, x);

std::cout << "nnz = " << H.nnz() << "\n";
std::cout << "star colors = " << H.coloring().n_colors() << "\n";

janus::NumericMatrix x_val(5, 1);
x_val << 0.0, 0.1, 0.3, 0.7, 1.0;

janus::NumericMatrix hess_nz = H.values(x_val);
```

For Hessians, Janus exposes CasADi's star coloring through `H.coloring()`.

### From `janus::Function` Blocks

Sparse derivative evaluators can also be built from an already-compiled `janus::Function`, selecting a specific output block and input block:

```cpp
janus::Function terms("terms", {x, u}, {defects, objective});

auto ddefects_dx = janus::sparse_jacobian(terms, 0, 0);
auto ddefects_du = janus::sparse_jacobian(terms, 0, 1);
auto hobjective_xx = janus::sparse_hessian(terms, 1, 0);
```

This is the most useful form for optimization pipelines where one compiled function already exposes multiple residual, constraint, and objective blocks.

### Reconstructing a Dense Matrix

When debugging, it is often useful to reconstruct the dense matrix from sparse values:

```cpp
auto nz = J.values(x_val);
auto [rows, cols] = J.sparsity().get_triplet();

janus::NumericMatrix dense =
    janus::NumericMatrix::Zero(J.sparsity().n_rows(), J.sparsity().n_cols());

for (Eigen::Index k = 0; k < nz.size(); ++k) {
    dense(rows[static_cast<size_t>(k)], cols[static_cast<size_t>(k)]) = nz(k);
}
```

In production you usually keep the structural ordering and pass the nonzero vector straight into a sparse solver or downstream callback.

### Reading Coloring Metadata

`janus::GraphColoring` exposes:
- `n_entries()` for the uncompressed derivative size
- `n_colors()` for the compressed directional count
- `compression_ratio()` for a quick summary
- `colorvec()` for the per-entry color assignment

This is useful when comparing derivative blocks and deciding whether sparse directional sweeps are worth it.

## Visualization

### ASCII Spy Plot
For quick debugging in the terminal, use `to_string()`:
```cpp
std::cout << sp.to_string() << std::endl;
```

Output:
```
Sparsity: 10x10, nnz=28 (density=28.000%)
┌──────────┐
│**........│
│***.......│
│.***......│
...
└──────────┘
```
Dots `.` represent zeros, asterisks `*` represent non-zeros.

### PDF Rendering (Graphviz)
For complex patterns or reports, generate a PDF grid visualization:
```cpp
sp.visualize_spy("my_pattern"); // Creates my_pattern.pdf
```
This renders a true "spy plot" where non-zeros are black squares.

### Interactive HTML
For exploring large matrices with pan/zoom and cell inspection:
```cpp
sp.export_spy_html("my_pattern", "My Jacobian"); // Creates my_pattern.html
```

The HTML output includes:
- **Pan/zoom** with mouse scroll and drag
- **Click cells** to see row/col in sidebar
- **Axis labels** with automatic tick marks
- **Stats panel** showing matrix size, nnz, density

## Example Walkthrough: `sparsity_intro.cpp`

The example `examples/intro/sparsity_intro.cpp` demonstrates four common structures found in optimization.

### 1. Simple Jacobian
Shows explicitly how dependencies map to non-zeros.
$f_0 = x_0^2$ depends only on $x_0$, so row 0 has non-zero at col 0.

### 2. Chain Structure (Tridiagonal)
```cpp
f = sum((x[i] - x[i+1])^2)
```
This optimization objective creates a tridiagonal Hessian. This "arrowhead" or band structure is typical in trajectory optimization (e.g., Direct Collocation) where each state depends only on its neighbors.

### 3. Independent Systems (Block Diagonal)
Two completely separate systems stacked together form a block-diagonal matrix. Solvers love this structure because it can be parallelized trivially. The example explicitly constructs this to show how Janus detects it.

### 4. 2D Laplacian (5-Point Stencil)
Typical in PDE constraints or grid-based problems. Each node $(i,j)$ depends on itself and its 4 neighbors (Up, Down, Left, Right).
The example uses `janus::sym_vec_pair` to easily handle 2D indexing while maintaining a valid symbolic input for the function:

```cpp
// 1. Create vector of symbols for easy indexing
auto [x_vec, x_mx] = janus::sym_vec_pair("x", n_vars);

// 2. Build equations using x_vec(k)
...

// 3. Create Function using raw x_mx
janus::Function f_pde({x_mx}, {janus::SymbolicScalar::vertcat(eqs)});
```

This structure produces a banded matrix with off-diagonal bands at distance $\pm 1$ and $\pm N$.

> [!TIP]
> [🔍 Explore the 2D Laplacian sparsity pattern interactively](examples/laplacian_2d.html) (open locally or via GitHub Pages)

---

## Sparse Pipeline Example Walkthrough: `sparse_derivative_pipeline.cpp`

The example `examples/math/sparse_derivative_pipeline.cpp` shows the full J-13 workflow:

1. Build a trajectory-style residual with local coupling.
2. Compile sparse Jacobian blocks and a sparse Hessian block.
3. Print `nnz` versus dense size.
4. Inspect forward, reverse, and star coloring counts.
5. Reuse the same sparse kernels at two different evaluation points.

This example is useful when you want to answer practical questions like:
- "How sparse is this block really?"
- "Should I expect forward or reverse directional sweeps to win?"
- "Can I evaluate only nonzero derivative entries and reuse the structure across solves?"

Build and run it with:

```bash
ninja -C build sparse_derivative_pipeline
./build/examples/sparse_derivative_pipeline
```

The output prints both the sparse nonzero vectors and reconstructed dense blocks so you can verify the ordering and the structure reuse story.

---

## NaN-Propagation Sparsity Detection

Sometimes you have **black-box functions** where symbolic sparsity analysis isn't possible:
- External library calls
- Non-traceable operations
- Functions with runtime branching

Janus provides `nan_propagation_sparsity()` for these cases.

### How It Works

1. Evaluate f(x) at a reference point
2. For each input i: set x[i] = NaN, evaluate f(x)
3. If output[j] becomes NaN → depends on input i → Jacobian(j, i) ≠ 0

### API

```cpp
// For lambda/callable functions
auto sp = janus::nan_propagation_sparsity(
    [](const NumericVector& x) {
        NumericVector y(x.size());
        for (int i = 0; i < x.size(); ++i) y(i) = x(i) * x(i);
        return y;
    },
    n_inputs, n_outputs);

// For janus::Function
janus::Function fn(...);
auto sp = janus::nan_propagation_sparsity(fn);

// With custom options (reference point)
NaNSparsityOptions opts;
opts.reference_point = NumericVector{{1.0, 2.0, 3.0}};
auto sp = nan_propagation_sparsity(fn, opts);
```

### Example: Verifying Symbolic Sparsity

```cpp
// Create symbolic function
auto x = janus::sym("x", 4);
auto f = x * x;  // Element-wise square

// Symbolic sparsity
auto sp_symbolic = janus::sparsity_of_jacobian(f, x);

// NaN-propagation sparsity (black-box equivalent)
janus::Function fn({x}, {f});
auto sp_nan = janus::nan_propagation_sparsity(fn);

// They should match!
assert(sp_symbolic == sp_nan);
```

> [!TIP]
> Use NaN-propagation to validate that your symbolic expressions have the expected structure, or when interfacing with external numerical code.
