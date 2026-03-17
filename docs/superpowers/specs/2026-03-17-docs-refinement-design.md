# Docs Refinement Design -- Janus v2.0.0

## Context

Janus v2.0.0 was just released with a massive feature PR adding polynomial chaos, quadrature, sparse derivative pipelines, sensitivity regime selection, matrix-free HVPs, multi-method root finding, implicit function builders, structural diagnostics/transforms, second-order and mass-matrix integrators, policy-driven linear solves, scaling diagnostics, and many examples/tests. Documentation needs a consistency and usability pass.

**Audience priority:** Contributors (C) > New users (A) > Upgraders (B)

## 1. Doxygen Comment Standard

Standardize comment style across all 44 headers in `include/janus/`.

### Format

```cpp
/// @file Interpolate.hpp
/// @brief N-dimensional interpolation (Linear, Hermite, BSpline, Nearest)

namespace janus {

/// @brief One-line description of the class/struct
///
/// Optional 1-2 sentence elaboration when the brief isn't self-explanatory.
/// @see related_function, OtherClass
struct InterpolationResult { ... };

/// @brief One-line description of what the function does
/// @tparam T Scalar type (NumericScalar or SymbolicScalar)
/// @param grid Evenly-spaced grid points
/// @param values Function values at grid points
/// @return Interpolated value at the query point
/// @see interp1d, interpnd
template <typename T>
T interp2d(const std::vector<T>& grid, const std::vector<T>& values);

/// @brief Short enum description
enum class InterpolationMethod {
    Linear,   ///< Piecewise linear
    Hermite,  ///< Cubic Hermite with derivative matching
    BSpline,  ///< B-spline (order configurable)
    Nearest   ///< Nearest-neighbor lookup
};

} // namespace janus
```

### Rules

- Every file gets `@file` + `@brief` at the top
- Every public class/struct/function gets `@brief`
- `@param`/`@return`/`@tparam` only where non-obvious (skip for self-documenting names)
- `@see` for cross-references between related APIs
- Enum values get `///<` inline comments
- No `@details` blocks -- keep it concise
- Existing doxygen comments that already match this style are left alone

## 2. User Guide Standard Template

Standardize structure across all 18 guides in `docs/user_guides/`.

### Template

```markdown
# [Feature Name]

One paragraph: what this feature does, when you'd use it, and whether it works
in numeric mode, symbolic mode, or both.

## Quick Start

Minimal working example (10-20 lines) that a user can copy-paste and run.

## Core API

Main functions/classes, organized by usage. Each entry:
- Signature or usage pattern
- Brief description
- Short code snippet if the signature alone isn't clear

## Usage Patterns

Common workflows and recipes. Real-world-ish examples with explanation.

## Advanced Usage

(Optional) Power-user features, performance tuning, non-obvious capabilities.

## Diagnostics & Troubleshooting

(Optional) Common errors, diagnostic output interpretation, edge cases.

## See Also

- Links to related user guides
- Links to relevant example files (examples/...)
- Links to relevant header files for API details
```

### Rules

- Every guide gets H1 -> Quick Start -> Core API -> Usage Patterns flow
- "See Also" section is mandatory -- minimum one cross-reference
- Code examples use `janus::` namespace (not `using namespace janus`)
- Guides that don't need Advanced/Diagnostics sections omit them
- No theory/math preamble before Quick Start

## 3. README.md Rewrite

Full rewrite as contributor-first landing page.

### Structure

```
# Janus

Elevator pitch (one paragraph).

## Building & Development

Prerequisites, nix develop, build/test/coverage scripts.

## Project Structure

Annotated directory tree (include/, examples/, tests/, docs/).

## Architecture Overview

5-10 sentences: template-first, dual backend, dispatch layer,
structural vs value logic. Link to docs/design_overview.md.

## Documentation

How docs are organized, how to generate (doxygen), link to GitHub Pages.
Table listing all 18 user guides with one-line descriptions.

## Key Features

Concise grouped list: Core, Math, Optimization, Interpolation,
Integration, Stochastic, Structural Analysis.

## Examples

Category breakdown of examples/ directory, how to build/run.

## Contributing

Code style, how to add tests/examples, PR workflow.
```

### Key changes from current

- Building & Development moves to top (contributor-first)
- Project structure expanded and prominent
- Architecture section added
- Feature list condensed and grouped
- Contributing section added
- Single usage example removed -- pointed to examples/ and guides

## 4. Design Overview & Usage Guide Refresh

### design_overview.md

- Update to reflect v2.0.0 scope (PCE, quadrature, structural diagnostics, sparsity, root finding, linear solve policies, second-order integrators, scaling)
- Keep philosophical structure (Code Transformations, Template-First, Dual Backend, Dispatch, Structural vs Value, Branching, Loops)
- Remove or update completed roadmap phases
- Add cross-references to new user guides
- Target ~150 lines max

### janus_usage_guide.md

- Audit against v2.0.0 API -- fix stale signatures, removed types, renamed enums
- Add sections for new features not currently covered
- Cross-reference user guides rather than duplicating content
- Standardize code examples to `janus::` namespace
- Serve as a map, not a manual

## 5. Doxyfile Configuration

- Add `PROJECT_NUMBER = 2.0.0`
- Verify `PROJECT_BRIEF` is set
- Ensure `USE_MDFILE_AS_MAINPAGE = README.md`
- Add `docs/janus_usage_guide.md` to INPUT
- Add `EXCLUDE_PATTERNS` for `docs/implementation_plans/` and `docs/saved_work/`
- Keep `EXTRACT_ALL=YES`
- Verify `GENERATE_TREEVIEW=YES`

## Scope Exclusions

- `docs/implementation_plans/` -- left as-is
- No CHANGELOG.md
- No new user guides -- only standardize existing ones
- Header comment content preserved -- only style standardized
