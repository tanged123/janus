# Error Handling Enhancement for Janus

**Goal**: Standardize and polish error handling across the Janus framework.
**Status**: ✅ Completed
**Created**: 2025-12-15

---

## Executive Summary

This plan addresses inconsistencies in error handling across Janus and proposes a unified approach using a custom exception hierarchy. The goal is to:

1. **Standardize exception types** with a `JanusError` base class
2. **Add domain-specific exceptions** for clearer debugging
3. **Improve error messages** with context and suggestions
4. **Fill validation gaps** where input validation is missing
5. **Ensure test coverage** for all error paths

---

## Current State Analysis

### Complete Audit of All 21 Header Files

**Files WITH throw statements (8 files, 19 throws):**

| File | throws | Exception Types |
|------|--------|-----------------|
| `Interpolate.hpp` | 5 | `std::invalid_argument`, `std::runtime_error` |
| `IntegrateDiscrete.hpp` | 3 | `std::invalid_argument` |
| `Calculus.hpp` | 3 | `std::invalid_argument` |
| `SurrogateModel.hpp` | 2 | `std::invalid_argument` |
| `FiniteDifference.hpp` | 2 | `std::invalid_argument` |
| `JanusIO.hpp` | 2 | `std::runtime_error` |
| `Logic.hpp` | 1 | `std::invalid_argument` |
| `Integrate.hpp` | 1 | `std::runtime_error` |

**Files WITHOUT throw statements (13 files):**

| File | Validation Gaps |
|------|-----------------|
| `Function.hpp` | None identified |
| `JanusConcepts.hpp` | None (concept definitions) |
| `JanusTypes.hpp` | None (type aliases) |
| `Arithmetic.hpp` | None (math functions) |
| `AutoDiff.hpp` | Uses `static_assert` correctly |
| `DiffOps.hpp` | Small wrapper file |
| `Linalg.hpp` | ⚠️ `cross()` assumes size 3 vectors |
| `Quaternion.hpp` | None (uses janus::where) |
| `Rotations.hpp` | ⚠️ Invalid axis falls through silently (L66-69) |
| `Spacing.hpp` | ⚠️ Silent handling of n < 2 |
| `Trig.hpp` | None (math dispatch) |
| `janus.hpp` | Master include only |
| `JanusMath.hpp` | Master include only |

### Identified Issues

1. **No custom exception hierarchy**: All errors use `std::{invalid_argument|runtime_error}`
2. **Inconsistent prefixes**: Some use `"JanusInterpolator: ..."`, others have no prefix
3. **Missing validation**: 
   - `Spacing.hpp` silently handles `n < 2` returning single-element vectors
   - `Rotations.hpp::rotation_matrix_3d` returns Identity for invalid axis (lines 66-69)
   - `Linalg.hpp::cross` assumes inputs are size 3 but doesn't validate
4. **No source context**: Errors don't indicate which file/function failed

---

## User Review Required

> [!IMPORTANT]
> **Design Decision: Exception Hierarchy vs. std Exceptions**
>
> Option A: Create `janus::Error` base class (recommended)
> - Consistent namespace, easier to catch Janus-specific errors
> - Can add context like function name, file location
>
> Option B: Continue using `std` exceptions
> - Simpler, no new types to learn
> - Less intrusive change

> [!WARNING]
> **Breaking Change Consideration**
>
> If we change exception types, code catching `std::invalid_argument` specifically will need updates. We can mitigate by deriving from `std::` base classes.

---

## Proposed Changes

### Component 1: Exception Hierarchy

#### [NEW] `include/janus/core/JanusError.hpp`

```cpp
#pragma once
#include <stdexcept>
#include <string>

namespace janus {

/**
 * @brief Base exception for all Janus errors
 * Derives from std::runtime_error for catch compatibility
 */
class JanusError : public std::runtime_error {
public:
    explicit JanusError(const std::string& what) 
        : std::runtime_error("[janus] " + what) {}
};

/**
 * @brief Input validation failed (e.g., mismatched sizes, invalid parameters)
 */
class InvalidArgument : public JanusError {
public:
    explicit InvalidArgument(const std::string& what)
        : JanusError(what) {}
};

/**
 * @brief Operation failed at runtime (e.g., CasADi eval with free variables)
 */
class RuntimeError : public JanusError {
public:
    explicit RuntimeError(const std::string& what)
        : JanusError(what) {}
};

/**
 * @brief Interpolation-specific errors
 */
class InterpolationError : public JanusError {
public:
    explicit InterpolationError(const std::string& what)
        : JanusError("Interpolation: " + what) {}
};

/**
 * @brief Integration/ODE solver errors
 */
class IntegrationError : public JanusError {
public:
    explicit IntegrationError(const std::string& what)
        : JanusError("Integration: " + what) {}
};

} // namespace janus
```

---

### Component 2: Refactor Existing Throws

#### [MODIFY] `include/janus/math/Interpolate.hpp`

| Line | Before | After |
|------|--------|-------|
| 40 | `throw std::invalid_argument("JanusInterpolator: x and y...")` | `throw janus::InterpolationError("x and y must have same size")` |
| 43 | `throw std::invalid_argument("JanusInterpolator: Need at least 2 points")` | `throw janus::InterpolationError("Need at least 2 grid points")` |
| 53 | `throw std::invalid_argument("JanusInterpolator: x grid must be sorted")` | `throw janus::InterpolationError("Grid points must be sorted")` |
| 71, 90 | `throw std::runtime_error("JanusInterpolator: Uninitialized")` | `throw janus::InterpolationError("Interpolator not initialized")` |

---

#### [MODIFY] `include/janus/math/IntegrateDiscrete.hpp`

| Line | Before | After |
|------|--------|-------|
| 236 | `throw std::invalid_argument("Invalid Simpson variant: " + method)` | `throw janus::IntegrationError("Invalid Simpson variant: " + method)` |
| 247 | `throw std::invalid_argument("Invalid integration method: " + method)` | `throw janus::IntegrationError("Unknown method: " + method + ". Use trapezoidal, simpson, or cubic.")` |
| 431 | `throw std::invalid_argument("Invalid squared curvature method...")` | `throw janus::IntegrationError("Unknown curvature method: " + method)` |

---

#### [MODIFY] `include/janus/math/Calculus.hpp`

| Line | Before | After |
|------|--------|-------|
| 124 | `throw std::invalid_argument("dx must be scalar, size N, or size N-1")` | `throw janus::InvalidArgument("gradient: dx must be scalar, size N, or size N-1")` |
| 177 | `throw std::invalid_argument("edge_order must be 1 or 2")` | `throw janus::InvalidArgument("gradient: edge_order must be 1 or 2")` |
| 190 | `throw std::invalid_argument("n must be 1 or 2")` | `throw janus::InvalidArgument("gradient: derivative order (n) must be 1 or 2")` |

---

#### [MODIFY] `include/janus/math/SurrogateModel.hpp`

| Line | Before | After |
|------|--------|-------|
| 32 | `throw std::invalid_argument("softmax requires at least one argument")` | `throw janus::InvalidArgument("softmax: requires at least one value")` |
| 35 | `throw std::invalid_argument("softmax softness must be positive")` | `throw janus::InvalidArgument("softmax: softness must be positive (got " + std::to_string(softness) + ")")` |

---

#### [MODIFY] `include/janus/core/JanusIO.hpp`

| Line | Before | After |
|------|--------|-------|
| 68 | `throw std::runtime_error("janus::eval failed (likely contains free variables): " + ...)` | `throw janus::RuntimeError("eval failed (expression contains free variables)")` |
| 84 | `throw std::runtime_error("janus::eval scalar failed: " + ...)` | `throw janus::RuntimeError("eval scalar failed: " + std::string(e.what()))` |

---

### Component 3: Add Missing Validation

#### [MODIFY] `include/janus/math/Spacing.hpp`

Add input validation to spacing functions:

```cpp
template <typename T> JanusVector<T> linspace(const T &start, const T &end, int n) {
    if (n < 1) {
        throw janus::InvalidArgument("linspace: n must be >= 1");
    }
    // ... existing code ...
}
```

Apply similar validation to `cosine_spacing`, `sinspace`, `logspace`, `geomspace`.

---

#### [MODIFY] `include/janus/math/Rotations.hpp`

Fix silent fallthrough on invalid axis (line 66-69):

```cpp
// Before (line 66-69):
default:
    // Simple fallback or error. For now, Identity.
    // In critical code, throw or assert.
    break;

// After:
default:
    throw janus::InvalidArgument("rotation_matrix_3d: axis must be 0 (X), 1 (Y), or 2 (Z)");
```

---

#### [MODIFY] `include/janus/math/Linalg.hpp`

Add validation to `cross()` function (line 115-125):

```cpp
template <typename DerivedA, typename DerivedB>
auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    if (a.size() != 3 || b.size() != 3) {
        throw janus::InvalidArgument("cross: both vectors must have exactly 3 elements");
    }
    // ... existing implementation ...
}
```

---

### Component 4: Update JanusMath.hpp Include

#### [MODIFY] `include/janus/math/JanusMath.hpp`

Add include for new error header:

```cpp
#include "janus/core/JanusError.hpp"
```

---

## Verification Plan

### Automated Tests

Existing error-path tests should continue to work since new exceptions derive from `std::runtime_error`:

```bash
# Run all tests
./scripts/test.sh

# Run specific error-handling tests
cd build && ctest --output-on-failure -R "Interpolate|Calculus|Surrogate|Logic"
```

### New Tests to Add

#### [NEW] `tests/core/test_janus_error.cpp`

Test the new exception hierarchy:

```cpp
TEST(JanusErrorTests, BaseErrorCatchable) {
    EXPECT_THROW(throw janus::JanusError("test"), std::runtime_error);
}

TEST(JanusErrorTests, InvalidArgumentCatchable) {
    EXPECT_THROW(throw janus::InvalidArgument("test"), janus::JanusError);
    EXPECT_THROW(throw janus::InvalidArgument("test"), std::runtime_error);
}

TEST(JanusErrorTests, InterpolationErrorCatchable) {
    EXPECT_THROW(throw janus::InterpolationError("test"), janus::JanusError);
}
```

#### Update Existing Tests

Existing `EXPECT_THROW` statements should still work (catching `std::invalid_argument` or `std::runtime_error`), but we should add explicit tests for new types:

```cpp
// In test_interpolate.cpp
TEST(InterpolateTests, ThrowsInterpolationError) {
    Eigen::VectorXd x(3), y(2);
    x << 0, 1, 2;
    y << 0, 1;
    EXPECT_THROW(janus::JanusInterpolator(x, y), janus::InterpolationError);
}
```

### Manual Verification

1. Build with `./scripts/build.sh` - verify no compilation errors
2. Run `./scripts/test.sh` - verify all tests pass
3. Intentionally trigger errors in a test file to verify message format

---

## Task Breakdown

### Phase 1: Core Infrastructure
- [ ] Create `JanusError.hpp` with exception hierarchy
- [ ] Add include to `JanusMath.hpp`
- [ ] Create basic tests for exception hierarchy

### Phase 2: Refactor Existing Throws
- [ ] Update `Interpolate.hpp` (5 throws)
- [ ] Update `IntegrateDiscrete.hpp` (3 throws)
- [ ] Update `Calculus.hpp` (3 throws)
- [ ] Update `SurrogateModel.hpp` (2 throws)
- [ ] Update `JanusIO.hpp` (2 throws)
- [ ] Update `FiniteDifference.hpp` (2 throws)
- [ ] Update `Logic.hpp` (1 throw)
- [ ] Update `Integrate.hpp` (1 throw)

### Phase 3: Add Missing Validation
- [ ] Add validation to `Spacing.hpp` functions (5 functions)
- [ ] Add validation to `Rotations.hpp::rotation_matrix_3d` (invalid axis)
- [ ] Add validation to `Linalg.hpp::cross` (vector size check)

### Phase 4: Test & Document
- [ ] Run full test suite
- [ ] Verify error messages are clear and actionable
- [ ] Update docs if needed

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing catch blocks | Medium | Derive from `std::runtime_error` for compatibility |
| Compile errors in user code | Low | New header is opt-in via `JanusMath.hpp` |
| Test failures | Low | All new exceptions are catchable as old types |

---

## Success Criteria

1. ✅ All exceptions use consistent `janus::` namespace types  
2. ✅ Error messages include context (function name, constraint violated)
3. ✅ All existing tests continue to pass
4. ✅ New exception hierarchy tests added and passing
5. ✅ Missing validation added to `Spacing.hpp`
