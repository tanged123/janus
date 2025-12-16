# Native Type Migration Plan

## Goal
Scrub the `include/janus` headers to replace direct usages of `Eigen::` and `casadi::` types with their Janus native aliases (defined in `JanusTypes.hpp`). This improves encapsulation and consistency.

## User Review Required
> [!NOTE]
> `Eigen::MatrixBase<Derived>` and `Eigen::Index` will generally be preserved as they are fundamental to the template mechanics that Janus relies on.
> `casadi::Opti` and `casadi::OptiSol` members in optimization classes will remain as they are wrappers, but their public interfaces will be updated to return/accept Janus types where possible.

## Proposed Changes

### Core Headers (`include/janus/core/`)

#### [MODIFY] [Function.hpp](file:///home/tanged/sources/janus/include/janus/core/Function.hpp)
- Replace `Eigen::MatrixXd` with `NumericMatrix` in return types and arguments.
- Replace `Eigen::Matrix<Scalar, ...>` with `JanusMatrix<Scalar>`.
- Replace `std::vector<Eigen::MatrixXd>` with `std::vector<NumericMatrix>`.

#### [MODIFY] [JanusIO.hpp](file:///home/tanged/sources/janus/include/janus/core/JanusIO.hpp)
- Replace `Eigen::MatrixXd` in internal evaluation logic with `NumericMatrix`.
- Ensure `JanusTypes.hpp` is included.

### Math Headers (`include/janus/math/`)
Apply changes to all math headers, including `Trig.hpp`, `Rotations.hpp`, `FiniteDifference.hpp`, `DiffOps.hpp`, etc.

#### [MODIFY] *All Math Files*
- Replace `casadi::MX` with `SymbolicScalar` where it refers to the type.
- Replace `Eigen::Matrix<Scalar, ...>` with `JanusMatrix<Scalar>` or `JanusVector<Scalar>`.
- Ensure `JanusTypes.hpp` is included.

### Optimization Headers (`include/janus/optimization/`)

#### [MODIFY] [Opti.hpp](file:///home/tanged/sources/janus/include/janus/optimization/Opti.hpp)
- Replace `casadi::MX` in public API with `SymbolicScalar`.
- Replace `Eigen::Matrix` return types with `JanusMatrix` / `JanusVector`.

#### [MODIFY] [OptiSol.hpp](file:///home/tanged/sources/janus/include/janus/optimization/OptiSol.hpp)
- Replace `Eigen::MatrixXd` in `value()` return type with `NumericMatrix`.
- Replace `casadi::MX` casts with `SymbolicScalar` casts where appropriate.

## Verification Plan

### Automated Tests
- Run full CI suite: `./scripts/ci.sh`
- Run validation suite (tests + examples): `./scripts/verify.sh`
- Ensure no compilation errors.
- Verify that aliases resolve correctly and binary compatibility is maintained.

### Manual Verification
- Spot check a few headers to ensure no raw `Eigen::` or `casadi::` types leaked into the update (except where intended).
