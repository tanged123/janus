/**
 * @file test_opti_solvers.cpp
 * @brief Tests for alternative solver backends (SNOPT, QPOASES)
 *
 * Tests cover:
 * - Solver availability detection
 * - SNOPT solver with Rosenbrock (if available)
 * - Graceful error when solver unavailable
 */

#include <gtest/gtest.h>
#include <janus/janus.hpp>

// =============================================================================
// Solver Availability Tests
// =============================================================================

TEST(OptiSolvers, SolverAvailable_IPOPT) {
    // IPOPT should always be available in a standard CasADi build
    EXPECT_TRUE(janus::solver_available(janus::Solver::IPOPT));
}

TEST(OptiSolvers, SolverName_ReturnsCorrectStrings) {
    EXPECT_STREQ(janus::solver_name(janus::Solver::IPOPT), "ipopt");
    EXPECT_STREQ(janus::solver_name(janus::Solver::SNOPT), "snopt");
    EXPECT_STREQ(janus::solver_name(janus::Solver::QPOASES), "qpoases");
}

// =============================================================================
// SNOPT Tests (skipped if unavailable)
// =============================================================================

TEST(OptiSolvers, SNOPT_Rosenbrock2D) {
    if (!janus::solver_available(janus::Solver::SNOPT)) {
        GTEST_SKIP() << "SNOPT not available in this CasADi build";
    }

    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    // Rosenbrock: (1-x)^2 + 100*(y - x^2)^2
    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.solver = janus::Solver::SNOPT, .verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-4);
}

TEST(OptiSolvers, SNOPT_Constrained) {
    if (!janus::solver_available(janus::Solver::SNOPT)) {
        GTEST_SKIP() << "SNOPT not available in this CasADi build";
    }

    janus::Opti opti;

    auto x = opti.variable(0.5);
    auto y = opti.variable(0.5);

    // Unit circle constraint
    opti.subject_to(x * x + y * y <= 1);

    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.solver = janus::Solver::SNOPT, .verbose = false});

    // Verify constraint is satisfied
    double x_opt = sol.value(x);
    double y_opt = sol.value(y);
    EXPECT_LE(x_opt * x_opt + y_opt * y_opt, 1.0 + 1e-6);
}

TEST(OptiSolvers, SNOPT_WithCustomOptions) {
    if (!janus::solver_available(janus::Solver::SNOPT)) {
        GTEST_SKIP() << "SNOPT not available in this CasADi build";
    }

    janus::Opti opti;

    auto x = opti.variable(0.0);
    opti.minimize(x * x);
    opti.subject_to(x >= 1);

    janus::OptiOptions opts;
    opts.solver = janus::Solver::SNOPT;
    opts.verbose = false;
    opts.snopt_opts.major_iterations_limit = 500;
    opts.snopt_opts.major_optimality_tolerance = 1e-8;

    auto sol = opti.solve(opts);

    EXPECT_NEAR(sol.value(x), 1.0, 1e-6);
}

// =============================================================================
// Solver Unavailable Error Tests
// =============================================================================

TEST(OptiSolvers, UnavailableSolverThrows) {
    // Create a synthetic test - if SNOPT is not available, verify error
    if (janus::solver_available(janus::Solver::SNOPT)) {
        GTEST_SKIP() << "SNOPT is available, cannot test unavailable case";
    }

    janus::Opti opti;
    auto x = opti.variable(0.0);
    opti.minimize(x * x);

    EXPECT_THROW(opti.solve({.solver = janus::Solver::SNOPT}), std::runtime_error);
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

TEST(OptiSolvers, OptiOptionsBuilder) {
    auto opts = janus::OptiOptions{}
                    .set_solver(janus::Solver::IPOPT)
                    .set_max_iter(500)
                    .set_tol(1e-10)
                    .set_verbose(false);

    EXPECT_EQ(opts.solver, janus::Solver::IPOPT);
    EXPECT_EQ(opts.max_iter, 500);
    EXPECT_DOUBLE_EQ(opts.tol, 1e-10);
    EXPECT_FALSE(opts.verbose);
}

TEST(OptiSolvers, SNOPTOptionsBuilder) {
    auto snopt_opts =
        janus::SNOPTOptions{}.set_major_iterations_limit(2000).set_major_optimality_tolerance(1e-9);

    EXPECT_EQ(snopt_opts.major_iterations_limit, 2000);
    EXPECT_DOUBLE_EQ(snopt_opts.major_optimality_tolerance, 1e-9);
}

// =============================================================================
// IPOPT Solver still works (regression test)
// =============================================================================

TEST(OptiSolvers, IPOPT_ExplicitSolver) {
    // Verify explicitly selecting IPOPT works
    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.solver = janus::Solver::IPOPT, .verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-4);
}
