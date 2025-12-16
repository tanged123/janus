/**
 * @file test_opti.cpp
 * @brief Tests for janus::Opti optimization interface
 *
 * Tests cover:
 * - Rosenbrock 2D unconstrained
 * - Rosenbrock 2D constrained (unit circle)
 * - N-dimensional Rosenbrock
 * - Derivative constraints for trajectory optimization
 */

#include <gtest/gtest.h>
#include <janus/janus.hpp>

// =============================================================================
// Rosenbrock Tests (based on AeroSandbox benchmarks)
// =============================================================================

TEST(OptiTest, Rosenbrock2D_Unconstrained) {
    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    // Rosenbrock: (1-x)^2 + 100*(y - x^2)^2
    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-4);
}

TEST(OptiTest, Rosenbrock2D_Constrained) {
    // Constrained to unit circle: x^2 + y^2 <= 1
    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    opti.subject_to(x * x + y * y <= 1);

    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.verbose = false});

    // Known solution from MATLAB/Julia
    EXPECT_NEAR(sol.value(x), 0.7864, 1e-3);
    EXPECT_NEAR(sol.value(y), 0.6177, 1e-3);
}

TEST(OptiTest, RosenbrockND) {
    // N-dimensional Rosenbrock with non-negativity constraint
    constexpr int N = 10;
    janus::Opti opti;

    auto x = opti.variable(N, 1.0); // init_guess = 1 (near solution)

    // Clean API: apply lower bound to all elements at once
    opti.subject_to_lower(x, 0.0);

    // Objective: sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    janus::SymbolicScalar obj = 0;
    for (int i = 0; i < N - 1; ++i) {
        obj = obj + 100 * janus::pow(x(i + 1) - x(i) * x(i), 2) + janus::pow(1 - x(i), 2);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // All elements should be ~1.0 at optimum
    janus::NumericVector x_opt = sol.value(x);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(x_opt(i), 1.0, 1e-4);
    }
}

// =============================================================================
// Constraint Tests
// =============================================================================

TEST(OptiTest, EqualityConstraint) {
    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    opti.subject_to(x + y == 2); // Equality constraint
    opti.minimize(x * x + y * y);

    auto sol = opti.solve({.verbose = false});

    // Minimum of x^2 + y^2 subject to x + y = 2 is at x = y = 1
    EXPECT_NEAR(sol.value(x), 1.0, 1e-6);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-6);
}

TEST(OptiTest, MultipleConstraints) {
    janus::Opti opti;

    auto x = opti.variable(0.5);
    auto y = opti.variable(0.5);

    opti.subject_to({x >= 0, y >= 0, x + y <= 1});

    opti.minimize(-x - y); // Maximize x + y

    auto sol = opti.solve({.verbose = false});

    // Maximum is at x + y = 1 boundary
    EXPECT_NEAR(sol.value(x) + sol.value(y), 1.0, 1e-6);
}

TEST(OptiTest, VariableBounds) {
    janus::Opti opti;

    auto x = opti.variable(0.0, std::nullopt, -1.0, 1.0); // -1 <= x <= 1

    opti.minimize(-x); // Maximize x

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-6);
}

// =============================================================================
// Parameter Tests
// =============================================================================

TEST(OptiTest, ParameterUsage) {
    janus::Opti opti;

    auto x = opti.variable(0.0);
    auto p = opti.parameter(5.0); // Fixed parameter

    opti.subject_to(x >= p); // x >= 5
    opti.minimize(x * x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 5.0, 1e-6);
}

// =============================================================================
// Derivative Helpers Tests (Trajectory Optimization)
// =============================================================================

TEST(OptiTest, DerivativeOf_Trapezoidal) {
    // Simple test: x(t) = t, so dx/dt = 1
    constexpr int N = 10;
    janus::Opti opti;

    janus::NumericVector t = janus::linspace(0.0, 1.0, N);
    auto x = opti.variable(t); // Init guess is t itself

    auto xdot = opti.derivative_of(x, t, 1.0); // Expect derivative ~= 1

    // Boundary conditions
    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);

    // Minimize deviation from constant derivative
    janus::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + (xdot(i) - 1) * (xdot(i) - 1);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    janus::NumericVector xdot_opt = sol.value(xdot);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(xdot_opt(i), 1.0, 0.1);
    }
}

TEST(OptiTest, ConstrainDerivative_DoubleIntegrator) {
    // Double integrator: position -> velocity -> acceleration(=0)
    // With a(t) = 0, v = constant, x = linear
    constexpr int N = 20;
    janus::Opti opti;

    janus::NumericVector t = janus::linspace(0.0, 1.0, N);
    auto x = opti.variable(N, 0.0); // Position
    auto v = opti.variable(N, 1.0); // Velocity
    auto a = opti.variable(N, 0.0); // Acceleration

    // Derivative constraints
    opti.constrain_derivative(v, x, t); // dx/dt = v
    opti.constrain_derivative(a, v, t); // dv/dt = a

    // Boundary conditions
    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);
    opti.subject_to(v(0) == 1);
    opti.subject_to(v(N - 1) == 1);

    // Minimize acceleration squared
    janus::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + a(i) * a(i);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // With constant velocity, acceleration should be ~0
    janus::NumericVector a_opt = sol.value(a);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(a_opt(i), 0.0, 0.1);
    }
}

// =============================================================================
// Maximize Test
// =============================================================================

TEST(OptiTest, Maximize) {
    janus::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to(x <= 5);
    opti.maximize(x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 5.0, 1e-6);
}

// =============================================================================
// Stats Test
// =============================================================================

TEST(OptiTest, SolverStats) {
    janus::Opti opti;

    auto x = opti.variable(0.0);
    opti.minimize(x * x);

    auto sol = opti.solve({.verbose = false});

    // Should have some iterations recorded
    EXPECT_GE(sol.num_iterations(), 0);
}

// =============================================================================
// Integration Tests: Function + Opti + Jacobian Synergies
// =============================================================================

/**
 * Test 1: Using janus::Function output as constraint in Opti
 *
 * Demonstrates: Pre-compiled symbolic functions can be reused in optimization
 */
TEST(OptiIntegration, FunctionAsConstraint) {
    // Define a reusable constraint function: circle constraint
    auto x_sym = janus::sym("x");
    auto y_sym = janus::sym("y");
    auto circle_expr = x_sym * x_sym + y_sym * y_sym;

    // Create compiled function
    janus::Function circle_fn("circle", {x_sym, y_sym}, {circle_expr});

    // Verify function works numerically
    auto result = circle_fn.eval(3.0, 4.0); // 3^2 + 4^2 = 25
    EXPECT_NEAR(result(0, 0), 25.0, 1e-10);

    // Now use in optimization
    janus::Opti opti;
    auto x = opti.variable(0.5);
    auto y = opti.variable(0.5);

    // Use symbolic evaluation to get constraint expression
    auto symbolic_result = circle_fn(x, y);
    opti.subject_to(symbolic_result[0](0, 0) <= 1); // x^2 + y^2 <= 1

    opti.minimize(-x - y); // Maximize x + y on unit circle

    auto sol = opti.solve({.verbose = false});

    // Optimal at x = y = 1/sqrt(2) ≈ 0.707
    EXPECT_NEAR(sol.value(x), 1.0 / std::sqrt(2.0), 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0 / std::sqrt(2.0), 1e-4);
}

/**
 * Test 2: Using janus::jacobian for gradient analysis
 *
 * Demonstrates: Computing explicit gradients of objective for analysis
 */
TEST(OptiIntegration, JacobianForGradientAnalysis) {
    // Define Rosenbrock function symbolically
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto rosenbrock = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);

    // Compute gradient symbolically using janus::jacobian
    auto gradient = janus::jacobian({rosenbrock}, {x, y});

    // Compile gradient into a function for efficient evaluation
    janus::Function grad_fn("rosenbrock_grad", {x, y}, {gradient});

    // Evaluate gradient at optimum (x=1, y=1)
    auto grad_at_opt = grad_fn.eval(1.0, 1.0);

    // At minimum, gradient should be [0, 0]
    EXPECT_NEAR(grad_at_opt(0, 0), 0.0, 1e-10); // df/dx at (1,1)
    EXPECT_NEAR(grad_at_opt(0, 1), 0.0, 1e-10); // df/dy at (1,1)

    // Evaluate gradient away from optimum (x=0, y=0)
    auto grad_away = grad_fn.eval(0.0, 0.0);

    // df/dx at (0,0) = -2(1-x) - 400x(y-x^2) = -2
    // df/dy at (0,0) = 200(y-x^2) = 0
    EXPECT_NEAR(grad_away(0, 0), -2.0, 1e-10);
    EXPECT_NEAR(grad_away(0, 1), 0.0, 1e-10);
}

/**
 * Test 3: Constraint Jacobian for sensitivity analysis
 *
 * Demonstrates: Computing constraint Jacobians explicitly for debugging
 */
TEST(OptiIntegration, ConstraintJacobianAnalysis) {
    // Variables
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto z = janus::sym("z");

    // Constraints:
    //   g1(x,y,z) = x + 2y + 3z - 6  (plane)
    //   g2(x,y,z) = x^2 + y^2 - z    (paraboloid)
    auto g1 = x + 2 * y + 3 * z - 6;
    auto g2 = x * x + y * y - z;

    // Compute constraint Jacobian
    auto constraint_jac = janus::jacobian({g1, g2}, {x, y, z});

    // Compile to function
    janus::Function jac_fn("constraint_jac", {x, y, z}, {constraint_jac});

    // Evaluate at point (1, 1, 1)
    auto J = jac_fn.eval(1.0, 1.0, 1.0);

    // Expected Jacobian:
    // | dg1/dx dg1/dy dg1/dz |   | 1   2   3 |
    // | dg2/dx dg2/dy dg2/dz | = | 2x 2y  -1 | at (1,1,1) = | 2 2 -1 |
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(J(0, 2), 3.0, 1e-10);
    EXPECT_NEAR(J(1, 0), 2.0, 1e-10); // 2*x at x=1
    EXPECT_NEAR(J(1, 1), 2.0, 1e-10); // 2*y at y=1
    EXPECT_NEAR(J(1, 2), -1.0, 1e-10);
}

/**
 * Test 4: Building physics model with Function, optimizing with Opti
 *
 * Demonstrates: Full workflow - define physics as Function, embed in Opti
 */
TEST(OptiIntegration, PhysicsModelInOptimization) {
    // Physics model: projectile motion
    // y(t) = v0*sin(theta)*t - 0.5*g*t^2
    // x(t) = v0*cos(theta)*t
    // Find angle theta to maximize range (x when y=0)

    constexpr double v0 = 10.0; // Initial velocity [m/s]
    constexpr double g = 9.81;  // Gravity [m/s^2]

    janus::Opti opti;

    auto theta = opti.variable(M_PI / 4); // Initial guess: 45 degrees
    opti.subject_to_bounds(janus::SymbolicVector::Constant(1, theta), 0.01, M_PI / 2 - 0.01);

    // Time of flight: t_f = 2*v0*sin(theta)/g
    auto t_flight = 2 * v0 * janus::sin(theta) / g;

    // Range: x(t_f) = v0*cos(theta)*t_f
    auto range = v0 * janus::cos(theta) * t_flight;

    opti.maximize(range);

    auto sol = opti.solve({.verbose = false});

    // Optimal angle for maximum range is 45 degrees = pi/4
    EXPECT_NEAR(sol.value(theta), M_PI / 4, 1e-3);
}

/**
 * Test 5: Hessian computation for second-order optimality
 *
 * Demonstrates: Computing Hessian for quadratic approximation analysis
 */
TEST(OptiIntegration, HessianComputation) {
    // Quadratic function: f(x,y) = x^2 + 2*y^2 + x*y
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto f = x * x + 2 * y * y + x * y;

    // First compute gradient
    auto grad = janus::jacobian({f}, {x, y}); // 1x2 row vector

    // Compute Hessian by differentiating gradient
    // grad = [df/dx, df/dy] as MX
    // We need to reshape or extract elements
    auto df_dx = grad(0, 0);
    auto df_dy = grad(0, 1);

    // Compute second derivatives
    auto d2f_dxx = janus::jacobian({df_dx}, {x});
    auto d2f_dxy = janus::jacobian({df_dx}, {y});
    auto d2f_dyx = janus::jacobian({df_dy}, {x});
    auto d2f_dyy = janus::jacobian({df_dy}, {y});

    // Compile to functions
    janus::Function hxx_fn({x, y}, {d2f_dxx});
    janus::Function hxy_fn({x, y}, {d2f_dxy});
    janus::Function hyy_fn({x, y}, {d2f_dyy});

    // Evaluate (should be constant for quadratic)
    auto hxx = hxx_fn.eval(0.0, 0.0);
    auto hxy = hxy_fn.eval(0.0, 0.0);
    auto hyy = hyy_fn.eval(0.0, 0.0);

    // Expected Hessian:
    // | d²f/dx² d²f/dxdy |   | 2 1 |
    // | d²f/dydx d²f/dy² | = | 1 4 |
    EXPECT_NEAR(hxx(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(hxy(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(hyy(0, 0), 4.0, 1e-10);
}

/**
 * Test 6: Reusing symbolic expressions between Opti and Function
 *
 * Demonstrates: Same symbolic expression used for optimization and evaluation
 */
TEST(OptiIntegration, SharedSymbolicExpressions) {
    // Create shared symbolic variables
    auto x = janus::sym("x");
    auto y = janus::sym("y");

    // Define shared objective expression
    auto objective_expr = x * x + y * y;

    // Create callable function from expression
    janus::Function obj_fn("objective", {x, y}, {objective_expr});

    // Verify numeric evaluation
    EXPECT_NEAR(obj_fn.eval(3.0, 4.0)(0, 0), 25.0, 1e-10);

    // Use SAME expression structure in optimization
    janus::Opti opti;
    auto opt_x = opti.variable(1.0);
    auto opt_y = opti.variable(1.0);

    // Substitute optimization variables into expression pattern
    auto opt_objective = opt_x * opt_x + opt_y * opt_y; // Same structure

    opti.subject_to(opt_x + opt_y == 1);
    opti.minimize(opt_objective);

    auto sol = opti.solve({.verbose = false});

    // Verify solution using the compiled function
    double x_opt = sol.value(opt_x);
    double y_opt = sol.value(opt_y);
    auto obj_at_opt = obj_fn.eval(x_opt, y_opt);

    EXPECT_NEAR(obj_at_opt(0, 0), 0.5, 1e-6); // x=y=0.5, obj = 0.25 + 0.25 = 0.5
}
