#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/Function.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/AutoDiff.hpp>
#include <janus/math/Linalg.hpp>
#include <janus/math/Trig.hpp>

// Test Jacobian of a simple quadratic function
TEST(AutoDiffTests, JacobianQuadratic) {
    // f(x) = x^2
    // df/dx = 2*x
    auto x = janus::sym("x");
    auto f = janus::pow(x, 2.0);

    auto J_sym = janus::jacobian(f, x);

    // Create function to evaluate Jacobian
    janus::Function J_fun({x}, {J_sym});

    // Evaluate at x = 3
    auto J = J_fun.eval(3.0);

    // Should be 2*3 = 6
    EXPECT_NEAR(J(0, 0), 6.0, 1e-10);
}

// Test Jacobian with multiple inputs
TEST(AutoDiffTests, JacobianMultipleInputs) {
    // f(x, y) = x^2 + 2*x*y + y^2
    // df/dx = 2*x + 2*y
    // df/dy = 2*x + 2*y
    auto x = janus::sym("x");
    auto y = janus::sym("y");

    auto f = janus::pow(x, 2.0) + 2.0 * x * y + janus::pow(y, 2.0);

    // Jacobian: [df/dx, df/dy]
    auto J_sym = janus::jacobian(f, x, y);

    janus::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=1, y=2
    auto J = J_fun.eval(1.0, 2.0);

    // df/dx = 2*1 + 2*2 = 6
    // df/dy = 2*1 + 2*2 = 6
    EXPECT_NEAR(J(0, 0), 6.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 6.0, 1e-10);
}

// Test Jacobian from drag coefficient example
TEST(AutoDiffTests, JacobianDragCoefficient) {
    // Drag model: Drag = 0.5 * rho * v^2 * S * (Cd0 + k * (Cl - Cl0)^2)
    auto v = janus::sym("v");
    auto Cl = janus::sym("Cl");

    // Constants
    double rho = 1.225;
    double S = 10.0;
    double Cd0 = 0.02;
    double k = 0.04;
    double Cl0 = 0.1;

    janus::SymbolicScalar rho_s = rho;
    janus::SymbolicScalar S_s = S;
    janus::SymbolicScalar Cd0_s = Cd0;
    janus::SymbolicScalar k_s = k;
    janus::SymbolicScalar Cl0_s = Cl0;

    auto q = 0.5 * rho_s * janus::pow(v, 2.0);
    auto Cd = Cd0_s + k_s * janus::pow(Cl - Cl0_s, 2.0);
    auto drag = q * S_s * Cd;

    // Compute Jacobian: [dDrag/dv, dDrag/dCl]
    auto J_sym = janus::jacobian(drag, v, Cl);

    janus::Function J_fun({v, Cl}, {J_sym});

    // Test at v=50, Cl=0.5
    double v_val = 50.0;
    double Cl_val = 0.5;
    auto J = J_fun.eval(v_val, Cl_val);

    // Analytic derivatives:
    // Cd = 0.02 + 0.04 * (0.5 - 0.1)^2 = 0.02 + 0.04 * 0.16 = 0.0264
    // q = 0.5 * 1.225 * 50^2 = 1531.25
    // dDrag/dv = rho * v * S * Cd = 1.225 * 50 * 10 * 0.0264 = 16.17
    // dDrag/dCl = q * S * 2 * k * (Cl - Cl0) = 1531.25 * 10 * 2 * 0.04 * 0.4 = 490

    EXPECT_NEAR(J(0, 0), 16.17, 0.01);
    EXPECT_NEAR(J(0, 1), 490.0, 1.0);
}

// Test Jacobian with vector of expressions and variables
TEST(AutoDiffTests, JacobianVectorInputs) {
    // f1(x, y) = x + y
    // f2(x, y) = x * y
    // J = [[df1/dx, df1/dy],
    //      [df2/dx, df2/dy]]
    //   = [[1, 1],
    //      [y, x]]
    auto x = janus::sym("x");
    auto y = janus::sym("y");

    auto f1 = x + y;
    auto f2 = x * y;

    auto J_sym = janus::jacobian({f1, f2}, {x, y});

    janus::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=3, y=4
    auto J = J_fun.eval(3.0, 4.0);

    // Expected: [[1, 1], [4, 3]]
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 1.0, 1e-10);
    EXPECT_NEAR(J(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(J(1, 1), 3.0, 1e-10);
}

// Test Jacobian of trigonometric functions
TEST(AutoDiffTests, JacobianTrigonometric) {
    // f(x) = sin(x)
    // df/dx = cos(x)
    auto x = janus::sym("x");
    auto f = janus::sin(x);

    auto J = janus::jacobian(f, x);

    janus::Function J_fun({x}, {J});

    // Evaluate at x=0
    auto J0 = J_fun.eval(0.0);

    // cos(0) = 1
    EXPECT_NEAR(J0(0, 0), 1.0, 1e-10);

    // Evaluate at x=pi/2
    auto J_pi2 = J_fun.eval(M_PI / 2.0);

    // cos(pi/2) ≈ 0
    EXPECT_NEAR(J_pi2(0, 0), 0.0, 1e-10);
}

// Test Jacobian with matrix output
TEST(AutoDiffTests, JacobianMatrixSymbolicArg) {
    // Test using SymbolicArg for matrix inputs
    using SymbolicMatrix = janus::SymbolicMatrix;

    auto x = janus::sym("x");
    auto y = janus::sym("y");

    // Create a 2x1 symbolic matrix
    SymbolicMatrix vec(2, 1);
    vec(0, 0) = x;
    vec(1, 0) = y;

    // f(vec) = sum of squares = x^2 + y^2
    auto f = janus::pow(x, 2.0) + janus::pow(y, 2.0);

    // Jacobian using SymbolicArg wrapper
    auto J_sym = janus::jacobian({f}, {x, y});

    janus::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=1, y=2
    auto J = J_fun.eval(1.0, 2.0);

    // Expected: [2*1, 2*2] = [2, 4]
    EXPECT_NEAR(J(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 4.0, 1e-10);
}

// ============================================================================
// Higher-Order Derivatives Tests (Phase 7)
// ============================================================================

TEST(AutoDiffTests, SymbolicGradient) {
    // f(x, y) = x^2 * y
    // grad(f) = [2xy, x^2] (column vector)
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto f = janus::pow(x, 2.0) * y;

    // Compute symbolic gradient
    auto g_sym = janus::sym_gradient(f, {x, y});

    // Verify it's a column vector
    EXPECT_EQ(g_sym.cols(), 1);
    EXPECT_EQ(g_sym.rows(), 2);

    // Evaluate
    janus::Function g_fun({x, y}, {g_sym});
    auto g_val = g_fun.eval(3.0, 2.0); // x=3, y=2

    // grad = [2*3*2, 3^2] = [12, 9]
    EXPECT_NEAR(g_val(0), 12.0, 1e-10);
    EXPECT_NEAR(g_val(1), 9.0, 1e-10);
}

TEST(AutoDiffTests, HessianMatrix) {
    // f(x, y) = x^2 + 3xy + y^3
    // grad(f) = [2x + 3y, 3x + 3y^2]
    // H(f) = [[2, 3],
    //         [3, 6y]]
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto f = janus::pow(x, 2.0) + 3.0 * x * y + janus::pow(y, 3.0);

    auto H_sym = janus::hessian(f, {x, y});

    // Verify dimensions
    EXPECT_EQ(H_sym.rows(), 2);
    EXPECT_EQ(H_sym.cols(), 2);

    // Evaluate
    janus::Function H_fun({x, y}, {H_sym});
    auto H_val = H_fun.eval(1.0, 2.0); // y=2

    // H = [[2, 3], [3, 6*2]] = [[2, 3], [3, 12]]
    EXPECT_NEAR(H_val(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(H_val(0, 1), 3.0, 1e-10);
    EXPECT_NEAR(H_val(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(H_val(1, 1), 12.0, 1e-10);
}

TEST(AutoDiffTests, HessianLagrangian) {
    // Minimize f(x) = x^2 subject to g(x) = x - 1 = 0
    // L = x^2 + lambda * (x - 1)
    // H_L = d^2L/dx^2 = 2

    auto x = janus::sym("x");
    auto lam = janus::sym("lam");

    auto obj = janus::pow(x, 2.0);
    auto constr = x - 1.0;

    auto HL_sym = janus::hessian_lagrangian(obj, constr, x, lam);

    janus::Function HL_fun({x, lam}, {HL_sym});
    auto HL_val = HL_fun.eval(0.0, 0.0);

    EXPECT_NEAR(HL_val(0, 0), 2.0, 1e-10);

    // Case 2: Multi-dimensional
    // f = x^2 + y^2, g = x+y
    // L = x^2 + y^2 + lam*(x+y)
    // H_L = diag(2, 2)
    auto y = janus::sym("y");
    auto obj2 = janus::pow(x, 2.0) + janus::pow(y, 2.0);
    auto constr2 = x + y;

    auto HL2_sym = janus::hessian_lagrangian(obj2, constr2, {x, y}, lam);
    janus::Function HL2_fun({x, y, lam}, {HL2_sym});
    auto HL2_val = HL2_fun.eval(0.0, 0.0, 0.0);

    EXPECT_NEAR(HL2_val(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(HL2_val(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(HL2_val(0, 1), 0.0, 1e-10);
}
