#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/core/Function.hpp> // Function is in core/Function.hpp not JanusFunction.hpp
#include <janus/math/RootFinding.hpp>

using namespace janus;

TEST(RootFindingTest, NumericSimpleQuadratic) {
    // F(x) = x^2 - 4 = 0
    auto x = sym("x");
    auto f_expr = x * x - 4.0;
    Function f("f", {x}, {f_expr});

    Eigen::VectorXd x0(1);
    x0 << 1.0; // Guess closer to 2

    auto res = rootfinder<double>(f, x0);

    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x(0), 2.0, 1e-6);

    // Guess closer to -2
    x0 << -1.0;
    res = rootfinder<double>(f, x0);
    EXPECT_NEAR(res.x(0), -2.0, 1e-6);
}

TEST(RootFindingTest, NumericSystem2D) {
    // F(x, y) = [x^2 + y^2 - 1, x - y] = 0
    // Intersection of unit circle and line y=x.
    // Solutions: (sqrt(0.5), sqrt(0.5)) and (-sqrt(0.5), -sqrt(0.5))

    auto x = sym("x");
    auto y = sym("y");
    auto r1 = x * x + y * y - 1.0;
    auto r2 = x - y;

    // We need input as a single vector or multiple args?
    // rootfinder expects F taking 1 input (vector x).
    // So we need to vertically concatenate x and y?
    // Or define F taking vector.

    auto xy = sym("xy", 2);
    auto r1_vec = xy(0) * xy(0) + xy(1) * xy(1) - 1.0;
    auto r2_vec = xy(0) - xy(1);

    Function f("f", {xy}, {casadi::MX::vertcat({r1_vec, r2_vec})});

    Eigen::VectorXd guess(2);
    guess << 0.5, 0.5;

    auto res = rootfinder<double>(f, guess);

    EXPECT_TRUE(res.converged);
    double expected = std::sqrt(0.5);
    EXPECT_NEAR(res.x(0), expected, 1e-6);
    EXPECT_NEAR(res.x(1), expected, 1e-6);
}

TEST(RootFindingTest, SymbolicGraph) {
    auto x = sym("x");
    auto f_expr = x * x - 9.0;
    Function f("f", {x}, {f_expr});

    SymbolicMatrix x0(1, 1);
    x0(0, 0) = casadi::MX(2.0); // Symbolic scalar 2.0

    auto res = rootfinder<SymbolicScalar>(f, x0);

    // res.x is SymbolicVector (Eigen<MX>)
    // Evaluate it
    EXPECT_NEAR(eval_scalar(res.x(0)), 3.0, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionCreation) {
    // G(x, p) = x^2 - p = 0  => x = sqrt(p)
    auto x = sym("x");
    auto p = sym("p");
    auto g_expr = x * x - p;
    Function g("g", {x, p}, {g_expr});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    auto implicit_fn = create_implicit_function(g, x_guess);

    // Eval at p=16 -> x=4
    Eigen::VectorXd p_val(1);
    p_val << 16.0;

    auto res = implicit_fn.eval(p_val);

    EXPECT_NEAR(res(0), 4.0, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionDerivative) {
    // Check if derivatives propagate through the implicit function
    // x = sqrt(p) -> dx/dp = 1/(2*sqrt(p))
    // At p=4, x=2, dx/dp = 1/4 = 0.25

    auto x = sym("x");
    auto p = sym("p");
    auto g_expr = x * x - p;
    Function g("g", {x, p}, {g_expr});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    auto implicit_fn = create_implicit_function(g, x_guess);

    // Compute Jacobian of implicit_fn w.r.t p
    // implicit_fn takes p, returns x.

    // We can use janus::jacobian? No, use CasADi jacobian on underlying
    auto fn_casadi = implicit_fn.casadi_function();
    auto J = fn_casadi.jacobian(); // Returns function computing jacobian

    // Eval J at p=4
    std::vector<double> args = {4.0};
    auto j_res = J(std::vector<casadi::DM>{casadi::DM(args), casadi::DM(2.0)});
    // Jacobian outputs are usually [J, x] or just J?
    // Function::jacobian returns function with inputs [in..., out...] and output [jac]

    // Easier way: Construct symbolic expression and diff
    auto p_sym = sym("p_sym");
    auto x_sym = implicit_fn(p_sym)[0]; // Symbolic output

    // Jacobian of x_sym w.r.t p_sym
    auto jac = casadi::MX::jacobian(janus::to_mx(x_sym), p_sym);

    // Eval jac at p=4
    casadi::Function j_fn("j_fn", {p_sym}, {jac});
    auto j_val = j_fn(std::vector<casadi::DM>{casadi::DM(4.0)});

    EXPECT_NEAR(double(j_val[0]), 0.25, 1e-6);
}

TEST(RootFindingTest, NewtonSolverClass) {
    auto x = sym("x");
    auto f_expr = x * x - 9.0; // Roots at +/- 3
    Function f("f", {x}, {f_expr});

    // Create persistent solver
    janus::NewtonSolver solver(f);

    // Solve with first guess
    Eigen::VectorXd x0(1);
    x0 << 1.0;
    auto res1 = solver.solve(x0);
    EXPECT_TRUE(res1.converged);
    EXPECT_NEAR(res1.x(0), 3.0, 1e-6);

    // Solve with second guess
    x0 << -1.0;
    auto res2 = solver.solve(x0);
    EXPECT_TRUE(res2.converged);
    EXPECT_NEAR(res2.x(0), -3.0, 1e-6);
}
