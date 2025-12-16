#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

/**
 * Root Finding Demo
 *
 * Demonstrates:
 * 1. Solving a 2D system of equations F(x)=0 using Persistent Newton Solver.
 * 2. Creating an implicit function z(p) from G(z, p)=0 and differentiating it.
 */
int main() {
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=== Root Finding Demo ===\n";

    // ---------------------------------------------------------
    // 1. Solving a system of equations
    // x^2 + y^2 - 1 = 0 (Circle)
    // x^2 - y = 0       (Parabola)
    // Intersection: y = x^2 => x^2 + x^4 - 1 = 0.
    // ---------------------------------------------------------

    std::cout << "\n--- 1. Solving 2D System F(q)=0 ---\n";
    std::cout << "System:\n  x^2 + y^2 - 1 = 0\n  x^2 - y = 0\n";

    // Define variable vector q = [x, y]
    // Use sym_vec_pair to get both:
    // 1. q (SymbolicVector) for building expressions with Eigen syntax
    // 2. q_mx (SymbolicScalar/MX) for passing as a function input primitive
    auto [q, q_mx] = janus::sym_vec_pair("q", 2);

    auto x = q(0);
    auto y = q(1);

    auto F1 = x * x + y * y - 1.0;
    auto F2 = x * x - y;

    // Residual vector [F1, F2]
    auto Res = janus::SymbolicVector(2);
    Res << F1, F2;

    // Define simple wrapper function F(q) -> Res
    // We pass the primitive q_mx as the input
    janus::Function F("System2D", {janus::SymbolicArg(q_mx)}, {janus::to_mx(Res)});

    // Configure Persistent Solver
    janus::RootFinderOptions opts;
    opts.verbose = true; // Show CasADi output
    janus::NewtonSolver solver(F, opts);

    // Guess 1: (0.5, 0.5) - close to positive intersection
    Eigen::VectorXd x0(2);
    x0 << 0.5, 0.5;

    std::cout << "Solving from initial guess: [" << x0.transpose() << "]\n";
    auto res = solver.solve(x0);

    if (res.converged) {
        std::cout << "Solution found: [" << res.x(0) << ", " << res.x(1) << "]\n";
        // Verification
        double x_sol = res.x(0);
        double y_sol = res.x(1);
        std::cout << "Residuals: " << std::abs(x_sol * x_sol + y_sol * y_sol - 1.0) << ", "
                  << std::abs(x_sol * x_sol - y_sol) << "\n";
    } else {
        std::cout << "Failed to converge: " << res.message << "\n";
    }

    // ---------------------------------------------------------
    // 2. Implicit Function Creation
    // Solve G(z, p) = z^3 + p*z + 1 = 0 for z(p)
    // ---------------------------------------------------------
    std::cout << "\n--- 2. Implicit Function z(p) from G(z, p)=0 ---\n";
    std::cout << "G(z, p) = z^3 + p*z + 1 = 0\n";

    auto z = janus::sym("z"); // State
    auto p = janus::sym("p"); // Parameter
    auto G_expr = z * z * z + p * z + 1.0;

    // Define G(z, p)
    janus::Function G("G", {z, p}, {G_expr});

    // Initial guess for z (at p=0, z=-1 is a root)
    Eigen::VectorXd z_guess(1);
    z_guess << -1.0;

    // Create function z(p) which implicitly solves G(z, p)=0
    auto z_of_p = janus::create_implicit_function(G, z_guess);

    // Evaluate z(0) -> should be -1
    auto res_0 = z_of_p(0.0)[0];
    std::cout << "z(0): " << res_0(0, 0) << " (Expected -1.0)\n";

    // Evaluate z(1) -> z^3 + z + 1 = 0  => z ~ -0.6823
    auto res_1 = z_of_p(1.0)[0];
    std::cout << "z(1): " << res_1(0, 0) << " (Expected ~ -0.6823)\n";

    // Differentiate: dz/dp
    // This demonstrates embedding the implicit solver in a symbolic graph!
    std::cout << "Computing dz/dp symbolically...\n";

    auto J_sym = janus::jacobian(z_of_p(p)[0](0), p);
    janus::Function J_fn("dzdp", {p}, {J_sym});

    // Analytic check at p=0, z=-1
    // dz/dp = -z / (3z^2 + p) = -(-1) / (3) = 1/3
    auto dzdp_0 = J_fn(0.0)[0];
    std::cout << "dz/dp at p=0: " << dzdp_0(0, 0) << " (Expected 0.33333)\n";

    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
