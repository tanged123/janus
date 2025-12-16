/**
 * @file rosenbrock.cpp
 * @brief Optimization example: Rosenbrock benchmark
 *
 * Demonstrates janus::Opti for nonlinear optimization using the classic
 * Rosenbrock "banana" function, a standard benchmark for optimization solvers.
 *
 * min f(x,y) = (1-x)^2 + 100*(y-x^2)^2
 *
 * Global minimum: (x,y) = (1,1) with f = 0
 */

#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

int main() {
    std::cout << "========================================\n";
    std::cout << "Rosenbrock Optimization Benchmark\n";
    std::cout << "========================================\n\n";

    // =========================================================================
    // Problem 1: 2D Rosenbrock (Unconstrained)
    // =========================================================================
    std::cout << "Problem 1: 2D Rosenbrock (Unconstrained)\n";
    std::cout << "  min (1-x)^2 + 100*(y-x^2)^2\n\n";

    {
        janus::Opti opti;

        // Decision variables with initial guess
        auto x = opti.variable(-1.0); // Start far from optimum
        auto y = opti.variable(-1.0);

        // Rosenbrock objective
        auto obj = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        opti.minimize(obj);

        // Solve
        auto sol = opti.solve({.verbose = false});

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Optimal x: " << sol.value(x) << " (expected: 1.0)\n";
        std::cout << "  Optimal y: " << sol.value(y) << " (expected: 1.0)\n";
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";
    }

    // =========================================================================
    // Problem 2: 2D Rosenbrock (Constrained)
    // =========================================================================
    std::cout << "Problem 2: 2D Rosenbrock with constraint x + y >= 2\n";

    {
        janus::Opti opti;

        auto x = opti.variable(0.5);
        auto y = opti.variable(0.5);

        auto obj = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        opti.minimize(obj);

        // Constraint pushes solution away from (1,1)
        opti.subject_to(x + y >= 2);

        auto sol = opti.solve({.verbose = false});

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Optimal x: " << sol.value(x) << "\n";
        std::cout << "  Optimal y: " << sol.value(y) << "\n";
        std::cout << "  x + y = " << sol.value(x) + sol.value(y) << " (should be >= 2)\n";
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";
    }

    // =========================================================================
    // Problem 3: N-Dimensional Rosenbrock
    // =========================================================================
    constexpr int N = 10;
    std::cout << "Problem 3: " << N << "-Dimensional Rosenbrock\n";
    std::cout << "  f(x) = sum_{i=1}^{N-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]\n\n";

    {
        janus::Opti opti;

        // Create N variables
        auto x = opti.variable(N, 0.0); // Vector of N variables, init at 0

        // Build N-D Rosenbrock objective
        janus::SymbolicScalar obj = 0;
        for (int i = 0; i < N - 1; ++i) {
            obj = obj + 100 * (x(i + 1) - x(i) * x(i)) * (x(i + 1) - x(i) * x(i));
            obj = obj + (1 - x(i)) * (1 - x(i));
        }
        opti.minimize(obj);

        auto sol = opti.solve({.verbose = false});

        std::cout << "  Optimal solution:\n";
        auto x_opt = sol.value(x);
        for (int i = 0; i < N; ++i) {
            std::cout << "    x[" << i << "] = " << std::fixed << std::setprecision(4) << x_opt(i)
                      << " (expected: 1.0)\n";
        }
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";
    }

    // =========================================================================
    // Problem 4: Using janus::Function with optimization
    // =========================================================================
    std::cout << "Problem 4: Using janus::Function + janus::jacobian\n";

    {
        // Define Rosenbrock symbolically
        auto x_sym = janus::sym("x");
        auto y_sym = janus::sym("y");
        auto rosenbrock =
            (1 - x_sym) * (1 - x_sym) + 100 * (y_sym - x_sym * x_sym) * (y_sym - x_sym * x_sym);

        // Compile to function
        janus::Function f_rosenbrock("rosenbrock", {x_sym, y_sym}, {rosenbrock});

        // Compute gradient
        auto grad = janus::jacobian({rosenbrock}, {x_sym, y_sym});
        janus::Function f_gradient("gradient", {x_sym, y_sym}, {grad});

        // Evaluate at optimum
        std::cout << "  At (1, 1):\n";
        auto f_val = f_rosenbrock.eval(1.0, 1.0);
        auto g_val = f_gradient.eval(1.0, 1.0);
        std::cout << "    f(1,1) = " << f_val(0, 0) << " (expected: 0)\n";
        std::cout << "    grad = [" << g_val(0, 0) << ", " << g_val(0, 1)
                  << "] (expected: [0, 0])\n\n";

        // Evaluate away from optimum
        std::cout << "  At (0, 0):\n";
        f_val = f_rosenbrock.eval(0.0, 0.0);
        g_val = f_gradient.eval(0.0, 0.0);
        std::cout << "    f(0,0) = " << f_val(0, 0) << " (expected: 1)\n";
        std::cout << "    grad = [" << g_val(0, 0) << ", " << g_val(0, 1)
                  << "] (expected: [-2, 0])\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Rosenbrock Optimization Complete!\n";
    std::cout << "========================================\n";

    return 0;
}
