/**
 * @file sparsity_intro.cpp
 * @brief Demonstration of Janus Sparsity Introspection
 *
 * Shows how to analyze sparsity patterns of Jacobians and Hessians,
 * which is critical for understanding the structure of optimization problems.
 */

#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

using namespace janus;

void print_section(const std::string &title) {
    std::cout << "\n=== " << title << " ===\n" << std::endl;
}

int main() {
    print_section("Sparsity Introspection Demo");

    // 1. Jacobian Sparsity of a Vector Function
    // -----------------------------------------
    // Let's analyze a simple physics-like function
    // f(x) = [x0^2, x0*x1, x2]
    {
        auto x = sym("x", 3);
        auto f = casadi::MX::vertcat({
            x(0) * x(0), // Depends only on x0
            x(0) * x(1), // Depends on x0 and x1
            x(2)         // Depends only on x2
        });

        auto sp = sparsity_of_jacobian(f, x);

        std::cout << "Function f(x) -> Jacobian J(x):\n";
        std::cout << "  f[0] = x0^2   (depends on x0)\n";
        std::cout << "  f[1] = x0*x1  (depends on x0, x1)\n";
        std::cout << "  f[2] = x2     (depends on x2)\n\n";

        std::cout << sp.to_string() << "\n";
    }

    // 2. Hessian Sparsity (The "Arrowhead" Pattern)
    // ---------------------------------------------
    // Common in optimal control: variables are coupled to neighbors
    // f = sum((x[i] - x[i+1])^2)
    {
        int N = 10;
        auto x = sym("x", N);

        SymbolicScalar f = 0;
        for (int i = 0; i < N - 1; ++i) {
            auto diff = x(i) - x(i + 1);
            f = f + diff * diff;
        }

        auto sp = sparsity_of_hessian(f, x);

        std::cout << "Tridiagonal Hessian (Chain Structure):\n";
        std::cout << sp.to_string() << "\n";
    }

    // 3. Block-Diagonal Structure
    // ---------------------------
    // Independent systems combined together
    {
        int N = 4;
        auto x = sym("x", N);
        auto y = sym("y", N);

        // System 1 depends only on x, System 2 depends only on y
        auto sys1 = casadi::MX::mtimes(casadi::DM::rand(N, N), x);
        auto sys2 = casadi::MX::mtimes(casadi::DM::rand(N, N), y);

        auto combined_out = casadi::MX::vertcat({sys1, sys2});
        auto combined_in = casadi::MX::vertcat({x, y});

        auto sp = sparsity_of_jacobian(combined_out, combined_in);

        std::cout << "Block-Diagonal System (Independent Subsystems):\n";
        std::cout << sp.to_string() << "\n";

        // Export to DOT and render to PDF
        std::cout << "Exporting 'block_diag.pdf' for visualization...\n";
        bool success = sp.visualize_spy("block_diag");
        if (success) {
            std::cout << "Successfully rendered block_diag.pdf\n";
        } else {
            std::cout << "Failed to render PDF (Graphviz 'dot' command might be missing)\n";
        }
    }

    // Example 4: 2D Laplacian (5-point stencil)
    // Structure typically found in PDE discretization
    {
        std::cout << "\n2D Laplacian (5-point Stencil on 5x5 grid):\n";
        int N = 5;
        int n_vars = N * N;

        // Use sym_vec_pair to get both the vector for indexing and the raw MX for function
        // definition
        auto [x_vec, x_mx] = janus::sym_vec_pair("x", n_vars);
        std::vector<janus::SymbolicScalar> eqs(n_vars);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int k = i * N + j;
                auto val = x_vec(k); // Access via Eigen vector
                // Neighbors: left, right, up, down
                if (j > 0)
                    val += x_vec(k - 1); // Left
                if (j < N - 1)
                    val += x_vec(k + 1); // Right
                if (i > 0)
                    val += x_vec(k - N); // Up
                if (i < N - 1)
                    val += x_vec(k + N); // Down
                eqs[k] = val;            // Simple linear dependency
            }
        }

        // Pass the raw MX symbol (x_mx) as the input
        janus::Function f_pde({x_mx}, {janus::SymbolicScalar::vertcat(eqs)});
        auto sp = janus::get_jacobian_sparsity(f_pde, 0, 0);

        std::cout << "Sparsity: " << sp.n_rows() << "x" << sp.n_cols() << ", nnz=" << sp.nnz()
                  << " (density=" << sp.density() << "%)" << std::endl;
        std::cout << sp.to_string() << "\n";

        // Visualize
        std::cout << "Exporting 'laplacian_2d.pdf'...\n";
        sp.visualize_spy("laplacian_2d");
    }
}
