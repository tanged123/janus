/**
 * @file collocation_demo.cpp
 * @brief Demonstration of DirectCollocation for trajectory optimization
 *
 * Shows how the DirectCollocation class simplifies optimal control problems
 * compared to manual collocation setup.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

using namespace janus;

// ============================================================================
// Dynamics: Brachistochrone (bead sliding under gravity)
// ============================================================================

/**
 * @brief Brachistochrone ODE
 *
 * State: [x, y, v] (position and velocity)
 * Control: theta (angle from vertical)
 *
 * dx/dt = v * sin(theta)
 * dy/dt = -v * cos(theta)
 * dv/dt = g * cos(theta)
 */
SymbolicVector brachistochrone_dynamics(const SymbolicVector &state, const SymbolicVector &control,
                                        const SymbolicScalar & /*t*/) {
    constexpr double g = 9.80665;

    SymbolicScalar v = state(2);
    SymbolicScalar theta = control(0);

    SymbolicVector dxdt(3);
    dxdt(0) = v * janus::sin(theta);  // x' = v * sin(θ)
    dxdt(1) = -v * janus::cos(theta); // y' = -v * cos(θ) (going down)
    dxdt(2) = g * janus::cos(theta);  // v' = g * cos(θ)

    return dxdt;
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    Brachistochrone via DirectCollocation                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

    // Problem setup
    NumericVector x0{{0.0, 10.0, 0.001}};  // Start: x=0, y=10, v≈0
    NumericVector xf_partial{{10.0, 5.0}}; // End: x=10, y=5, v=free

    std::cout << "Problem:\n";
    std::cout << "  Start: (x=" << x0(0) << ", y=" << x0(1) << "), v=0\n";
    std::cout << "  End:   (x=" << xf_partial(0) << ", y=" << xf_partial(1) << "), v=free\n";
    std::cout << "  Minimize: Total time T\n\n";

    // =========================================================================
    // CLEAN COLLOCATION SETUP
    // =========================================================================

    Opti opti;
    DirectCollocation dc(opti);

    // Final time is a decision variable
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    // Setup collocation with Hermite-Simpson (4th order)
    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = 31;

    auto [x, u, tau] = dc.setup(3, 1, 0.0, T, opts);

    // Set dynamics
    dc.set_dynamics(brachistochrone_dynamics);

    // Add collocation constraints
    dc.add_defect_constraints();

    // Boundary conditions
    dc.set_initial_state(x0);
    dc.set_final_state(0, xf_partial(0)); // Final x
    dc.set_final_state(1, xf_partial(1)); // Final y
    // Final velocity is free

    // Control bounds: theta in (0, π)
    constexpr double theta_min = 0.01;
    constexpr double theta_max = M_PI - 0.01;
    opti.subject_to_bounds(u.col(0), theta_min, theta_max);

    // Velocity must be non-negative
    opti.subject_to_lower(x.col(2), 0.0);

    // =========================================================================
    // Minimize time
    // =========================================================================
    opti.minimize(T);

    std::cout << "Solving (Hermite-Simpson collocation, " << opts.n_nodes << " nodes)...\n";
    auto sol = opti.solve({.max_iter = 500, .verbose = true});

    // =========================================================================
    // Results
    // =========================================================================
    double T_opt = sol.value(T);
    auto x_sol = sol.value(x);
    auto u_sol = sol.value(u);

    std::cout << "\n=== RESULTS ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Optimal time: T* = " << T_opt << " seconds\n";
    std::cout << "Dymos reference: T* ≈ 1.8019 seconds\n";
    std::cout << "Error: " << std::abs(T_opt - 1.80185) / 1.80185 * 100 << "%\n\n";

    // Print trajectory samples
    std::cout << "Trajectory (sampled):\n";
    std::cout << std::setw(8) << "t [s]" << std::setw(10) << "x [m]" << std::setw(10) << "y [m]"
              << std::setw(10) << "v [m/s]" << std::setw(12) << "θ [deg]" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int k = 0; k < dc.n_nodes(); k += 5) {
        double t = tau(k) * T_opt;
        std::cout << std::setw(8) << t << std::setw(10) << x_sol(k, 0) << std::setw(10)
                  << x_sol(k, 1) << std::setw(10) << x_sol(k, 2) << std::setw(12)
                  << u_sol(k, 0) * 180.0 / M_PI << "\n";
    }
    int last = dc.n_nodes() - 1;
    std::cout << std::setw(8) << T_opt << std::setw(10) << x_sol(last, 0) << std::setw(10)
              << x_sol(last, 1) << std::setw(10) << x_sol(last, 2) << std::setw(12)
              << u_sol(last, 0) * 180.0 / M_PI << "\n";

    std::cout << "\n=== COMPARISON WITH MANUAL COLLOCATION ===\n";
    std::cout << "DirectCollocation reduces ~50 lines of manual setup to ~10 lines!\n";
    std::cout << "  - Automatic decision variable creation\n";
    std::cout << "  - Automatic defect constraint generation\n";
    std::cout << "  - Built-in Hermite-Simpson for 4th-order accuracy\n";

    return 0;
}
