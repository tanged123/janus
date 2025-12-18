/**
 * @file multishoot_comparison.cpp
 * @brief Comparison of Direct Collocation vs Multiple Shooting
 *
 * Solves the Brachistochrone problem using both transcription methods
 * to demonstrate differences in convergence and accuracy.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

using namespace janus;

// ============================================================================
// Brachistochrone Dynamics
// ============================================================================

SymbolicVector brachistochrone_ode(const SymbolicVector &state, const SymbolicVector &control,
                                   const SymbolicScalar & /*t*/) {
    constexpr double g = 9.80665;
    SymbolicScalar v = state(2);
    SymbolicScalar theta = control(0);

    SymbolicVector dxdt(3);
    dxdt(0) = v * janus::sin(theta);
    dxdt(1) = -v * janus::cos(theta);
    dxdt(2) = g * janus::cos(theta);
    return dxdt;
}

// ============================================================================
// Helper: Solve with Collocation
// ============================================================================

double solve_collocation(int nodes) {
    std::cout << "\n[Direct Collocation (Hermite-Simpson)]\n";
    Opti opti;
    DirectCollocation dc(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    // Hermite-Simpson scheme (4th order)
    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = nodes;

    auto [x, u, tau] = dc.setup(3, 1, 0.0, T, opts);

    dc.set_dynamics(brachistochrone_ode);
    dc.add_defect_constraints();

    dc.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    dc.set_final_state(0, 10.0);
    dc.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);

    opti.minimize(T);

    try {
        auto sol = opti.solve({.verbose = false});
        double t_opt = sol.value(T);
        std::cout << "  Nodes: " << nodes << " | Vars: " << opti.casadi_opti().x().numel()
                  << " | Constr: " << opti.casadi_opti().g().numel() << "\n";

        int iters = static_cast<int>(sol.stats().at("iter_count"));
        std::cout << "  Time: " << t_opt << " s | Iterations: " << iters << "\n";
        return t_opt;
    } catch (...) {
        std::cout << "  Solver failed.\n";
        return -1.0;
    }
}

// ============================================================================
// Helper: Solve with Multiple Shooting
// ============================================================================

double solve_multishoot(int intervals) {
    std::cout << "\n[Multiple Shooting (CVODES - variable order)]\n";
    Opti opti;
    MultipleShooting ms(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    MultiShootingOptions opts;
    opts.n_intervals = intervals;
    opts.integrator = "cvodes";
    opts.tol = 1e-6; // High accuracy integration

    auto [x, u, tau] = ms.setup(3, 1, 0.0, T, opts);

    ms.set_dynamics(brachistochrone_ode);
    ms.add_continuity_constraints();

    ms.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    ms.set_final_state(0, 10.0);
    ms.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);

    opti.minimize(T);

    try {
        auto sol = opti.solve({.verbose = false});
        double t_opt = sol.value(T);
        std::cout << "  Intervals: " << intervals << " | Vars: " << opti.casadi_opti().x().numel()
                  << " | Constr: " << opti.casadi_opti().g().numel() << "\n";

        int iters = static_cast<int>(sol.stats().at("iter_count"));
        std::cout << "  Time: " << t_opt << " s | Iterations: " << iters << "\n";
        return t_opt;
    } catch (...) {
        std::cout << "  Solver failed.\n";
        return -1.0;
    }
}

int main() {
    std::cout << "===========================================================\n";
    std::cout << "  Trajectory Transcription Comparison: Brachistochrone\n";
    std::cout << "  Ref: T* = 1.80185 s\n";
    std::cout << "===========================================================\n";

    double ref = 1.80185;

    // Case 1: Collocation with 30 nodes
    double t_col = solve_collocation(30);
    double err_col = std::abs(t_col - ref);

    // Case 2: Multiple Shooting with 20 intervals
    double t_ms = solve_multishoot(20);
    double err_ms = std::abs(t_ms - ref);

    std::cout << "\n[Summary]\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Collocation Error (Hermite-Simpson, N=30): " << err_col << " s\n";
    std::cout << "Multiple Shooting Error (CVODES, N=20):    " << err_ms << " s\n";

    if (err_ms < err_col) {
        std::cout << "-> Multiple Shooting is " << (err_col / err_ms)
                  << "x more accurate for same grid size.\n";
    }

    return 0;
}
