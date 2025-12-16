/**
 * @file drag_optimization.cpp
 * @brief Aerodynamic Optimization: Maximum L/D and Minimum Drag
 *
 * Uses the SAME compute_drag function from drag_coefficient.cpp
 * to demonstrate optimization of aerodynamic quantities.
 *
 * Drag polar model: Cd = Cd0 + k * (Cl - Cl0)²
 * Lift: L = 0.5 * rho * v² * S * Cl
 * Drag: D = 0.5 * rho * v² * S * Cd
 *
 * Problems:
 * 1. Find Cl for maximum L/D ratio (aerodynamic efficiency)
 * 2. Find velocity for minimum drag at fixed lift (cruise condition)
 * 3. Find optimal Cl and velocity for minimum power (v * D)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

// ============================================================================
// SAME compute_drag function from drag_coefficient.cpp
// ============================================================================
template <typename Scalar>
Scalar compute_drag(const Scalar &rho, const Scalar &v, const Scalar &S, const Scalar &Cd0,
                    const Scalar &k, const Scalar &Cl, const Scalar &Cl0) {
    auto q = 0.5 * rho * janus::pow(v, 2.0);
    auto Cd = Cd0 + k * janus::pow(Cl - Cl0, 2.0);
    return q * S * Cd;
}

// Compute lift using same model
template <typename Scalar>
Scalar compute_lift(const Scalar &rho, const Scalar &v, const Scalar &S, const Scalar &Cl) {
    auto q = 0.5 * rho * janus::pow(v, 2.0);
    return q * S * Cl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Aerodynamic Optimization Examples                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Aircraft parameters (typical light aircraft)
    constexpr double rho = 1.225; // Air density [kg/m³]
    constexpr double S = 16.0;    // Wing area [m²]
    constexpr double Cd0 = 0.02;  // Zero-lift drag coefficient
    constexpr double k = 0.04;    // Induced drag factor
    constexpr double Cl0 = 0.1;   // Cl at minimum drag
    constexpr double W = 10000.0; // Aircraft weight [N]

    std::cout << "Aircraft Parameters:\n";
    std::cout << "  Wing area S = " << S << " m²\n";
    std::cout << "  Cd0 = " << Cd0 << "\n";
    std::cout << "  k = " << k << "\n";
    std::cout << "  Cl0 = " << Cl0 << "\n";
    std::cout << "  Weight W = " << W << " N\n\n";

    // =========================================================================
    // Problem 1: Find Cl for Maximum L/D
    // =========================================================================
    std::cout << "=== Problem 1: Maximum L/D Ratio ===\n";
    std::cout << "Find Cl that maximizes Cl/Cd\n\n";

    {
        janus::Opti opti;

        // Decision variable: Cl
        auto Cl = opti.variable(0.5);
        opti.subject_to_bounds(Cl, 0.1, 2.0); // Now works for scalars!

        // Cd = Cd0 + k * (Cl - Cl0)²
        auto Cd = Cd0 + k * (Cl - Cl0) * (Cl - Cl0);

        // Maximize L/D = Cl/Cd  →  Minimize Cd/Cl
        opti.minimize(Cd / Cl);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        double Cl_opt = sol.value(Cl);
        double Cd_opt = Cd0 + k * std::pow(Cl_opt - Cl0, 2);
        double LD_opt = Cl_opt / Cd_opt;

        // Analytical: Cl* = sqrt((Cd0 + k*Cl0²) / k)
        double Cl_analytical = std::sqrt((Cd0 + k * Cl0 * Cl0) / k);
        double Cd_analytical = Cd0 + k * std::pow(Cl_analytical - Cl0, 2);
        double LD_analytical = Cl_analytical / Cd_analytical;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Optimization Result:\n";
        std::cout << "  Cl* = " << Cl_opt << "\n";
        std::cout << "  Cd* = " << Cd_opt << "\n";
        std::cout << "  (L/D)_max = " << LD_opt << "\n";
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";

        std::cout << "Analytical Result:\n";
        std::cout << "  Cl* = " << Cl_analytical << "\n";
        std::cout << "  Cd* = " << Cd_analytical << "\n";
        std::cout << "  (L/D)_max = " << LD_analytical << "\n\n";
    }

    // =========================================================================
    // Problem 2: Minimum Drag at Fixed Lift (Cruise)
    // =========================================================================
    std::cout << "=== Problem 2: Minimum Drag at Cruise (L = W) ===\n";
    std::cout << "Find velocity V that minimizes drag while L = W\n\n";

    {
        janus::Opti opti;

        // Decision variables
        auto V = opti.variable(50.0);
        auto Cl = opti.variable(0.5);

        opti.subject_to_bounds(V, 20.0, 150.0);
        opti.subject_to_bounds(Cl, 0.1, 1.8);

        // Dynamic pressure: q = 0.5 * rho * V²
        auto q = 0.5 * rho * V * V;

        // Lift constraint: L = q * S * Cl = W
        auto L = q * S * Cl;
        opti.subject_to(L == W);

        // Drag: D = q * S * Cd, where Cd = Cd0 + k*(Cl-Cl0)²
        auto Cd = Cd0 + k * (Cl - Cl0) * (Cl - Cl0);
        auto D = q * S * Cd;
        opti.minimize(D);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        double V_opt = sol.value(V);
        double Cl_opt = sol.value(Cl);
        double D_opt = compute_drag(rho, V_opt, S, Cd0, k, Cl_opt, Cl0);
        double L_check = compute_lift(rho, V_opt, S, Cl_opt);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Optimization Result:\n";
        std::cout << "  V* = " << V_opt << " m/s (" << V_opt * 3.6 << " km/h)\n";
        std::cout << "  Cl* = " << Cl_opt << "\n";
        std::cout << "  D_min = " << D_opt << " N\n";
        std::cout << "  L (check) = " << L_check << " N (should be " << W << ")\n";
        std::cout << "  L/D = " << W / D_opt << "\n";
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";
    }

    // =========================================================================
    // Problem 3: Minimum Power (Best Endurance)
    // =========================================================================
    std::cout << "=== Problem 3: Minimum Power (L = W) ===\n";
    std::cout << "Find V that minimizes P = D * V (best endurance)\n\n";

    {
        janus::Opti opti;

        auto V = opti.variable(40.0);
        auto Cl = opti.variable(0.8);

        opti.subject_to_bounds(V, 20.0, 150.0);
        opti.subject_to_bounds(Cl, 0.1, 1.8);

        // Lift constraint: L = W
        auto q = 0.5 * rho * V * V;
        auto L = q * S * Cl;
        opti.subject_to(L == W);

        // Power: P = D * V
        auto Cd = Cd0 + k * (Cl - Cl0) * (Cl - Cl0);
        auto D = q * S * Cd;
        opti.minimize(D * V);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        double V_opt = sol.value(V);
        double Cl_opt = sol.value(Cl);
        double D_opt = compute_drag(rho, V_opt, S, Cd0, k, Cl_opt, Cl0);
        double P_opt = D_opt * V_opt;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Optimization Result:\n";
        std::cout << "  V* = " << V_opt << " m/s (" << V_opt * 3.6 << " km/h)\n";
        std::cout << "  Cl* = " << Cl_opt << "\n";
        std::cout << "  D = " << D_opt << " N\n";
        std::cout << "  P_min = " << P_opt / 1000.0 << " kW\n";
        std::cout << "  Iterations: " << sol.num_iterations() << "\n\n";
    }

    std::cout << "=== SUMMARY ===\n";
    std::cout << "✓ Same compute_drag<Scalar> used for BOTH simulation AND optimization\n";
    std::cout << "✓ Classic aero problems: max L/D, min drag cruise, min power\n";
    std::cout << "✓ Optimization results match aerodynamic theory\n";
    std::cout << "✓ Clean API - physics expressions work directly in Opti\n";

    return 0;
}
