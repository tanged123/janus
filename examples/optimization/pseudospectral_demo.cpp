/**
 * @file pseudospectral_demo.cpp
 * @brief Demonstration of Pseudospectral transcription for trajectory optimization
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>

using namespace janus;

SymbolicVector brachistochrone_dynamics(const SymbolicVector &state, const SymbolicVector &control,
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

int main() {
    std::cout << "===============================================================\n";
    std::cout << "  Brachistochrone via Pseudospectral (LGL)\n";
    std::cout << "===============================================================\n\n";

    NumericVector x0{{0.0, 10.0, 0.001}};
    NumericVector xf_partial{{10.0, 5.0}};

    Opti opti;
    Pseudospectral ps(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = 31;

    auto [x, u, tau] = ps.setup(3, 1, 0.0, T, opts);

    ps.set_dynamics(brachistochrone_dynamics);
    ps.add_dynamics_constraints();

    ps.set_initial_state(x0);
    ps.set_final_state(0, xf_partial(0));
    ps.set_final_state(1, xf_partial(1));

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);
    opti.minimize(T);

    std::cout << "Solving (LGL, " << opts.n_nodes << " nodes)...\n";
    auto sol = opti.solve({.max_iter = 500, .verbose = true});

    const double T_opt = sol.value(T);
    auto x_sol = sol.value(x);
    auto u_sol = sol.value(u);

    std::cout << "\n=== RESULTS ===\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Optimal time: T* = " << T_opt << " s\n";
    std::cout << "Reference: T* ≈ 1.80185 s\n";
    std::cout << "Relative error: " << std::abs(T_opt - 1.80185) / 1.80185 * 100.0 << "%\n\n";

    std::cout << "Trajectory samples:\n";
    std::cout << std::setw(8) << "t [s]" << std::setw(10) << "x [m]" << std::setw(10) << "y [m]"
              << std::setw(10) << "v [m/s]" << std::setw(12) << "theta [deg]\n";
    std::cout << std::string(50, '-') << "\n";

    for (int k = 0; k < ps.n_nodes(); k += 5) {
        double t = tau(k) * T_opt;
        std::cout << std::setw(8) << t << std::setw(10) << x_sol(k, 0) << std::setw(10)
                  << x_sol(k, 1) << std::setw(10) << x_sol(k, 2) << std::setw(12)
                  << u_sol(k, 0) * 180.0 / M_PI << "\n";
    }
    const int last = ps.n_nodes() - 1;
    std::cout << std::setw(8) << T_opt << std::setw(10) << x_sol(last, 0) << std::setw(10)
              << x_sol(last, 1) << std::setw(10) << x_sol(last, 2) << std::setw(12)
              << u_sol(last, 0) * 180.0 / M_PI << "\n";

    return 0;
}
