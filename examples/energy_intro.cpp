#include <iostream>
#include <janus/janus.hpp>

// 1. Define a generic physics model
template <typename Scalar> Scalar compute_energy(const Scalar &v, const Scalar &m) {
    // Use janus:: math functions for dual-backend support
    return 0.5 * m * janus::pow(v, 2.0);
}

int main() {
    std::cout << "--- Janus Intro Example: Kinetic Energy & Autodiff ---\n";

    // 2. Numeric Mode (Fast Standard Execution)
    double v = 10.0, m = 2.0;
    double E = compute_energy(v, m);

    std::cout << "Numeric Energy (v=" << v << ", m=" << m << "): " << E << "\n";

    // 3. Symbolic Mode (Graph Generation & Derivatives)
    auto v_sym = janus::sym("v");
    auto m_sym = janus::sym("m");
    auto E_sym = compute_energy(v_sym, m_sym);

    // Automatic Differentiation (Compute dE/dv)
    auto dE_dv = janus::jacobian({E_sym}, {v_sym});

    // Create Callable Function (wraps CasADi)
    janus::Function f_grad({v_sym, m_sym}, {dE_dv});

    // Evaluate derivatives numerically
    auto dE_dv_result = f_grad.eval(10.0, 2.0);

    std::cout << "Symbolic dE/dv (evaluated at v=10, m=2): " << dE_dv_result(0, 0) << "\n";
    std::cout << "Analytic Check (m*v): " << 2.0 * 10.0 << "\n";

    return 0;
}
