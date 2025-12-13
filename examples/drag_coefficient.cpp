#include <janus/janus.hpp>
#include <iostream>
#include <vector>

// Physics model: Drag Coefficient
// Cd = Cd0 + k * (Cl - Cl0)^2
// Drag = 0.5 * rho * v^2 * S * Cd

template <typename Scalar>
Scalar compute_drag(const Scalar& rho, const Scalar& v, const Scalar& S, 
                    const Scalar& Cd0, const Scalar& k, const Scalar& Cl, const Scalar& Cl0) {
    auto q = 0.5 * rho * janus::pow(v, 2.0);
    auto Cd = Cd0 + k * janus::pow(Cl - Cl0, 2.0);
    return q * S * Cd;
}

int main() {
    // Numeric Mode
    double rho = 1.225;
    double v = 50.0;
    double S = 10.0;
    double Cd0 = 0.02;
    double k = 0.04;
    double Cl = 0.5;
    double Cl0 = 0.1;
    
    double drag_numeric = compute_drag(rho, v, S, Cd0, k, Cl, Cl0);
    std::cout << "Numeric Drag: " << drag_numeric << " N" << std::endl;
    
    // Symbolic Mode
    janus::SymbolicScalar v_sym = casadi::MX::sym("v");
    janus::SymbolicScalar Cl_sym = casadi::MX::sym("Cl");
    
    // Constants for symbolic evaluation
    janus::SymbolicScalar rho_s = rho;
    janus::SymbolicScalar S_s = S;
    janus::SymbolicScalar Cd0_s = Cd0;
    janus::SymbolicScalar k_s = k;
    janus::SymbolicScalar Cl0_s = Cl0;

    auto drag_sym = compute_drag(rho_s, v_sym, S_s, Cd0_s, k_s, Cl_sym, Cl0_s);
    
    // Create function: f(v, Cl) -> drag
    std::vector<janus::SymbolicScalar> inputs = {v_sym, Cl_sym};
    std::vector<janus::SymbolicScalar> outputs = {drag_sym};
    casadi::Function drag_fun("drag_fun", inputs, outputs);
    
    // Evaluate symbolic function at numeric point
    auto res = drag_fun(std::vector<casadi::DM>{v, Cl});
    std::cout << "Symbolic Drag (evaluated): " << res[0] << " N" << std::endl;
    
    return 0;
}
