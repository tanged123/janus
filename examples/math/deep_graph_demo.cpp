#include <iostream>
#include <janus/janus.hpp>

/**
 * @brief Deep Graph Visualization Demo
 *
 * Demonstrates the difference between shallow (MX-based) and deep (SX-based)
 * graph visualization. Deep graphs show all primitive operations by expanding
 * nested function calls.
 */

int main() {
    std::cout << "=== Deep Graph Visualization Demo ===\n\n";

    // ============================================================
    // Example 1: Simple expression - compare shallow vs deep
    // ============================================================
    std::cout << "1. Simple Expression: sin(x)^2 + cos(x)^2\n";

    auto x = janus::sym("x");
    auto expr1 = janus::sin(x) * janus::sin(x) + janus::cos(x) * janus::cos(x);

    // Shallow graph (MX-based) - existing function
    janus::export_graph_dot(expr1, "shallow_trig", "ShallowTrig");
    janus::export_graph_html(expr1, "shallow_trig", "ShallowTrig");
    std::cout << "   Shallow: shallow_trig.html\n";

    // Create a Function and export deep graph
    janus::Function f1("trig_identity", {x}, {expr1});
    janus::export_graph_deep(f1.casadi_function(), "deep_trig", janus::DeepGraphFormat::HTML,
                             "DeepTrig");
    janus::export_graph_deep(f1.casadi_function(), "deep_trig", janus::DeepGraphFormat::DOT,
                             "DeepTrig");
    std::cout << "   Deep:    deep_trig.html\n\n";

    // ============================================================
    // Example 2: Nested function calls - this is where deep shines
    // ============================================================
    std::cout << "2. Nested Functions: Demonstrates function expansion\n";

    auto y = janus::sym("y");
    auto z = janus::sym("z");

    // Inner function: computes magnitude
    auto magnitude = janus::sqrt(x * x + y * y + z * z);
    janus::Function mag_fn("magnitude", {x, y, z}, {magnitude});

    // Outer function: normalizes a vector using the inner magnitude function
    // Call the magnitude function symbolically
    auto mag_result = mag_fn(x, y, z);
    auto mag_call = mag_result[0](0, 0); // Get the scalar result

    auto nx = x / mag_call;
    auto ny = y / mag_call;
    auto nz = z / mag_call;

    // Create function that uses the nested call
    janus::Function normalize_fn("normalize", {x, y, z}, {nx, ny, nz});

    // Shallow graph - shows magnitude as opaque function call
    janus::export_graph_html(nx, "shallow_normalize", "ShallowNormalize");
    std::cout << "   Shallow: shallow_normalize.html (magnitude appears as single node)\n";

    // Deep graph - expands magnitude function to show sqrt, sq, add operations
    janus::export_graph_deep(normalize_fn.casadi_function(), "deep_normalize",
                             janus::DeepGraphFormat::HTML, "DeepNormalize");
    std::cout << "   Deep:    deep_normalize.html (shows sqrt, *, +, / operations)\n\n";

    // ============================================================
    // Example 3: Physics simulation - 2-body gravitational dynamics
    // ============================================================
    std::cout << "3. Two-Body Gravitational Dynamics\n";

    // State: position (x, y, z) and velocity (vx, vy, vz)
    auto px = janus::sym("px");
    auto py = janus::sym("py");
    auto pz = janus::sym("pz");
    auto vx = janus::sym("vx");
    auto vy = janus::sym("vy");
    auto vz = janus::sym("vz");

    // Gravitational parameter
    auto mu = janus::sym("mu");

    // Compute r^3 for gravitational acceleration
    auto r_squared = px * px + py * py + pz * pz;
    auto r = janus::sqrt(r_squared);
    auto r_cubed = r * r_squared;

    // Gravitational acceleration: a = -mu/r^3 * r_vec
    auto ax = -mu * px / r_cubed;
    auto ay = -mu * py / r_cubed;
    auto az = -mu * pz / r_cubed;

    // State derivative: [v, a]
    janus::Function dynamics("gravity_dynamics", {px, py, pz, vx, vy, vz, mu},
                             {vx, vy, vz, ax, ay, az});

    // Deep graph shows all the mathematical operations
    janus::export_graph_deep(dynamics.casadi_function(), "deep_gravity",
                             janus::DeepGraphFormat::HTML, "GravityDynamics");
    janus::export_graph_deep(dynamics.casadi_function(), "deep_gravity",
                             janus::DeepGraphFormat::DOT, "GravityDynamics");
    std::cout << "   Deep:    deep_gravity.html\n";
    std::cout << "   Shows: sqrt, sq (square), *, /, - operations\n\n";

    // ============================================================
    // Example 4: Jacobian of dynamics - automatic differentiation
    // ============================================================
    std::cout << "4. Jacobian of Gravitational Dynamics (Automatic Differentiation)\n";

    // Compute Jacobian of acceleration w.r.t. position using janus::jacobian
    // which takes vectors of symbolic scalars
    auto jacobian_a_r = janus::jacobian({ax, ay, az}, {px, py, pz});

    janus::Function jac_fn("gravity_jacobian", {px, py, pz, mu}, {jacobian_a_r});
    janus::export_graph_deep(jac_fn.casadi_function(), "deep_gravity_jacobian",
                             janus::DeepGraphFormat::HTML, "GravityJacobian");
    std::cout << "   Deep:    deep_gravity_jacobian.html\n";
    std::cout << "   Shows the full derivative computation graph\n\n";

    // ============================================================
    // Example 5: Control system - PID controller
    // ============================================================
    std::cout << "5. PID Controller\n";

    auto error = janus::sym("error");
    auto error_integral = janus::sym("error_int");
    auto error_derivative = janus::sym("error_dot");
    auto Kp = janus::sym("Kp");
    auto Ki = janus::sym("Ki");
    auto Kd = janus::sym("Kd");

    // PID output: u = Kp*e + Ki*integral(e) + Kd*de/dt
    auto u = Kp * error + Ki * error_integral + Kd * error_derivative;

    // With saturation (smooth clamp)
    auto u_max = janus::sym("u_max");
    auto u_saturated = u_max * janus::tanh(u / u_max);

    janus::Function pid_fn("pid_saturated",
                           {error, error_integral, error_derivative, Kp, Ki, Kd, u_max},
                           {u_saturated});

    janus::export_graph_deep(pid_fn.casadi_function(), "deep_pid", janus::DeepGraphFormat::HTML,
                             "PIDController");
    std::cout << "   Deep:    deep_pid.html\n";
    std::cout << "   Shows: *, +, tanh, / operations\n\n";

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "=== Generated Files ===\n";
    std::cout << "Shallow graphs (MX-based, may have opaque function nodes):\n";
    std::cout << "  - shallow_trig.html\n";
    std::cout << "  - shallow_normalize.html\n\n";

    std::cout << "Deep graphs (SX-based, all operations expanded):\n";
    std::cout << "  - deep_trig.html\n";
    std::cout << "  - deep_normalize.html\n";
    std::cout << "  - deep_gravity.html\n";
    std::cout << "  - deep_gravity_jacobian.html\n";
    std::cout << "  - deep_pid.html\n\n";

    std::cout << "Open any .html file in a browser for interactive visualization.\n";
    std::cout << "Features: pan (drag), zoom (scroll), click nodes for details.\n\n";

    std::cout << "Node colors in deep graphs:\n";
    std::cout << "  - Green ellipse:  Symbolic inputs (x, y, mu, etc.)\n";
    std::cout << "  - Orange ellipse: Constants (0, 1, numeric values)\n";
    std::cout << "  - Light blue box: Arithmetic (+, -, *, /, neg)\n";
    std::cout << "  - Plum box:       Trigonometric (sin, cos, tan, etc.)\n";
    std::cout << "  - Pink box:       Power/Exp (sqrt, sq, exp, log, pow)\n";
    std::cout << "  - Gold circle:    Output nodes\n";

    return 0;
}
