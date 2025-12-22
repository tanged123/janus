/**
 * @file integration_demo.cpp
 * @brief Demo: Using Janus integrators with Icarus-style component models
 *
 * This example explores two patterns for integrating ODE state with component models:
 * 1. Monolithic state vector (traditional approach)
 * 2. Component-based state management (Icarus-style)
 */

#include <iostream>
#include <janus/janus.hpp>

using namespace janus;

// ============================================================================
// Pattern 1: Monolithic State Vector (Traditional)
// ============================================================================
// All state variables in one vector, one dynamics function.
// Good for small systems, trajectory optimization.

void monolithic_example() {
    std::cout << "=== Pattern 1: Monolithic State Vector ===\n";

    // Harmonic oscillator: y'' = -ω²y
    // State: [y, v] where y'=v, v'=-ω²y
    double omega = 2.0;

    auto dynamics = [omega](double t, const NumericVector &state) {
        NumericVector dydt(2);
        dydt << state(1), -omega * omega * state(0);
        return dydt;
    };

    NumericVector state(2);
    state << 1.0, 0.0; // y=1, v=0
    double t = 0.0;
    double dt = 0.01;

    std::cout << "t=0: y=" << state(0) << ", v=" << state(1) << "\n";

    // Simulate for 1 second
    for (int i = 0; i < 100; ++i) {
        state = rk4_step(dynamics, state, t, dt);
        t += dt;
    }

    std::cout << "t=1: y=" << state(0) << " (expected: " << std::cos(omega) << ")\n\n";
}

// ============================================================================
// Pattern 2: Component-Based State (Icarus-Style)
// ============================================================================
// Each component owns its state and provides derivative computation.

/**
 * @brief Base class for integrable state variables
 *
 * Components inherit from this to expose their ODE state to the integrator.
 */
template <typename Scalar> class IntegrableState {
  public:
    virtual ~IntegrableState() = default;

    /// Get current state as a vector
    virtual JanusVector<Scalar> get_state() const = 0;

    /// Set state from a vector
    virtual void set_state(const JanusVector<Scalar> &state) = 0;

    /// Compute derivative: dx/dt = f(t, x)
    virtual JanusVector<Scalar> compute_derivative(Scalar t) const = 0;

    /// Number of state variables
    virtual int state_dim() const = 0;
};

/**
 * @brief Example: 6DOF Body State Component
 *
 * Manages position, velocity, orientation, angular velocity.
 * In real Icarus, this would receive forces/torques from other components.
 */
class RigidBody : public IntegrableState<double> {
  public:
    // State: [x, y, z, vx, vy, vz] (simplified - no rotation for demo)
    Vec3<double> position{0, 0, 0};
    Vec3<double> velocity{0, 0, 0};

    // Inputs (set by other components)
    Vec3<double> acceleration{0, 0, 0}; // From forces

    int state_dim() const override { return 6; }

    JanusVector<double> get_state() const override {
        NumericVector s(6);
        s << position(0), position(1), position(2), velocity(0), velocity(1), velocity(2);
        return s;
    }

    void set_state(const JanusVector<double> &state) override {
        position << state(0), state(1), state(2);
        velocity << state(3), state(4), state(5);
    }

    JanusVector<double> compute_derivative(double t) const override {
        NumericVector dydt(6);
        // dx/dt = v, dv/dt = a
        dydt << velocity(0), velocity(1), velocity(2), acceleration(0), acceleration(1),
            acceleration(2);
        return dydt;
    }
};

/**
 * @brief Example: Gravity Model Component
 *
 * Computes gravitational acceleration for a body.
 */
class GravityModel {
  public:
    double mu = 3.986e14; // Earth GM

    void update(RigidBody &body) {
        double r = body.position.norm();
        if (r > 0) {
            body.acceleration = -mu / (r * r * r) * body.position;
        }
    }
};

/**
 * @brief Integrator that works with IntegrableState components
 */
template <typename Scalar> class ComponentIntegrator {
  public:
    void step_rk4(IntegrableState<Scalar> &component, Scalar t, Scalar dt) {
        // Wrap component's derivative function for janus::rk4_step
        auto dynamics = [&component](Scalar t, const JanusVector<Scalar> &state) {
            // Temporarily set state for derivative computation
            auto saved_state = component.get_state();
            component.set_state(state);
            auto deriv = component.compute_derivative(t);
            component.set_state(saved_state); // Restore original
            return deriv;
        };

        auto new_state = rk4_step(dynamics, component.get_state(), t, dt);
        component.set_state(new_state);
    }
};

void component_based_example() {
    std::cout << "=== Pattern 2: Component-Based State ===\n";

    // Create components
    RigidBody satellite;
    satellite.position << 7000e3, 0, 0; // LEO altitude
    satellite.velocity << 0, 7500, 0;   // Circular orbit velocity

    GravityModel gravity;
    ComponentIntegrator<double> integrator;

    double t = 0.0;
    double dt = 10.0; // 10 second steps
    double period = 2 * M_PI * std::sqrt(std::pow(7000e3, 3) / 3.986e14);

    std::cout << "Orbital period: " << period << " seconds\n";
    std::cout << "Initial position: [" << satellite.position.transpose() << "]\n";

    // Simulate half an orbit
    int steps = static_cast<int>(period / 2 / dt);
    for (int i = 0; i < steps; ++i) {
        gravity.update(satellite); // Compute forces
        integrator.step_rk4(satellite, t, dt);
        t += dt;
    }

    std::cout << "After half orbit: [" << satellite.position.transpose() << "]\n";
    std::cout << "(Expected x ≈ -7000 km)\n\n";
}

// ============================================================================
// Pattern 3: ODE Variable Wrapper (Alternative Design)
// ============================================================================

/**
 * @brief Lightweight wrapper for a single ODE variable
 *
 * Alternative to full IntegrableState - just wraps value and derivative.
 * Good for simple scalar or vector states without component overhead.
 */
template <typename T> struct OdeVar {
    T value;
    std::function<T(double, const T &)> derivative;

    void step_euler(double t, double dt) { value = value + dt * derivative(t, value); }

    void step_rk4(double t, double dt) {
        T k1 = derivative(t, value);
        T k2 = derivative(t + dt / 2, value + dt / 2 * k1);
        T k3 = derivative(t + dt / 2, value + dt / 2 * k2);
        T k4 = derivative(t + dt, value + dt * k3);
        value = value + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
};

void ode_var_example() {
    std::cout << "=== Pattern 3: ODE Variable Wrapper ===\n";

    // Simple exponential decay
    OdeVar<double> temperature{.value = 100.0,
                               .derivative = [](double t, const double &T) { return -0.1 * T; }};

    double t = 0.0;
    double dt = 0.1;

    std::cout << "t=0: T=" << temperature.value << "\n";

    for (int i = 0; i < 100; ++i) {
        temperature.step_rk4(t, dt);
        t += dt;
    }

    std::cout << "t=10: T=" << temperature.value << " (expected: " << 100 * std::exp(-1) << ")\n\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Janus Integration Demo for Icarus-Style Simulations\n";
    std::cout << "===================================================\n\n";

    monolithic_example();
    component_based_example();
    ode_var_example();

    std::cout << "Recommendation for Icarus:\n";
    std::cout << "--------------------------\n";
    std::cout << "Use Pattern 2 (Component-Based) for simulation.\n";
    std::cout << "Each component implements IntegrableState interface.\n";
    std::cout << "Integrator collects all components and steps them together.\n";

    return 0;
}
