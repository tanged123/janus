#pragma once

#include "OptiCache.hpp"
#include "OptiOptions.hpp"
#include "OptiSol.hpp"
#include "OptiSweep.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Calculus.hpp"
#include "janus/math/FiniteDifference.hpp"
#include <algorithm>
#include <casadi/casadi.hpp>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace janus {

/**
 * @brief Options for variable creation
 *
 * Allows specifying category and freeze status for optimization variables.
 *
 * Example:
 * @code
 *   auto x = opti.variable(1.0, {.category = "Wing", .freeze = false});
 *   auto y = opti.variable(2.0, {.category = "Wing", .freeze = true});  // Frozen at 2.0
 * @endcode
 */
struct VariableOptions {
    std::string category = "Uncategorized"; ///< Category for grouping variables
    bool freeze = false;                    ///< If true, variable is fixed at init_guess
};

/**
 * @brief Main optimization environment class
 *
 * Wraps CasADi's Opti interface to provide Janus-native types
 * and a clean C++ API for nonlinear programming with IPOPT backend.
 *
 * Example:
 * @code
 *   janus::Opti opti;
 *   auto x = opti.variable(0.0);  // scalar, init_guess=0
 *   auto y = opti.variable(0.0);
 *   opti.minimize((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x));
 *   auto sol = opti.solve();
 *   double x_opt = sol.value(x);  // ~1.0
 * @endcode
 *
 * Rosenbrock Benchmark:
 * @code
 *   janus::Opti opti;
 *   auto x = opti.variable(10, 0.0);  // 10 variables
 *   opti.subject_to(x >= 0);
 *   // minimize sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
 *   SymbolicScalar obj = 0;
 *   for (int i = 0; i < 9; ++i) {
 *       obj = obj + 100 * janus::pow(x(i+1) - x(i)*x(i), 2)
 *                 + janus::pow(1 - x(i), 2);
 *   }
 *   opti.minimize(obj);
 *   auto sol = opti.solve();  // All x[i] ~= 1.0
 * @endcode
 */
class Opti {
  public:
    /**
     * @brief Construct a new optimization environment
     */
    Opti() : opti_(), categories_to_freeze_(), categories_() {}

    /**
     * @brief Construct with frozen categories
     *
     * Variables created with a category in `categories_to_freeze` will be
     * automatically frozen at their initial guess values.
     *
     * @param categories_to_freeze List of category names to freeze
     */
    explicit Opti(const std::vector<std::string> &categories_to_freeze)
        : opti_(), categories_to_freeze_(categories_to_freeze), categories_() {}

    ~Opti() = default;

    // =========================================================================
    // Decision Variables
    // =========================================================================

    /**
     * @brief Create a scalar decision variable
     *
     * @param init_guess Initial guess value (where optimizer starts)
     * @param scale Optional scale for numerical conditioning (default: |init_guess| or 1)
     * @param lower_bound Optional lower bound constraint
     * @param upper_bound Optional upper bound constraint
     * @return Symbolic scalar representing the optimization variable
     */
    SymbolicScalar variable(double init_guess = 0.0, std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        return variable(init_guess, VariableOptions{}, scale, lower_bound, upper_bound);
    }

    /**
     * @brief Create a scalar decision variable with options
     *
     * @param init_guess Initial guess value (where optimizer starts)
     * @param opts Variable options (category, freeze)
     * @param scale Optional scale for numerical conditioning (default: |init_guess| or 1)
     * @param lower_bound Optional lower bound constraint
     * @param upper_bound Optional upper bound constraint
     * @return Symbolic scalar representing the variable (or frozen parameter)
     */
    SymbolicScalar variable(double init_guess, const VariableOptions &opts,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        // Check if variable should be frozen (explicit or via category)
        bool should_freeze = opts.freeze || is_category_frozen(opts.category);

        // Determine scale
        double s = scale.value_or(std::abs(init_guess) > 0 ? std::abs(init_guess) : 1.0);

        SymbolicScalar scaled_var;

        if (should_freeze) {
            // Create as parameter (not optimized)
            SymbolicScalar param = opti_.parameter();
            opti_.set_value(param, init_guess);
            scaled_var = param;
        } else {
            // Create scaled variable
            SymbolicScalar raw_var = opti_.variable();
            scaled_var = s * raw_var;

            // Set initial guess
            opti_.set_initial(raw_var, init_guess / s);

            // Apply bounds if specified
            if (lower_bound.has_value()) {
                opti_.subject_to(scaled_var >= lower_bound.value());
            }
            if (upper_bound.has_value()) {
                opti_.subject_to(scaled_var <= upper_bound.value());
            }
        }

        // Track in category
        categories_[opts.category].push_back(scaled_var);

        return scaled_var;
    }

    /**
     * @brief Create a vector of decision variables with scalar init guess
     *
     * @param n_vars Number of variables
     * @param init_guess Initial guess applied to all elements
     * @param scale Optional scale for numerical conditioning
     * @param lower_bound Optional lower bound for all elements
     * @param upper_bound Optional upper bound for all elements
     * @return Symbolic vector (column vector) of optimization variables
     */
    SymbolicVector variable(int n_vars, double init_guess = 0.0,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        // Determine scale
        double s = scale.value_or(std::abs(init_guess) > 0 ? std::abs(init_guess) : 1.0);

        // Create scaled variable
        SymbolicScalar raw_var = opti_.variable(n_vars, 1);
        SymbolicScalar scaled_var = s * raw_var;

        // Set initial guess (constant vector)
        opti_.set_initial(raw_var, init_guess / s);

        // Apply bounds if specified
        if (lower_bound.has_value()) {
            opti_.subject_to(scaled_var >= lower_bound.value());
        }
        if (upper_bound.has_value()) {
            opti_.subject_to(scaled_var <= upper_bound.value());
        }

        return janus::to_eigen(scaled_var);
    }

    /**
     * @brief Create a vector of decision variables with per-element init guess
     *
     * @param init_guess Vector of initial guesses (size determines n_vars)
     * @param scale Optional scale for numerical conditioning
     * @param lower_bound Optional lower bound for all elements
     * @param upper_bound Optional upper bound for all elements
     * @return Symbolic vector of optimization variables
     */
    SymbolicVector variable(const NumericVector &init_guess,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        int n_vars = static_cast<int>(init_guess.size());

        // Determine scale from mean of absolute values
        double mean_abs = init_guess.cwiseAbs().mean();
        double s = scale.value_or(mean_abs > 0 ? mean_abs : 1.0);

        // Create scaled variable
        SymbolicScalar raw_var = opti_.variable(n_vars, 1);
        SymbolicScalar scaled_var = s * raw_var;

        // Set initial guess (convert Eigen to std::vector)
        std::vector<double> init_vec(init_guess.data(), init_guess.data() + init_guess.size());
        for (auto &v : init_vec) {
            v /= s;
        }
        opti_.set_initial(raw_var, init_vec);

        // Apply bounds if specified
        if (lower_bound.has_value()) {
            opti_.subject_to(scaled_var >= lower_bound.value());
        }
        if (upper_bound.has_value()) {
            opti_.subject_to(scaled_var <= upper_bound.value());
        }

        return janus::to_eigen(scaled_var);
    }

    // =========================================================================
    // Parameters (fixed values during optimization)
    // =========================================================================

    /**
     * @brief Create a scalar parameter
     *
     * Parameters are fixed values that can be changed between solves.
     *
     * @param value Parameter value
     * @return Symbolic scalar representing the parameter
     */
    SymbolicScalar parameter(double value) {
        SymbolicScalar param = opti_.parameter();
        opti_.set_value(param, value);
        return param;
    }

    /**
     * @brief Create a vector parameter
     *
     * @param value Vector of parameter values
     * @return Symbolic vector representing the parameters
     */
    SymbolicVector parameter(const NumericVector &value) {
        int n = static_cast<int>(value.size());
        SymbolicScalar param = opti_.parameter(n, 1);
        std::vector<double> vals(value.data(), value.data() + value.size());
        opti_.set_value(param, vals);
        return janus::to_eigen(param);
    }

    // =========================================================================
    // Constraints
    // =========================================================================

    /**
     * @brief Add a scalar constraint
     *
     * Supports both equality and inequality constraints:
     * @code
     *   opti.subject_to(x >= 0);      // inequality
     *   opti.subject_to(x == 1);      // equality
     *   opti.subject_to(x * x + y * y <= 1);  // nonlinear
     * @endcode
     *
     * @param constraint Symbolic inequality/equality expression
     */
    void subject_to(const SymbolicScalar &constraint) { opti_.subject_to(constraint); }

    /**
     * @brief Add multiple constraints
     *
     * @param constraints Vector of constraint expressions
     */
    void subject_to(const std::vector<SymbolicScalar> &constraints) {
        for (const auto &c : constraints) {
            opti_.subject_to(c);
        }
    }

    /**
     * @brief Add constraints from initializer list
     *
     * Example:
     * @code
     *   opti.subject_to({
     *       x >= 0,
     *       y >= 0,
     *       x + y <= 10
     *   });
     * @endcode
     */
    void subject_to(std::initializer_list<SymbolicScalar> constraints) {
        for (const auto &c : constraints) {
            opti_.subject_to(c);
        }
    }

    // ---- Scalar Constraint Helpers ----

    /**
     * @brief Apply lower bound to a scalar variable
     * @param scalar Symbolic scalar
     * @param lower_bound Lower bound
     */
    void subject_to_lower(const SymbolicScalar &scalar, double lower_bound) {
        opti_.subject_to(scalar >= lower_bound);
    }

    /**
     * @brief Apply upper bound to a scalar variable
     * @param scalar Symbolic scalar
     * @param upper_bound Upper bound
     */
    void subject_to_upper(const SymbolicScalar &scalar, double upper_bound) {
        opti_.subject_to(scalar <= upper_bound);
    }

    /**
     * @brief Apply both lower and upper bounds to a scalar variable
     * @param scalar Symbolic scalar
     * @param lower_bound Lower bound
     * @param upper_bound Upper bound
     */
    void subject_to_bounds(const SymbolicScalar &scalar, double lower_bound, double upper_bound) {
        subject_to_lower(scalar, lower_bound);
        subject_to_upper(scalar, upper_bound);
    }

    // ---- Vector Constraint Helpers ----

    /**
     * @brief Apply lower bound to all elements of a vector
     *
     * Example:
     * @code
     *   auto x = opti.variable(10, 0.0);
     *   opti.subject_to_lower(x, 0.0);  // All x[i] >= 0
     * @endcode
     *
     * @param vec Symbolic vector
     * @param lower_bound Lower bound for all elements
     */
    void subject_to_lower(const SymbolicVector &vec, double lower_bound) {
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            opti_.subject_to(vec(i) >= lower_bound);
        }
    }

    /**
     * @brief Apply upper bound to all elements of a vector
     *
     * @param vec Symbolic vector
     * @param upper_bound Upper bound for all elements
     */
    void subject_to_upper(const SymbolicVector &vec, double upper_bound) {
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            opti_.subject_to(vec(i) <= upper_bound);
        }
    }

    /**
     * @brief Apply both lower and upper bounds to all elements
     *
     * Example:
     * @code
     *   auto x = opti.variable(10, 0.5);
     *   opti.subject_to_bounds(x, 0.0, 1.0);  // 0 <= x[i] <= 1
     * @endcode
     *
     * @param vec Symbolic vector
     * @param lower_bound Lower bound for all elements
     * @param upper_bound Upper bound for all elements
     */
    void subject_to_bounds(const SymbolicVector &vec, double lower_bound, double upper_bound) {
        subject_to_lower(vec, lower_bound);
        subject_to_upper(vec, upper_bound);
    }

    // =========================================================================
    // Objective
    // =========================================================================

    /**
     * @brief Set objective to minimize
     *
     * @param objective Symbolic expression to minimize
     */
    void minimize(const SymbolicScalar &objective) { opti_.minimize(objective); }

    /**
     * @brief Set objective to maximize
     *
     * @param objective Symbolic expression to maximize
     */
    void maximize(const SymbolicScalar &objective) {
        opti_.minimize(-objective); // CasADi only has minimize
    }

    // =========================================================================
    // Solve
    // =========================================================================

    /**
     * @brief Solve the optimization problem
     *
     * Uses the solver specified in options (default: IPOPT).
     *
     * @param options Solver configuration (solver, max_iter, verbose, etc.)
     * @return OptiSol containing optimized values
     * @throws std::runtime_error if selected solver unavailable or fails
     */
    OptiSol solve(const OptiOptions &options = {}) {
        // Verify solver availability
        if (!solver_available(options.solver)) {
            throw std::runtime_error(std::string("Solver '") + solver_name(options.solver) +
                                     "' is not available in this CasADi build");
        }

        casadi::Dict solver_opts;

        // Configure solver-specific options
        switch (options.solver) {
        case Solver::SNOPT:
            configure_snopt_opts(solver_opts, options);
            break;
        case Solver::QPOASES:
            configure_qpoases_opts(solver_opts, options);
            break;
        case Solver::IPOPT:
        default:
            configure_ipopt_opts(solver_opts, options);
            break;
        }

        // Common options
        if (options.jit) {
            solver_opts["jit"] = true;
            solver_opts["jit_options"] = casadi::Dict{{"flags", std::vector<std::string>{"-O3"}}};
        }

        // Set solver and solve
        opti_.solver(solver_name(options.solver), solver_opts);
        return OptiSol(opti_.solve());
    }

    /**
     * @brief Perform parametric sweep over a parameter
     *
     * Solves the optimization problem for each value in the sweep range,
     * automatically warm-starting subsequent solves from the previous solution.
     *
     * Example:
     * @code
     *   auto rho = opti.parameter(1.225);  // Air density
     *   // ... define problem using rho ...
     *
     *   auto result = opti.solve_sweep(rho, {1.0, 1.1, 1.2, 1.3});
     *   for (size_t i = 0; i < result.size(); ++i) {
     *       std::cout << "rho=" << result.param_values[i]
     *                 << " x*=" << result.solutions[i].value(x) << "\n";
     *   }
     * @endcode
     *
     * @param param Parameter to sweep (from opti.parameter())
     * @param values Vector of parameter values to try
     * @param options Solver options (applied to all solves)
     * @return SweepResult containing all solutions
     */
    SweepResult solve_sweep(const SymbolicScalar &param, const std::vector<double> &values,
                            const OptiOptions &options = {}) {
        // Verify solver availability
        if (!solver_available(options.solver)) {
            throw std::runtime_error(std::string("Solver '") + solver_name(options.solver) +
                                     "' is not available in this CasADi build");
        }

        SweepResult result;
        result.param_values = values;
        result.all_converged = true;

        // Configure solver once
        casadi::Dict solver_opts;

        switch (options.solver) {
        case Solver::SNOPT:
            configure_snopt_opts(solver_opts, options);
            break;
        case Solver::QPOASES:
            configure_qpoases_opts(solver_opts, options);
            break;
        case Solver::IPOPT:
        default:
            configure_ipopt_opts(solver_opts, options);
            // Enable warm-start for parameter sweeps
            solver_opts["ipopt.warm_start_init_point"] = "yes";
            break;
        }

        opti_.solver(solver_name(options.solver), solver_opts);

        for (double val : values) {
            // Update parameter value
            opti_.set_value(param, val);

            try {
                // Solve (CasADi automatically warm-starts from previous solution)
                auto sol = opti_.solve();
                result.solutions.emplace_back(sol);
                result.iterations.push_back(sol.stats().count("iter_count")
                                                ? static_cast<int>(sol.stats().at("iter_count"))
                                                : -1);
            } catch (const std::exception &) {
                result.all_converged = false;
                // Store empty solution marker - could also rethrow
                break;
            }
        }

        return result;
    }
    // =========================================================================
    // Derivative Helpers (for trajectory optimization)
    // =========================================================================

    /**
     * @brief Create a derivative variable constrained by integration
     *
     * Returns a new variable that is constrained to be the derivative of
     * `variable` with respect to `with_respect_to`. This is the core
     * mechanism for trajectory optimization via direct collocation.
     *
     * Example:
     * @code
     *   NumericVector time = janus::linspace(0, 1, 100);
     *   auto position = opti.variable(100, 0.0);
     *   auto velocity = opti.derivative_of(position, time, 0.0);
     *   // velocity is now constrained: d(position)/dt = velocity
     * @endcode
     *
     * @param var The quantity to differentiate
     * @param with_respect_to Independent variable (e.g., time array)
     * @param derivative_init_guess Initial guess for derivative values
     * @param method Integration method: "trapezoidal", "forward_euler", "backward_euler"
     * @return Symbolic vector representing the derivative
     */
    SymbolicVector derivative_of(const SymbolicVector &var, const NumericVector &with_respect_to,
                                 double derivative_init_guess,
                                 const std::string &method = "trapezoidal") {
        int n = static_cast<int>(var.size());
        if (n != static_cast<int>(with_respect_to.size())) {
            throw std::invalid_argument(
                "derivative_of: variable and with_respect_to must have same size");
        }

        // Create derivative variable
        auto deriv = variable(n, derivative_init_guess);

        // Add integration constraints
        constrain_derivative(deriv, var, with_respect_to, method);

        return deriv;
    }

    /**
     * @brief Constrain an existing variable to be a derivative
     *
     * Adds constraints: d(variable)/d(with_respect_to) == derivative
     *
     * @param derivative The derivative variable
     * @param var The variable being differentiated
     * @param with_respect_to Independent variable (e.g., time)
     * @param method Integration method (string or IntegrationMethod enum)
     */
    void constrain_derivative(const SymbolicVector &derivative, const SymbolicVector &var,
                              const NumericVector &with_respect_to,
                              const std::string &method = "trapezoidal") {
        constrain_derivative(derivative, var, with_respect_to, parse_integration_method(method));
    }

    /**
     * @brief Constrain an existing variable to be a derivative (enum version)
     */
    void constrain_derivative(const SymbolicVector &derivative, const SymbolicVector &var,
                              const NumericVector &with_respect_to, IntegrationMethod method) {
        int n = static_cast<int>(var.size());

        // Use janus::diff from Calculus.hpp
        NumericVector dt = janus::diff(with_respect_to);

        switch (method) {
        case IntegrationMethod::Trapezoidal:
        case IntegrationMethod::Midpoint:
            // Second-order: x[i+1] - x[i] = 0.5 * (xdot[i] + xdot[i+1]) * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = 0.5 * (derivative(i) + derivative(i + 1)) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;

        case IntegrationMethod::ForwardEuler:
            // First-order forward: x[i+1] - x[i] = xdot[i] * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = derivative(i) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;

        case IntegrationMethod::BackwardEuler:
            // First-order backward: x[i+1] - x[i] = xdot[i+1] * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = derivative(i + 1) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;
        }
    }

    // =========================================================================
    // Access
    // =========================================================================

    /**
     * @brief Access the underlying CasADi Opti object
     *
     * For advanced usage when CasADi-specific features are needed.
     */
    casadi::Opti &casadi_opti() { return opti_; }
    const casadi::Opti &casadi_opti() const { return opti_; }

    // =========================================================================
    // Category & Freezing
    // =========================================================================

    /**
     * @brief Get all variables in a category
     *
     * @param category Category name
     * @return Vector of symbolic scalars in that category (empty if not found)
     */
    std::vector<SymbolicScalar> get_category(const std::string &category) const {
        auto it = categories_.find(category);
        if (it != categories_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * @brief Get all registered category names
     * @return Vector of category names
     */
    std::vector<std::string> get_category_names() const {
        std::vector<std::string> names;
        names.reserve(categories_.size());
        for (const auto &[name, _] : categories_) {
            names.push_back(name);
        }
        return names;
    }

    /**
     * @brief Check if a category is marked for freezing
     * @param category Category name
     * @return True if the category was specified in constructor to be frozen
     */
    bool is_category_frozen(const std::string &category) const {
        return std::find(categories_to_freeze_.begin(), categories_to_freeze_.end(), category) !=
               categories_to_freeze_.end();
    }

  private:
    casadi::Opti opti_;
    std::vector<std::string> categories_to_freeze_;
    std::map<std::string, std::vector<SymbolicScalar>> categories_;

    // =========================================================================
    // Solver Configuration Helpers
    // =========================================================================

    void configure_ipopt_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        solver_opts["ipopt.max_iter"] = options.max_iter;
        solver_opts["ipopt.max_cpu_time"] = options.max_cpu_time;
        solver_opts["ipopt.tol"] = options.tol;
        solver_opts["ipopt.mu_strategy"] = options.mu_strategy;
        solver_opts["ipopt.sb"] = "yes"; // Suppress banner

        if (options.verbose) {
            solver_opts["ipopt.print_level"] = 5;
        } else {
            solver_opts["print_time"] = false;
            solver_opts["ipopt.print_level"] = 0;
        }

        if (options.detect_simple_bounds) {
            solver_opts["detect_simple_bounds"] = true;
        }
    }

    void configure_snopt_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        const auto &snopt = options.snopt_opts;

        // SNOPT option names in CasADi use underscores
        solver_opts["snopt.Major_iterations_limit"] = snopt.major_iterations_limit;
        solver_opts["snopt.Minor_iterations_limit"] = snopt.minor_iterations_limit;
        solver_opts["snopt.Major_optimality_tolerance"] = snopt.major_optimality_tolerance;
        solver_opts["snopt.Major_feasibility_tolerance"] = snopt.major_feasibility_tolerance;

        if (options.verbose) {
            solver_opts["snopt.Major_print_level"] = 1;
            solver_opts["snopt.Minor_print_level"] = 1;
        } else {
            solver_opts["print_time"] = false;
            solver_opts["snopt.Major_print_level"] = snopt.print_level;
            solver_opts["snopt.Minor_print_level"] = 0;
        }
    }

    void configure_qpoases_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        solver_opts["qpoases.nWSR"] = options.max_iter;
        solver_opts["qpoases.CPUtime"] = options.max_cpu_time;

        if (!options.verbose) {
            solver_opts["print_time"] = false;
            solver_opts["qpoases.printLevel"] = "none";
        }
    }
};

} // namespace janus
