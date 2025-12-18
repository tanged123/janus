#pragma once

#include "JanusTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <string>
#include <vector>

#include <atomic>
#include <tuple>
#include <utility>

namespace janus {

/**
 * @brief Wrapper around casadi::Function to provide Janus-native types (Eigen)
 */
class Function {
  public:
    /**
     * @brief Construct a new Function object
     *
     * @param name Name of the function
     * @param inputs Vector of symbolic arguments (scalars or matrices) describing the inputs
     * @param outputs Vector of symbolic arguments (scalars or matrices) describing the outputs
     */
    Function(const std::string &name, const std::vector<SymbolicArg> &inputs,
             const std::vector<SymbolicArg> &outputs)
        : fn_(name, convert_args(inputs), convert_args(outputs)) {}

    /**
     * @brief Constructor with auto-generated name
     * @param inputs Vector of symbolic arguments (scalars or matrices) describing the inputs
     * @param outputs Vector of symbolic arguments (scalars or matrices) describing the outputs
     */
    Function(const std::vector<SymbolicArg> &inputs, const std::vector<SymbolicArg> &outputs)
        : fn_(generate_unique_name(), convert_args(inputs), convert_args(outputs)) {}

  private:
    static std::string generate_unique_name() {
        static std::atomic<uint64_t> counter{0};
        return "janus_fn_" + std::to_string(counter.fetch_add(1));
    }

    static std::vector<SymbolicScalar> convert_args(const std::vector<SymbolicArg> &args) {
        std::vector<SymbolicScalar> ret;
        ret.reserve(args.size());
        for (const auto &arg : args) {
            ret.push_back(arg.get()); // or implicit cast
        }
        return ret;
    }

  public:
  private:
    // Helper traits to detect symbolic arguments
    template <typename T> struct is_symbolic_type : std::false_type {};
    template <> struct is_symbolic_type<casadi::MX> : std::true_type {};

    // Specialization for Eigen::Matrix
    template <typename S, int R, int C, int O, int MR, int MC>
    struct is_symbolic_type<Eigen::Matrix<S, R, C, O, MR, MC>>
        : std::conditional_t<std::is_same_v<S, casadi::MX>, std::true_type, std::false_type> {};

    // Helper to normalize input to MX
    template <typename T> casadi::MX normalize_arg(const T &val) const {
        if constexpr (std::is_arithmetic_v<std::decay_t<T>> ||
                      std::is_same_v<std::decay_t<T>, casadi::MX> ||
                      std::is_same_v<std::decay_t<T>, casadi::DM>) {
            return casadi::MX(val);
        } else {
            return janus::to_mx(val);
        }
    }

    // Helper to normalize input to DM (strictly numeric)
    template <typename T> casadi::DM normalize_arg_dm(const T &val) const {
        using DecayT = std::decay_t<T>;
        if constexpr (std::is_arithmetic_v<DecayT>) {
            return casadi::DM(double(val));
        } else if constexpr (std::is_same_v<DecayT, std::vector<double>>) {
            return casadi::DM(val);
        } else {
            // Assume Eigen numeric matrix
            casadi::DM m(val.rows(), val.cols());
            for (int i = 0; i < val.rows(); ++i)
                for (int j = 0; j < val.cols(); ++j)
                    m(i, j) = double(val(i, j));
            return m;
        }
    }

  public:
    /**
     * @brief Evaluate function with arbitrary arguments (scalars or Eigen matrices)
     * Handles both numeric (double/MatrixXd) and symbolic (MX/SymbolicMatrix) inputs.
     *
     * @tparam Args Argument types
     * @param args Variadic arguments matching the function inputs
     * @return Vector of result matrices (NumericMatrix or SymbolicMatrix)
     */
    template <typename... Args> auto operator()(Args &&...args) const {
        constexpr bool is_symbolic = (is_symbolic_type<std::decay_t<Args>>::value || ...);

        if constexpr (is_symbolic) {
            std::vector<casadi::MX> mx_args;
            mx_args.reserve(sizeof...(args));
            (mx_args.push_back(normalize_arg(std::forward<Args>(args))), ...);

            std::vector<casadi::MX> res_mx = fn_(mx_args);
            return to_eigen_vector<casadi::MX>(res_mx);
        } else {
            std::vector<casadi::DM> dm_args;
            dm_args.reserve(sizeof...(args));
            (dm_args.push_back(normalize_arg_dm(std::forward<Args>(args))), ...);

            std::vector<casadi::DM> res_dm = fn_(dm_args);
            return to_eigen_vector<double>(res_dm);
        }
    }

    /**
     * @brief Evaluate function and return first output
     */
    template <typename... Args> auto eval(Args &&...args) const {
        auto results = operator()(std::forward<Args>(args)...);
        return results[0];
    }

    // Explicit numeric vector overload
    std::vector<NumericMatrix> operator()(std::vector<double> &args) const {
        return operator()(const_cast<const std::vector<double> &>(args));
    }

    std::vector<NumericMatrix> operator()(const std::vector<double> &args) const {
        std::vector<casadi::DM> dm_args;
        dm_args.reserve(args.size());
        for (double val : args) {
            dm_args.push_back(casadi::DM(val));
        }

        auto res_dm = fn_(dm_args);
        return to_eigen_vector<double>(res_dm);
    }

    /**
     * @brief Access the underlying CasADi function
     * @return const reference to casadi::Function
     */
    const casadi::Function &casadi_function() const { return fn_; }

  private:
    casadi::Function fn_;

    template <typename Scalar, typename CasadiType>
    std::vector<JanusMatrix<Scalar>> to_eigen_vector(const std::vector<CasadiType> &dms) const {
        std::vector<JanusMatrix<Scalar>> ret;
        ret.reserve(dms.size());
        for (const auto &dm : dms) {
            // Use generic converter if possible, or manual
            if constexpr (std::is_same_v<CasadiType, casadi::MX>) {
                ret.push_back(janus::to_eigen(dm));
            } else {
                using MatType = JanusMatrix<Scalar>;
                MatType mat(dm.size1(), dm.size2());
                std::vector<double> elements = static_cast<std::vector<double>>(dm);
                for (Eigen::Index j = 0; j < mat.cols(); ++j) {
                    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
                        mat(i, j) = elements[j * mat.rows() + i];
                    }
                }
                ret.push_back(mat);
            }
        }
        return ret;
    }
};

// =============================================================================
// Lambda-Style Function Construction
// =============================================================================

namespace detail {

/**
 * @brief Helper to detect if a type is a std::tuple
 */
template <typename T> struct is_tuple : std::false_type {};
template <typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T> inline constexpr bool is_tuple_v = is_tuple<T>::value;

/**
 * @brief Convert a tuple of symbolic scalars to a vector of SymbolicArg
 */
template <typename Tuple, std::size_t... Is>
std::vector<SymbolicArg> tuple_to_outputs_impl(const Tuple &t, std::index_sequence<Is...>) {
    return {std::get<Is>(t)...};
}

template <typename... Ts> std::vector<SymbolicArg> tuple_to_outputs(const std::tuple<Ts...> &t) {
    return tuple_to_outputs_impl(t, std::index_sequence_for<Ts...>{});
}

/**
 * @brief Invoke lambda with unpacked symbolic arguments
 */
template <typename Func, std::size_t... Is>
auto invoke_with_symbols_impl(Func &&fn, const std::vector<SymbolicScalar> &syms,
                              std::index_sequence<Is...>) {
    return fn(syms[Is]...);
}

template <int NInputs, typename Func>
auto invoke_with_symbols(Func &&fn, const std::vector<SymbolicScalar> &syms) {
    return invoke_with_symbols_impl(std::forward<Func>(fn), syms,
                                    std::make_index_sequence<NInputs>{});
}

} // namespace detail

/**
 * @brief Create a Function from a lambda expression
 *
 * Automatically creates symbolic input variables and constructs a Function
 * from the lambda's return value(s).
 *
 * @code
 * // Single output
 * auto f = janus::make_function<2, 1>("f", [](auto x, auto y) {
 *     return x*x + y*y;
 * });
 *
 * // Multiple outputs (return a tuple)
 * auto g = janus::make_function<2, 2>("g", [](auto x, auto y) {
 *     return std::make_tuple(x + y, x - y);
 * });
 * @endcode
 *
 * @tparam NInputs Number of scalar inputs
 * @tparam NOutputs Number of scalar outputs (for validation)
 * @tparam Func Lambda type
 * @param name Function name
 * @param fn Lambda taking NInputs symbolic scalars
 * @return Function constructed from the lambda
 */
template <int NInputs, int NOutputs, typename Func>
Function make_function(const std::string &name, Func &&fn) {
    static_assert(NInputs > 0, "NInputs must be positive");
    static_assert(NOutputs > 0, "NOutputs must be positive");

    // Create symbolic inputs with auto-generated names
    std::vector<SymbolicScalar> inputs;
    inputs.reserve(NInputs);
    for (int i = 0; i < NInputs; ++i) {
        inputs.push_back(sym("_x" + std::to_string(i)));
    }

    // Invoke lambda with unpacked symbols
    auto result = detail::invoke_with_symbols<NInputs>(std::forward<Func>(fn), inputs);

    // Convert result to output vector
    std::vector<SymbolicArg> outputs;
    if constexpr (detail::is_tuple_v<decltype(result)>) {
        outputs = detail::tuple_to_outputs(result);
    } else {
        // Single output
        outputs = {result};
    }

    // Convert inputs to SymbolicArg vector
    std::vector<SymbolicArg> input_args(inputs.begin(), inputs.end());

    return Function(name, input_args, outputs);
}

/**
 * @brief Create a Function from a lambda with named inputs
 *
 * @code
 * auto f = janus::make_function<2>("f", {"x", "y"}, [](auto x, auto y) {
 *     return x*x + y*y;
 * });
 * @endcode
 *
 * @tparam NInputs Number of inputs (must match input_names.size())
 * @tparam Func Lambda type
 * @param name Function name
 * @param input_names Names for input variables
 * @param fn Lambda taking NInputs symbolic scalars
 * @return Function constructed from the lambda
 */
template <int NInputs, typename Func>
Function make_function(const std::string &name, const std::vector<std::string> &input_names,
                       Func &&fn) {
    static_assert(NInputs > 0, "NInputs must be positive");
    if (static_cast<int>(input_names.size()) != NInputs) {
        throw std::invalid_argument("make_function: input_names.size() must equal NInputs");
    }

    // Create symbolic inputs with provided names
    std::vector<SymbolicScalar> inputs;
    inputs.reserve(NInputs);
    for (const auto &iname : input_names) {
        inputs.push_back(sym(iname));
    }

    // Invoke lambda with unpacked symbols
    auto result = detail::invoke_with_symbols<NInputs>(std::forward<Func>(fn), inputs);

    // Convert result to output vector
    std::vector<SymbolicArg> outputs;
    if constexpr (detail::is_tuple_v<decltype(result)>) {
        outputs = detail::tuple_to_outputs(result);
    } else {
        outputs = {result};
    }

    // Convert inputs to SymbolicArg vector
    std::vector<SymbolicArg> input_args(inputs.begin(), inputs.end());

    return Function(name, input_args, outputs);
}

} // namespace janus
