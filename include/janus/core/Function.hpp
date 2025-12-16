#pragma once

#include "JanusTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <string>
#include <vector>

#include <atomic>

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

} // namespace janus
