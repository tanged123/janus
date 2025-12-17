#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <limits>

namespace Eigen {
/**
 * @brief NumTraits specialization for casadi::MX
 * Required for Eigen's operator<< and other scalar-dependent operations.
 */
template <> struct NumTraits<casadi::MX> : GenericNumTraits<casadi::MX> {
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };

    static inline int digits10() { return std::numeric_limits<double>::digits10; }
    static inline int min_exponent() { return std::numeric_limits<double>::min_exponent; }
    static inline int max_exponent() { return std::numeric_limits<double>::max_exponent; }
    static inline casadi::MX epsilon() { return std::numeric_limits<double>::epsilon(); }
    static inline casadi::MX dummy_precision() { return 1e-5; }
    static inline casadi::MX highest() { return std::numeric_limits<double>::max(); }
    static inline casadi::MX lowest() { return std::numeric_limits<double>::lowest(); }
};
} // namespace Eigen

namespace janus {

// --- Matrix Types ---
/**
 * Universal Matrix Template
 *
 * Provides a unified matrix type for both numeric and symbolic backends.
 * Uses dynamic sizing for maximum flexibility.
 */
template <typename Scalar>
using JanusMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar> using JanusVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// --- Fixed-Size Types ---
/**
 * Fixed-size vectors and matrices for performance-critical code.
 * Stack-allocated, no heap overhead, SIMD-friendly.
 */
template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar> using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

template <typename Scalar> using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
template <typename Scalar> using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar> using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

// Numeric Backend
using NumericScalar = double;
using NumericMatrix = JanusMatrix<NumericScalar>; // Equivalent to Eigen::MatrixXd
using NumericVector = JanusVector<NumericScalar>; // Equivalent to Eigen::VectorXd

// Symbolic Backend
using SymbolicScalar = casadi::MX;
using SymbolicMatrix = JanusMatrix<SymbolicScalar>;
using SymbolicVector = JanusVector<SymbolicScalar>;

// --- Symbolic Variable Creation ---

/**
 * @brief Create a named symbolic scalar variable
 * @param name Name of the variable
 * @return SymbolicScalar (casadi::MX)
 */
inline SymbolicScalar sym(const std::string &name) { return casadi::MX::sym(name); }

/**
 * @brief Create a named symbolic matrix variable
 * @param name Name of the variable
 * @param rows Number of rows
 * @param cols Number of columns (default 1)
 * @return SymbolicScalar (casadi::MX) representing a matrix
 */
inline SymbolicScalar sym(const std::string &name, int rows, int cols = 1) {
    return casadi::MX::sym(name, rows, cols);
}

/**
 * @brief Create a named symbolic vector (returns SymbolicVector)
 *
 * Convenience overload that returns SymbolicVector (Eigen container) directly.
 * This is syntactic sugar for `sym_vec(name, size)`.
 *
 * @code
 * auto x = janus::sym_vector("x", 3);  // Returns SymbolicVector
 * @endcode
 *
 * @param name Name of the variable
 * @param size Number of elements
 * @return SymbolicVector with MX elements
 */
inline SymbolicVector sym_vector(const std::string &name, int size) {
    casadi::MX mx = casadi::MX::sym(name, size, 1);
    SymbolicVector v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = mx(i);
    }
    return v;
}

/**
 * @brief Create a symbolic vector as SymbolicVector (Eigen container of MX)
 *
 * Creates a single CasADi MX vector internally, then unpacks it into an
 * Eigen container. This preserves the connection to the original symbolic
 * primitive, which is required for correct jacobian computation.
 *
 * @code
 * auto state = janus::sym_vec("state", 3);    // Returns SymbolicVector
 * auto dydt = my_ode(state, theta);           // Works with JanusVector<Scalar>
 *
 * // For jacobian, use to_mx to get a compatible representation
 * auto jac = janus::jacobian({janus::to_mx(dydt)}, {janus::to_mx(state)});
 * @endcode
 *
 * @param name Name of the variable
 * @param size Number of elements
 * @return SymbolicVector with MX elements from single underlying vector
 */
inline SymbolicVector sym_vec(const std::string &name, int size) {
    // Delegate to sym_vector - they are equivalent
    return sym_vector(name, size);
}

/**
 * @brief Create symbolic vector and return both SymbolicVector and underlying MX
 *
 * Use this when you need both:
 * - The SymbolicVector for templated functions
 * - The original MX for janus::Function or janus::jacobian
 *
 * @code
 * auto [state_vec, state_mx] = janus::sym_vec_pair("state", 3);
 * auto dydt = my_ode(state_vec, theta);
 * auto jac = janus::jacobian({janus::to_mx(dydt)}, {state_mx, theta});
 * @endcode
 */
inline std::pair<SymbolicVector, SymbolicScalar> sym_vec_pair(const std::string &name, int size) {
    casadi::MX mx = casadi::MX::sym(name, size, 1);
    SymbolicVector v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = mx(i);
    }
    return {v, mx};
}

/**
 * @brief Get the underlying MX representation of a SymbolicVector
 *
 * This packs an Eigen container of MX elements back into a single CasADi MX
 * for use with janus::Function or janus::jacobian.
 *
 * @param v SymbolicVector to convert
 * @return Single casadi::MX representing the vector
 */
inline SymbolicScalar as_mx(const SymbolicVector &v) {
    casadi::MX m(v.size(), 1);
    for (int i = 0; i < v.size(); ++i) {
        m(i) = v(i);
    }
    return m;
}

// --- Universal Conversion Helpers ---

/**
 * @brief Convert Eigen matrix of MX (or numeric) to CasADi MX
 * @tparam Derived Eigen matrix type
 * @param e Input Eigen matrix
 * @return CasADi MX (dense)
 */
template <typename Derived> casadi::MX to_mx(const Eigen::MatrixBase<Derived> &e) {
    if (e.size() == 0)
        return casadi::MX(e.rows(), e.cols());

    // Create an MX of correct shape
    casadi::MX m(e.rows(), e.cols());
    // Fill it element-wise
    for (Eigen::Index i = 0; i < e.rows(); ++i) {
        for (Eigen::Index j = 0; j < e.cols(); ++j) {
            if constexpr (std::is_same_v<typename Derived::Scalar, casadi::MX>) {
                m(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
            } else {
                m(static_cast<int>(i), static_cast<int>(j)) = casadi::MX(e(i, j));
            }
        }
    }
    return m;
}

/**
 * @brief Convert CasADi MX to Eigen matrix of MX
 * @param m Input CasADi MX
 * @return Eigen matrix (dynamic size)
 */
inline Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> to_eigen(const casadi::MX &m) {
    Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> e(m.size1(), m.size2());
    for (int i = 0; i < m.size1(); ++i) {
        for (int j = 0; j < m.size2(); ++j) {
            e(i, j) = m(i, j);
        }
    }
    return e;
}

/**
 * @brief Convert CasADi MX vector to SymbolicVector (Eigen container of MX)
 * @param m Input CasADi MX (column vector)
 * @return SymbolicVector (Eigen::Matrix<casadi::MX, Dynamic, 1>)
 */
inline SymbolicVector as_vector(const casadi::MX &m) {
    SymbolicVector v(m.size1());
    for (int i = 0; i < m.size1(); ++i) {
        v(i) = m(i);
    }
    return v;
}

// Backwards compatibility alias
inline SymbolicVector to_eigen_vec(const casadi::MX &m) { return as_vector(m); }

/**
 * @brief Universal Symbolic Argument Wrapper
 *
 * Allows automatic flattening of Eigen matrices (SymbolicMatrix) and Scalars (SymbolicScalar)
 * into a single casadi::MX type for function definitions and Jacobians.
 * Enables mixed initializer lists: {scalar_sym, matrix_sym}.
 */
class SymbolicArg {
  public:
    // From Scalar (MX)
    /**
     * @brief Construct from single symbolic scalar (MX)
     */
    SymbolicArg(const SymbolicScalar &s) : mx_(s) {}

    // From Matrix (Eigen<MX>)
    /**
     * @brief Construct from Eigen matrix of symbolic scalars
     * Flattens the matrix into a CasADi DM/MX structure if needed or fills an MX.
     */
    template <typename Derived> SymbolicArg(const Eigen::MatrixBase<Derived> &e) {
        if (e.size() == 0) {
            mx_ = casadi::MX(e.rows(), e.cols());
            return;
        }
        mx_ = casadi::MX(e.rows(), e.cols());
        for (Eigen::Index i = 0; i < e.rows(); ++i) {
            for (Eigen::Index j = 0; j < e.cols(); ++j) {
                mx_(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
            }
        }
    }

    // Implicit conversion to MX
    /**
     * @brief Implicit conversion to CasADi MX
     */
    operator SymbolicScalar() const { return mx_; }

    /**
     * @brief Get underlying CasADi MX object
     */
    SymbolicScalar get() const { return mx_; }

  private:
    SymbolicScalar mx_;
};

} // namespace janus
