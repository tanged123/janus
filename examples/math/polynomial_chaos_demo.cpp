/**
 * @file polynomial_chaos_demo.cpp
 * @brief Demonstrate polynomial chaos basis construction, projection, and regression.
 *
 * This example shows:
 * 1. Multidimensional basis construction and multi-index ordering.
 * 2. Projection-based coefficient recovery for a scalar response.
 * 3. Symbolic mean/variance gradients through fitted PCE coefficients.
 */

#include <iomanip>
#include <iostream>
#include <janus/janus.hpp>
#include <string>
#include <vector>

using namespace janus;

namespace {

NumericMatrix make_sample_matrix(const NumericVector &samples) {
    NumericMatrix out(samples.size(), 1);
    out.col(0) = samples;
    return out;
}

void print_multi_index(const std::vector<int> &multi_index) {
    std::cout << "[";
    for (std::size_t i = 0; i < multi_index.size(); ++i) {
        if (i != 0u) {
            std::cout << ", ";
        }
        std::cout << multi_index[i];
    }
    std::cout << "]";
}

double max_abs_error(const NumericMatrix &a, const NumericMatrix &b) {
    return (a - b).array().abs().maxCoeff();
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Polynomial Chaos Demo ===\n\n";

    {
        std::cout << "Case 1: total-order basis construction in two stochastic dimensions\n";

        PolynomialChaosBasis basis({legendre_dimension(), hermite_dimension()}, 2);
        std::cout << "  dimension   = " << basis.dimension() << "\n";
        std::cout << "  order       = " << basis.order() << "\n";
        std::cout << "  term count  = " << basis.size() << "\n";
        std::cout << "  normalized  = " << (basis.normalized() ? "true" : "false") << "\n";
        std::cout << "  terms:\n";
        for (std::size_t i = 0; i < basis.terms().size(); ++i) {
            std::cout << "    psi[" << i << "] multi-index = ";
            print_multi_index(basis.terms()[i].multi_index);
            std::cout << ", squared norm = " << basis.terms()[i].squared_norm << "\n";
        }

        NumericVector point(2);
        point << 0.25, -0.4;
        NumericVector basis_values = basis.evaluate(point);
        std::cout << "  basis values at xi = [0.25, -0.4]^T:\n"
                  << basis_values.transpose() << "\n\n";
    }

    {
        std::cout << "Case 2: projection and regression with symbolic sample values\n";

        PolynomialChaosBasis basis({legendre_dimension()}, 2);

        const NumericVector projection_nodes = lgl_nodes(5);
        const NumericVector projection_weights = 0.5 * lgl_weights(5, projection_nodes);
        const NumericMatrix projection_samples = make_sample_matrix(projection_nodes);

        NumericVector regression_nodes(6);
        regression_nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;
        const NumericMatrix regression_samples = make_sample_matrix(regression_nodes);

        SymbolicScalar a = sym("a");

        auto make_symbolic_samples = [&](const NumericVector &nodes) {
            SymbolicVector values(nodes.size());
            for (Eigen::Index i = 0; i < nodes.size(); ++i) {
                const double xi = nodes(i);
                values(i) = (1.0 + a) * pce_polynomial(legendre_dimension(), 0, xi) +
                            0.5 * pce_polynomial(legendre_dimension(), 1, xi) +
                            (0.25 * a) * pce_polynomial(legendre_dimension(), 2, xi);
            }
            return values;
        };

        SymbolicVector coeffs_projection = pce_projection_coefficients(
            basis, projection_samples, projection_weights, make_symbolic_samples(projection_nodes));
        SymbolicVector coeffs_regression = pce_regression_coefficients(
            basis, regression_samples, make_symbolic_samples(regression_nodes), 0.0);

        SymbolicScalar mean = pce_mean(coeffs_projection);
        SymbolicScalar variance = pce_variance(basis, coeffs_projection);

        Function summary("pce_summary", {a},
                         {to_mx(coeffs_projection), to_mx(coeffs_regression), mean, variance,
                          jacobian(mean, a), jacobian(variance, a)});

        const auto outputs = summary(2.0);
        const NumericMatrix coeffs_projection_val = outputs[0];
        const NumericMatrix coeffs_regression_val = outputs[1];

        std::cout << "  projection coefficients at a = 2:\n"
                  << coeffs_projection_val.transpose() << "\n";
        std::cout << "  regression coefficients at a = 2:\n"
                  << coeffs_regression_val.transpose() << "\n";
        std::cout << "  max coefficient mismatch = "
                  << max_abs_error(coeffs_projection_val, coeffs_regression_val) << "\n";
        std::cout << "  mean(a=2)      = " << outputs[2](0, 0) << "\n";
        std::cout << "  variance(a=2)  = " << outputs[3](0, 0) << "\n";
        std::cout << "  dmean/da       = " << outputs[4](0, 0) << "\n";
        std::cout << "  dvariance/da   = " << outputs[5](0, 0) << "\n\n";
    }

    std::cout << "Takeaway:\n";
    std::cout << "  - PolynomialChaosBasis builds the multi-index set and evaluates the basis\n";
    std::cout << "  - pce_projection_coefficients(...) and pce_regression_coefficients(...)\n";
    std::cout << "    recover coefficients from weighted quadrature samples or plain collocation\n";
    std::cout << "  - because the fitted coefficients remain symbolic, pce_mean(...) and\n";
    std::cout << "    pce_variance(...) can be differentiated directly with CasADi/Janus\n";

    return 0;
}
