#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <janus/janus.hpp>
#include <limits>

namespace {

std::pair<casadi::Function, casadi::Function>
make_identity_root_functions(const std::string &name) {
    casadi::MX x = casadi::MX::sym(name + "_x");
    casadi::MX residual = x;

    return {
        casadi::Function(name + "_residual", {x}, {residual}),
        casadi::Function(name + "_jacobian", {x}, {casadi::MX::jacobian(residual, x)}),
    };
}

janus::detail::NumericState make_scalar_state(double x_value, double residual_value,
                                              double jacobian_value) {
    janus::detail::NumericState state;
    state.x = Eigen::VectorXd::Constant(1, x_value);
    state.residual = Eigen::VectorXd::Constant(1, residual_value);
    state.jacobian = Eigen::MatrixXd::Constant(1, 1, jacobian_value);
    state.residual_norm = std::abs(residual_value);
    state.merit = 0.5 * residual_value * residual_value;
    return state;
}

} // namespace

// ======================================================================
// FiniteDifference.hpp Coverage
// ======================================================================

TEST(FiniteDifferenceCoverage, ErrorChecks) {
    janus::JanusVector<double> x(3);
    x << 0, 1, 2;

    // Invalid degree
    EXPECT_THROW(janus::finite_difference_coefficients(x, 0.0, -1), janus::InvalidArgument);

    // Too few points
    // Degree 3 requires 4 points
    EXPECT_THROW(janus::finite_difference_coefficients(x, 0.0, 3), janus::InvalidArgument);
}

// ======================================================================
// Integrate.hpp Coverage
// ======================================================================

TEST(IntegrateCoverage, SymbolicLambdaError) {
    auto x = janus::sym("x");
    // Use quad(func, a, b) signature with Symbolic arguments to trigger the template runtime error
    janus::SymbolicScalar a(0.0), b(1.0);
    EXPECT_THROW(janus::quad([](janus::SymbolicScalar s) { return s; }, a, b),
                 janus::IntegrationError);
}

// ======================================================================
// Spacing.hpp Coverage
// ======================================================================

TEST(SpacingCoverage, InvalidN) {
    // n < 1 should throw for all
    EXPECT_THROW(janus::linspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::cosine_spacing(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::sinspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::logspace(0.0, 1.0, 0), janus::InvalidArgument);
    EXPECT_THROW(janus::geomspace(1.0, 10.0, 0), janus::InvalidArgument);
}

// ======================================================================
// Linalg.hpp Coverage
// ======================================================================

TEST(LinalgCoverage, CrossError) {
    janus::JanusVector<double> a(2);
    a << 1, 2;
    janus::JanusVector<double> b(3);
    b << 1, 2, 3;

    EXPECT_THROW(janus::cross(a, b), janus::InvalidArgument);
}

TEST(RootFindingCoverage, HelperFunctionsAndValidationErrors) {
    EXPECT_EQ(janus::detail::method_name(janus::RootSolveMethod::None), "none");
    EXPECT_EQ(janus::detail::method_name(janus::RootSolveMethod::TrustRegionNewton),
              "trust-region Newton");
    EXPECT_EQ(janus::detail::method_name(janus::RootSolveMethod::LineSearchNewton),
              "line-search Newton");
    EXPECT_EQ(janus::detail::method_name(janus::RootSolveMethod::QuasiNewtonBroyden),
              "quasi-Newton Broyden");
    EXPECT_EQ(janus::detail::method_name(janus::RootSolveMethod::PseudoTransientContinuation),
              "pseudo-transient continuation");
    EXPECT_THROW(janus::detail::method_name(static_cast<janus::RootSolveMethod>(-1)),
                 janus::InvalidArgument);

    EXPECT_EQ(janus::detail::strategy_to_method(janus::RootSolveStrategy::Auto),
              janus::RootSolveMethod::None);
    EXPECT_EQ(janus::detail::strategy_to_method(janus::RootSolveStrategy::TrustRegionNewton),
              janus::RootSolveMethod::TrustRegionNewton);
    EXPECT_EQ(janus::detail::strategy_to_method(janus::RootSolveStrategy::LineSearchNewton),
              janus::RootSolveMethod::LineSearchNewton);
    EXPECT_EQ(janus::detail::strategy_to_method(janus::RootSolveStrategy::QuasiNewtonBroyden),
              janus::RootSolveMethod::QuasiNewtonBroyden);
    EXPECT_EQ(
        janus::detail::strategy_to_method(janus::RootSolveStrategy::PseudoTransientContinuation),
        janus::RootSolveMethod::PseudoTransientContinuation);
    EXPECT_THROW(janus::detail::strategy_to_method(static_cast<janus::RootSolveStrategy>(-1)),
                 janus::InvalidArgument);

    const std::string name0 = janus::detail::unique_name("rf");
    const std::string name1 = janus::detail::unique_name("rf");
    EXPECT_NE(name0, name1);
    EXPECT_EQ(name0.rfind("rf_", 0), 0u);

    janus::RootFinderOptions opts;
    opts.verbose = true;
    opts.linear_solver_options["pivot"] = 1;
    const casadi::Dict dict = janus::detail::opts_to_dict(opts);
    EXPECT_NE(dict.find("linear_solver_options"), dict.end());
    EXPECT_NE(dict.find("verbose"), dict.end());
    EXPECT_NE(dict.find("print_in"), dict.end());
    EXPECT_NE(dict.find("print_out"), dict.end());

    Eigen::VectorXd x(2);
    x << 1.0, -2.0;
    const casadi::DM x_dm = janus::detail::vector_to_dm(x);
    EXPECT_TRUE(janus::detail::dm_to_vector(x_dm).isApprox(x, 1e-12));

    casadi::DM M(2, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(1, 0) = 3.0;
    M(1, 1) = 4.0;
    janus::NumericMatrix M_expected(2, 2);
    M_expected << 1.0, 2.0, 3.0, 4.0;
    EXPECT_TRUE(janus::detail::dm_to_matrix(M).isApprox(M_expected, 1e-12));

    testing::internal::CaptureStdout();
    janus::detail::maybe_log(opts, "coverage");
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("coverage"), std::string::npos);

    janus::RootFinderOptions bad_opts;
    bad_opts.abstol = 0.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.abstolStep = 0.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.max_iter = 0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.trust_region_initial_damping = 0.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.trust_region_damping_increase = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.trust_region_damping_decrease = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.line_search_contraction = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.line_search_sufficient_decrease = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.max_backtracking_steps = 0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.broyden_jacobian_refresh = -1;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.pseudo_transient_dt0 = 0.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.pseudo_transient_dt_growth = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    bad_opts = janus::RootFinderOptions();
    bad_opts.pseudo_transient_dt0 = 2.0;
    bad_opts.pseudo_transient_dt_max = 1.0;
    EXPECT_THROW(janus::detail::validate_root_options(bad_opts, "ctx"), janus::InvalidArgument);

    EXPECT_TRUE(janus::detail::solve_linear_system(M_expected, x)
                    .isApprox(M_expected.colPivHouseholderQr().solve(x), 1e-12));
}

TEST(RootFindingCoverage, ProblemShapeValidationAndBranchErrors) {
    casadi::MX x = casadi::MX::sym("x");
    casadi::MX y = casadi::MX::sym("y");
    casadi::MX x_row = casadi::MX::sym("x_row", 1, 2);
    casadi::MX x_col = casadi::MX::sym("x_col", 2, 1);

    const casadi::Function bad_arity("bad_arity", {x, y}, {x});
    EXPECT_THROW(janus::detail::validate_root_problem(bad_arity, "ctx"), janus::InvalidArgument);

    const casadi::Function bad_input_shape("bad_input_shape", {x_row}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(janus::detail::validate_root_problem(bad_input_shape, "ctx"),
                 janus::InvalidArgument);

    const casadi::Function bad_output_shape("bad_output_shape", {x_col}, {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(janus::detail::validate_root_problem(bad_output_shape, "ctx"),
                 janus::InvalidArgument);

    const casadi::Function bad_dimensions("bad_dimensions", {x_col}, {casadi::MX::zeros(1, 1)});
    EXPECT_THROW(janus::detail::validate_root_problem(bad_dimensions, "ctx"),
                 janus::InvalidArgument);

    EXPECT_EQ(janus::detail::implicit_function_name(bad_dimensions), "bad_dimensions_implicit");

    janus::ImplicitFunctionOptions implicit_opts;
    Eigen::VectorXd x_guess = Eigen::VectorXd::Zero(2);

    implicit_opts.implicit_input_index = 2;
    EXPECT_THROW(janus::detail::validate_implicit_problem(bad_dimensions, x_guess, implicit_opts),
                 janus::InvalidArgument);

    implicit_opts = janus::ImplicitFunctionOptions();
    implicit_opts.implicit_output_index = 1;
    EXPECT_THROW(janus::detail::validate_implicit_problem(bad_dimensions, x_guess, implicit_opts),
                 janus::InvalidArgument);

    const casadi::Function bad_impl_input("bad_impl_input", {x_row, y}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(janus::detail::validate_implicit_problem(bad_impl_input, x_guess,
                                                          janus::ImplicitFunctionOptions()),
                 janus::InvalidArgument);

    const casadi::Function bad_impl_output("bad_impl_output", {x_col, y},
                                           {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(janus::detail::validate_implicit_problem(bad_impl_output, x_guess,
                                                          janus::ImplicitFunctionOptions()),
                 janus::InvalidArgument);

    EXPECT_THROW(janus::detail::validate_implicit_problem(bad_dimensions, x_guess,
                                                          janus::ImplicitFunctionOptions()),
                 janus::InvalidArgument);

    Eigen::VectorXd short_guess = Eigen::VectorXd::Zero(1);
    const casadi::Function good_impl("good_impl", {x_col, y}, {x_col});
    EXPECT_THROW(janus::detail::validate_implicit_problem(good_impl, short_guess,
                                                          janus::ImplicitFunctionOptions()),
                 janus::InvalidArgument);

    const casadi::Function bad_residual_eval("bad_residual_eval", {x}, {x, x});
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 1.0);
    EXPECT_THROW(janus::detail::evaluate_residual_only(bad_residual_eval, x0), janus::JanusError);

    const casadi::Function good_residual_eval("good_residual_eval", {x}, {x});
    const casadi::Function bad_jacobian_eval("bad_jacobian_eval", {x}, {x, x});
    EXPECT_THROW(
        janus::detail::evaluate_state(good_residual_eval, bad_jacobian_eval, x0, "rootfinder"),
        janus::JanusError);

    const casadi::Function nan_jacobian_eval(
        "nan_jacobian_eval", {x}, {casadi::MX(std::numeric_limits<double>::quiet_NaN())});
    EXPECT_THROW(
        janus::detail::evaluate_state(good_residual_eval, nan_jacobian_eval, x0, "rootfinder"),
        janus::JanusError);

    auto root_fns = make_identity_root_functions("root_stage");
    janus::RootFinderOptions opts;
    opts.abstol = 1e-12;
    opts.abstolStep = 1e-12;

    const auto converged_state = janus::detail::evaluate_state(
        root_fns.first, root_fns.second, Eigen::VectorXd::Zero(1), "root_stage");
    const auto tr_converged = janus::detail::solve_trust_region(root_fns.first, root_fns.second,
                                                                converged_state, opts, 5);
    EXPECT_TRUE(tr_converged.converged);
    EXPECT_EQ(tr_converged.message, "initial iterate satisfies tolerance");

    const auto nonconverged_state = janus::detail::evaluate_state(
        root_fns.first, root_fns.second, Eigen::VectorXd::Ones(1), "root_stage");
    const auto ls_none = janus::detail::solve_line_search(root_fns.first, root_fns.second,
                                                          nonconverged_state, opts, 0);
    EXPECT_FALSE(ls_none.converged);
    EXPECT_EQ(ls_none.message, "no line-search iterations remaining");

    const auto br_none =
        janus::detail::solve_broyden(root_fns.first, root_fns.second, nonconverged_state, opts, 0);
    EXPECT_FALSE(br_none.converged);
    EXPECT_EQ(br_none.message, "no Broyden iterations remaining");

    const auto pt_none = janus::detail::solve_pseudo_transient(root_fns.first, root_fns.second,
                                                               nonconverged_state, opts, 0);
    EXPECT_FALSE(pt_none.converged);
    EXPECT_EQ(pt_none.message, "no pseudo-transient iterations remaining");

    const auto nan_state = make_scalar_state(0.0, 1.0, std::numeric_limits<double>::quiet_NaN());
    const auto tr_nan =
        janus::detail::solve_trust_region(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(tr_nan.message.find("non-finite"), std::string::npos);

    const auto ls_nan =
        janus::detail::solve_line_search(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(ls_nan.message.find("non-finite"), std::string::npos);

    const auto br_nan =
        janus::detail::solve_broyden(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(br_nan.message.find("non-finite"), std::string::npos);

    const auto pt_tiny = janus::detail::solve_pseudo_transient(
        root_fns.first, root_fns.second, make_scalar_state(0.0, 1.0, 1e20), opts, 1);
    EXPECT_NE(pt_tiny.message.find("tiny step"), std::string::npos);

    auto x_sym = janus::sym("x_root");
    janus::Function f_identity("f_identity", {x_sym}, {x_sym});
    janus::NewtonSolver solver(f_identity);

    Eigen::VectorXd bad_guess = Eigen::VectorXd::Zero(2);
    EXPECT_THROW(solver.solve(bad_guess), janus::InvalidArgument);

    Eigen::VectorXd zero_guess = Eigen::VectorXd::Zero(1);
    const auto solved = solver.solve(zero_guess);
    EXPECT_TRUE(solved.converged);
    EXPECT_NE(solved.message.find("Initial guess"), std::string::npos);

    janus::Function f_no_root("f_no_root", {x_sym}, {x_sym * x_sym + 1.0});
    janus::RootFinderOptions fail_opts;
    fail_opts.max_iter = 2;
    fail_opts.strategy = janus::RootSolveStrategy::TrustRegionNewton;
    const auto failed = janus::rootfinder<double>(f_no_root, zero_guess, fail_opts);
    EXPECT_FALSE(failed.converged);
    EXPECT_EQ(failed.method, janus::RootSolveMethod::TrustRegionNewton);
    EXPECT_NE(failed.message.find("Failed to converge"), std::string::npos);

    auto p_sym = janus::sym("p_root");
    auto x_row_sym = janus::sym("x_row_root", 1, 2);
    auto x_col_sym = janus::sym("x_col_root", 2, 1);
    janus::Function impl_bad_input("impl_bad_input", {x_row_sym, p_sym}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(janus::create_implicit_function(impl_bad_input, x_guess), janus::InvalidArgument);

    janus::Function impl_bad_output("impl_bad_output", {x_col_sym, p_sym},
                                    {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(janus::create_implicit_function(impl_bad_output, x_guess), janus::InvalidArgument);

    janus::Function impl_bad_dim("impl_bad_dim", {x_col_sym, p_sym}, {casadi::MX::zeros(1, 1)});
    EXPECT_THROW(janus::create_implicit_function(impl_bad_dim, x_guess), janus::InvalidArgument);

    janus::Function impl_good("impl_good", {x_col_sym, p_sym}, {x_col_sym});
    EXPECT_THROW(janus::create_implicit_function(impl_good, short_guess), janus::InvalidArgument);
}

TEST(QuadratureCoverage, HelperFunctionsAndValidationErrors) {
    EXPECT_THROW(janus::detail::validate_order(0, "quad"), janus::InvalidArgument);
    EXPECT_THROW(janus::detail::validate_level(0, "quad"), janus::InvalidArgument);

    EXPECT_DOUBLE_EQ(janus::detail::binomial(5, 2), 10.0);
    EXPECT_DOUBLE_EQ(janus::detail::binomial(4, -1), 0.0);
    EXPECT_DOUBLE_EQ(janus::detail::binomial(4, 5), 0.0);
    EXPECT_EQ(janus::detail::clenshaw_curtis_order_from_level(1), 1);
    EXPECT_EQ(janus::detail::clenshaw_curtis_order_from_level(3), 5);
    EXPECT_EQ(janus::detail::gauss_order_from_level(4), 4);
    EXPECT_TRUE(janus::detail::is_bounded_support(janus::legendre_dimension()));
    EXPECT_TRUE(janus::detail::is_bounded_support(janus::jacobi_dimension(1.0, 2.0)));
    EXPECT_FALSE(janus::detail::is_bounded_support(janus::hermite_dimension()));

    EXPECT_DOUBLE_EQ(janus::detail::standard_normal_moment(3), 0.0);
    EXPECT_NEAR(janus::detail::standard_normal_moment(4), 3.0, 1e-12);
    EXPECT_NEAR(janus::detail::shifted_beta_moment(1, 1.0, 2.0), 0.2, 1e-12);

    EXPECT_THROW(janus::detail::probability_moment(janus::legendre_dimension(), -1),
                 janus::InvalidArgument);
    EXPECT_NEAR(janus::detail::probability_moment(janus::hermite_dimension(), 4), 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(janus::detail::probability_moment(janus::legendre_dimension(), 3), 0.0);
    EXPECT_GT(janus::detail::probability_moment(janus::jacobi_dimension(1.0, 2.0), 2), 0.0);
    EXPECT_NEAR(janus::detail::probability_moment(janus::laguerre_dimension(), 3), 6.0, 1e-12);

    janus::PolynomialChaosDimension bad_dimension = janus::legendre_dimension();
    bad_dimension.family = static_cast<janus::PolynomialChaosFamily>(999);
    EXPECT_THROW(janus::detail::probability_moment(bad_dimension, 0), janus::InvalidArgument);

    janus::NumericVector singleton_node(1);
    singleton_node << 0.0;
    EXPECT_THROW(janus::detail::clenshaw_curtis_probability_weights(janus::hermite_dimension(),
                                                                    singleton_node),
                 janus::InvalidArgument);
    const auto singleton_weights = janus::detail::clenshaw_curtis_probability_weights(
        janus::jacobi_dimension(1.0, 2.0), singleton_node);
    EXPECT_DOUBLE_EQ(singleton_weights(0), 1.0);

    const janus::NumericVector jacobi_nodes = janus::cgl_nodes(3);
    const janus::NumericVector jacobi_weights = janus::detail::clenshaw_curtis_probability_weights(
        janus::jacobi_dimension(1.0, 2.0), jacobi_nodes);
    EXPECT_NEAR(jacobi_weights.sum(), 1.0, 1e-12);
}

TEST(QuadratureCoverage, RuleConstructionAndSparseGridValidation) {
    const auto hermite_rule = janus::detail::gauss_rule(janus::hermite_dimension(), 3, 2);
    const auto jacobi_rule = janus::detail::gauss_rule(janus::jacobi_dimension(1.0, 2.0), 3, 2);
    const auto laguerre_rule = janus::detail::gauss_rule(janus::laguerre_dimension(), 3, 2);
    EXPECT_EQ(hermite_rule.nodes.size(), 3);
    EXPECT_EQ(jacobi_rule.nodes.size(), 3);
    EXPECT_EQ(laguerre_rule.nodes.size(), 3);

    janus::PolynomialChaosDimension bad_dimension = janus::legendre_dimension();
    bad_dimension.family = static_cast<janus::PolynomialChaosFamily>(999);
    EXPECT_THROW(janus::detail::gauss_rule(bad_dimension, 2, 1), janus::InvalidArgument);

    EXPECT_THROW(janus::detail::gauss_kronrod_rule(janus::hermite_dimension(), 7, 1),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::detail::gauss_kronrod_rule(janus::legendre_dimension(), 9, 1),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::detail::clenshaw_curtis_rule(janus::hermite_dimension(), 3, 1),
                 janus::InvalidArgument);

    const auto cc_rule = janus::detail::clenshaw_curtis_rule(janus::legendre_dimension(), 1, 1);
    ASSERT_EQ(cc_rule.nodes.size(), 1);
    EXPECT_DOUBLE_EQ(cc_rule.nodes(0), 0.0);
    EXPECT_DOUBLE_EQ(cc_rule.weights(0), 1.0);

    const auto compositions = janus::detail::positive_compositions(3, 5);
    ASSERT_FALSE(compositions.empty());
    for (const auto &composition : compositions) {
        EXPECT_EQ(std::accumulate(composition.begin(), composition.end(), 0), 5);
    }

    janus::NumericVector point(2);
    point << 0.25, -0.25;
    EXPECT_THROW(janus::detail::sample_key(point, 0.0), janus::InvalidArgument);
    EXPECT_FALSE(janus::detail::sample_key(point, 1e-3).empty());

    EXPECT_THROW(
        janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3,
                                          static_cast<janus::StochasticQuadratureRule>(999)),
        janus::InvalidArgument);
    EXPECT_THROW(
        janus::stochastic_quadrature_level(janus::legendre_dimension(), 3,
                                           janus::StochasticQuadratureRule::GaussKronrod15),
        janus::InvalidArgument);

    EXPECT_THROW(janus::tensor_product_quadrature({}), janus::InvalidArgument);

    janus::UnivariateQuadratureRule bad_rule;
    bad_rule.nodes.resize(0);
    bad_rule.weights.resize(0);
    EXPECT_THROW(janus::tensor_product_quadrature({bad_rule}), janus::InvalidArgument);

    bad_rule.nodes.resize(1);
    bad_rule.nodes << 0.0;
    bad_rule.weights.resize(2);
    bad_rule.weights << 1.0, 0.0;
    EXPECT_THROW(janus::tensor_product_quadrature({bad_rule}), janus::InvalidArgument);

    janus::SmolyakQuadratureOptions smolyak_opts;
    smolyak_opts.merge_tolerance = 0.0;
    EXPECT_THROW(janus::smolyak_sparse_grid({janus::legendre_dimension()}, 1, smolyak_opts),
                 janus::InvalidArgument);

    smolyak_opts = janus::SmolyakQuadratureOptions();
    smolyak_opts.zero_weight_tolerance = -1.0;
    EXPECT_THROW(janus::smolyak_sparse_grid({janus::legendre_dimension()}, 1, smolyak_opts),
                 janus::InvalidArgument);

    smolyak_opts = janus::SmolyakQuadratureOptions();
    smolyak_opts.zero_weight_tolerance = 2.0;
    EXPECT_THROW(janus::smolyak_sparse_grid({janus::legendre_dimension()}, 1, smolyak_opts),
                 janus::RuntimeError);

    janus::PolynomialChaosBasis multivariate_basis(
        {janus::legendre_dimension(), janus::hermite_dimension()}, 1);
    const auto rule = janus::stochastic_quadrature_rule(janus::legendre_dimension(), 3);
    janus::NumericVector values = janus::NumericVector::Ones(rule.nodes.size());
    EXPECT_THROW(janus::pce_projection_coefficients(multivariate_basis, rule, values),
                 janus::InvalidArgument);

    janus::NumericMatrix matrix_values(rule.nodes.size(), 1);
    matrix_values.setOnes();
    EXPECT_THROW(janus::pce_projection_coefficients(multivariate_basis, rule, matrix_values),
                 janus::InvalidArgument);
}

TEST(LinalgCoverage, ValidationAndSymbolicSmallMatrixBranches) {
    EXPECT_THROW(janus::detail::validate_linear_solve_dims(0, 1, 0, "solve"),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::detail::validate_linear_solve_dims(2, 2, 1, "solve"),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::detail::validate_square_required(2, 3, "solve", "LLT"),
                 janus::InvalidArgument);

    janus::LinearSolvePolicy iterative_policy = janus::LinearSolvePolicy::iterative();
    iterative_policy.tolerance = 0.0;
    EXPECT_THROW(janus::detail::validate_iterative_policy(iterative_policy, "solve"),
                 janus::InvalidArgument);
    iterative_policy = janus::LinearSolvePolicy::iterative();
    iterative_policy.max_iterations = 0;
    EXPECT_THROW(janus::detail::validate_iterative_policy(iterative_policy, "solve"),
                 janus::InvalidArgument);
    iterative_policy = janus::LinearSolvePolicy::iterative();
    iterative_policy.gmres_restart = 0;
    EXPECT_THROW(janus::detail::validate_iterative_policy(iterative_policy, "solve"),
                 janus::InvalidArgument);

    janus::NumericMatrix dense(2, 2);
    dense << 0.0, 0.0, 0.0, 2.0;
    const janus::SparseMatrix sparse = janus::to_sparse(dense);
    janus::NumericVector rhs(2);
    rhs << 1.0, 2.0;

    const auto none_prec = janus::detail::make_preconditioner(
        sparse, janus::LinearSolvePolicy::iterative(janus::IterativeKrylovSolver::BiCGSTAB,
                                                    janus::IterativePreconditioner::None));
    EXPECT_TRUE(none_prec(rhs).isApprox(rhs, 1e-12));

    const auto diag_prec = janus::detail::make_preconditioner(
        sparse, janus::LinearSolvePolicy::iterative(janus::IterativeKrylovSolver::BiCGSTAB,
                                                    janus::IterativePreconditioner::Diagonal));
    const janus::NumericVector scaled_rhs = diag_prec(rhs);
    EXPECT_DOUBLE_EQ(scaled_rhs(0), 1.0);
    EXPECT_DOUBLE_EQ(scaled_rhs(1), 1.0);

    janus::LinearSolvePolicy bad_prec_policy = janus::LinearSolvePolicy::iterative();
    bad_prec_policy.iterative_preconditioner = static_cast<janus::IterativePreconditioner>(999);
    EXPECT_THROW(janus::detail::make_preconditioner(sparse, bad_prec_policy),
                 janus::InvalidArgument);

    janus::NumericMatrix nonsquare(2, 3);
    nonsquare << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    EXPECT_THROW(janus::solve(nonsquare, rhs,
                              janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::FullPivLU)),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::solve(nonsquare, rhs,
                              janus::LinearSolvePolicy::sparse_direct(
                                  janus::SparseDirectLinearSolver::SimplicialLLT)),
                 janus::InvalidArgument);

    janus::NumericMatrix indefinite(2, 2);
    indefinite << 0.0, 1.0, 1.0, 0.0;
    EXPECT_THROW(janus::solve(indefinite, rhs,
                              janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::LLT)),
                 janus::InvalidArgument);

    janus::NumericVector zero_vec = janus::NumericVector::Zero(2);
    EXPECT_THROW(janus::detail::normalize_vector(zero_vec), janus::InvalidArgument);

    janus::NumericMatrix numeric_scalar_matrix(1, 1);
    numeric_scalar_matrix << 7.0;
    const auto numeric_eig = janus::eig(numeric_scalar_matrix);
    EXPECT_DOUBLE_EQ(numeric_eig.eigenvalues(0), 7.0);
    EXPECT_DOUBLE_EQ(numeric_eig.eigenvectors(0, 0), 1.0);

    janus::SymbolicMatrix symbolic_1x1(1, 1);
    symbolic_1x1(0, 0) = 5.0;
    const auto sym_eig_1x1 = janus::eig_symmetric(symbolic_1x1);
    EXPECT_DOUBLE_EQ(janus::eval(sym_eig_1x1.eigenvalues)(0), 5.0);
    EXPECT_DOUBLE_EQ(janus::eval(sym_eig_1x1.eigenvectors)(0, 0), 1.0);

    janus::SymbolicMatrix symbolic_2x2(2, 2);
    symbolic_2x2 << 3.0, 1.0, 1.0, 3.0;
    const auto sym_eig_2x2 = janus::eig_symmetric(symbolic_2x2);
    const janus::NumericVector sym_values_2x2 = janus::eval(sym_eig_2x2.eigenvalues);
    EXPECT_NEAR(sym_values_2x2(0), 2.0, 1e-12);
    EXPECT_NEAR(sym_values_2x2(1), 4.0, 1e-12);

    janus::SymbolicMatrix symbolic_4x4(4, 4);
    symbolic_4x4.setZero();
    for (int i = 0; i < 4; ++i) {
        symbolic_4x4(i, i) = 1.0;
    }
    EXPECT_THROW(janus::eig_symmetric(symbolic_4x4), janus::InvalidArgument);
}

TEST(IntegrateCoverage, DetailValidationAndMassMatrixBranches) {
    EXPECT_STREQ(janus::detail::method_name(janus::SecondOrderIntegratorMethod::StormerVerlet),
                 "Stormer-Verlet");
    EXPECT_STREQ(janus::detail::method_name(janus::SecondOrderIntegratorMethod::RungeKuttaNystrom4),
                 "RKN4");
    EXPECT_THROW(janus::detail::method_name(static_cast<janus::SecondOrderIntegratorMethod>(999)),
                 janus::InvalidArgument);

    EXPECT_STREQ(janus::detail::method_name(janus::MassMatrixIntegratorMethod::RosenbrockEuler),
                 "Rosenbrock-Euler");
    EXPECT_STREQ(janus::detail::method_name(janus::MassMatrixIntegratorMethod::Bdf1), "BDF1");
    EXPECT_THROW(janus::detail::method_name(static_cast<janus::MassMatrixIntegratorMethod>(999)),
                 janus::InvalidArgument);

    EXPECT_THROW(janus::detail::validate_eval_count("ivp", 1), janus::IntegrationError);

    janus::SecondOrderIvpOptions second_order_opts;
    second_order_opts.substeps = 0;
    EXPECT_THROW(janus::detail::validate_second_order_options(second_order_opts, "ivp"),
                 janus::IntegrationError);

    janus::MassMatrixIvpOptions mass_opts;
    mass_opts.substeps = 0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);
    mass_opts = janus::MassMatrixIvpOptions();
    mass_opts.abstol = 0.0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);
    mass_opts = janus::MassMatrixIvpOptions();
    mass_opts.reltol = 0.0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);
    mass_opts = janus::MassMatrixIvpOptions();
    mass_opts.finite_difference_epsilon = 0.0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);
    mass_opts = janus::MassMatrixIvpOptions();
    mass_opts.max_newton_iterations = 0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);
    mass_opts = janus::MassMatrixIvpOptions();
    mass_opts.newton_tolerance = 0.0;
    EXPECT_THROW(janus::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 janus::IntegrationError);

    janus::NumericVector empty_q0(0);
    janus::NumericVector q0(1);
    q0 << 1.0;
    janus::NumericVector v0(1);
    v0 << 0.0;
    janus::NumericVector bad_v0(2);
    bad_v0 << 0.0, 0.0;
    EXPECT_THROW(janus::detail::validate_second_order_initial_state(empty_q0, v0),
                 janus::IntegrationError);
    EXPECT_THROW(janus::detail::validate_second_order_initial_state(q0, bad_v0),
                 janus::IntegrationError);

    EXPECT_DOUBLE_EQ(janus::detail::inf_norm(q0), 1.0);
    EXPECT_TRUE(janus::detail::is_constant_zero(janus::SymbolicScalar(0.0)));
    auto symbolic_t = janus::sym("t_cov");
    EXPECT_FALSE(janus::detail::is_constant_zero(symbolic_t));

    janus::NumericVector x(2);
    x << 1.0, -2.0;
    const janus::NumericMatrix J = janus::detail::finite_difference_jacobian(
        [](const janus::NumericVector &state) { return (2.0 * state).eval(); }, x, 1e-6);
    EXPECT_TRUE(J.isApprox(2.0 * janus::NumericMatrix::Identity(2, 2), 1e-5));

    EXPECT_THROW(janus::detail::evaluate_mass_matrix(
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size() + 1, state.size() + 1);
                     },
                     0.0, x, "mass"),
                 janus::IntegrationError);

    janus::MassMatrixIvpOptions opts;
    EXPECT_THROW(janus::detail::rosenbrock_euler_step(
                     [](double, const janus::NumericVector &) {
                         janus::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size(), state.size());
                     },
                     x, 0.0, 0.1, opts),
                 janus::IntegrationError);

    EXPECT_THROW(janus::detail::bdf1_step(
                     [](double, const janus::NumericVector &) {
                         janus::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size(), state.size());
                     },
                     x, 0.0, 0.1, opts),
                 janus::IntegrationError);

    EXPECT_THROW(janus::solve_second_order_ivp(
                     [](double, const janus::NumericVector &state) { return (-state).eval(); },
                     {0.0, 1.0}, empty_q0, v0, 10),
                 janus::IntegrationError);
    EXPECT_THROW(janus::solve_second_order_ivp(
                     [](double, const janus::NumericVector &state) { return (-state).eval(); },
                     {0.0, 1.0}, q0, bad_v0, 10),
                 janus::IntegrationError);

    janus::NumericVector empty_y0(0);
    EXPECT_THROW(janus::solve_ivp_mass_matrix(
                     [](double, const janus::NumericVector &state) { return state; },
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size(), state.size());
                     },
                     {0.0, 1.0}, empty_y0, 10),
                 janus::IntegrationError);

    janus::NumericVector y0(2);
    y0 << 1.0, 0.0;
    EXPECT_THROW(janus::solve_ivp_mass_matrix(
                     [](double, const janus::NumericVector &) {
                         janus::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size(), state.size());
                     },
                     {0.0, 1.0}, y0, 2),
                 janus::IntegrationError);

    EXPECT_THROW(janus::solve_ivp_mass_matrix(
                     [](double, const janus::NumericVector &state) { return state; },
                     [](double, const janus::NumericVector &state) {
                         return janus::NumericMatrix::Identity(state.size() + 1, state.size() + 1);
                     },
                     {0.0, 1.0}, y0, 2),
                 janus::IntegrationError);
}

TEST(IntegrateCoverage, SymbolicMassMatrixExprErrorAndOdeOnlyBranches) {
    auto t = janus::sym("t_cov_mass");
    auto y = janus::sym("y_cov_mass");
    janus::NumericVector y0(1);
    y0 << 1.0;

    janus::MassMatrixIvpOptions opts;
    opts.symbolic_integrator_options["max_num_steps"] = 1000;
    const auto ode_only = janus::solve_ivp_mass_matrix_expr(-y, casadi::MX::ones(1, 1), t, y,
                                                            {0.0, 1.0}, y0, 8, opts);
    EXPECT_TRUE(ode_only.success);
    EXPECT_NEAR(ode_only.y(0, ode_only.y.cols() - 1), std::exp(-1.0), 1e-4);

    EXPECT_THROW(
        janus::solve_ivp_mass_matrix_expr(-y, casadi::MX::zeros(1, 1), t, y, {0.0, 1.0}, y0, 8),
        janus::IntegrationError);

    EXPECT_THROW(
        janus::solve_ivp_mass_matrix_expr(-y, casadi::MX::zeros(2, 2), t, y, {0.0, 1.0}, y0, 8),
        janus::IntegrationError);

    EXPECT_THROW(janus::solve_ivp_mass_matrix_expr(casadi::MX::zeros(1, 2), casadi::MX::ones(1, 1),
                                                   t, y, {0.0, 1.0}, y0, 8),
                 janus::IntegrationError);
}

TEST(PolynomialChaosCoverage, ValidationErrorsAndMatrixRegressionPaths) {
    EXPECT_THROW(janus::detail::validate_degree(-1, "pce"), janus::InvalidArgument);
    EXPECT_THROW(janus::detail::validate_dimension(janus::jacobi_dimension(-1.0, 0.0), "pce"),
                 janus::InvalidArgument);
    EXPECT_THROW(janus::detail::validate_dimension(janus::laguerre_dimension(-1.0), "pce"),
                 janus::InvalidArgument);

    EXPECT_DOUBLE_EQ(janus::detail::raw_jacobi_polynomial(0, 0.25, 1.0, 2.0), 1.0);
    EXPECT_DOUBLE_EQ(janus::detail::raw_jacobi_polynomial(1, 0.25, 1.0, 2.0), 0.125);
    EXPECT_DOUBLE_EQ(janus::detail::raw_laguerre_polynomial(0, 0.2, 0.5), 1.0);
    EXPECT_DOUBLE_EQ(janus::detail::raw_laguerre_polynomial(1, 0.2, 0.5), 1.3);
    EXPECT_DOUBLE_EQ(janus::pce_squared_norm(janus::legendre_dimension(), 2), 1.0);

    janus::NumericMatrix underdetermined(2, 3);
    underdetermined << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    EXPECT_THROW(janus::detail::regression_operator(underdetermined, 0.0, "pce"),
                 janus::InvalidArgument);
    EXPECT_THROW(
        janus::detail::regression_operator(janus::NumericMatrix::Identity(3, 3), -1.0, "pce"),
        janus::InvalidArgument);

    EXPECT_THROW(janus::PolynomialChaosBasis({}, 1), janus::InvalidArgument);
    EXPECT_THROW(janus::PolynomialChaosBasis({janus::legendre_dimension()}, -1),
                 janus::InvalidArgument);

    janus::PolynomialChaosBasis basis({janus::legendre_dimension()}, 2);

    janus::NumericVector bad_point(2);
    bad_point << 0.0, 1.0;
    EXPECT_THROW(basis.evaluate(bad_point), janus::InvalidArgument);

    janus::NumericMatrix empty_samples(0, 1);
    EXPECT_THROW(basis.evaluate(empty_samples), janus::InvalidArgument);

    janus::NumericMatrix bad_samples(1, 2);
    bad_samples << 0.0, 1.0;
    EXPECT_THROW(basis.evaluate(bad_samples), janus::InvalidArgument);

    janus::NumericMatrix samples(4, 1);
    samples << -1.0, -0.5, 0.5, 1.0;

    janus::NumericVector vector_values(4);
    vector_values << 1.0, 0.5, 0.5, 1.0;
    janus::NumericMatrix matrix_values(4, 2);
    matrix_values.col(0) = vector_values;
    matrix_values.col(1) = 2.0 * vector_values;
    const janus::NumericMatrix regressed =
        janus::pce_regression_coefficients(basis, samples, matrix_values, 1e-3);
    EXPECT_EQ(regressed.rows(), basis.size());
    EXPECT_EQ(regressed.cols(), 2);

    janus::NumericVector bad_vector_values(3);
    bad_vector_values << 1.0, 2.0, 3.0;
    EXPECT_THROW(janus::pce_regression_coefficients(basis, samples, bad_vector_values, 1e-3),
                 janus::InvalidArgument);

    janus::NumericMatrix bad_matrix_values(3, 2);
    bad_matrix_values.setOnes();
    EXPECT_THROW(janus::pce_regression_coefficients(basis, samples, bad_matrix_values, 1e-3),
                 janus::InvalidArgument);

    janus::NumericVector bad_weights(3);
    bad_weights.setOnes();
    EXPECT_THROW(janus::pce_projection_coefficients(basis, samples, bad_weights, vector_values),
                 janus::InvalidArgument);

    janus::NumericMatrix projection_matrix_values(3, 1);
    projection_matrix_values.setOnes();
    janus::NumericVector good_weights = janus::NumericVector::Ones(samples.rows());
    EXPECT_THROW(
        janus::pce_projection_coefficients(basis, samples, good_weights, projection_matrix_values),
        janus::InvalidArgument);

    janus::NumericVector empty_coeffs(0);
    EXPECT_THROW(janus::pce_mean(empty_coeffs), janus::InvalidArgument);

    janus::NumericVector wrong_size_coeffs(2);
    wrong_size_coeffs.setZero();
    EXPECT_THROW(janus::pce_variance(basis, wrong_size_coeffs), janus::InvalidArgument);
}
