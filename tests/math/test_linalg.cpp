#include "../utils/TestUtils.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <janus/core/Function.hpp>
#include <janus/core/JanusIO.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Linalg.hpp>

template <typename Scalar> void test_linalg_ops() {
    using Matrix = janus::JanusMatrix<Scalar>;
    using Vector = janus::JanusVector<Scalar>;

    // Test solve
    // A = [[2, 1], [1, 2]]
    // b = [3, 3]
    // x = [1, 1]
    Matrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    Vector b(2);
    b(0) = 3.0;
    b(1) = 3.0;

    auto x = janus::solve(A, b);

    // Test norm
    Vector v(2);
    v(0) = 3.0;
    v(1) = 4.0;
    auto n = janus::norm(v);

    // Test outer
    Vector v1(2);
    v1(0) = 1.0;
    v1(1) = 2.0;
    Vector v2(2);
    v2(0) = 3.0;
    v2(1) = 4.0;                   // v2 = [3, 4]
    auto M = janus::outer(v1, v2); // [[3, 4], [6, 8]]

    // Test dot
    auto d = janus::dot(v, v); // 3*3 + 4*4 = 25

    // Test cross (3D)
    Vector c1(3);
    c1 << 1.0, 0.0, 0.0;
    Vector c2(3);
    c2 << 0.0, 1.0, 0.0;
    auto c3 = janus::cross(c1, c2); // [0, 0, 1]

    // Test inner
    auto i_prod = janus::inner(v, v); // 25.0

    // Test pinv
    // A singular = [[1, 1], [2, 2]]
    Matrix A_sing(2, 2);
    A_sing << 1.0, 1.0, 2.0, 2.0;
    auto A_pinv = janus::pinv(A_sing);

    // Test extended norms
    Vector v_norm(3);
    v_norm << -1.0, 2.0, -3.0;

    auto n_1 = janus::norm(v_norm, janus::NormType::L1);    // 1+2+3 = 6
    auto n_inf = janus::norm(v_norm, janus::NormType::Inf); // 3

    // Existing inv / det tests
    auto A_inv = janus::inv(A); // inv([[2, 1], [1, 2]]) = 1/3 * [[2, -1], [-1, 2]]
    auto A_det = janus::det(A); // 4 - 1 = 3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(x(0), 1.0, 1e-6);
        EXPECT_NEAR(x(1), 1.0, 1e-6);
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(M(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(d, 25.0);

        EXPECT_DOUBLE_EQ(c3(0), 0.0);
        EXPECT_DOUBLE_EQ(c3(1), 0.0);
        EXPECT_DOUBLE_EQ(c3(2), 1.0);

        EXPECT_DOUBLE_EQ(i_prod, 25.0);

        // Pinv of [[1, 1], [2, 2]]
        // SVD based, roughly [[0.1, 0.2], [0.1, 0.2]]
        // Check A * pinv * A = A
        auto recon = A_sing * A_pinv * A_sing;
        EXPECT_TRUE(recon.isApprox(A_sing, 1e-5));

        EXPECT_DOUBLE_EQ(n_1, 6.0);
        EXPECT_DOUBLE_EQ(n_inf, 3.0);

        EXPECT_NEAR(A_det, 3.0, 1e-9);
        EXPECT_NEAR(A_inv(0, 0), 2.0 / 3.0, 1e-9);
    } else {
        auto x_eval = janus::eval(x);
        EXPECT_NEAR(x_eval(0), 1.0, 1e-6);
        EXPECT_NEAR(x_eval(1), 1.0, 1e-6);

        EXPECT_DOUBLE_EQ(janus::eval(n), 5.0);

        auto M_eval = janus::eval(M);
        EXPECT_DOUBLE_EQ(M_eval(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(janus::eval(d), 25.0);

        auto c3_eval = janus::eval(c3);
        EXPECT_NEAR(c3_eval(0), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(2), 1.0, 1e-9);

        EXPECT_DOUBLE_EQ(janus::eval(i_prod), 25.0);

        // Pinv test symbolic
        auto A_pinv_inv = janus::pinv(A);
        auto A_pinv_eval = janus::eval(A_pinv_inv);
        EXPECT_NEAR(A_pinv_eval(0, 0), 2.0 / 3.0, 1e-6);

        EXPECT_DOUBLE_EQ(janus::eval(n_1), 6.0);
        EXPECT_DOUBLE_EQ(janus::eval(n_inf), 3.0);

        auto A_inv_eval = janus::eval(A_inv);
        if (std::abs(A_inv_eval(0, 0) - (2.0 / 3.0)) > 1e-9) {
            janus::print("A_inv (Symbolic)", A_inv);
            janus::print("A_inv (Evaluated)", A_inv_eval);
        }
        EXPECT_NEAR(A_inv_eval(0, 0), 2.0 / 3.0, 1e-9);
    }
}

TEST(LinalgTests, Numeric) { test_linalg_ops<double>(); }

TEST(LinalgTests, Symbolic) { test_linalg_ops<janus::SymbolicScalar>(); }

TEST(LinalgTests, SolveDensePolicyBackends) {
    janus::NumericMatrix A(3, 3);
    A << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;

    janus::NumericMatrix B(3, 2);
    B << 1.0, 2.0, 0.0, -1.0, 3.0, 1.0;

    janus::NumericMatrix x_qr = janus::solve(
        A, B, janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::ColPivHouseholderQR));
    janus::NumericMatrix x_lu =
        janus::solve(A, B, janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::PartialPivLU));
    janus::NumericMatrix x_llt =
        janus::solve(A, B, janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::LLT));
    janus::NumericMatrix x_ldlt =
        janus::solve(A, B, janus::LinearSolvePolicy::dense(janus::DenseLinearSolver::LDLT));

    EXPECT_TRUE((A * x_qr).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_lu).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_llt).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_ldlt).isApprox(B, 1e-10));
}

TEST(LinalgTests, SolveSparseDirectBackends) {
    janus::NumericMatrix dense(3, 3);
    dense << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;
    janus::SparseMatrix sparse = janus::to_sparse(dense);

    janus::NumericVector b(3);
    b << 1.0, 0.0, 3.0;

    janus::NumericVector x_default = janus::solve(sparse, b);
    janus::NumericVector x_sparse_lu = janus::solve(
        sparse, b,
        janus::LinearSolvePolicy::sparse_direct(janus::SparseDirectLinearSolver::SparseLU));
    janus::NumericVector x_sparse_qr = janus::solve(
        sparse, b,
        janus::LinearSolvePolicy::sparse_direct(janus::SparseDirectLinearSolver::SparseQR));
    janus::NumericVector x_sparse_ldlt = janus::solve(
        sparse, b,
        janus::LinearSolvePolicy::sparse_direct(janus::SparseDirectLinearSolver::SimplicialLDLT));
    janus::NumericVector x_dense_input_sparse_policy = janus::solve(
        dense, b,
        janus::LinearSolvePolicy::sparse_direct(janus::SparseDirectLinearSolver::SparseLU));

    EXPECT_TRUE((dense * x_default).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_lu).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_qr).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_ldlt).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_dense_input_sparse_policy).isApprox(b, 1e-10));
}

TEST(LinalgTests, SolveIterativeBackendsAndPreconditionerHook) {
    janus::NumericMatrix dense(4, 4);
    dense << 10.0, 1.0, 0.0, 0.0, 1.0, 7.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 5.0;
    janus::SparseMatrix sparse = janus::to_sparse(dense);

    janus::NumericVector b(4);
    b << 1.0, 2.0, 3.0, 4.0;

    auto bicg_policy = janus::LinearSolvePolicy::iterative(janus::IterativeKrylovSolver::BiCGSTAB,
                                                           janus::IterativePreconditioner::Diagonal)
                           .set_tolerance(1e-12)
                           .set_max_iterations(200);
    janus::NumericVector x_bicg = janus::solve(sparse, b, bicg_policy);
    EXPECT_TRUE((dense * x_bicg).isApprox(b, 1e-8));

    janus::NumericVector inv_diag = dense.diagonal().cwiseInverse();
    int preconditioner_calls = 0;
    auto gmres_policy = janus::LinearSolvePolicy::iterative(janus::IterativeKrylovSolver::GMRES,
                                                            janus::IterativePreconditioner::None)
                            .set_tolerance(1e-12)
                            .set_max_iterations(200)
                            .set_gmres_restart(8)
                            .set_preconditioner_hook([&](const janus::NumericVector &rhs) {
                                ++preconditioner_calls;
                                return (inv_diag.array() * rhs.array()).matrix();
                            });
    janus::NumericVector x_gmres = janus::solve(sparse, b, gmres_policy);
    EXPECT_GT(preconditioner_calls, 0);
    EXPECT_TRUE((dense * x_gmres).isApprox(b, 1e-8));
}

TEST(LinalgTests, SolveSymbolicPolicyAndValidationErrors) {
    janus::SymbolicMatrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    janus::SymbolicVector b(2);
    b(0) = 3.0;
    b(1) = 3.0;

    auto symbolic_policy = janus::LinearSolvePolicy();
    symbolic_policy.set_symbolic_solver("qr");
    auto x = janus::solve(A, b, symbolic_policy);
    auto x_eval = janus::eval(x);
    EXPECT_NEAR(x_eval(0), 1.0, 1e-10);
    EXPECT_NEAR(x_eval(1), 1.0, 1e-10);

    auto bad_symbolic_policy = janus::LinearSolvePolicy();
    bad_symbolic_policy.symbolic_options["mock_option"] = 1;
    EXPECT_THROW(janus::solve(A, b, bad_symbolic_policy), janus::InvalidArgument);

    auto bad_iterative = janus::LinearSolvePolicy::iterative().set_max_iterations(0);
    janus::NumericMatrix dense(2, 2);
    dense << 2.0, 1.0, 1.0, 2.0;
    janus::NumericVector rhs(2);
    rhs << 3.0, 3.0;
    EXPECT_THROW(janus::solve(dense, rhs, bad_iterative), janus::InvalidArgument);
}

TEST(LinalgTests, CoverageEdges) {
    // 1. Empty to_mx coverage
    janus::NumericMatrix empty(0, 0);
    casadi::MX empty_mx = janus::to_mx(empty);
    EXPECT_TRUE(empty_mx.is_empty());

    // 2. Numeric Norm Edges
    janus::NumericVector v(3);
    v << 1.0, 2.0, 2.0; // norm = 3

    // Frobenius (same as L2 for vectors)
    EXPECT_DOUBLE_EQ(janus::norm(v, janus::NormType::Frobenius), 3.0);

    // Default (invalid enum)
    EXPECT_DOUBLE_EQ(janus::norm(v, static_cast<janus::NormType>(999)), 3.0);

    // 3. Symbolic Norm Edges (Default branch)
    janus::SymbolicVector vs = janus::as_vector(janus::to_mx(v));
    auto n_def = janus::norm(vs, static_cast<janus::NormType>(999));
    EXPECT_DOUBLE_EQ(janus::eval(n_def), 3.0);
}

namespace {

void expect_eigen_residual(const janus::NumericMatrix &A, const janus::NumericVector &eigenvalues,
                           const janus::NumericMatrix &eigenvectors, double tol) {
    EXPECT_TRUE((A * eigenvectors).isApprox(eigenvectors * eigenvalues.asDiagonal(), tol));
}

std::array<double, 6> pack_symmetric_lower_column_major(const janus::NumericMatrix &A) {
    return {A(0, 0), A(1, 0), A(2, 0), A(1, 1), A(2, 1), A(2, 2)};
}

janus::NumericMatrix unpack_symmetric_lower_column_major(const std::array<double, 6> &packed) {
    janus::NumericMatrix A(3, 3);
    A << packed[0], packed[1], packed[2], packed[1], packed[3], packed[4], packed[2], packed[4],
        packed[5];
    return A;
}

} // namespace

TEST(LinalgTests, EigSymmetricNumeric) {
    janus::NumericMatrix A(2, 2);
    A << 2.0, 1.0, 1.0, 2.0;

    auto decomp = janus::eig_symmetric(A);

    ASSERT_EQ(decomp.eigenvalues.size(), 2);
    ASSERT_EQ(decomp.eigenvectors.rows(), 2);
    ASSERT_EQ(decomp.eigenvectors.cols(), 2);
    EXPECT_NEAR(decomp.eigenvalues(0), 1.0, 1e-12);
    EXPECT_NEAR(decomp.eigenvalues(1), 3.0, 1e-12);
    expect_eigen_residual(A, decomp.eigenvalues, decomp.eigenvectors, 1e-12);
    EXPECT_TRUE((decomp.eigenvectors.transpose() * decomp.eigenvectors)
                    .isApprox(janus::NumericMatrix::Identity(2, 2), 1e-12));
}

TEST(LinalgTests, EigNumericRealSpectrum) {
    janus::NumericMatrix A(2, 2);
    A << 1.0, 1.0, 0.0, 2.0;

    auto decomp = janus::eig(A);

    ASSERT_EQ(decomp.eigenvalues.size(), 2);
    EXPECT_NEAR(decomp.eigenvalues(0), 1.0, 1e-12);
    EXPECT_NEAR(decomp.eigenvalues(1), 2.0, 1e-12);
    expect_eigen_residual(A, decomp.eigenvalues, decomp.eigenvectors, 1e-12);
}

TEST(LinalgTests, EigRejectsComplexSpectrum) {
    janus::NumericMatrix A(2, 2);
    A << 0.0, -1.0, 1.0, 0.0;

    EXPECT_THROW(janus::eig(A), janus::InvalidArgument);
}

TEST(LinalgTests, EigSymmetricRejectsNonsymmetricNumeric) {
    janus::NumericMatrix A(2, 2);
    A << 1.0, 2.0, 0.0, 1.0;

    EXPECT_THROW(janus::eig_symmetric(A), janus::InvalidArgument);
}

TEST(LinalgTests, EigSymmetricSymbolic3x3) {
    auto t = janus::sym("t");
    janus::SymbolicMatrix A(3, 3);
    A << 4.0, t, 0.0, t, 2.0, 0.0, 0.0, 0.0, 1.0;

    auto decomp = janus::eig_symmetric(A);
    janus::Function f({t}, {janus::to_mx(decomp.eigenvalues), janus::to_mx(decomp.eigenvectors)});
    auto outputs = f(0.5);

    ASSERT_EQ(outputs.size(), 2);
    janus::NumericVector eigenvalues = outputs[0].col(0);
    const janus::NumericMatrix eigenvectors = outputs[1];

    EXPECT_NEAR(eigenvalues(0), 1.0, 1e-9);
    EXPECT_NEAR(eigenvalues(1), 3.0 - 0.5 * std::sqrt(5.0), 1e-9);
    EXPECT_NEAR(eigenvalues(2), 3.0 + 0.5 * std::sqrt(5.0), 1e-9);

    janus::NumericMatrix A_eval(3, 3);
    A_eval << 4.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 1.0;
    expect_eigen_residual(A_eval, eigenvalues, eigenvectors, 1e-8);
    EXPECT_TRUE((eigenvectors.transpose() * eigenvectors)
                    .isApprox(janus::NumericMatrix::Identity(3, 3), 1e-8));
}

TEST(LinalgTests, EigSymbolicRejectsGeneralMatrices) {
    auto A_mx = janus::sym("A", 2, 2);
    auto A = janus::to_eigen(A_mx);

    EXPECT_THROW(janus::eig(A), janus::InvalidArgument);
}

TEST(LinalgTests, InvSymmetric3x3ExplicitPackedOrderMatchesInv) {
    const std::array<janus::NumericMatrix, 4> cases = [] {
        std::array<janus::NumericMatrix, 4> mats;

        mats[0].resize(3, 3);
        mats[0] << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

        mats[1].resize(3, 3);
        mats[1] << 4.0, 1.0, 1.5, 1.0, 5.0, 2.0, 1.5, 2.0, 6.0;

        mats[2].resize(3, 3);
        mats[2] << 7.0, -2.5, 3.0, -2.5, 8.0, 1.5, 3.0, 1.5, 9.0;

        mats[3].resize(3, 3);
        mats[3] << 3.5, -0.4, 0.8, -0.4, 2.2, -0.6, 0.8, -0.6, 4.1;

        return mats;
    }();

    for (const auto &A : cases) {
        const auto A_inv = janus::inv(A);
        const auto packed_expected = pack_symmetric_lower_column_major(A_inv);

        const auto packed_actual_tuple =
            janus::inv_symmetric_3x3_explicit(A(0, 0), A(1, 1), A(2, 2), A(0, 1), A(1, 2), A(0, 2));
        const std::array<double, 6> packed_actual = {
            std::get<0>(packed_actual_tuple), std::get<1>(packed_actual_tuple),
            std::get<2>(packed_actual_tuple), std::get<3>(packed_actual_tuple),
            std::get<4>(packed_actual_tuple), std::get<5>(packed_actual_tuple)};

        for (std::size_t i = 0; i < packed_actual.size(); ++i) {
            EXPECT_NEAR(packed_actual[i], packed_expected[i], 1e-12);
        }

        EXPECT_TRUE(unpack_symmetric_lower_column_major(packed_actual).isApprox(A_inv, 1e-12));
    }
}

// =============================================================================
// Sparse Matrix Tests
// =============================================================================

TEST(LinalgTests, SparseFromTriplets) {
    std::vector<janus::SparseTriplet> triplets;
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(1, 1, 2.0);
    triplets.emplace_back(2, 2, 3.0);

    auto sp = janus::sparse_from_triplets(3, 3, triplets);

    EXPECT_EQ(sp.rows(), 3);
    EXPECT_EQ(sp.cols(), 3);
    EXPECT_EQ(sp.nonZeros(), 3);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sp.coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(sp.coeff(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 1), 0.0);
}

TEST(LinalgTests, ToSparse) {
    janus::NumericMatrix dense(3, 3);
    dense << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;

    auto sp = janus::to_sparse(dense);

    EXPECT_EQ(sp.nonZeros(), 3);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sp.coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(sp.coeff(2, 2), 3.0);
}

TEST(LinalgTests, ToSparseWithTolerance) {
    janus::NumericMatrix dense(2, 2);
    dense << 1.0, 1e-10, 1e-10, 1.0;

    // With tol=0, small values are kept
    auto sp1 = janus::to_sparse(dense, 0.0);
    EXPECT_EQ(sp1.nonZeros(), 4);

    // With tol=1e-9, small values are ignored
    auto sp2 = janus::to_sparse(dense, 1e-9);
    EXPECT_EQ(sp2.nonZeros(), 2);
}

TEST(LinalgTests, ToDense) {
    std::vector<janus::SparseTriplet> triplets;
    triplets.emplace_back(0, 0, 5.0);
    triplets.emplace_back(1, 1, 10.0);

    auto sp = janus::sparse_from_triplets(2, 2, triplets);
    auto dense = janus::to_dense(sp);

    EXPECT_EQ(dense.rows(), 2);
    EXPECT_EQ(dense.cols(), 2);
    EXPECT_DOUBLE_EQ(dense(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(dense(1, 1), 10.0);
    EXPECT_DOUBLE_EQ(dense(0, 1), 0.0);
}

TEST(LinalgTests, SparseIdentity) {
    auto I = janus::sparse_identity(4);

    EXPECT_EQ(I.rows(), 4);
    EXPECT_EQ(I.cols(), 4);
    EXPECT_EQ(I.nonZeros(), 4);

    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(I.coeff(i, i), 1.0);
    }
}
