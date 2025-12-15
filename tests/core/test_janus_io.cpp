#include <gtest/gtest.h>
#include <janus/core/JanusIO.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Linalg.hpp>
#include <sstream>

TEST(JanusIOTests, PrintNumeric) {
    Eigen::MatrixXd m(2, 2);
    m << 1, 2, 3, 4;

    testing::internal::CaptureStdout();
    janus::print("Numeric Matrix", m);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Numeric Matrix:") != std::string::npos);
    EXPECT_TRUE(output.find("1 2") != std::string::npos);
    EXPECT_TRUE(output.find("3 4") != std::string::npos);
}

TEST(JanusIOTests, PrintSymbolic) {
    janus::SymbolicMatrix m = Eigen::MatrixXd::Identity(2, 2).cast<janus::SymbolicScalar>();

    testing::internal::CaptureStdout();
    janus::print("Symbolic Matrix", m);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Symbolic Matrix:") != std::string::npos);
    // Exact output depends on CasADi formatting but usually contains matrix dimensions or content
}

TEST(JanusIOTests, DispAlias) {
    Eigen::VectorXd v(2);
    v << 1, 2;
    testing::internal::CaptureStdout();
    janus::disp("Vector", v);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(output.find("Vector:") != std::string::npos);
}

TEST(JanusIOTests, EvalNumeric) {
    // Eval double
    double d = 5.0;
    EXPECT_DOUBLE_EQ(janus::eval(d), 5.0);

    // Eval int (arithmetic)
    int i = 3;
    EXPECT_EQ(janus::eval(i), 3);

    // Eval Eigen Matrix
    Eigen::MatrixXd m(2, 1);
    m << 1.0, 2.0;
    auto res = janus::eval(m);
    EXPECT_DOUBLE_EQ(res(0), 1.0);
    EXPECT_DOUBLE_EQ(res(1), 2.0);
}

TEST(JanusIOTests, EvalSymbolic) {
    // Eval Symbolic Scalar (constant)
    janus::SymbolicScalar s = 10.0;
    EXPECT_DOUBLE_EQ(janus::eval(s), 10.0);

    // Eval Symbolic Matrix (constant)
    Eigen::MatrixXd m_ref(2, 2);
    m_ref << 1, 2, 3, 4;
    janus::SymbolicMatrix m = m_ref.cast<janus::SymbolicScalar>();
    auto res = janus::eval(m);

    EXPECT_TRUE(res.isApprox(m_ref));
}

TEST(JanusIOTests, EvalError) {
    // Eval Symbolic Variable (not constant) should fail
    janus::SymbolicScalar x = janus::sym("x");
    EXPECT_THROW(janus::eval(x), std::runtime_error);

    janus::SymbolicMatrix M(1, 1);
    M(0, 0) = x;
    EXPECT_THROW(janus::eval(M), std::runtime_error);
}
