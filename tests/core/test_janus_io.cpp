#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <janus/core/JanusIO.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Linalg.hpp>
#include <janus/math/Trig.hpp>
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

// ======================================================================
// Graph Visualization Tests
// ======================================================================

TEST(JanusIOTests, ExportGraphDot) {
    // Create simple symbolic expression
    auto x = janus::sym("x");
    auto y = x * x + 2.0 * x + 1.0;

    // Export to DOT
    std::string dot_file = "/tmp/test_janus_graph";
    janus::export_graph_dot(y, dot_file, "quadratic");

    // Verify DOT file was created
    std::ifstream f(dot_file + ".dot");
    EXPECT_TRUE(f.good()) << "DOT file should exist";

    // Read content
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Verify DOT structure (new deep graph format)
    EXPECT_TRUE(content.find("digraph") != std::string::npos) << "Should contain digraph";
    EXPECT_TRUE(content.find("quadratic") != std::string::npos) << "Should contain graph name";
    EXPECT_TRUE(content.find("node_") != std::string::npos) << "Should contain graph nodes";
    EXPECT_TRUE(content.find("Output") != std::string::npos) << "Should contain output marker";
    EXPECT_TRUE(content.find("->") != std::string::npos) << "Should contain edges";

    // Cleanup
    std::remove((dot_file + ".dot").c_str());
}

TEST(JanusIOTests, ExportGraphDotMultipleInputs) {
    // Expression with multiple inputs
    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto z = x * y + x + y;

    std::string dot_file = "/tmp/test_janus_multi";
    janus::export_graph_dot(z, dot_file, "multi_input");

    std::ifstream f(dot_file + ".dot");
    EXPECT_TRUE(f.good());

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Should have variable nodes labeled with x and y (deep graph shows actual variable names)
    EXPECT_TRUE(content.find("\"x\"") != std::string::npos) << "Should contain x variable";
    EXPECT_TRUE(content.find("\"y\"") != std::string::npos) << "Should contain y variable";

    std::remove((dot_file + ".dot").c_str());
}

TEST(JanusIOTests, ExportGraphDotConstant) {
    // Expression with no free variables (constant)
    janus::SymbolicScalar c = 42.0;

    std::string dot_file = "/tmp/test_janus_const";
    janus::export_graph_dot(c, dot_file, "constant");

    std::ifstream f(dot_file + ".dot");
    EXPECT_TRUE(f.good());

    std::remove((dot_file + ".dot").c_str());
}

TEST(JanusIOTests, RenderGraph) {
    // Only run if graphviz is available
    if (std::system("which dot > /dev/null 2>&1") != 0) {
        GTEST_SKIP() << "Graphviz not available";
    }

    auto x = janus::sym("x");
    auto y = x * x;

    std::string base = "/tmp/test_janus_render";
    janus::export_graph_dot(y, base);
    bool success = janus::render_graph(base + ".dot", base + ".pdf");

    EXPECT_TRUE(success) << "Rendering should succeed";

    // Verify PDF was created
    std::ifstream pdf(base + ".pdf");
    EXPECT_TRUE(pdf.good()) << "PDF file should exist";

    // Cleanup
    std::remove((base + ".dot").c_str());
    std::remove((base + ".pdf").c_str());
}

TEST(JanusIOTests, VisualizeGraph) {
    // Only run if graphviz is available
    if (std::system("which dot > /dev/null 2>&1") != 0) {
        GTEST_SKIP() << "Graphviz not available";
    }

    auto x = janus::sym("x");
    auto y = janus::sym("y");
    auto expr = x * y + janus::sin(x);

    std::string base = "/tmp/test_janus_viz";
    bool success = janus::visualize_graph(expr, base);

    EXPECT_TRUE(success) << "visualize_graph should succeed";

    // Both files should exist
    std::ifstream dot(base + ".dot");
    std::ifstream pdf(base + ".pdf");
    EXPECT_TRUE(dot.good());
    EXPECT_TRUE(pdf.good());

    // Cleanup
    std::remove((base + ".dot").c_str());
    std::remove((base + ".pdf").c_str());
}

TEST(JanusIOTests, RenderGraphPNG) {
    // Only run if graphviz is available
    if (std::system("which dot > /dev/null 2>&1") != 0) {
        GTEST_SKIP() << "Graphviz not available";
    }

    auto x = janus::sym("x");
    std::string base = "/tmp/test_janus_png";
    janus::export_graph_dot(x, base);
    bool success = janus::render_graph(base + ".dot", base + ".png");

    EXPECT_TRUE(success);

    std::ifstream png(base + ".png");
    EXPECT_TRUE(png.good());

    std::remove((base + ".dot").c_str());
    std::remove((base + ".png").c_str());
}
