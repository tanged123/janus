#include "../utils/TestUtils.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <janus/janus.hpp>

// ======================================================================
// JanusIO.hpp Coverage
// ======================================================================

TEST(JanusIOCoverage, ExportGraphError) {
    auto x = janus::sym("x");
    // Invalid path should throw
    // Assuming /root or /invalid is not writable
    EXPECT_THROW(janus::export_graph_dot(x, "/invalid/path/graph"), janus::RuntimeError);
}

TEST(JanusIOCoverage, VisualizeGraphError) {
    auto x = janus::sym("x");
    // Should catch exception and return false
    bool success = janus::visualize_graph(x, "/invalid/path/graph");
    EXPECT_FALSE(success);
}

TEST(JanusIOCoverage, RenderGraphFailure) {
    // If dot is missing, this returns false
    // If dot exists but file is bad, returns false
    // We just want to ensure it doesn't crash and returns false
    bool success = janus::render_graph("nonexistent.dot", "out.pdf");
    EXPECT_FALSE(success);
}

TEST(JanusIOCoverage, DetailHelpers) {
    // Test escape_dot_label with special chars
    // We can't access detail:: directly usually, but it's in header.
    // If it's private/implementation detail, we might skip.
    // But it's in janus::detail, which is accessible.

    std::string complex_label = "a\"b\\c\nd<e>{f}";
    std::string escaped = janus::detail::escape_dot_label(complex_label);

    EXPECT_NE(escaped.find("\\\""), std::string::npos);
    EXPECT_NE(escaped.find("\\\\"), std::string::npos);
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_NE(escaped.find("&lt;"), std::string::npos);

    // Test truncation
    std::string long_label(100, 'a');
    std::string truncated = janus::detail::escape_dot_label(long_label);
    EXPECT_TRUE(truncated.find("...") != std::string::npos);
    EXPECT_LT(truncated.size(), 100);
}

TEST(JanusIOCoverage, GetOpName) {
    // Test get_op_name for various types
    using namespace janus::detail;

    auto x = janus::sym("x");
    EXPECT_EQ(get_op_name(x), "x");

    auto c = casadi::MX(42.0);
    EXPECT_EQ(get_op_name(c), "42");

    auto expr = janus::sin(x);
    EXPECT_EQ(get_op_name(expr), "sin");

    // Binary op truncation
    // Create a deeply nested expression to force long string representation
    auto long_expr = x;
    for (int i = 0; i < 10; ++i)
        long_expr = long_expr + long_expr;

    // get_op_name truncates if > 30 chars
    std::string name = get_op_name(long_expr);
    // Just verify it runs safely
    EXPECT_FALSE(name.empty());
}
