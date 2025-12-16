#pragma once
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

/**
 * @file JanusIO.hpp
 * @brief IO Utilities and Traits for Janus
 *
 * Provides:
 * 1. Helper functions for printing/displaying matrices with wrappers.
 * 2. Evaluation utilities (eval).
 * 3. Graph visualization utilities for CasADi expressions.
 *
 * Note: Eigen::NumTraits definitions are in JanusTypes.hpp usually.
 */

namespace janus {

// Forward declaration
class Function;

/**
 * @brief Print a Matrix (Numeric or Symbolic) to stdout with a label
 */
template <typename Derived>
void print(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    std::cout << label << ":\n" << mat << "\n" << std::endl;
}

/**
 * @brief Deprecated alias for print
 */
template <typename Derived>
void disp(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    print(label, mat);
}

/**
 * @brief Evaluation helper for symbolic matrices
 * Evaluates a symbolic matrix to a numeric Eigen matrix.
 * Assumes the matrix contains no free variables (i.e. is constant).
 * Throws if evaluation fails.
 */
template <typename Derived> auto eval(const Eigen::MatrixBase<Derived> &mat) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        // Flatten to MX, evaluate, map back
        SymbolicScalar flat = SymbolicScalar::zeros(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                flat(i, j) = mat(i, j);
            }
        }

        try {
            casadi::Function f("f", {}, {flat});
            auto res = f(std::vector<casadi::DM>{});
            casadi::DM res_dm = res[0];

            NumericMatrix res_eigen(mat.rows(), mat.cols());
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    res_eigen(i, j) = static_cast<double>(res_dm(i, j));
                }
            }
            return res_eigen;
        } catch (const std::exception &e) {
            throw RuntimeError("eval failed (expression contains free variables): " +
                               std::string(e.what()));
        }
    } else {
        return mat.eval();
    }
}

// Explicit overload for SymbolicScalar scalar
inline double eval(const SymbolicScalar &val) {
    try {
        casadi::Function f("f", {}, {val});
        auto res = f(std::vector<casadi::DM>{});
        casadi::DM res_dm = res[0];
        return static_cast<double>(res_dm);
    } catch (const std::exception &e) {
        throw RuntimeError("eval scalar failed: " + std::string(e.what()));
    }
}

// Overload for numeric types (passthrough)
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0> T eval(const T &val) {
    return val;
}

// ======================================================================
// Graph Visualization
// ======================================================================

namespace detail {

/**
 * @brief Escape special characters for DOT format
 */
inline std::string escape_dot_label(const std::string &s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '<':
            result += "&lt;";
            break;
        case '>':
            result += "&gt;";
            break;
        case '{':
        case '}':
            result += ' ';
            break;
        default:
            result += c;
        }
    }
    // Truncate long labels
    if (result.size() > 40) {
        result = result.substr(0, 37) + "...";
    }
    return result;
}

/**
 * @brief Get a short description of an MX operation type
 */
inline std::string get_op_name(const SymbolicScalar &mx) {
    if (mx.is_symbolic()) {
        std::ostringstream ss;
        mx.disp(ss, false);
        return ss.str();
    }
    if (mx.is_constant()) {
        std::ostringstream ss;
        mx.disp(ss, false);
        std::string s = ss.str();
        if (s.size() > 20)
            s = s.substr(0, 17) + "...";
        return s;
    }
    // Try to get operation name from string representation
    std::ostringstream ss;
    mx.disp(ss, false);
    std::string s = ss.str();

    // Common operations
    if (s.find("sq(") == 0)
        return "sq";
    if (s.find("sin(") == 0)
        return "sin";
    if (s.find("cos(") == 0)
        return "cos";
    if (s.find("exp(") == 0)
        return "exp";
    if (s.find("log(") == 0)
        return "log";
    if (s.find("sqrt(") == 0)
        return "sqrt";
    if (s.find("tanh(") == 0)
        return "tanh";

    // For binary ops, return truncated expression
    if (s.size() > 30)
        s = s.substr(0, 27) + "...";
    return s;
}

} // namespace detail

/**
 * @brief Export a symbolic expression to DOT format for visualization
 *
 * Creates a DOT file representing the computational graph of the expression.
 * Traverses the full expression tree showing all intermediate operations.
 *
 * @param expr The symbolic expression to visualize
 * @param filename Output filename (without extension, .dot will be added)
 * @param name Optional graph name for DOT file
 */
inline void export_graph_dot(const SymbolicScalar &expr, const std::string &filename,
                             const std::string &name = "expression") {
    // Get all free variables
    std::vector<SymbolicScalar> free_vars = SymbolicScalar::symvar(expr);

    // Build DOT file
    std::string dot_filename = filename + ".dot";
    std::ofstream out(dot_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + dot_filename);
    }

    out << "digraph " << name << " {\n";
    out << "  rankdir=BT;\n"; // Bottom to top (inputs at bottom)
    out << "  splines=ortho;\n";
    out << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";
    out << "  edge [color=\"#666666\", arrowsize=0.7];\n\n";

    // Title
    out << "  labelloc=\"t\";\n";
    out << "  label=\"" << detail::escape_dot_label(name) << "\";\n";
    out << "  fontsize=16;\n\n";

    // Collect nodes by traversing the expression
    std::map<std::string, int> node_ids;    // ptr-based ID -> sequential ID
    std::vector<std::pair<int, int>> edges; // edges as sequential IDs
    std::set<const void *> visited;
    int node_counter = 0;

    // BFS traversal to collect all nodes
    std::vector<SymbolicScalar> queue = {expr};
    std::map<const void *, int> ptr_to_id;

    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = node_counter++;
        ptr_to_id[ptr] = current_id;

        // Get dependencies
        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            SymbolicScalar dep = current.dep(i);
            queue.push_back(dep);
        }
    }

    // Second pass: generate nodes and edges
    visited.clear();
    queue = {expr};

    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = ptr_to_id[ptr];
        std::string label = detail::get_op_name(current);

        // Determine node style based on type
        std::string color = "#87CEEB"; // light blue default
        std::string shape = "box";

        if (current.is_symbolic()) {
            color = "#90EE90"; // light green for inputs
            shape = "ellipse";
        } else if (current.is_constant()) {
            color = "#FFE4B5"; // moccasin for constants
            shape = "ellipse";
        } else if (current.n_dep() == 0) {
            color = "#DDA0DD"; // plum for leaf operations
        }

        out << "  node_" << current_id << " [label=\"" << detail::escape_dot_label(label)
            << "\", fillcolor=\"" << color << "\", shape=" << shape << "];\n";

        // Add edges from dependencies
        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            SymbolicScalar dep = current.dep(i);
            const void *dep_ptr = dep.get();
            int dep_id = ptr_to_id[dep_ptr];
            out << "  node_" << dep_id << " -> node_" << current_id << ";\n";
            queue.push_back(dep);
        }
    }

    // Mark output node specially
    if (!ptr_to_id.empty()) {
        const void *out_ptr = expr.get();
        int out_id = ptr_to_id[out_ptr];
        out << "\n  // Output marker\n";
        out << "  output [label=\"Output\", shape=doublecircle, fillcolor=\"#FFD700\"];\n";
        out << "  node_" << out_id << " -> output;\n";
    }

    out << "}\n";
    out.close();
}

/**
 * @brief Export a janus::Function to DOT format for visualization
 *
 * @param func The function to visualize (uses underlying CasADi function)
 * @param filename Output filename (without extension)
 */
// Note: This overload is implemented after Function class is defined
// Use the casadi::Function version directly for now

/**
 * @brief Render a DOT file to an image using Graphviz
 *
 * Requires Graphviz to be installed and `dot` command available in PATH.
 *
 * @param dot_file Path to the DOT file (with .dot extension)
 * @param output_file Path for output image (extension determines format: .pdf, .png, .svg)
 * @return true if rendering succeeded, false otherwise
 */
inline bool render_graph(const std::string &dot_file, const std::string &output_file) {
    // Determine output format from extension
    std::string format = "pdf"; // default
    size_t dot_pos = output_file.rfind('.');
    if (dot_pos != std::string::npos) {
        format = output_file.substr(dot_pos + 1);
    }

    // Build command: dot -Tformat input.dot -o output.ext
    std::string cmd = "dot -T" + format + " \"" + dot_file + "\" -o \"" + output_file + "\"";

    int result = std::system(cmd.c_str());
    return result == 0;
}

/**
 * @brief Convenience function: export expression to DOT and render to PDF
 *
 * Creates both a .dot file and a .pdf file with the given base name.
 *
 * @param expr The symbolic expression to visualize
 * @param output_base Base filename (creates output_base.dot and output_base.pdf)
 * @return true if both export and render succeeded
 */
inline bool visualize_graph(const SymbolicScalar &expr, const std::string &output_base) {
    try {
        export_graph_dot(expr, output_base);
        return render_graph(output_base + ".dot", output_base + ".pdf");
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace janus
