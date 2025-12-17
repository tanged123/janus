#pragma once

#include "JanusIO.hpp"
#include "JanusTypes.hpp"
#include <casadi/casadi.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace janus {

// Forward declaration
class Function;

/**
 * @brief Wrapper around CasADi Sparsity for pattern analysis
 *
 * Provides query interface for Jacobian/Hessian sparsity patterns useful for
 * debugging, optimization configuration, and visualization.
 *
 * @example
 * ```cpp
 * auto x = janus::sym("x", 3);
 * auto y = janus::sym("y", 3);
 * auto f = casadi::MX::dot(x, y);  // x'*y
 *
 * auto sp = janus::jacobian_sparsity(f, casadi::MX::vertcat({x, y}));
 * std::cout << "Jacobian size: " << sp.n_rows() << "x" << sp.n_cols() << "\n";
 * std::cout << "Non-zeros: " << sp.nnz() << "\n";
 * std::cout << sp.to_string() << "\n";  // ASCII spy plot
 * ```
 */
class SparsityPattern {
  public:
    // Default constructor creates an empty 0x0 sparsity (not null)
    SparsityPattern() : sp_(casadi::Sparsity(0, 0)) {}

    /**
     * @brief Construct from CasADi Sparsity object
     */
    explicit SparsityPattern(const casadi::Sparsity &sp) : sp_(sp) {}

    /**
     * @brief Construct from CasADi MX (extracts its sparsity)
     */
    explicit SparsityPattern(const SymbolicScalar &mx) : sp_(mx.sparsity()) {}

    // === Query Interface ===

    /**
     * @brief Number of rows in the pattern
     */
    int n_rows() const { return static_cast<int>(sp_.size1()); }

    /**
     * @brief Number of columns in the pattern
     */
    int n_cols() const { return static_cast<int>(sp_.size2()); }

    /**
     * @brief Number of structural non-zeros
     */
    int nnz() const { return static_cast<int>(sp_.nnz()); }

    /**
     * @brief Sparsity density (nnz / total elements)
     */
    double density() const {
        int total = n_rows() * n_cols();
        return total > 0 ? static_cast<double>(nnz()) / total : 0.0;
    }

    /**
     * @brief Check if a specific element is structurally non-zero
     */
    bool has_nz(int row, int col) const { return sp_.has_nz(row, col); }

    /**
     * @brief Get all non-zero positions as (row, col) pairs
     */
    std::vector<std::pair<int, int>> nonzeros() const {
        std::vector<std::pair<int, int>> result;
        result.reserve(nnz());

        auto [row_idx, col_idx] = get_triplet();
        for (size_t i = 0; i < row_idx.size(); ++i) {
            result.emplace_back(row_idx[i], col_idx[i]);
        }
        return result;
    }

    // === Export Formats ===

    /**
     * @brief Get triplet format (row indices, column indices)
     */
    std::tuple<std::vector<int>, std::vector<int>> get_triplet() const {
        std::vector<casadi_int> rows, cols;
        sp_.get_triplet(rows, cols);

        // Convert casadi_int to int
        std::vector<int> row_int(rows.begin(), rows.end());
        std::vector<int> col_int(cols.begin(), cols.end());
        return {row_int, col_int};
    }

    /**
     * @brief Get Compressed Row Storage (CRS) format
     * @return (row_ptr, col_idx) where row_ptr has size n_rows+1
     */
    std::tuple<std::vector<int>, std::vector<int>> get_crs() const {
        auto crs_sp = sp_.T(); // Transpose to get CRS from CCS
        std::vector<casadi_int> colind = crs_sp.get_colind();
        std::vector<casadi_int> row = crs_sp.get_row();

        std::vector<int> row_ptr(colind.begin(), colind.end());
        std::vector<int> col_idx(row.begin(), row.end());
        return {row_ptr, col_idx};
    }

    /**
     * @brief Get Compressed Column Storage (CCS) format
     * @return (col_ptr, row_idx) where col_ptr has size n_cols+1
     */
    std::tuple<std::vector<int>, std::vector<int>> get_ccs() const {
        std::vector<casadi_int> colind = sp_.get_colind();
        std::vector<casadi_int> row = sp_.get_row();

        std::vector<int> col_ptr(colind.begin(), colind.end());
        std::vector<int> row_idx(row.begin(), row.end());
        return {col_ptr, row_idx};
    }

    // === Visualization ===

    /**
     * @brief Generate ASCII spy plot of sparsity pattern
     *
     * Returns a string representation where '*' = non-zero, '.' = zero
     *
     * @param max_rows Maximum rows to display (default 40)
     * @param max_cols Maximum cols to display (default 80)
     * @return ASCII art string
     */
    std::string to_string(int max_rows = 40, int max_cols = 80) const {
        std::ostringstream oss;

        int rows = n_rows();
        int cols = n_cols();

        // Header
        oss << "Sparsity: " << rows << "x" << cols << ", nnz=" << nnz()
            << " (density=" << std::fixed << std::setprecision(3) << density() * 100 << "%)\n";

        // Limit display size
        int disp_rows = std::min(rows, max_rows);
        int disp_cols = std::min(cols, max_cols);

        // Top border
        oss << "┌";
        for (int j = 0; j < disp_cols; ++j)
            oss << "─";
        if (cols > max_cols)
            oss << "…";
        oss << "┐\n";

        // Rows
        for (int i = 0; i < disp_rows; ++i) {
            oss << "│";
            for (int j = 0; j < disp_cols; ++j) {
                oss << (has_nz(i, j) ? '*' : '.');
            }
            if (cols > max_cols)
                oss << "…";
            oss << "│\n";
        }

        if (rows > max_rows) {
            oss << "│";
            for (int j = 0; j < disp_cols; ++j)
                oss << "⋮";
            if (cols > max_cols)
                oss << "…";
            oss << "│\n";
        }

        // Bottom border
        oss << "└";
        for (int j = 0; j < disp_cols; ++j)
            oss << "─";
        if (cols > max_cols)
            oss << "…";
        oss << "┘\n";

        return oss.str();
    }

    /**
     * @brief Export sparsity pattern to DOT format as a matrix grid (spy plot)
     *
     * Creates a DOT file that renders as a grid where non-zeros are colored squares.
     * This mimics standard spy() plots found in MATLAB/Python.
     *
     * @param filename Output filename (without .dot extension)
     * @param name Graph name
     */
    void export_spy_dot(const std::string &filename, const std::string &name = "sparsity") const {
        std::ofstream file(filename + ".dot");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename + ".dot");
        }

        file << "digraph " << name << " {\n";
        file << "  node [shape=plaintext];\n";
        file << "  labelloc=\"t\";\n";
        file << "  label=\"" << name << " (" << n_rows() << "x" << n_cols() << ", nnz=" << nnz()
             << ")\";\n\n";

        file << "  matrix [label=<\n";
        file << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";

        // Iterate rows (HTML tables are row-major)
        for (int i = 0; i < n_rows(); ++i) {
            file << "      <TR>\n";
            for (int j = 0; j < n_cols(); ++j) {
                if (has_nz(i, j)) {
                    // Non-zero element: Black (or dark blue)
                    file << "        <TD BGCOLOR=\"black\" WIDTH=\"20\" HEIGHT=\"20\"></TD>\n";
                } else {
                    // Zero element: White (or transparent)
                    file << "        <TD BGCOLOR=\"white\" WIDTH=\"20\" HEIGHT=\"20\"></TD>\n";
                }
            }
            file << "      </TR>\n";
        }

        file << "    </TABLE>\n";
        file << "  >];\n";
        file << "}\n";
        file.close();
    }

    /**
     * @brief Export/Render sparsity pattern to PDF via Graphviz
     *
     * Creates a DOT file and renders it to PDF using the `dot` command.
     * Use this for high-quality visualization of sparsity patterns.
     *
     * @param output_base Base filename (creates output_base.dot and output_base.pdf)
     * @return true if successful (requires Graphviz installed)
     */
    bool visualize_spy(const std::string &output_base) const {
        try {
            export_spy_dot(output_base, output_base);
            // render_graph is defined in JanusIO.hpp
            return render_graph(output_base + ".dot", output_base + ".pdf");
        } catch (...) {
            return false;
        }
    }

    // === Underlying Access ===

    /**
     * @brief Access underlying CasADi Sparsity object
     */
    const casadi::Sparsity &casadi_sparsity() const { return sp_; }

    // === Operators ===

    bool operator==(const SparsityPattern &other) const { return sp_ == other.sp_; }
    bool operator!=(const SparsityPattern &other) const { return sp_ != other.sp_; }

  private:
    casadi::Sparsity sp_;
};

// === Sparsity Query Functions ===

/**
 * @brief Get Jacobian sparsity without computing the full Jacobian
 *
 * This is more efficient than computing the Jacobian and extracting its sparsity.
 * Named sparsity_of_jacobian to avoid collision with casadi::jacobian_sparsity.
 *
 * @param expr Expression to differentiate (output)
 * @param vars Variables to differentiate with respect to (input)
 * @return SparsityPattern of the Jacobian
 */
inline SparsityPattern sparsity_of_jacobian(const SymbolicScalar &expr,
                                            const SymbolicScalar &vars) {
    casadi::Sparsity sp = casadi::MX::jacobian_sparsity(expr, vars);
    return SparsityPattern(sp);
}

/**
 * @brief Get Hessian sparsity of a scalar expression
 *
 * Named sparsity_of_hessian to avoid collision with potential CasADi functions.
 *
 * @param expr Scalar expression
 * @param vars Variables
 * @return SparsityPattern of the Hessian (symmetric)
 */
inline SparsityPattern sparsity_of_hessian(const SymbolicScalar &expr, const SymbolicScalar &vars) {
    // Hessian sparsity = Jacobian sparsity of the gradient
    auto grad = casadi::MX::gradient(expr, vars);
    casadi::Sparsity sp = casadi::MX::jacobian_sparsity(grad, vars);
    return SparsityPattern(sp);
}

/**
 * @brief Get sparsity of a janus::Function Jacobian
 *
 * @param fn The function
 * @param output_idx Output index (default 0)
 * @param input_idx Input index (default 0)
 * @return SparsityPattern of the function's Jacobian
 */
inline SparsityPattern get_jacobian_sparsity(const Function &fn, int output_idx = 0,
                                             int input_idx = 0) {
    // Get the Jacobian function and extract its output sparsity
    casadi::Function jac_fn = fn.casadi_function().jacobian();
    casadi::Sparsity sp = jac_fn.sparsity_out(0);
    return SparsityPattern(sp);
}

/**
 * @brief Get input sparsity of a janus::Function
 */
inline SparsityPattern get_sparsity_in(const Function &fn, int input_idx = 0) {
    casadi::Sparsity sp = fn.casadi_function().sparsity_in(input_idx);
    return SparsityPattern(sp);
}

/**
 * @brief Get output sparsity of a janus::Function
 */
inline SparsityPattern get_sparsity_out(const Function &fn, int output_idx = 0) {
    casadi::Sparsity sp = fn.casadi_function().sparsity_out(output_idx);
    return SparsityPattern(sp);
}

} // namespace janus
