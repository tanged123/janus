#pragma once

#include "JanusIO.hpp"
#include "JanusTypes.hpp"
#include <casadi/casadi.hpp>
#include <cmath>
#include <limits>
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

    /**
     * @brief Export sparsity pattern to an interactive HTML file
     *
     * Creates a self-contained HTML file with an SVG spy plot.
     * Supports pan, zoom, and hover info showing (row, col) coordinates.
     *
     * @param filename Output filename (without extension, .html will be added)
     * @param name Optional title for the visualization
     */
    void export_spy_html(const std::string &filename, const std::string &name = "sparsity") const {
        std::string html_filename = filename + ".html";
        std::ofstream out(html_filename);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open file: " + html_filename);
        }

        int rows = n_rows();
        int cols = n_cols();
        int cell_size = 12; // pixels per cell

        // Limit cell size for very large matrices
        if (rows > 100 || cols > 100) {
            cell_size = std::max(3, 1200 / std::max(rows, cols));
        }

        // Add margins for axis labels
        int margin_left = 50;
        int margin_top = 40;
        int svg_width = cols * cell_size + margin_left + 20;
        int svg_height = rows * cell_size + margin_top + 20;

        // Determine tick interval based on matrix size
        int tick_interval = 1;
        if (rows > 50 || cols > 50)
            tick_interval = 10;
        else if (rows > 20 || cols > 20)
            tick_interval = 5;

        out << R"SPYHTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>)SPYHTML"
            << name << R"SPYHTML(</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; width: 100%; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; display: flex; }
        #controls { position: fixed; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.8);
                    padding: 12px; border-radius: 8px; }
        #controls button { margin: 2px; padding: 8px 14px; cursor: pointer; border: none;
                           border-radius: 4px; background: #4a4a6a; color: white; font-size: 13px; }
        #controls button:hover { background: #6a6a8a; }
        #container { flex: 1; cursor: grab; overflow: hidden; height: 100%; }
        #container:active { cursor: grabbing; }
        #sidebar { width: 280px; height: 100%; background: #16213e; padding: 16px; overflow-y: auto; 
                   border-left: 2px solid #0f3460; }
        #sidebar h2 { color: #e94560; margin-bottom: 12px; font-size: 16px; }
        #sidebar .section { margin-bottom: 14px; }
        #sidebar .label { color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
        #sidebar .value { background: #0f3460; padding: 10px; border-radius: 6px; font-family: monospace;
                          font-size: 14px; }
        #sidebar .stat { display: flex; justify-content: space-between; padding: 6px 0;
                         border-bottom: 1px solid #0f3460; }
        #sidebar .stat-value { color: #4dc3ff; font-weight: bold; }
        #info { position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8);
                padding: 10px; border-radius: 4px; font-size: 12px; }
        .nz { cursor: pointer; }
        .nz:hover { fill: #e94560 !important; }
        .axis-label { font-family: monospace; font-size: 10px; fill: #666; }
        .grid-line { stroke: #ddd; stroke-width: 0.5; }
    </style>
</head>
<body>
    <div id="controls">
        <button onclick="zoomIn()">Zoom +</button>
        <button onclick="zoomOut()">Zoom -</button>
        <button onclick="resetView()">Reset</button>
        <button onclick="fitToScreen()">Fit</button>
    </div>
    <div id="container">
        <svg id="spy" width=")SPYHTML"
            << svg_width << R"SPYHTML(" height=")SPYHTML" << svg_height
            << R"SPYHTML(" style="background:#ffffff;">
            <g id="matrix" transform="translate()SPYHTML"
            << margin_left << "," << margin_top << R"SPYHTML()">
)SPYHTML";

        // Draw grid lines at tick intervals
        for (int r = 0; r <= rows; r += tick_interval) {
            out << "                <line class=\"grid-line\" x1=\"0\" y1=\"" << (r * cell_size)
                << "\" x2=\"" << (cols * cell_size) << "\" y2=\"" << (r * cell_size) << "\"/>\n";
        }
        for (int c = 0; c <= cols; c += tick_interval) {
            out << "                <line class=\"grid-line\" x1=\"" << (c * cell_size)
                << "\" y1=\"0\" x2=\"" << (c * cell_size) << "\" y2=\"" << (rows * cell_size)
                << "\"/>\n";
        }

        // Draw axis labels
        for (int r = 0; r <= rows; r += tick_interval) {
            out << "                <text class=\"axis-label\" x=\"-5\" y=\""
                << (r * cell_size + cell_size / 2)
                << "\" text-anchor=\"end\" dominant-baseline=\"middle\">" << r << "</text>\n";
        }
        for (int c = 0; c <= cols; c += tick_interval) {
            out << "                <text class=\"axis-label\" x=\""
                << (c * cell_size + cell_size / 2) << "\" y=\"-8\" text-anchor=\"middle\">" << c
                << "</text>\n";
        }

        // Output non-zero cells as rectangles
        auto [row_idx, col_idx] = get_triplet();
        for (size_t i = 0; i < row_idx.size(); ++i) {
            int r = row_idx[i];
            int c = col_idx[i];
            out << "                <rect class=\"nz\" x=\"" << (c * cell_size) << "\" y=\""
                << (r * cell_size) << "\" width=\"" << cell_size << "\" height=\"" << cell_size
                << "\" fill=\"#1e3a5f\" data-row=\"" << r << "\" data-col=\"" << c << "\"/>\n";
        }

        out << R"SPYHTML(            </g>
        </svg>
    </div>
    <div id="sidebar">
        <h2>Sparsity Info</h2>
        <div class="section">
            <div class="stat"><span>Matrix Size</span><span class="stat-value">)SPYHTML"
            << rows << " x " << cols << R"SPYHTML(</span></div>
            <div class="stat"><span>Non-zeros</span><span class="stat-value">)SPYHTML"
            << nnz() << R"SPYHTML(</span></div>
            <div class="stat"><span>Density</span><span class="stat-value">)SPYHTML";

        // Format density nicely
        double dens = density() * 100;
        out << std::fixed << std::setprecision(2) << dens;

        out << R"SPYHTML(%</span></div>
        </div>
        <div class="section">
            <div class="label">Selected Cell</div>
            <div class="value" id="cell-info">Click on a non-zero cell</div>
        </div>
        <div class="section">
            <div class="label">Notation</div>
            <div class="value" id="notation-info">A(row, col)</div>
        </div>
    </div>
    <div id="info">Scroll to zoom - Drag to pan - Click cells for info</div>
    <script>
        let scale = 1, panX = 0, panY = 0, isDragging = false, startX, startY;
        const container = document.getElementById('container');
        const svg = document.getElementById('spy');
        const cellInfo = document.getElementById('cell-info');
        const notationInfo = document.getElementById('notation-info');
        const svgWidth = )SPYHTML"
            << svg_width << R"SPYHTML(;
        const svgHeight = )SPYHTML"
            << svg_height << R"SPYHTML(;
        const matrixRows = )SPYHTML"
            << rows << R"SPYHTML(;
        const matrixCols = )SPYHTML"
            << cols << R"SPYHTML(;

        fitToScreen();

        function updateTransform() {
            svg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
            svg.style.transformOrigin = '0 0';
        }
        function zoomIn() { scale *= 1.3; updateTransform(); }
        function zoomOut() { scale /= 1.3; updateTransform(); }
        function resetView() { scale = 1; panX = 0; panY = 0; updateTransform(); }
        function fitToScreen() {
            const availWidth = window.innerWidth - 280;
            const scaleX = (availWidth - 40) / svgWidth;
            const scaleY = (window.innerHeight - 40) / svgHeight;
            scale = Math.min(scaleX, scaleY);
            panX = (availWidth - svgWidth * scale) / 2;
            panY = (window.innerHeight - svgHeight * scale) / 2;
            updateTransform();
        }
        container.addEventListener('wheel', e => {
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
            panX = mouseX - (mouseX - panX) * zoomFactor;
            panY = mouseY - (mouseY - panY) * zoomFactor;
            scale *= zoomFactor;
            updateTransform();
        });
        container.addEventListener('mousedown', e => { 
            if (e.target.classList.contains('nz')) return;
            isDragging = true; startX = e.clientX - panX; startY = e.clientY - panY; 
        });
        container.addEventListener('mousemove', e => { if (isDragging) { panX = e.clientX - startX; panY = e.clientY - startY; updateTransform(); } });
        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('mouseleave', () => isDragging = false);

        // Cell interaction
        let selectedCell = null;
        svg.querySelectorAll('.nz').forEach(rect => {
            rect.addEventListener('click', e => {
                e.stopPropagation();
                if (selectedCell) selectedCell.style.stroke = 'none';
                selectedCell = e.target;
                selectedCell.style.stroke = '#e94560';
                selectedCell.style.strokeWidth = '2';
                
                const r = parseInt(e.target.dataset.row);
                const c = parseInt(e.target.dataset.col);
                cellInfo.textContent = `Row: ${r}, Col: ${c}`;
                notationInfo.textContent = `A(${r}, ${c}) = nonzero`;
            });
            rect.addEventListener('mouseenter', e => {
                const r = e.target.dataset.row;
                const c = e.target.dataset.col;
                cellInfo.textContent = `Row: ${r}, Col: ${c}`;
            });
        });
    </script>
</body>
</html>
)SPYHTML";
        out.close();
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

// =============================================================================
// NaN-Propagation Sparsity Detection
// =============================================================================

/**
 * @brief Options for NaN-propagation sparsity detection
 */
struct NaNSparsityOptions {
    NumericVector reference_point; ///< Point to evaluate at (default: zeros)
    double perturbation = 1e-7;    ///< Reserved for future finite-difference methods
};

/**
 * @brief Detect Jacobian sparsity using NaN propagation
 *
 * This provides black-box sparsity detection for functions where symbolic
 * sparsity analysis is not available. The algorithm:
 *
 * 1. Evaluate f(x) at a reference point to get baseline outputs
 * 2. For each input i: set x[i] = NaN, evaluate f(x_perturbed)
 * 3. If output[j] becomes NaN, then df[j]/dx[i] ≠ 0 (structurally)
 *
 * @note This is a conservative estimate - it may report false positives
 *       if NaN propagates through unused code paths, but won't miss
 *       actual dependencies.
 *
 * @code
 * // Detect sparsity of element-wise function
 * auto sp = janus::nan_propagation_sparsity(
 *     [](const NumericVector& x) {
 *         NumericVector y(x.size());
 *         for (int i = 0; i < x.size(); ++i) y(i) = x(i) * x(i);
 *         return y;
 *     },
 *     3, 3);  // 3 inputs, 3 outputs
 *
 * EXPECT_EQ(sp.nnz(), 3);  // Diagonal pattern
 * @endcode
 *
 * @tparam Func Function type: (const NumericVector&) -> NumericVector
 * @param fn Function to analyze
 * @param n_inputs Number of scalar inputs
 * @param n_outputs Number of scalar outputs
 * @param opts Options (reference point, etc.)
 * @return SparsityPattern of the Jacobian (n_outputs x n_inputs)
 */
template <typename Func>
SparsityPattern nan_propagation_sparsity(Func &&fn, int n_inputs, int n_outputs,
                                         const NaNSparsityOptions &opts = {}) {
    // Use reference point or default to zeros
    NumericVector x0(n_inputs);
    if (opts.reference_point.size() == n_inputs) {
        x0 = opts.reference_point;
    } else {
        x0.setZero();
    }

    // Evaluate at reference point to verify the function works
    NumericVector y0 = fn(x0);
    if (y0.size() != n_outputs) {
        throw std::invalid_argument("nan_propagation_sparsity: function returned " +
                                    std::to_string(y0.size()) + " outputs, expected " +
                                    std::to_string(n_outputs));
    }

    // Build sparsity pattern by probing each input with NaN
    std::vector<casadi_int> row_indices;
    std::vector<casadi_int> col_indices;

    for (int i = 0; i < n_inputs; ++i) {
        // Create perturbed input with NaN at position i
        NumericVector x_perturbed = x0;
        x_perturbed(i) = std::numeric_limits<double>::quiet_NaN();

        // Evaluate function
        NumericVector y_perturbed = fn(x_perturbed);

        // Check which outputs became NaN
        for (int j = 0; j < n_outputs; ++j) {
            if (std::isnan(y_perturbed(j))) {
                // Output j depends on input i
                row_indices.push_back(j);
                col_indices.push_back(i);
            }
        }
    }

    // Construct CasADi sparsity from triplets
    casadi::Sparsity sp = casadi::Sparsity::triplet(n_outputs, n_inputs, row_indices, col_indices);
    return SparsityPattern(sp);
}

/**
 * @brief Detect Jacobian sparsity of a janus::Function using NaN propagation
 *
 * Convenience overload that uses the Function's internal structure.
 *
 * @param fn Function to analyze (must have single vector input/output)
 * @param opts Options for NaN propagation
 * @return SparsityPattern of the Jacobian
 */
inline SparsityPattern nan_propagation_sparsity(const Function &fn,
                                                const NaNSparsityOptions &opts = {}) {
    // Get input/output dimensions from the function
    auto sp_in = fn.casadi_function().sparsity_in(0);
    auto sp_out = fn.casadi_function().sparsity_out(0);

    int n_inputs = static_cast<int>(sp_in.nnz());
    int n_outputs = static_cast<int>(sp_out.nnz());

    // Create wrapper that evaluates the function with a single vector input
    auto eval_fn = [&fn](const NumericVector &x) -> NumericVector {
        // Pass as a single Eigen vector (not as separate scalars)
        auto results = fn(x);
        // Flatten results to a single vector
        int total_size = 0;
        for (const auto &m : results) {
            total_size += static_cast<int>(m.size());
        }
        NumericVector y(total_size);
        int offset = 0;
        for (const auto &m : results) {
            for (int i = 0; i < m.size(); ++i) {
                y(offset + i) = m(i);
            }
            offset += static_cast<int>(m.size());
        }
        return y;
    };

    return nan_propagation_sparsity(eval_fn, n_inputs, n_outputs, opts);
}

} // namespace janus
