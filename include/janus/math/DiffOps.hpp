#pragma once
#include "janus/core/JanusConcepts.hpp"
#include <Eigen/Dense>

namespace janus {

// --- diff(vector) ---
// Returns adjacent differences: out[i] = v[i+1] - v[i]
// Returns a vector of size N-1
template <typename Derived>
auto diff(const Eigen::MatrixBase<Derived>& v) {
    // Return expression template for efficiency
    return (v.tail(v.size() - 1) - v.head(v.size() - 1));
}

// --- trapz(y, x) ---
// Trapezoidal integration: sum(0.5 * (y[i+1]+y[i]) * (x[i+1]-x[i]))
template <typename DerivedY, typename DerivedX>
auto trapz(const Eigen::MatrixBase<DerivedY>& y, const Eigen::MatrixBase<DerivedX>& x) {
    auto dx = (x.tail(x.size() - 1) - x.head(x.size() - 1));
    auto mean_y = 0.5 * (y.tail(y.size() - 1) + y.head(y.size() - 1));
    
    // Sum of element-wise product
    return (mean_y.array() * dx.array()).sum();
}

// --- gradient_1d(y, x) ---
// Central difference gradient
// Returns vector of same size as inputs
template <typename DerivedY, typename DerivedX>
auto gradient_1d(const Eigen::MatrixBase<DerivedY>& y, const Eigen::MatrixBase<DerivedX>& x) {
    Eigen::Index n = y.size();
    typename DerivedY::PlainObject grad(n);
    
    if (n < 2) {
        if (n > 0) grad.setZero();
        return grad;
    }

    // Interior points: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    if (n > 2) {
        grad.segment(1, n-2) = (y.tail(n-2) - y.head(n-2)).array() / 
                               (x.tail(n-2) - x.head(n-2)).array();
    }
    
    // Boundaries (Forward/Backward difference)
    // Forward at 0
    grad(0) = (y(1) - y(0)) / (x(1) - x(0));
    // Backward at n-1
    grad(n-1) = (y(n-1) - y(n-2)) / (x(n-1) - x(n-2));
    
    return grad;
}

} // namespace janus
