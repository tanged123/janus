#pragma once
#include "janus/core/JanusConcepts.hpp"
#include "janus/math/Trig.hpp" // for cos, pi
#include <Eigen/Dense>
#include <numbers>

namespace janus {

// --- linspace ---
// Returns a vector of n points linearly spaced between start and end (inclusive)
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> linspace(const T& start, const T& end, int n) {
    if (n < 2) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> ret(1);
        ret(0) = start; // degenerate case
        return ret;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(n);
    // Explicit loop to support symbolic types (cannot use .setLinSpaced)
    // step = (end - start) / (n - 1)
    T step = (end - start) / static_cast<double>(n - 1);
    
    for (int i = 0; i < n; ++i) {
        ret(i) = start + static_cast<double>(i) * step;
    }
    // Ensure exact end point
    ret(n - 1) = end; 
    return ret;
}

// --- cosine_spacing ---
// Returns a vector of n points with cosine spacing (denser at ends)
// x_i = 0.5 * (start + end) - 0.5 * (end - start) * cos(pi * i / (n - 1))
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> cosine_spacing(const T& start, const T& end, int n) {
    if (n < 2) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> ret(1);
        ret(0) = start;
        return ret;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(n);
    T center = 0.5 * (start + end);
    T radius = 0.5 * (end - start);
    double pi = std::numbers::pi_v<double>;

    for (int i = 0; i < n; ++i) {
        // We use std::cos here because the argument is always double (setup logic), 
        // BUT we want to support symbolic start/end.
        // If we want consistency, we can use janus::cos if we cast argument to T, 
        // but typically the angle fraction is purely numeric.
        // Actually, let's keep the fraction numeric.
        double angle = pi * static_cast<double>(i) / static_cast<double>(n - 1);
        // We need T's cos if T is symbolic? No, angle is double. 
        // Wait, if T is symbolic, `radius * cos(angle)` involves mixing T and double. 
        // This is fine for CasADi (MX * double is valid).
        ret(i) = center - radius * std::cos(angle);
    }
    return ret;
}

} // namespace janus
