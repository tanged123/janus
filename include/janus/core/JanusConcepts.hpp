/// @file JanusConcepts.hpp
/// @brief C++20 concepts constraining valid Janus scalar types
#pragma once
#include <casadi/casadi.hpp>
#include <concepts>

namespace janus {
/**
 * @brief Concept for valid Janus scalars
 * @tparam T Type to check (must be floating-point or casadi::MX)
 */
template <typename T>
concept JanusScalar = std::floating_point<T> || std::same_as<T, casadi::MX>;
} // namespace janus
