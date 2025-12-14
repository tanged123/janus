#pragma once
#include <casadi/casadi.hpp>
#include <concepts>

namespace janus {
/**
 * @brief Concenpt for valid Janus scalars
 *
 * Satisfied by floating point types (double, float) or CasADi symbolic types (casadi::MX).
 */
template <typename T>
concept JanusScalar = std::floating_point<T> || std::same_as<T, casadi::MX>;
} // namespace janus
