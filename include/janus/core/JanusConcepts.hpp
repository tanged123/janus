#pragma once
#include <casadi/casadi.hpp>
#include <concepts>

namespace janus {
template <typename T>
concept JanusScalar = std::floating_point<T> || std::same_as<T, casadi::MX>;
}
