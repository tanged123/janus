#pragma once
/**
 * @file JanusError.hpp
 * @brief Custom exception hierarchy for Janus framework
 *
 * Provides consistent error handling with contextual messages.
 * All exceptions derive from std::runtime_error for backward compatibility.
 */

#include <stdexcept>
#include <string>

namespace janus {

/**
 * @brief Base exception for all Janus errors
 * @see InvalidArgument, RuntimeError, InterpolationError, IntegrationError
 */
class JanusError : public std::runtime_error {
  public:
    /**
     * @brief Construct with a descriptive message
     * @param what Error description (automatically prefixed with "[janus]")
     */
    explicit JanusError(const std::string &what) : std::runtime_error("[janus] " + what) {}
};

/**
 * @brief Input validation failed (e.g., mismatched sizes, invalid parameters)
 */
class InvalidArgument : public JanusError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description
    explicit InvalidArgument(const std::string &what) : JanusError(what) {}
};

/**
 * @brief Operation failed at runtime (e.g., CasADi eval with free variables)
 */
class RuntimeError : public JanusError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description
    explicit RuntimeError(const std::string &what) : JanusError(what) {}
};

/**
 * @brief Interpolation-specific errors
 */
class InterpolationError : public JanusError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description (automatically prefixed with "Interpolation:")
    explicit InterpolationError(const std::string &what) : JanusError("Interpolation: " + what) {}
};

/**
 * @brief Integration/ODE solver errors
 */
class IntegrationError : public JanusError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description (automatically prefixed with "Integration:")
    explicit IntegrationError(const std::string &what) : JanusError("Integration: " + what) {}
};

} // namespace janus
