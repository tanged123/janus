#pragma once

/**
 * @file JanusMath.hpp
 * @brief Master header for Janus Math library
 * Includes all math sub-modules.
 */

#include "janus/core/JanusError.hpp"
#include "janus/math/Arithmetic.hpp"
#include "janus/math/AutoDiff.hpp"
#include "janus/math/Calculus.hpp"
#include "janus/math/FiniteDifference.hpp"
#include "janus/math/Integrate.hpp"
#include "janus/math/IntegrateDiscrete.hpp"
#include "janus/math/Interpolate.hpp"
#include "janus/math/Linalg.hpp"
#include "janus/math/Logic.hpp"
#include "janus/math/Quaternion.hpp"
#include "janus/math/RootFinding.hpp"
#include "janus/math/Rotations.hpp"
#include "janus/math/Spacing.hpp"
#include "janus/math/SurrogateModel.hpp"
#include "janus/math/Trig.hpp"

// Deprecated headers (for backward compatibility if needed, but discouraged in new code)
// #include "janus/math/DiffOps.hpp"
