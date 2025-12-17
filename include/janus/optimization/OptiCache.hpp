#pragma once

#include "janus/utils/JsonUtils.hpp"
#include <map>
#include <string>
#include <vector>

namespace janus {

/**
 * @brief Optimization solution caching utilities
 *
 * Provides methods to load solutions. Saving is handled by OptiSol::save().
 */
class OptiCache {
  public:
    /**
     * @brief Load solution data from JSON file
     *
     * @param filename JSON file path
     * @return Map of variable names to value vectors
     * @throws std::runtime_error if file cannot be read or parsed
     */
    static std::map<std::string, std::vector<double>> load(const std::string &filename) {
        return janus::utils::read_json(filename);
    }
};

} // namespace janus
