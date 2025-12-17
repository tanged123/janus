#pragma once

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace janus::utils {

/**
 * @brief Simple JSON writer for map of vectors
 *
 * Writes strictly formatted JSON:
 * {
 *   "key1": [1.1, 2.2],
 *   "key2": [3.3]
 * }
 */
inline void write_json(const std::string &filename,
                       const std::map<std::string, std::vector<double>> &data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file << "{\n";
    auto it = data.begin();
    while (it != data.end()) {
        file << "  \"" << it->first << "\": [";
        const auto &vec = it->second;
        for (size_t i = 0; i < vec.size(); ++i) {
            file << std::scientific << std::setprecision(16) << vec[i];
            if (i < vec.size() - 1) {
                file << ", ";
            }
        }
        file << "]";

        if (++it != data.end()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

/**
 * @brief Tiny JSON parser for flat string->vector<double> maps
 *
 * Very limited parser. Expects:
 * - Top-level object {}
 * - Keys are strings
 * - Values are arrays of numbers
 * - No nested objects or other types
 */
inline std::map<std::string, std::vector<double>> read_json(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    std::map<std::string, std::vector<double>> data;

    // Very naive parser
    size_t pos = 0;
    while (pos < content.length()) {
        // Find key
        size_t quote_start = content.find('"', pos);
        if (quote_start == std::string::npos)
            break;

        size_t quote_end = content.find('"', quote_start + 1);
        if (quote_end == std::string::npos)
            throw std::runtime_error("Malformed JSON: Unclosed string");

        std::string key = content.substr(quote_start + 1, quote_end - quote_start - 1);
        pos = quote_end + 1;

        // Find colon
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos)
            throw std::runtime_error("Malformed JSON: Missing colon");
        pos = colon + 1;

        // Find value (array)
        size_t bracket_start = content.find('[', pos);
        if (bracket_start == std::string::npos)
            throw std::runtime_error("Malformed JSON: Missing array start [");
        pos = bracket_start + 1;

        size_t bracket_end = content.find(']', pos);
        if (bracket_end == std::string::npos)
            throw std::runtime_error("Malformed JSON: Missing array end ]");

        // Parse array content
        std::string array_content = content.substr(pos, bracket_end - pos);
        std::vector<double> vec;
        std::stringstream ss(array_content);
        std::string number_str;

        while (std::getline(ss, number_str, ',')) {
            // Trim whitespace
            number_str.erase(0, number_str.find_first_not_of(" \t\n\r"));
            number_str.erase(number_str.find_last_not_of(" \t\n\r") + 1);
            if (!number_str.empty()) {
                try {
                    vec.push_back(std::stod(number_str));
                } catch (...) {
                    // Ignore parsing errors for now or throw
                }
            }
        }
        data[key] = vec;

        pos = bracket_end + 1;
    }

    return data;
}

} // namespace janus::utils
