#ifndef SEP_UTIL_PATTERN_PROCESSING_HPP
#define SEP_UTIL_PATTERN_PROCESSING_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace sep {
namespace util {

inline void extract_bitstream_from_pattern_id(const std::string& pattern_id, std::vector<uint8_t>& bitstream) {
    bitstream.clear();
    for (char c : pattern_id) {
        uint8_t char_val = static_cast<uint8_t>(c);
        for (int bit = 0; bit < 8; ++bit) {
            bitstream.push_back((char_val >> bit) & 1);
        }
    }
}

} // namespace util
} // namespace sep

#endif // SEP_UTIL_PATTERN_PROCESSING_HPP