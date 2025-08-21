#include "compression.h"

#include "core/standard_includes.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace sep {
namespace core {

// Default compression implementation using simple RLE
class DefaultCompressor : public CompressionStrategy {
public:
    std::vector<uint8_t> compress(const void* data, size_t size) override
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        std::vector<uint8_t> compressed;
        compressed.reserve(size);

        for (size_t i = 0; i < size; i++) {
            uint8_t count = 1;
            uint8_t current = bytes[i];

            while (i + 1 < size && bytes[i + 1] == current && count < 255) {
                count++;
                i++;
            }

            compressed.push_back(count);
            compressed.push_back(current);
        }

        return compressed;
    }

    bool decompress(const std::vector<uint8_t>& compressed, void* output,
                    size_t outputSize) override
    {
        uint8_t* out = static_cast<uint8_t*>(output);
        size_t outPos = 0;
        
        for (size_t i = 0; i < compressed.size(); i += 2) {
            if (i + 1 >= compressed.size()) return false;
            
            uint8_t count = compressed[i];
            uint8_t value = compressed[i + 1];
            
            if (outPos + count > outputSize) return false;
            
            std::fill_n(out + outPos, count, value);
            outPos += count;
        }
        
        return outPos == outputSize;
    }

    sep::memory::CompressionMethod getMethod() const override
    {
        return sep::memory::CompressionMethod::None;
    }

    CompressionStats getStats() const override {
        return stats_;
    }

private:
    CompressionStats stats_{};
};

// Factory method implementation
std::unique_ptr<CompressionStrategy> CompressionFactory::create(
    sep::memory::CompressionMethod method)
{
    switch (method) {
        case sep::memory::CompressionMethod::ZSTD:
        case sep::memory::CompressionMethod::None:
        default:
            // Only a simple RLE compressor is implemented in this minimal build
            return std::make_unique<DefaultCompressor>();
    }
}

sep::memory::CompressionMethod CompressionFactory::analyzeData(const void* /*data*/,
                                                               size_t /*size*/)
{
    // Minimal heuristic: always return None
    return sep::memory::CompressionMethod::None;
}

float CompressionFactory::estimateCompressionRatio(const void* data, size_t size,
                                                   sep::memory::CompressionMethod method)
{
    if (!data || size == 0) {
        return 0.0f;
    }

    // Simple method-based estimation to eliminate unused parameter warning
    switch (method) {
        case sep::memory::CompressionMethod::ZSTD:
            return 0.6f;  // ZSTD typically achieves ~40% compression
        case sep::memory::CompressionMethod::None:
        default:
            return 1.0f;  // No compression
    }
}

// Utility functions implementation
namespace compression_utils {

float calculateEntropy(const void* data, size_t size) {
    if (!data || size == 0)
        return 0.0f;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    std::unordered_map<uint8_t, size_t> counts;

    for (size_t i = 0; i < size; ++i) {
        counts[bytes[i]]++;
    }

    float entropy = 0.0f;
    for (const auto& pair : counts) {
        float p = static_cast<float>(pair.second) / size;
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

float calculateNormalizedEntropy(const void* data, size_t size) {
    float entropy = calculateEntropy(data, size);
    return entropy / 8.0f;  // Normalize to [0,1] for 8-bit data
}

bool hasRepeatingPatterns(const void* data, size_t size) {
    if (!data || size < 2)
        return false;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    size_t repeats = 0;

    for (size_t i = 0; i < size - 1; ++i) {
        if (bytes[i] == bytes[i + 1]) {
            repeats++;
        }
    }

    return (repeats * 4) > size;  // >25% adjacent repeats
}

std::vector<uint8_t> downsample(const void* data, size_t size, size_t factor) {
    if (!data || size == 0 || factor == 0) {
        return {};
    }

    if (factor == 1) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        return std::vector<uint8_t>(bytes, bytes + size);
    }

    std::vector<uint8_t> result;
    result.reserve(size / factor + 1);

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; i += factor) {
        result.push_back(bytes[i]);
    }

    return result;
}

std::vector<uint8_t> upsample(const std::vector<uint8_t>& data, size_t original_size,
                              size_t factor) {
    if (data.empty() || factor == 0) {
        return {};
    }

    if (factor == 1) {
        return data;
    }

    std::vector<uint8_t> result;
    result.reserve(original_size);

    for (size_t i = 0; i < data.size() && result.size() < original_size; ++i) {
        for (size_t j = 0; j < factor && result.size() < original_size; ++j) {
            result.push_back(data[i]);
        }
    }

    return result;
}

}  // namespace compression_utils

} // namespace core
} // namespace sep