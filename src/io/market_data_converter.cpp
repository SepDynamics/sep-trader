#include "io/market_data_converter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace sep {
namespace connectors {

std::vector<uint8_t> MarketDataConverter::candlesToByteStream(
    const std::vector<OandaCandle>& candles) {
    std::vector<uint8_t> stream;

    // Reserve space for efficient memory allocation
    stream.reserve(candles.size() * 64);  // Estimate 64 bytes per candle

    // Extract price vectors for normalization
    std::vector<double> opens, highs, lows, closes, volumes;
    for (const auto& candle : candles) {
        opens.push_back(candle.open);
        highs.push_back(candle.high);
        lows.push_back(candle.low);
        closes.push_back(candle.close);
        volumes.push_back(static_cast<double>(candle.volume));

        appendTimestamp(stream,
                        candle.time);  // Use 'time' field which is the primary timestamp field
        appendDouble(stream, candle.open);
        appendDouble(stream, candle.high);
        appendDouble(stream, candle.low);
        appendDouble(stream, candle.close);
        appendUint64(stream, static_cast<uint64_t>(candle.volume));
    }

    return stream;
}

std::vector<uint8_t> MarketDataConverter::convertToBitstream(const std::vector<double>& prices) {
    if (prices.empty()) {
        return {};
    }

    // Calculate statistics for normalization
    auto [mean, std] = calculateMeanStd(prices);

    std::vector<uint8_t> bitstream;
    bitstream.reserve(prices.size() * 8);  // Each double -> 8 bytes

    // Convert each price to normalized binary representation
    for (double price : prices) {
        double normalized = normalizeValue(price, mean, std);

        // Convert to bit representation using IEEE 754 double precision
        uint64_t bits = *reinterpret_cast<const uint64_t*>(&normalized);

        // Store as 8 bytes (64 bits)
        for (int i = 0; i < 8; ++i) {
            bitstream.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
        }
    }

    return bitstream;
}

std::vector<uint8_t> MarketDataConverter::orderBookToByteStream(
    const std::vector<std::pair<double, double>>& bid_book,
    const std::vector<std::pair<double, double>>& ask_book) {
    std::vector<uint8_t> stream;

    // Encode bid book size
    appendUint64(stream, bid_book.size());

    // Encode bid book (price, volume pairs)
    for (const auto& [price, volume] : bid_book) {
        appendDouble(stream, price);
        appendDouble(stream, volume);
    }

    // Encode ask book size
    appendUint64(stream, ask_book.size());

    // Encode ask book (price, volume pairs)
    for (const auto& [price, volume] : ask_book) {
        appendDouble(stream, price);
        appendDouble(stream, volume);
    }

    return stream;
}

std::vector<uint8_t> MarketDataConverter::createCompositeStream(
    const std::vector<OandaCandle>& candles, const MarketData& market_data, size_t window_size) {
    std::vector<uint8_t> stream;

    // Limit candles to window size
    size_t start_idx = candles.size() > window_size ? candles.size() - window_size : 0;
    std::vector<OandaCandle> windowed_candles(candles.begin() + start_idx, candles.end());

    // Convert candle data
    auto candle_stream = candlesToByteStream(windowed_candles);
    stream.insert(stream.end(), candle_stream.begin(), candle_stream.end());

    // Add market data
    appendDouble(stream, market_data.bid);
    appendDouble(stream, market_data.ask);
    appendDouble(stream, market_data.mid);
    appendDouble(stream, market_data.volume);
    appendUint64(stream, market_data.timestamp);

    // Add technical indicators
    appendDouble(stream, market_data.atr);
    appendUint64(stream, static_cast<uint64_t>(market_data.volatility_level));
    appendDouble(stream, market_data.spread);
    appendDouble(stream, market_data.daily_change);

    // Add order book data if available
    if (!market_data.bid_book.empty() && !market_data.ask_book.empty()) {
        // Convert flat vectors to price/volume pairs (assuming alternating price/volume)
        std::vector<std::pair<double, double>> bid_pairs, ask_pairs;

        for (size_t i = 0; i + 1 < market_data.bid_book.size(); i += 2) {
            bid_pairs.emplace_back(market_data.bid_book[i], market_data.bid_book[i + 1]);
        }

        for (size_t i = 0; i + 1 < market_data.ask_book.size(); i += 2) {
            ask_pairs.emplace_back(market_data.ask_book[i], market_data.ask_book[i + 1]);
        }

        auto orderbook_stream = orderBookToByteStream(bid_pairs, ask_pairs);
        stream.insert(stream.end(), orderbook_stream.begin(), orderbook_stream.end());
    }

    return stream;
}

// Private helper methods

void MarketDataConverter::appendDouble(std::vector<uint8_t>& stream, double value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    stream.insert(stream.end(), bytes, bytes + sizeof(double));
}

void MarketDataConverter::appendUint64(std::vector<uint8_t>& stream, uint64_t value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    stream.insert(stream.end(), bytes, bytes + sizeof(uint64_t));
}

void MarketDataConverter::appendTimestamp(std::vector<uint8_t>& stream,
                                          const std::string& timestamp) {
    std::tm tm{};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        appendUint64(stream, 0);
        return;
    }
#ifdef _WIN32
    std::time_t time = _mkgmtime(&tm);
#else
    std::time_t time = timegm(&tm);
#endif
    uint64_t unix_time = static_cast<uint64_t>(time);
    appendUint64(stream, unix_time);
}

std::pair<double, double> MarketDataConverter::calculateMeanStd(const std::vector<double>& values) {
    if (values.empty()) {
        return {0.0, 1.0};
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();

    double std_dev = std::sqrt(variance);
    return {mean, std_dev == 0.0 ? 1.0 : std_dev};
}

double MarketDataConverter::normalizeValue(double value, double mean, double std) {
    return (value - mean) / std;
}

}  // namespace connectors
}  // namespace sep