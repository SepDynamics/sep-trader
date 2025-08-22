#ifndef SEP_OANDA_CONSTANTS_H
#define SEP_OANDA_CONSTANTS_H

namespace sep {
namespace oanda_constants {

constexpr char SANDBOX_API_URL[] = "https://api-fxpractice.oanda.com";
constexpr char SANDBOX_STREAM_URL[] = "https://stream-fxpractice.oanda.com";
constexpr char LIVE_API_URL[] = "https://api-fxtrade.oanda.com";
constexpr char LIVE_STREAM_URL[] = "https://stream-fxtrade.oanda.com";
constexpr int DEFAULT_CANDLE_COUNT = 2880;

} // namespace oanda_constants
} // namespace sep

#endif // SEP_OANDA_CONSTANTS_H