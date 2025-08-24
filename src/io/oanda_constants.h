#ifndef SEP_OANDA_CONSTANTS_H
#define SEP_OANDA_CONSTANTS_H

namespace sep {
namespace oanda_constants {

constexpr char SANDBOX_API_URL[] = "https://api-fxpractice.oanda.com";
constexpr char SANDBOX_STREAM_URL[] = "https://stream-fxpractice.oanda.com";
constexpr char LIVE_API_URL[] = "https://api-fxtrade.oanda.com";
constexpr char LIVE_STREAM_URL[] = "https://stream-fxtrade.oanda.com";
constexpr int DEFAULT_CANDLE_COUNT = 2880;
constexpr double MIN_FOREX_ATR = 0.0001;  // Minimum ATR value (1 pip)
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int SECONDS_PER_HOUR = 3600;
constexpr int SECONDS_PER_DAY = 86400;

}  // namespace oanda_constants
}  // namespace sep

#endif  // SEP_OANDA_CONSTANTS_H