// SEP Live Tuner Header
// Real-time parameter tuning system

#ifndef LIVE_TUNER_HPP
#define LIVE_TUNER_HPP

namespace sep {
namespace training {

class LiveTuner {
public:
    LiveTuner() = default;
    ~LiveTuner() = default;
    
    bool startTuning();
};

} // namespace training
} // namespace sep

#endif // LIVE_TUNER_HPP
