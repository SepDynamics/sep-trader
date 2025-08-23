#pragma once

#include <cstdint>
#include <string>
#include <span>
#include <vector>
#include <unordered_map>
#include <limits>
#include <cmath>

namespace sep {

struct SignalProbe {
  std::string pair;
  uint64_t t;
  double v;
};

struct SignalOut {
  std::string id;
  uint64_t t;
  double score;
  uint8_t state;
};

enum class SignalState : uint8_t { None, Watch, Enter, Exit, Halt };

struct RuleInput {
  double ma;
  double ema;
  double vol;
  double rupture;
  double coherence;
};

struct RuleParams {
  double ma_thresh;
  double vol_min;
  double coherence_min;
};

inline SignalState evaluateSignal(const RuleInput& xin, const RuleParams& p) {
  auto clamp = [](double v) {
    if (!std::isfinite(v)) return std::numeric_limits<double>::quiet_NaN();
    return v < 0.0 ? 0.0 : v;
  };
  RuleInput x{clamp(xin.ma), clamp(xin.ema), clamp(xin.vol),
              clamp(xin.rupture), clamp(xin.coherence)};
  if (!std::isfinite(x.ma) || !std::isfinite(x.ema) ||
      !std::isfinite(x.vol) || !std::isfinite(x.rupture) ||
      !std::isfinite(x.coherence))
    return SignalState::None;
  if (x.vol < p.vol_min) return SignalState::None;
  if (x.coherence < p.coherence_min) return SignalState::Watch;
  if (x.ma > p.ma_thresh && x.rupture == 0.0) return SignalState::Enter;
  if (x.rupture > 0.0) return SignalState::Exit;
  return SignalState::Watch;
}

struct FusionCfg {
  int bucket_ms;
  double enter_hyst;
  double exit_hyst;
  std::unordered_map<std::string, double> weights;
};

class IQuantumPipeline {
public:
  virtual bool evaluate_batch(std::span<const SignalProbe> in,
                              std::vector<SignalOut>& out) = 0;
  virtual ~IQuantumPipeline() = default;
};

} // namespace sep

