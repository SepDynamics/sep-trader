#pragma once

#include "core/signal_types.h"
#include <string>
#include <unordered_map>

namespace sep {

struct EngineConfig {
  int version{1};
  RuleParams rules{};
  FusionCfg fusion{};
  std::unordered_map<std::string, std::string> error_templates;
};

bool load_engine_config(const std::string& json, EngineConfig& cfg, std::string* err = nullptr);
std::string save_engine_config(const EngineConfig& cfg);

} // namespace sep

