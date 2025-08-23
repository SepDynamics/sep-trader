#include "engine_config_io.h"
#include "util/nlohmann_json_safe.h"
#include <map>

namespace sep {

using ordered_json = nlohmann::ordered_json;

bool load_engine_config(const std::string& json, EngineConfig& cfg, std::string* err) {
  ordered_json j;
  try {
    j = ordered_json::parse(json);
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
  if (j.value("version", 0) != 1) {
    if (err) *err = "bad version";
    return false;
  }
  if (!j.contains("rules")) {
    if (err) *err = "rules missing";
    return false;
  }
  auto& r = j["rules"];
  if (!(r.contains("ma_thresh") && r.contains("vol_min") && r.contains("coherence_min"))) {
    if (err) *err = "rules incomplete";
    return false;
  }
  cfg.version = 1;
  cfg.rules.ma_thresh = r["ma_thresh"].get<double>();
  cfg.rules.vol_min = r["vol_min"].get<double>();
  cfg.rules.coherence_min = r["coherence_min"].get<double>();
  if (!j.contains("fusion")) {
    if (err) *err = "fusion missing";
    return false;
  }
  auto& f = j["fusion"];
  if (!(f.contains("bucket_ms") && f.contains("enter_hyst") && f.contains("exit_hyst") && f.contains("weights"))) {
    if (err) *err = "fusion incomplete";
    return false;
  }
  cfg.fusion.bucket_ms = f["bucket_ms"].get<int>();
  cfg.fusion.enter_hyst = f["enter_hyst"].get<double>();
  cfg.fusion.exit_hyst = f["exit_hyst"].get<double>();
  cfg.fusion.weights.clear();
  for (auto it = f["weights"].begin(); it != f["weights"].end(); ++it) {
    cfg.fusion.weights[it.key()] = it.value().get<double>();
  }
  cfg.error_templates.clear();
  if (j.contains("error_templates")) {
    for (auto it = j["error_templates"].begin(); it != j["error_templates"].end(); ++it) {
      cfg.error_templates[it.key()] = it.value().get<std::string>();
    }
  }
  return true;
}

std::string save_engine_config(const EngineConfig& cfg) {
  ordered_json j;
  j["version"] = cfg.version;
  ordered_json rules;
  rules["ma_thresh"] = cfg.rules.ma_thresh;
  rules["vol_min"] = cfg.rules.vol_min;
  rules["coherence_min"] = cfg.rules.coherence_min;
  j["rules"] = rules;
  ordered_json fusion;
  fusion["bucket_ms"] = cfg.fusion.bucket_ms;
  fusion["enter_hyst"] = cfg.fusion.enter_hyst;
  fusion["exit_hyst"] = cfg.fusion.exit_hyst;
  ordered_json weights;
  std::map<std::string,double> sorted_w(cfg.fusion.weights.begin(), cfg.fusion.weights.end());
  for (const auto& [k,v] : sorted_w) {
    weights[k] = v;
  }
  fusion["weights"] = weights;
  j["fusion"] = fusion;
  ordered_json errs;
  std::map<std::string,std::string> sorted_e(cfg.error_templates.begin(), cfg.error_templates.end());
  for (const auto& [k,v] : sorted_e) {
    errs[k] = v;
  }
  j["error_templates"] = errs;
  return j.dump(2);
}

} // namespace sep

