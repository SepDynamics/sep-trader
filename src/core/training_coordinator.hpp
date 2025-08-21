#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sep::train {

// ===== Strong types =====
enum class Mode : uint8_t { Quick, Full, LiveTune };
enum class Quality : uint8_t { Low, Medium, High, Unknown };

struct Instrument {
    std::string symbol;
};  // "EUR_USD"
struct RunId {
    std::string value;
};  // UUID/sha
struct Commit {
    std::string value;
};
struct DockerSha {
    std::string value;
};

struct Horizon {
    int fixed_minutes{30}; // 30 minutes
};

struct CostModelPips {
    double spread_pips = 0.8;
    double commission_per_lot_per_side_usd = 0.50;
    double slippage_pips = 0.1;
    double lot_size = 100'000.0;  // FX
};

// ===== Data IO ports (adapters implement) =====
struct Tick {
    std::chrono::system_clock::time_point ts;
    double bid{}, ask{};
    double mid() const {
        return 0.5 * (bid + ask);
    }
};

class DataFeed {
  public:
    virtual ~DataFeed() = default;
    virtual std::vector<Tick> get_ticks_lookback(const Instrument&,
                                                 int lookback_hours) = 0;
    virtual std::vector<Tick> get_ticks_range(const Instrument&,
                                              std::chrono::system_clock::time_point from,
                                              std::chrono::system_clock::time_point to) = 0;
};

class RemoteClient {
  public:
    virtual ~RemoteClient() = default;
    virtual bool ping() = 0;
    virtual bool push_model(const std::string& pair, const std::string& artifact_path) = 0;
    virtual bool pull_params(const std::string& pair, std::string& json_out) = 0;
};

// ===== Core modules =====
struct BTHSnapshot {
    double C{}, S{}, H{};                // coherence, stability, entropy [0,1]
    double flip_rate{}, rupture_rate{};  // [0,1]
    std::vector<double> fwht_topK;       // e.g., K=8
};

class FeatureEncoder {
  public:
    virtual ~FeatureEncoder() = default;
    // Convert ticks → bitstates + compute BTH snapshot/feats
    virtual BTHSnapshot encode(const std::vector<Tick>& ticks) = 0;
};

struct TrainConfig {
    bool use_cuda = true;
    int max_iters = 1000;
    uint64_t seed = 1337;
};

struct TrainedModel {
    std::string artifact_path;  // disk path or registry key
    std::string model_hash;     // sha256 of artifact bytes
    BTHSnapshot snapshot;       // last-window features used
};

class Trainer {
  public:
    virtual ~Trainer() = default;
    virtual TrainedModel train(const Instrument&, const std::vector<Tick>& train_ticks,
                               const TrainConfig&) = 0;
};

struct Metrics {
    // after-cost metrics on test split
    double hit_rate{};          // %
    double hit_ci_halfwidth{};  // % (95% CI half width)
    double expectancy_pips{};
    double profit_factor{};
    double sharpe{};
    double dsr{};  // deflated sharpe (optional)
    double max_dd_pct{};
    size_t n_signals{};
    size_t n_trades{};
};

class Evaluator {
  public:
    virtual ~Evaluator() = default;
    virtual Metrics evaluate(const Instrument&, const std::vector<Tick>& test_ticks,
                             const TrainedModel&, const Horizon&, const CostModelPips&) = 0;
};

// ===== Registry (artifacts & metadata) =====
struct ArtifactMeta {
    std::string pair;
    std::string model_hash;
    std::string artifact_path;
    std::string timestamp_iso8601;
    RunId run;
    Commit commit;
    DockerSha docker;
    Metrics metrics;
    Quality quality = Quality::Unknown;
};

class ModelRegistry {
  public:
    virtual ~ModelRegistry() = default;
    virtual bool save(const ArtifactMeta&) = 0;
    virtual std::optional<ArtifactMeta> load_best(const std::string& pair) = 0;
};

// ===== Policy & config =====
struct Policy {
    // quality thresholds
    double min_hit_pct = 60.0;
    double min_pf = 1.10;
    double min_sharpe = 1.2;
    // label the artifact
    Quality classify(const Metrics& m) const {
        if (m.hit_rate >= min_hit_pct + 5 && m.profit_factor >= 1.2 && m.sharpe >= 1.6)
            return Quality::High;
        if (m.hit_rate >= min_hit_pct && m.profit_factor >= min_pf)
            return Quality::Medium;
        return Quality::Low;
    }
};

struct OrchestratorConfig {
    std::vector<std::string> pairs{"EUR_USD", "GBP_USD", "USD_JPY",
                                   "AUD_USD", "USD_CHF", "USD_CAD"};
    int lookback_hours{24};
    int lookback{24};  // Legacy compatibility
    Horizon horizon{};
    CostModelPips costs{};
    TrainConfig trainer{};
    Policy policy{};
    RunId run{};
    Commit commit{};
    DockerSha docker{};
    int parallel_jobs = 2;
    bool push_to_remote = true;
};

// ===== Results =====
struct TrainResult {
    std::string pair;
    ArtifactMeta artifact;
    bool pushed = false;
    
    // Additional fields expected by status display
    double accuracy = 0.0;
    Quality quality = Quality::Unknown;
    std::string timestamp;
};

// ===== Orchestrator =====
class Orchestrator {
  public:
    Orchestrator(OrchestratorConfig cfg, std::shared_ptr<DataFeed> feed,
                 std::shared_ptr<FeatureEncoder> encoder, std::shared_ptr<Trainer> trainer,
                 std::shared_ptr<Evaluator> evaluator, std::shared_ptr<ModelRegistry> registry,
                 std::shared_ptr<RemoteClient> remote);

    // one-shot per pair
    TrainResult train_pair(const std::string& pair, Mode mode);

    // batch
    std::vector<TrainResult> train_all(Mode mode);

    // live tuning background session
    bool start_live_tuning(const std::vector<std::string>& pairs);
    bool stop_live_tuning();
    bool live_tuning_active() const {
        return live_active_.load();
    }

    // health
    bool remote_ok() const;
    bool isRemoteTraderConnected() const {
        return remote_ok();
    }

  private:
    TrainResult do_train_(const std::string& pair, Mode mode);
    static std::string now_iso8601_();

  private:
    OrchestratorConfig cfg_;
    std::shared_ptr<DataFeed> feed_;
    std::shared_ptr<FeatureEncoder> encoder_;
    std::shared_ptr<Trainer> trainer_;
    std::shared_ptr<Evaluator> evaluator_;
    std::shared_ptr<ModelRegistry> registry_;
    std::shared_ptr<RemoteClient> remote_;

    std::thread tuner_thread_;
    std::atomic<bool> live_active_{false};
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex m_;
};

}  // namespace sep::train
