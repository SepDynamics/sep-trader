#include "core/training_coordinator.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace sep::train {

// ---------- Helper functions ----------

std::string Orchestrator::now_iso8601_() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// ---------- Orchestrator Implementation ----------

Orchestrator::Orchestrator(OrchestratorConfig cfg, std::shared_ptr<DataFeed> feed,
                          std::shared_ptr<FeatureEncoder> encoder, std::shared_ptr<Trainer> trainer,
                          std::shared_ptr<Evaluator> evaluator, std::shared_ptr<ModelRegistry> registry,
                          std::shared_ptr<RemoteClient> remote)
    : cfg_(std::move(cfg))
    , feed_(std::move(feed))
    , encoder_(std::move(encoder))
    , trainer_(std::move(trainer))
    , evaluator_(std::move(evaluator))
    , registry_(std::move(registry))
    , remote_(std::move(remote)) {
    
    // Initialize run metadata if not set
    if (cfg_.run.value == "unset") {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "run_" << std::put_time(std::gmtime(&time_t), "%Y%m%d_%H%M%S");
        cfg_.run.value = ss.str();
    }
    if (cfg_.commit.value == "unset") {
        cfg_.commit.value = "dev-build";
    }
    if (cfg_.docker.value == "unset") {
        cfg_.docker.value = "local-dev";
    }
}

TrainResult Orchestrator::train_pair(const std::string& pair, Mode mode) {
    return do_train_(pair, mode);
}

std::vector<TrainResult> Orchestrator::train_all(Mode mode) {
    std::vector<TrainResult> results;
    results.reserve(cfg_.pairs.size());
    
    // Simple serial execution - could be parallelized with thread pool
    for (const auto& pair : cfg_.pairs) {
        results.push_back(do_train_(pair, mode));
    }
    
    return results;
}

TrainResult Orchestrator::do_train_(const std::string& pair, Mode mode) {
    TrainResult result;
    result.pair = pair;
    
    try {
        // 1. Load data
        Instrument instrument{pair};
        auto ticks = feed_->get_ticks_lookback(instrument, cfg_.lookback);
        
        if (ticks.empty()) {
            // Log warning and return empty result
            return result;
        }
        
        // 2. Encode features (BTH snapshot)
        auto bth_snapshot = encoder_->encode(ticks);
        
        // 3. Split train/test (simple rolling split)
        const auto split_idx = ticks.size() * 4 / 5;  // 80/20 split
        std::vector<Tick> train_ticks(ticks.begin(), ticks.begin() + split_idx);
        std::vector<Tick> test_ticks(ticks.begin() + split_idx, ticks.end());
        
        // 4. Train model
        TrainConfig train_config = cfg_.trainer;
        
        // Adjust config based on mode
        switch (mode) {
            case Mode::Quick:
                train_config.max_iters = std::min(train_config.max_iters, 100);
                break;
            case Mode::Full:
                // Use full configuration
                break;
            case Mode::LiveTune:
                train_config.max_iters = std::min(train_config.max_iters, 50);
                break;
        }
        
        auto trained_model = trainer_->train(instrument, train_ticks, train_config);
        trained_model.snapshot = bth_snapshot;
        
        // 5. Evaluate model performance
        auto metrics = evaluator_->evaluate(instrument, test_ticks, trained_model, cfg_.horizon, cfg_.costs);
        
        // 6. Create artifact metadata
        ArtifactMeta meta;
        meta.pair = pair;
        meta.model_hash = trained_model.model_hash;
        meta.artifact_path = trained_model.artifact_path;
        meta.timestamp_iso8601 = now_iso8601_();
        meta.run = cfg_.run;
        meta.commit = cfg_.commit;
        meta.docker = cfg_.docker;
        meta.metrics = metrics;
        meta.quality = cfg_.policy.classify(metrics);
        
        // 7. Persist to registry
        if (registry_) {
            registry_->save(meta);
        }
        
        result.artifact = meta;
        
        // 8. Optionally push to remote
        if (cfg_.push_to_remote && remote_ && remote_->ping() && meta.quality != Quality::Low) {
            result.pushed = remote_->push_model(pair, meta.artifact_path);
        }
        
        // Log results (in production would use proper logging)
        // spdlog::info("[{}] hit={:.1f}% PF={:.2f} sharpe={:.2f} qual={}",
        //              pair, metrics.hit_rate, metrics.profit_factor, metrics.sharpe,
        //              static_cast<int>(meta.quality));
        
    } catch (const std::exception& e) {
        // Log error and return empty result
        // spdlog::error("Training failed for {}: {}", pair, e.what());
    }
    
    return result;
}

bool Orchestrator::start_live_tuning(const std::vector<std::string>& pairs) {
    if (live_active_.exchange(true)) {
        return false;  // Already running
    }
    
    auto pair_list = pairs.empty() ? cfg_.pairs : pairs;
    
    stop_requested_ = false;
    tuner_thread_ = std::thread([this, pair_list]() {
        while (!stop_requested_.load() && live_active_.load()) {
            for (const auto& pair : pair_list) {
                if (stop_requested_.load()) break;
                
                // Perform live tuning (quick mode)
                (void)do_train_(pair, Mode::LiveTune);
            }
            
            // Sleep between tuning cycles
            std::this_thread::sleep_for(std::chrono::minutes(5));
        }
    });
    
    return true;
}

bool Orchestrator::stop_live_tuning() {
    if (!live_active_.exchange(false)) {
        return false;  // Not running
    }
    
    if (tuner_thread_.joinable()) {
        stop_requested_ = true;
        tuner_thread_.join();
    }
    
    return true;
}

bool Orchestrator::remote_ok() const {
    return remote_ ? remote_->ping() : false;
}

} // namespace sep::train