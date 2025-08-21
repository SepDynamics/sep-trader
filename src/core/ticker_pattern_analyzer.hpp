#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

// C++20 compatibility
#if __cplusplus >= 202002L
    #include <span>
    #include <stop_token>
#else
    #include "core/cpp20_compatibility.h"
#endif

#include "core/qfh.h"                         // if you keep legacy QFH/QBSA interop
#include "core/quantum_manifold_optimizer.h"  // GAO impl lives here or new header
#include "core/result_types.h"                // sep::Result<T>
#include "core/timeframe.h"

namespace sep::engine {

// ---------- Strong enums / IDs ----------

using ::sep::Timeframe;
enum class Side : uint8_t { Buy, Sell, Flat };
enum class Strength : uint8_t { None, Weak, Moderate, Strong, VeryStrong };

struct InstrumentId {
    std::string symbol;
};  // "EUR/USD" etc.
struct RunId {
    std::string value;
};  // UUID/sha
struct CommitHash {
    std::string value;
};
struct DockerSha {
    std::string value;
};

// ---------- Cost & horizon models ----------

struct CostModelPips {
    // After-cost round-turn pips: spread + commission per-side *2 + slippage
    double median_spread_pips = 0.8;
    double commission_per_lot_per_side_usd = 0.50;  // not pips; convert at runtime if needed
    double slippage_pips = 0.1;
    double lot_size = 100'000.0;  // standard FX lot for conversions
};

struct Horizon {
    std::chrono::minutes fixed = std::chrono::minutes{30};
};

// ---------- Decay / weighting ----------

struct Decay {
    // exponential kernel w_k = lambda * (1 - lambda)^k
    double lambda = 0.10;    // 0<lambda<=1
    double ema_beta = 0.10;  // for S/H estimates
    int window_L = 64;       // FWHT window
    int top_K = 8;           // FWHT bands surfaced
};

// ---------- BTH / BRS / GAO / Evolution configs ----------

struct BTHConfig {
    Decay decay{};
    bool compute_fwht = true;
};

struct BRSConfig {
    double tau = 0.85;  // reliability threshold on normalized Hamming similarity
    bool calibrate_on_disjoint_split = true;
    double max_ece = 0.02;  // bound Expected Calibration Error
};

struct GAOConfig {
    // Riemannian-style diagonal metric: G(θ)=diag(gC,gS,gH), g•=ε+EMA[(∇•J)^2]
    double lr = 0.02;
    double epsilon = 1e-8;                         // ensure PD metric
    std::array<double, 3> box_min{0.0, 0.0, 0.0};  // bounds for (C,S,H)
    std::array<double, 3> box_max{1.0, 1.0, 1.0};
    int max_iters = 64;
};

struct EvolutionConfig {
    int pop_size = 16;
    int elite_k = 4;
    double mutation_sigma = 0.05;
    int eval_minutes = 60;  // rolling evaluation window
    bool online = true;     // evolve during session
    uint64_t seed = 42;
};

// ---------- Analysis configuration (per-call overrides allowed) ----------

struct EngineConfig {
    // analysis
    int lookback_hours = 24;
    std::vector<Timeframe> timeframes{Timeframe::M1, Timeframe::M5, Timeframe::M15, Timeframe::H1};
    bool cross_tf_validation = true;
    double min_tf_agreement = 0.60;

    // modules
    BTHConfig bth{};
    BRSConfig brs{};
    GAOConfig gao{};
    EvolutionConfig evo{};

    // constraints & risk
    double min_signal_confidence = 0.65;
    double max_risk_per_trade = 0.02;
    double max_pair_correlation = 0.80;  // cap exposure vs correlated instruments
    bool drawdown_protection = true;

    // perf
    bool use_cuda = true;
    int threads =
        std::thread::hardware_concurrency() ? int(std::thread::hardware_concurrency()) : 8;
    bool cache_intermediates = true;

    // repro/meta
    RunId run_id{"unset"};
    CommitHash commit{"unset"};
    DockerSha docker{"unset"};
    uint64_t rng_seed = 1337;
};

// ---------- Data feed & execution ports ----------

struct Tick {
    std::chrono::system_clock::time_point ts;
    double bid;
    double ask;
};

class MarketDataFeed {
  public:
    virtual ~MarketDataFeed() = default;
    virtual sep::Result<std::vector<Tick>> get_ticks(
        const InstrumentId& instrument, std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) = 0;

    virtual sep::Result<std::vector<Tick>> get_ticks_lookback(
        const InstrumentId& instrument, std::chrono::system_clock::time_point end,
        std::chrono::hours lookback) = 0;
};

struct OrderIntent {
    InstrumentId instrument;
    Side side = Side::Flat;
    double quantity_lots = 1.0;
    std::chrono::system_clock::time_point expires_at{};
};

class TradeExecutor {
  public:
    virtual ~TradeExecutor() = default;
    virtual sep::Result<std::string> submit(const OrderIntent& intent) = 0;
    virtual sep::Result<void> cancel_all(const InstrumentId& instrument) = 0;
};

// ---------- Core feature & state snapshots ----------

struct BitState64 {
    std::array<uint64_t, 1> w{};
};  // 64 thresholded features packed

struct BTHSnapshot {
    double C;                       // coherence in [0,1]
    double S;                       // stability in [0,1]
    double H;                       // entropy  in [0,1]
    double flip_rate;               // f_t in [0,1]
    double rupture_rate;            // u_t in [0,1]
    std::vector<double> fwht_topK;  // size K
};

struct ReliabilityReport {
    double brs;      // normalized Hamming similarity
    double ece;      // calibration error on held-out split
    bool gate_pass;  // brs >= tau
};

struct Signal {
    Side side = Side::Flat;
    Strength strength = Strength::None;
    double confidence = 0.0;
    Horizon horizon{};
    std::chrono::system_clock::time_point issued_at{};
    std::chrono::system_clock::time_point expires_at{};
    std::string reason;  // short reason code(s)
};

enum class PatternType : uint8_t {
    TrendingUp,
    TrendingDown,
    Ranging,
    BreakoutPending,
    ReversalForming,
    HighVolatility,
    Consolidation,
    Unknown
};

// ---------- Request/Result ----------

struct AnalysisRequest {
    InstrumentId instrument;
    Horizon horizon{};
    CostModelPips costs{};
    EngineConfig overrides{};  // optional overrides (leave as defaults to use engine cfg)
    std::optional<std::chrono::system_clock::time_point> asof;  // default=now
};

struct RiskMetrics {
    double estimated_risk_0_1 = 0.0;
    double drawdown_risk = 0.0;  // pips
    double suggested_position_lots = 0.0;
};

struct PerformanceFootprint {
    std::chrono::microseconds p50_latency{0};
    std::chrono::microseconds p95_latency{0};
    size_t states_processed = 0;
};

struct AnalysisResult {
    InstrumentId instrument;
    std::chrono::system_clock::time_point asof{};
    bool success = true;
    std::string error;

    // Core metrics & features
    BTHSnapshot bth{};
    ReliabilityReport brs{};
    PatternType dominant = PatternType::Unknown;
    std::vector<PatternType> secondary;

    // Signals (primary + per timeframe)
    Signal primary{};
    std::vector<std::pair<Timeframe, Signal>> by_timeframe;

    // Risk & capacity
    RiskMetrics risk{};
    double capacity_slope_pips_per_0p1_slip = -0.05;

    // Evolution lineage
    struct EvolutionTag {
        int generation = 0;
        uint64_t parent_hash = 0;
        uint64_t param_hash = 0;
    };
    EvolutionTag evo{};

    // Diagnostics
    size_t ticks_used = 0;
    PerformanceFootprint perf{};
    RunId run_id{"unset"};
    CommitHash commit{"unset"};
    DockerSha docker{"unset"};
};

// ---------- Subsystems (interfaces) ----------

class BTHEngine {
  public:
    virtual ~BTHEngine() = default;
    virtual BTHSnapshot compute(const std::vector<Tick>& ticks, const BTHConfig& cfg) = 0;
};

class ReliabilityGate {
  public:
    virtual ~ReliabilityGate() = default;
    virtual ReliabilityReport score(const BitState64& predicted, const BitState64& observed,
                                    const BRSConfig& cfg) = 0;
};

class GAO {
  public:
    virtual ~GAO() = default;
    // Optimize (C,S,H) or other θ against objective J derived from BTHSnapshot / costs / horizon
    virtual std::array<double, 3> optimize(std::array<double, 3> theta0, const GAOConfig& cfg) = 0;
};

class EvolutionEngine {
  public:
    virtual ~EvolutionEngine() = default;
    virtual std::array<double, 3> step(std::array<double, 3> theta, const EvolutionConfig& cfg,
                                       uint64_t seed_override = 0) = 0;
};

// ---------- Engine (orchestrator) ----------

class SepEngine {
  public:
    explicit SepEngine(EngineConfig cfg, std::shared_ptr<MarketDataFeed> feed,
                       std::shared_ptr<TradeExecutor> exec, std::unique_ptr<BTHEngine> bth,
                       std::unique_ptr<ReliabilityGate> brs, std::unique_ptr<GAO> gao,
                       std::unique_ptr<EvolutionEngine> evo);

    ~SepEngine();

    // Single-shot analysis (pure; no orders placed)
    sep::Result<AnalysisResult> analyze(const AnalysisRequest& req);

    // Batch
    std::vector<sep::Result<AnalysisResult>> analyze_many(
        const std::vector<AnalysisRequest>& requests);

    // Realtime session (non-blocking)
    struct SessionId {
        std::string value;
    };
    sep::Result<SessionId> start_session(const InstrumentId& instrument, Horizon horizon,
                                         CostModelPips costs);

    sep::Result<void> stop_session(const SessionId& id);

    std::optional<AnalysisResult> latest(const InstrumentId& instrument) const;

    // Execution (optional): convert signal → order intent via policy
    sep::Result<std::string> act_on(const AnalysisResult& res, double lots);

    // Config
    void update_config(const EngineConfig& cfg);
    EngineConfig config() const;

    // Introspection - Non-atomic stats struct for return values
    struct Stats {
        uint64_t analyses = 0;
        uint64_t ok = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        std::chrono::system_clock::time_point last_reset{};
    };
    Stats stats() const;
    void reset_stats();

  private:
    AnalysisResult pipeline_(const InstrumentId& instrument,
#if __cplusplus >= 202002L
                            std::span<const Tick> ticks,
#else
                            sep_compat::span<const Tick> ticks,
#endif
                             const AnalysisRequest& req);

    // helpers
    static std::string timeframe_str(Timeframe tf);
    BitState64 make_bitstate64(const Tick& t) const;  // your thresholding/feature map

  private:
    EngineConfig cfg_;
    std::shared_ptr<MarketDataFeed> feed_;
    std::shared_ptr<TradeExecutor> exec_;
    std::unique_ptr<BTHEngine> bth_;
    std::unique_ptr<ReliabilityGate> brs_;
    std::unique_ptr<GAO> gao_;
    std::unique_ptr<EvolutionEngine> evo_;

    // realtime: one thread per instrument
    struct Session {
        SessionId id;
        InstrumentId instrument;
        Horizon horizon;
        CostModelPips costs;
        std::thread th;  // Use std::thread consistently
        std::atomic<bool> running{false};

        // Custom constructors to handle atomic member
        Session() = default;

        // Delete copy operations due to atomic member
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;

        // Custom move constructor
        Session(Session&& other) noexcept
            : id(std::move(other.id)),
              instrument(std::move(other.instrument)),
              horizon(std::move(other.horizon)),
              costs(std::move(other.costs)),
              th(std::move(other.th)),
              running(other.running.load()) {}

        // Custom move assignment
        Session& operator=(Session&& other) noexcept {
            if (this != &other) {
                // Stop and join current thread if active
                if (running.load() && th.joinable()) {
                    running.store(false);
                    th.join();
                }

                id = std::move(other.id);
                instrument = std::move(other.instrument);
                horizon = std::move(other.horizon);
                costs = std::move(other.costs);
                th = std::move(other.th);
                running.store(other.running.load());
            }
            return *this;
        }
    };
    mutable std::mutex m_sessions_;
    std::unordered_map<std::string, Session> sessions_;  // key=instrument.symbol

    // latest analyses
    mutable std::mutex m_latest_;
    std::unordered_map<std::string, AnalysisResult> latest_;  // key=instrument.symbol

    // cache / stats - use atomic members for thread safety
    mutable std::mutex m_stats_;
    struct AtomicStats {
        std::atomic<uint64_t> analyses{0};
        std::atomic<uint64_t> ok{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::chrono::system_clock::time_point last_reset{};
    };
    AtomicStats stats_;
};

}  // namespace sep::engine
