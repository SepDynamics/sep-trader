# Application Services Overview

## DataAccessService
**Files:** `DataAccessService.h`, `DataAccessService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<std::string> storeObject(const std::string& collection, const std::map<std::string, std::any>& data, const std::string& id = "")`
- `Result<std::map<std::string, std::any>> retrieveObject(const std::string& collection, const std::string& id)`
- `Result<void> updateObject(const std::string& collection, const std::string& id, const std::map<std::string, std::any>& data)`
- `Result<void> deleteObject(const std::string& collection, const std::string& id)`
- `Result<std::vector<std::map<std::string, std::any>>> queryObjects(const std::string& collection, const std::vector<QueryFilter>& filters = {}, const std::vector<SortSpec>& sortSpecs = {}, int limit = 0, int skip = 0)`
- `Result<int> countObjects(const std::string& collection, const std::vector<QueryFilter>& filters = {})`
- `Result<std::shared_ptr<TransactionContext>> beginTransaction()`
- `Result<void> executeTransaction(std::function<Result<void>(std::shared_ptr<TransactionContext>)> operations)`
- `int registerChangeListener(const std::string& collection, std::function<void(const std::string&, const std::string&)> callback)`
- `Result<void> unregisterChangeListener(int subscriptionId)`
- `Result<void> createCollection(const std::string& collection, const std::string& schema = "")`
- `Result<void> deleteCollection(const std::string& collection)`
- `Result<std::vector<std::string>> getCollections()`

**TODO / Mock notes**
- Filtering currently supports only basic equality; full query support remains a TODO.

## MemoryTierService
**Files:** `MemoryTierService.h`, `MemoryTierService.cpp`

**Key functions**
- `Result<sep::memory::MemoryBlock*> allocate(std::size_t size, sep::memory::MemoryTierEnum tier)`
- `Result<void> deallocate(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> findBlockByPtr(void* ptr)`
- `Result<sep::memory::MemoryTier*> getTier(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTierUtilization(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTierFragmentation(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTotalUtilization()`
- `Result<float> getTotalFragmentation()`
- `Result<void> defragmentTier(sep::memory::MemoryTierEnum tier)`
- `Result<void> optimizeBlocks()`
- `Result<void> optimizeTiers()`
- `Result<sep::memory::MemoryBlock*> promoteBlock(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> demoteBlock(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> updateBlockMetrics(sep::memory::MemoryBlock* block, float coherence, float stability, uint32_t generation, float contextScore)`
- `Result<std::string> getMemoryAnalytics()`
- `Result<std::string> getMemoryVisualization()`
- `Result<void> configureTierPolicies(const sep::memory::MemoryThresholdConfig& config)`
- `Result<void> optimizeRedisIntegration(int optimizationLevel)`
- `Result<std::string> allocateBlock(uint64_t size, MemoryTierLevel tier, const std::string& contentType, const std::vector<uint8_t>& initialData = {}, const std::map<std::string, std::string>& tags = {})`
- `Result<void> deallocateBlock(const std::string& blockId)`
- `Result<void> storeData(const std::string& blockId, const std::vector<uint8_t>& data, uint64_t offset = 0)`
- `Result<std::vector<uint8_t>> retrieveData(const std::string& blockId, uint64_t size, uint64_t offset = 0)`
- `Result<MemoryBlockMetadata> getBlockMetadata(const std::string& blockId)`
- `Result<void> moveBlockToTier(const std::string& blockId, MemoryTierLevel destinationTier, const std::string& reason = "Manual transition")`
- `Result<TierStatistics> getTierStatistics(MemoryTierLevel tier)`
- `Result<std::map<MemoryTierLevel, TierStatistics>> getAllTierStatistics()`
- `Result<void> configureTier(MemoryTierLevel tier, uint64_t totalCapacity, const std::map<std::string, std::string>& policies = {})`
- `Result<void> optimizeTiers(bool aggressive = false)`
- `int registerTransitionCallback(std::function<void(const TierTransitionRecord&)> callback)`
- `Result<void> unregisterTransitionCallback(int subscriptionId)`
- `Result<std::vector<TierTransitionRecord>> getTransitionHistory(int maxRecords = 100)`
- `Result<std::vector<MemoryAccessPattern>> getAccessPatterns(uint32_t minFrequency = 5)`

**TODO / Mock notes**
- No outstanding TODOs noted.

## PatternRecognitionService
**Files:** `PatternRecognitionService.h`, `PatternRecognitionService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<std::string> registerPattern(const Pattern& pattern)`
- `Result<Pattern> getPattern(const std::string& patternId)`
- `Result<void> updatePattern(const std::string& patternId, const Pattern& pattern)`
- `Result<void> deletePattern(const std::string& patternId)`
- `Result<PatternClassification> classifyPattern(const Pattern& pattern)`
- `Result<std::vector<PatternMatch>> findSimilarPatterns(const Pattern& pattern, int maxResults = 10, float minScore = 0.7f)`
- `Result<PatternEvolution> getPatternEvolution(const std::string& patternId)`
- `Result<void> addEvolutionStage(const std::string& patternId, const Pattern& newStage)`
- `Result<std::vector<PatternCluster>> clusterPatterns(const std::vector<std::string>& patternIds = {}, int numClusters = 0)`
- `Result<float> calculateCoherence(const Pattern& pattern)`
- `Result<float> calculateStability(const Pattern& pattern)`
- `int registerChangeListener(std::function<void(const std::string&, const Pattern&)> callback)`
- `Result<void> unregisterChangeListener(int subscriptionId)`

**TODO / Mock notes**
- No outstanding TODOs noted.

## QuantumProcessingService
**Files:** `QuantumProcessingService.h`, `QuantumProcessingService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<BinaryStateVector> processBinaryStateAnalysis(const QuantumState& state)`
- `Result<std::vector<QuantumFourierComponent>> applyQuantumFourierHierarchy(const QuantumState& state, int hierarchyLevels)`
- `Result<CoherenceMatrix> calculateCoherence(const QuantumState& state)`
- `Result<StabilityMetrics> determineStability(const QuantumState& state, const std::vector<QuantumState>& historicalStates)`
- `Result<QuantumState> evolveQuantumState(const QuantumState& state, const std::map<std::string, double>& evolutionParameters)`
- `Result<QuantumState> runQuantumPipeline(const QuantumState& state)`
- `std::map<std::string, std::string> getAvailableAlgorithms() const`

**TODO / Mock notes**
- Initialization check for `runQuantumPipeline` is temporarily skipped to avoid diamond inheritance issues.

## TradingLogicService
**Files:** `TradingLogicService.h`, `TradingLogicService.cpp`

**Key functions**
- `Result<void> processMarketData(const MarketDataPoint& dataPoint)`
- `Result<void> processMarketDataBatch(const std::vector<MarketDataPoint>& dataPoints)`
- `Result<OHLCVCandle> updateOHLCVCandle(const std::string& symbol, TradingTimeframe timeframe, const MarketDataPoint& dataPoint)`
- `Result<std::vector<OHLCVCandle>> getHistoricalCandles(const std::string& symbol, TradingTimeframe timeframe, int count, std::chrono::system_clock::time_point endTime)`
- `Result<std::vector<TradingSignal>> generateSignals(const MarketContext& context, const std::vector<std::string>& patternIds)`
- `Result<std::vector<TradingSignal>> generateSignalsFromPatterns(const std::vector<std::shared_ptr<Pattern>>& patterns, const MarketContext& context)`
- `Result<std::vector<TradingDecision>> makeDecisions(const std::vector<TradingSignal>& signals, const MarketContext& context)`
- `Result<PerformanceMetrics> evaluatePerformance(const std::vector<TradingDecision>& decisions, const MarketContext& currentContext)`
- `Result<PerformanceMetrics> backtestStrategy(const std::map<std::string, std::vector<OHLCVCandle>>& historicalData, const std::map<std::string, double>& parameters)`
- `int registerSignalCallback(std::function<void(const TradingSignal&)> callback)`
- `Result<void> unregisterSignalCallback(int subscriptionId)`
- `std::map<std::string, std::string> getAvailableStrategies() const`
- `Result<MarketContext> getCurrentMarketContext() const`

**TODO / Mock notes**
- No outstanding TODOs noted.

## MarketModelCache
**Files:** `market_model_cache.hpp`, `market_model_cache.cpp`

**Key functions**
- `bool ensureCacheForLastWeek(const std::string& instrument = "EUR_USD")`
- `const std::map<std::string, sep::trading::QuantumTradingSignal>& getSignalMap() const`
- `bool loadCache(const std::string& filepath)`
- `bool saveCache(const std::string& filepath) const`
- `void processAndCacheData(const std::vector<Candle>& raw_candles, const std::string& filepath)`
- `std::string getCacheFilepathForLastWeek(const std::string& instrument) const`

**TODO / Mock notes**
- Signal generation uses a placeholder based on simple price movement; integrate the full quantum pipeline.

## EnhancedMarketModelCache
**Files:** `enhanced_market_model_cache.hpp`, `enhanced_market_model_cache.cpp`

**Key functions**
- `bool ensureEnhancedCacheForInstrument(const std::string& instrument, TimeFrame timeframe = TimeFrame::M1)`
- `ProcessedSignal generateCorrelationEnhancedSignal(const std::string& target_asset, const std::string& timestamp)`
- `void updateCorrelatedAssets(const std::string& primary_asset)`
- `void optimizeCacheHierarchy()`
- `bool loadEnhancedCache(const std::string& filepath)`
- `bool saveEnhancedCache(const std::string& filepath) const`
- `CrossAssetCorrelation calculateCrossAssetCorrelation(const std::string& primary_asset, const std::vector<std::string>& correlated_assets)`
- `double calculatePairwiseCorrelation(const std::vector<double>& asset1_prices, const std::vector<double>& asset2_prices, std::chrono::milliseconds& optimal_lag)`
- `const std::unordered_map<std::string, CacheEntry>& getCacheEntries() const`
- `std::vector<ProcessedSignal> getCorrelationEnhancedSignals(const std::string& instrument) const`
- `CachePerformanceMetrics getPerformanceMetrics() const`

**TODO / Mock notes**
- No outstanding TODOs noted.

## MultiAssetSignalFusion
**Files:** `multi_asset_signal_fusion.hpp`, `multi_asset_signal_fusion.cpp`

**Key functions**
- `FusedSignal generateFusedSignal(const std::string& target_asset)`
- `std::vector<std::string> getCorrelatedAssets(const std::string& target_asset)`
- `CrossAssetCorrelation calculateDynamicCorrelation(const std::string& asset1, const std::string& asset2)`
- `double calculateCrossAssetBoost(const sep::trading::QuantumIdentifiers& signal, const CrossAssetCorrelation& correlation)`
- `FusedSignal fuseSignals(const std::vector<AssetSignal>& asset_signals)`
- `std::vector<double> calculateCorrelationMatrix(const std::vector<std::string>& assets)`
- `double calculateCrossAssetCoherence(const std::vector<AssetSignal>& signals)`
- `void updateCorrelationCache()`
- `void invalidateCorrelationCache()`
- `void logFusionDetails(const FusedSignal& signal)`
- `std::string serializeFusionResult(const FusedSignal& signal)`

**TODO / Mock notes**
- Correlation calculations currently use default values; historical data fetching via connector is a TODO.

## MarketRegimeAdaptiveProcessor
**Files:** `market_regime_adaptive.hpp`, `market_regime_adaptive.cpp`

**Key functions**
- `AdaptiveThresholds calculateRegimeOptimalThresholds(const std::string& asset)`
- `MarketRegime detectCurrentRegime(const std::string& asset)`
- `VolatilityLevel calculateVolatilityLevel(const std::vector<Candle>& data)`
- `TrendStrength calculateTrendStrength(const std::vector<Candle>& data)`
- `LiquidityLevel calculateLiquidityLevel(const std::string& asset)`
- `NewsImpactLevel calculateNewsImpact()`
- `QuantumCoherenceLevel calculateQuantumCoherence(const std::vector<Candle>& data)`
- `AdaptiveThresholds adaptThresholdsForRegime(const MarketRegime& regime)`
- `double calculateVolatilityAdjustment(VolatilityLevel volatility)`
- `double calculateTrendAdjustment(TrendStrength trend)`
- `double calculateLiquidityAdjustment(LiquidityLevel liquidity)`
- `double calculateNewsAdjustment(NewsImpactLevel news)`
- `double calculateCoherenceAdjustment(QuantumCoherenceLevel coherence)`
- `double calculateATR(const std::vector<Candle>& data, int periods = 14)`
- `double calculateRSI(const std::vector<Candle>& data, int periods = 14)`
- `double calculateSMA(const std::vector<Candle>& data, int periods)`
- `bool isLondonSession()`
- `bool isNewYorkSession()`
- `bool isTokyoSession()`
- `void logRegimeDetails(const MarketRegime& regime, const AdaptiveThresholds& thresholds)`
- `std::string serializeRegimeData(const MarketRegime& regime, const AdaptiveThresholds& thresholds)`
- `void updateRegimeCache(const std::string& asset)`
- `void invalidateRegimeCache()`

**TODO / Mock notes**
- No outstanding TODOs noted.

## Health Monitor (C interface)
**Files:** `health_monitor_c_wrapper.h`, `health_monitor_c_wrapper.cpp`

**Key functions**
- `int c_health_monitor_init()`
- `int c_health_monitor_get_status(CHealthStatus* status)`
- `void c_health_monitor_cleanup()`

**TODO / Mock notes**
- Lightweight C-style wrapper intended for integration; no TODOs currently documented.

