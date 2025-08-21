#include "TradingLogicService.h"
#include "core/result.h"
#include "core/result_types.h"
#include "core/quantum_types.h"
#include <chrono>
#include <iterator>

using namespace sep::services;

sep::Result<std::vector<TradingSignal>> TradingLogicService::generateSignalsFromPatterns(
    const std::vector<std::shared_ptr<Pattern>>& patterns, const MarketContext& context) {
    if (!isReady()) {
        return sep::Result<std::vector<TradingSignal>>(
            sep::Error(sep::Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::vector<TradingSignal> signals;
    
    // Process each pattern
    for (const auto& pattern : patterns) {
        // Basic implementation - in a real system, this would use pattern attributes
        // to determine appropriate trading actions
        
        if (pattern->stability > 0.7 && pattern->coherence > 0.6) {
            // Create a signal based on the pattern
            TradingSignal signal;
            signal.signalId = "SIG_" + pattern->id + "_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            signal.patternId = pattern->id;
            
            // Mock logic to determine action type based on pattern attributes
            // Enhanced with market context information
            bool bullishSignal = false;
            
            if (!pattern->attributeScores.empty()) {
                bullishSignal = pattern->attributeScores[0] > 0;
            } else {
                // Use market indicators from context if available
                if (!context.indicators.empty() && context.indicators.find("trend") != context.indicators.end()) {
                    bullishSignal = context.indicators.at("trend") > 0;
                } else {
                    // Fallback to pattern attributes
                    bullishSignal = pattern->coherence > pattern->stability;
                }
            }
            
            signal.actionType = bullishSignal ? TradingActionType::Buy : TradingActionType::Sell;
            
            // Enhance confidence calculation with market context
            double marketConfidenceModifier = 1.0;
            if (!context.marketMetrics.empty() && context.marketMetrics.find("volatility") != context.marketMetrics.end()) {
                double volatility = context.marketMetrics.at("volatility");
                // Lower confidence in high volatility markets
                marketConfidenceModifier = std::max(0.5, 1.0 - volatility);
            }
            
            // Set signal confidence enhanced by market context
            signal.confidence = pattern->coherence * pattern->stability * marketConfidenceModifier;
            
            // Use context to set symbol if available
            if (!context.currentPrices.empty()) {
                signal.symbol = context.currentPrices.begin()->first;
            }
            
            // Set timing
            signal.generatedTime = std::chrono::system_clock::now();
            signal.expirationTime = signal.generatedTime + std::chrono::hours(24);
            
            // Add to signals list
            signals.push_back(signal);
            
            // Note: notifySignalCallbacks would be implemented in the full service
            // For now, we skip this callback notification
        }
    }

    return sep::Result<std::vector<TradingSignal>>(signals);
}