Result<std::vector<TradingSignal>> TradingLogicService::generateSignalsFromPatterns(
    const std::vector<std::shared_ptr<Pattern>>& patterns,
    const MarketContext& context) {
    
    if (!isReady()) {
        return Result<std::vector<TradingSignal>>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::vector<TradingSignal> signals;
    
    // Process each pattern
    for (const auto& pattern : patterns) {
        // Basic implementation - in a real system, this would use pattern attributes
        // to determine appropriate trading actions
        
        if (pattern->stability > 0.7 && pattern->coherence > 0.6) {
            // Create a signal based on the pattern
            TradingSignal signal;
            signal.signalId = generateUniqueId("SIG");
            signal.patternId = pattern->id;
            
            // Mock logic to determine action type based on pattern attributes
            // Since Pattern doesn't have an 'evolution' property, we'll use other attributes
            
            // Analyze pattern features to determine direction
            // For simplicity, we'll use the first feature value if available, or fallback to coherence comparison
            bool bullishSignal = false;
            
            if (!pattern->features.empty()) {
                bullishSignal = pattern->features[0] > 0;
            } else {
                // Arbitrary logic for demo purposes - in a real system this would be more sophisticated
                bullishSignal = pattern->coherence > pattern->stability;
            }
            
            signal.actionType = bullishSignal ? TradingActionType::Buy : TradingActionType::Sell;
            
            // Set other signal properties
            signal.confidence = pattern->coherence * pattern->stability;
            signal.generatedTime = std::chrono::system_clock::now();
            signal.expirationTime = signal.generatedTime + std::chrono::hours(24);
            
            // Add to signals list
            signals.push_back(signal);
            
            // Notify callbacks
            notifySignalCallbacks(signal);
        }
    }
    
    return Result<std::vector<TradingSignal>>(signals);
}