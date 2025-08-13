#pragma once

#include "IPatternRecognitionService.h"
#include "ServiceBase.h"
#include "nlohmann_json_safe.h"
#include <array>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <vector>

namespace sep {
namespace services {

/**
 * Implementation of the Pattern Recognition Service
 * Provides pattern analysis, classification, matching, and evolutionary tracking capabilities
 */
class PatternRecognitionService : public IPatternRecognitionService, public ServiceBase {
public:
    PatternRecognitionService();
    virtual ~PatternRecognitionService();
    
    // IService interface
    bool isReady() const override;
    
    // IPatternRecognitionService interface
    Result<std::string> registerPattern(const Pattern& pattern) override;
    Result<Pattern> getPattern(const std::string& patternId) override;
    Result<void> updatePattern(const std::string& patternId, const Pattern& pattern) override;
    Result<void> deletePattern(const std::string& patternId) override;
    Result<PatternClassification> classifyPattern(const Pattern& pattern) override;
    Result<std::vector<PatternMatch>> findSimilarPatterns(
        const Pattern& pattern, 
        int maxResults = 10, 
        float minScore = 0.7f) override;
    Result<PatternEvolution> getPatternEvolution(const std::string& patternId) override;
    Result<void> addEvolutionStage(const std::string& patternId, const Pattern& newStage) override;
    Result<std::vector<PatternCluster>> clusterPatterns(
        const std::vector<std::string>& patternIds = {}, 
        int numClusters = 0) override;
    Result<float> calculateCoherence(const Pattern& pattern) override;
    Result<float> calculateStability(const Pattern& pattern) override;
    int registerChangeListener(
        std::function<void(const std::string&, const Pattern&)> callback) override;
    Result<void> unregisterChangeListener(int subscriptionId) override;
    
protected:
    // ServiceBase overrides
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;
    
private:
    // Helper methods
    std::string generateUniqueId();
    void notifyChangeListeners(const std::string& patternId, const Pattern& pattern);
    float computePatternSimilarity(const Pattern& pattern1, const Pattern& pattern2);
    
    // Pattern storage
    std::unordered_map<std::string, Pattern> patterns_;
    std::unordered_map<std::string, PatternEvolution> evolutions_;
    std::vector<PatternCluster> clusters_;
    
    // Change listeners
    std::unordered_map<int, std::function<void(const std::string&, const Pattern&)>> changeListeners_;
    
    // Synchronization
    std::mutex mutex_;
    
    // ID generators
    std::atomic<int> nextPatternId_{1};
    std::atomic<int> nextSubscriptionId_{1};
};

} // namespace services
} // namespace sep