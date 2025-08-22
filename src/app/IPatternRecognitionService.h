#pragma once

#ifndef SRC_APP_IPATTERNRECOGNITIONSERVICE_MERGED_H
#define SRC_APP_IPATTERNRECOGNITIONSERVICE_MERGED_H

#include "core/standard_includes.h"
#include "core/result_types.h"
#include "core/types.h"

#include "IService.h"
#include "PatternTypes.h"
#include "QuantumTypes.h"

namespace sep {
namespace services {

class IPatternRecognitionService : public IService {
public:
    // Methods from src/app/include/IPatternRecognitionService.h
    virtual Result<std::vector<std::shared_ptr<Pattern>>> extractPatterns(const std::vector<float>& inputData, const std::map<std::string, float>& extractionParameters = {}) = 0;
    virtual Result<std::vector<std::shared_ptr<Pattern>>> extractPatternsFromQuantumState(const QuantumState& state) = 0;
    virtual Result<PatternClassification> classifyPattern(const Pattern& pattern) = 0;
    virtual Result<std::vector<PatternMatch>> matchPattern(const Pattern& pattern, float matchThreshold = 0.7f, int maxResults = 10) = 0;
    virtual Result<PatternEvolution> trackPatternEvolution(const std::string& patternId, int historyLength = 10) = 0;
    virtual Result<std::vector<PatternCluster>> clusterPatterns(const std::vector<std::shared_ptr<Pattern>>& patterns, const std::map<std::string, float>& clusteringParameters = {}) = 0;
    virtual std::map<std::string, std::string> getAvailableClassifiers() const = 0;
    virtual std::map<std::string, std::string> getAvailableExtractionAlgorithms() const = 0;
    virtual Result<std::string> storePattern(const Pattern& pattern) = 0;
    virtual Result<std::shared_ptr<Pattern>> retrievePattern(const std::string& patternId) = 0;

    // Methods from src/app/include/pattern/IPatternRecognitionService.h
    virtual Result<std::string> registerPattern(const Pattern& pattern) = 0;
    virtual Result<Pattern> getPattern(const std::string& patternId) = 0;
    virtual Result<void> updatePattern(const std::string& patternId, const Pattern& pattern) = 0;
    virtual Result<void> deletePattern(const std::string& patternId) = 0;
    virtual Result<std::vector<PatternMatch>> findSimilarPatterns(const Pattern& pattern, int maxResults = 10, float minScore = 0.7f) = 0;
    virtual Result<PatternEvolution> getPatternEvolution(const std::string& patternId) = 0;
    virtual Result<void> addEvolutionStage(const std::string& patternId, const Pattern& newStage) = 0;
    virtual Result<std::vector<PatternCluster>> clusterPatterns(const std::vector<std::string>& patternIds = {}, int numClusters = 0) = 0;
    virtual Result<float> calculateCoherence(const Pattern& pattern) = 0;
    virtual Result<float> calculateStability(const Pattern& pattern) = 0;
    virtual int registerChangeListener(std::function<void(const std::string&, const Pattern&)> callback) = 0;
    virtual Result<void> unregisterChangeListener(int subscriptionId) = 0;
};

} // namespace services
} // namespace sep

#endif // SRC_APP_IPATTERNRECOGNITIONSERVICE_MERGED_H
