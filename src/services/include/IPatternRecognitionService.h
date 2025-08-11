#pragma once

#include "IService.h"
#include "PatternTypes.h"
#include "QuantumTypes.h"
#include <vector>
#include <string>
#include <memory>

namespace sep {
namespace services {

/**
 * Interface for the Pattern Recognition Service
 * Responsible for pattern identification, classification, clustering,
 * and evolution tracking
 */
class IPatternRecognitionService : public IService {
public:
    /**
     * Extract patterns from raw input data
     * @param inputData Raw input data as a vector of floats
     * @param extractionParameters Optional parameters to control extraction
     * @return Result containing extracted patterns or error
     */
    virtual Result<std::vector<std::shared_ptr<Pattern>>> extractPatterns(
        const std::vector<float>& inputData,
        const std::map<std::string, float>& extractionParameters = {}) = 0;
    
    /**
     * Extract patterns from a quantum state
     * @param state Quantum state to extract patterns from
     * @return Result containing extracted patterns or error
     */
    virtual Result<std::vector<std::shared_ptr<Pattern>>> extractPatternsFromQuantumState(
        const QuantumState& state) = 0;
    
    /**
     * Classify a pattern
     * @param pattern Pattern to classify
     * @return Result containing classification result or error
     */
    virtual Result<PatternClassification> classifyPattern(
        const Pattern& pattern) = 0;
    
    /**
     * Match a pattern against the pattern database
     * @param pattern Pattern to match
     * @param matchThreshold Minimum score to consider a match
     * @param maxResults Maximum number of matches to return
     * @return Result containing matched patterns or error
     */
    virtual Result<std::vector<PatternMatch>> matchPattern(
        const Pattern& pattern,
        float matchThreshold = 0.7f,
        int maxResults = 10) = 0;
    
    /**
     * Track pattern evolution over time
     * @param patternId Pattern ID to track
     * @param historyLength Maximum number of evolution stages to track
     * @return Result containing pattern evolution history or error
     */
    virtual Result<PatternEvolution> trackPatternEvolution(
        const std::string& patternId,
        int historyLength = 10) = 0;
    
    /**
     * Cluster patterns into groups
     * @param patterns Patterns to cluster
     * @param clusteringParameters Parameters to control clustering
     * @return Result containing pattern clusters or error
     */
    virtual Result<std::vector<PatternCluster>> clusterPatterns(
        const std::vector<std::shared_ptr<Pattern>>& patterns,
        const std::map<std::string, float>& clusteringParameters = {}) = 0;
    
    /**
     * Get list of available pattern classifiers
     * @return Map of classifier names to descriptions
     */
    virtual std::map<std::string, std::string> getAvailableClassifiers() const = 0;
    
    /**
     * Get list of available pattern extraction algorithms
     * @return Map of algorithm names to descriptions
     */
    virtual std::map<std::string, std::string> getAvailableExtractionAlgorithms() const = 0;
    
    /**
     * Store a pattern in the pattern database
     * @param pattern Pattern to store
     * @return Result containing stored pattern ID or error
     */
    virtual Result<std::string> storePattern(const Pattern& pattern) = 0;
    
    /**
     * Retrieve a pattern from the pattern database
     * @param patternId ID of the pattern to retrieve
     * @return Result containing retrieved pattern or error
     */
    virtual Result<std::shared_ptr<Pattern>> retrievePattern(const std::string& patternId) = 0;
};

} // namespace services
} // namespace sep