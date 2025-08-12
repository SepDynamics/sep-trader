#pragma once

#include "../IService.h"
#include "../PatternTypes.h"
#include "../Result.h"
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <functional>

namespace sep {
namespace services {

/**
 * Interface for the Pattern Recognition Service
 * Provides pattern analysis, classification, matching, and evolutionary tracking capabilities
 */
class IPatternRecognitionService : public IService {
public:
    /**
     * Register a new pattern for recognition
     * @param pattern Pattern to register
     * @return Result containing pattern ID or error
     */
    virtual Result<std::string> registerPattern(const Pattern& pattern) = 0;
    
    /**
     * Retrieve a pattern by ID
     * @param patternId Pattern identifier
     * @return Result containing pattern or error
     */
    virtual Result<Pattern> getPattern(const std::string& patternId) = 0;
    
    /**
     * Update an existing pattern
     * @param patternId Pattern identifier
     * @param pattern Updated pattern data
     * @return Result<void> Success or error
     */
    virtual Result<void> updatePattern(const std::string& patternId, const Pattern& pattern) = 0;
    
    /**
     * Delete a pattern
     * @param patternId Pattern identifier
     * @return Result<void> Success or error
     */
    virtual Result<void> deletePattern(const std::string& patternId) = 0;
    
    /**
     * Classify a pattern
     * @param pattern Pattern to classify
     * @return Result containing classification or error
     */
    virtual Result<PatternClassification> classifyPattern(const Pattern& pattern) = 0;
    
    /**
     * Find patterns similar to the input pattern
     * @param pattern Pattern to match against
     * @param maxResults Maximum number of results to return (0 for unlimited)
     * @param minScore Minimum match score threshold (0.0-1.0)
     * @return Result containing matches or error
     */
    virtual Result<std::vector<PatternMatch>> findSimilarPatterns(
        const Pattern& pattern, 
        int maxResults = 10, 
        float minScore = 0.7f) = 0;
    
    /**
     * Track pattern evolution over time
     * @param patternId Pattern identifier
     * @return Result containing evolution history or error
     */
    virtual Result<PatternEvolution> getPatternEvolution(const std::string& patternId) = 0;
    
    /**
     * Add a pattern evolution stage
     * @param patternId Pattern identifier
     * @param newStage New evolution stage of the pattern
     * @return Result<void> Success or error
     */
    virtual Result<void> addEvolutionStage(const std::string& patternId, const Pattern& newStage) = 0;
    
    /**
     * Cluster patterns based on similarity
     * @param patternIds Vector of pattern IDs to cluster (empty for all patterns)
     * @param numClusters Desired number of clusters (0 for auto-determine)
     * @return Result containing clusters or error
     */
    virtual Result<std::vector<PatternCluster>> clusterPatterns(
        const std::vector<std::string>& patternIds = {}, 
        int numClusters = 0) = 0;
    
    /**
     * Calculate pattern coherence
     * @param pattern Pattern to analyze
     * @return Result containing coherence value or error
     */
    virtual Result<float> calculateCoherence(const Pattern& pattern) = 0;
    
    /**
     * Calculate pattern stability
     * @param pattern Pattern to analyze
     * @return Result containing stability value or error
     */
    virtual Result<float> calculateStability(const Pattern& pattern) = 0;
    
    /**
     * Register a pattern change listener
     * @param callback Function to call when patterns change
     * @return Subscription ID for the callback
     */
    virtual int registerChangeListener(
        std::function<void(const std::string&, const Pattern&)> callback) = 0;
    
    /**
     * Unregister a pattern change listener
     * @param subscriptionId ID returned from registerChangeListener
     * @return Result<void> Success or error
     */
    virtual Result<void> unregisterChangeListener(int subscriptionId) = 0;
};

} // namespace services
} // namespace sep