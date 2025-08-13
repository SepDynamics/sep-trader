#pragma once

#include "../../nlohmann_json_safe.h"
#include <array>
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <memory>

namespace sep {
namespace services {

/**
 * Represents a pattern in the SEP Engine
 */
struct Pattern {
    std::string id;
    std::vector<float> features;
    float coherence;
    float stability;
    std::string classificationLabel;
    std::map<std::string, float> attributeScores;
    uint64_t timestamp;
    
    Pattern() : coherence(0.0f), stability(0.0f), timestamp(0) {}
};

/**
 * Pattern classification result
 */
struct PatternClassification {
    std::string primaryClass;
    std::map<std::string, float> classProbabilities;
    float confidence;
    
    PatternClassification() : confidence(0.0f) {}
};

/**
 * Pattern matching result
 */
struct PatternMatch {
    std::string patternId;
    float matchScore;
    std::vector<std::pair<int, int>> featureMatches;
    
    PatternMatch() : matchScore(0.0f) {}
};

/**
 * Pattern evolution history
 */
struct PatternEvolution {
    std::string patternId;
    std::vector<std::shared_ptr<Pattern>> evolutionStages;
    std::vector<float> stabilityHistory;
    std::vector<float> coherenceHistory;
    
    PatternEvolution() {}
};

/**
 * Pattern cluster
 */
struct PatternCluster {
    std::string clusterId;
    std::vector<std::string> patternIds;
    std::vector<float> centroidFeatures;
    float cohesion;
    float separation;
    
    PatternCluster() : cohesion(0.0f), separation(0.0f) {}
};

} // namespace services
} // namespace sep