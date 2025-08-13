#include "PatternRecognitionService.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <chrono>

namespace sep {
namespace services {

PatternRecognitionService::PatternRecognitionService()
    : ServiceBase("PatternRecognitionService", "1.0.0") {
}

PatternRecognitionService::~PatternRecognitionService() {
    if (isReady()) {
        ServiceBase::shutdown();
    }
}

bool PatternRecognitionService::isReady() const {
    return ServiceBase::isReady();
}

Result<void> PatternRecognitionService::onInitialize() {
    // Initialize pattern recognition subsystems
    // For now, just return success
    return Result<void>();
}

Result<void> PatternRecognitionService::onShutdown() {
    // Clean up resources
    std::lock_guard<std::mutex> lock(mutex_);
    patterns_.clear();
    evolutions_.clear();
    clusters_.clear();
    changeListeners_.clear();
    return Result<void>();
}

Result<std::string> PatternRecognitionService::registerPattern(const Pattern& pattern) {
    if (!isReady()) {
        return Result<std::string>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create a copy of the pattern with a new ID if not provided
    Pattern newPattern = pattern;
    if (newPattern.id.empty()) {
        newPattern.id = generateUniqueId();
    } else if (patterns_.find(newPattern.id) != patterns_.end()) {
        return Result<std::string>(Error(Error::Code::InvalidArgument, "Pattern ID already exists"));
    }
    
    // Calculate coherence and stability if not provided
    if (newPattern.coherence <= 0.0f) {
        auto coherenceResult = calculateCoherence(newPattern);
        if (!coherenceResult.isError()) {
            newPattern.coherence = coherenceResult.value();
        }
    }
    
    if (newPattern.stability <= 0.0f) {
        auto stabilityResult = calculateStability(newPattern);
        if (!stabilityResult.isError()) {
            newPattern.stability = stabilityResult.value();
        }
    }
    
    // Set timestamp if not provided
    if (newPattern.timestamp == 0) {
        newPattern.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    // Store the pattern
    patterns_[newPattern.id] = newPattern;
    
    // Create evolution entry if it doesn't exist
    if (evolutions_.find(newPattern.id) == evolutions_.end()) {
        PatternEvolution evolution;
        evolution.patternId = newPattern.id;
        evolution.evolutionStages.push_back(std::make_shared<Pattern>(newPattern));
        evolution.stabilityHistory.push_back(newPattern.stability);
        evolution.coherenceHistory.push_back(newPattern.coherence);
        evolutions_[newPattern.id] = evolution;
    }
    
    // Notify listeners
    notifyChangeListeners(newPattern.id, newPattern);
    
    return Result<std::string>(newPattern.id);
}

Result<Pattern> PatternRecognitionService::getPattern(const std::string& patternId) {
    if (!isReady()) {
        return Result<Pattern>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = patterns_.find(patternId);
    if (it == patterns_.end()) {
        return Result<Pattern>(Error(Error::Code::NotFound, "Pattern not found"));
    }
    
    return Result<Pattern>(it->second);
}

Result<void> PatternRecognitionService::updatePattern(const std::string& patternId, const Pattern& pattern) {
    if (!isReady()) {
        return Result<void>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = patterns_.find(patternId);
    if (it == patterns_.end()) {
        return Result<void>(Error(Error::Code::NotFound, "Pattern not found"));
    }
    
    // Create a copy of the new pattern with the existing ID
    Pattern updatedPattern = pattern;
    updatedPattern.id = patternId;
    
    // Preserve the original timestamp if the new one is not provided
    if (updatedPattern.timestamp == 0) {
        updatedPattern.timestamp = it->second.timestamp;
    }
    
    // Update pattern
    patterns_[patternId] = updatedPattern;
    
    // Add to evolution history
    auto evolIt = evolutions_.find(patternId);
    if (evolIt != evolutions_.end()) {
        evolIt->second.evolutionStages.push_back(std::make_shared<Pattern>(updatedPattern));
        evolIt->second.stabilityHistory.push_back(updatedPattern.stability);
        evolIt->second.coherenceHistory.push_back(updatedPattern.coherence);
    }
    
    // Notify listeners
    notifyChangeListeners(patternId, updatedPattern);
    
    return Result<void>();
}

Result<void> PatternRecognitionService::deletePattern(const std::string& patternId) {
    if (!isReady()) {
        return Result<void>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = patterns_.find(patternId);
    if (it == patterns_.end()) {
        return Result<void>(Error(Error::Code::NotFound, "Pattern not found"));
    }
    
    // Store the pattern for notification
    Pattern deletedPattern = it->second;
    
    // Remove pattern and evolution history
    patterns_.erase(it);
    evolutions_.erase(patternId);
    
    // Remove from clusters
    for (auto& cluster : clusters_) {
        auto& patternIds = cluster.patternIds;
        patternIds.erase(
            std::remove(patternIds.begin(), patternIds.end(), patternId),
            patternIds.end()
        );
    }
    
    // Clean up empty clusters
    clusters_.erase(
        std::remove_if(clusters_.begin(), clusters_.end(),
            [](const PatternCluster& cluster) {
                return cluster.patternIds.empty();
            }),
        clusters_.end()
    );
    
    // Notify listeners about deletion
    notifyChangeListeners(patternId, deletedPattern);
    
    return Result<void>();
}

Result<PatternClassification> PatternRecognitionService::classifyPattern(const Pattern& pattern) {
    if (!isReady()) {
        return Result<PatternClassification>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    // Simple classification based on similarity to existing patterns
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (patterns_.empty()) {
        return Result<PatternClassification>(Error(Error::Code::ResourceUnavailable, "No patterns available for classification"));
    }
    
    PatternClassification classification;
    classification.confidence = 0.0f;
    
    // Group patterns by class
    std::map<std::string, std::vector<Pattern>> classBuckets;
    for (const auto& pair : patterns_) {
        if (!pair.second.classificationLabel.empty()) {
            classBuckets[pair.second.classificationLabel].push_back(pair.second);
        }
    }
    
    if (classBuckets.empty()) {
        return Result<PatternClassification>(Error(Error::Code::ResourceUnavailable, "No classified patterns available"));
    }
    
    // Calculate average similarity to each class
    std::map<std::string, float> classScores;
    for (const auto& classPair : classBuckets) {
        float totalSimilarity = 0.0f;
        for (const auto& classPattern : classPair.second) {
            totalSimilarity += computePatternSimilarity(pattern, classPattern);
        }
        classScores[classPair.first] = totalSimilarity / classPair.second.size();
    }
    
    // Find the class with highest similarity
    std::string bestClass;
    float bestScore = 0.0f;
    for (const auto& scorePair : classScores) {
        if (scorePair.second > bestScore) {
            bestScore = scorePair.second;
            bestClass = scorePair.first;
        }
    }
    
    // Set classification result
    classification.primaryClass = bestClass;
    classification.classProbabilities = classScores;
    classification.confidence = bestScore;
    
    return Result<PatternClassification>(classification);
}

Result<std::vector<PatternMatch>> PatternRecognitionService::findSimilarPatterns(
    const Pattern& pattern, 
    int maxResults, 
    float minScore) {
    
    if (!isReady()) {
        return Result<std::vector<PatternMatch>>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (patterns_.empty()) {
        return Result<std::vector<PatternMatch>>(Error(Error::Code::ResourceUnavailable, "No patterns available for matching"));
    }
    
    std::vector<PatternMatch> matches;
    
    // Compare with all existing patterns
    for (const auto& pair : patterns_) {
        float similarity = computePatternSimilarity(pattern, pair.second);
        
        if (similarity >= minScore) {
            PatternMatch match;
            match.patternId = pair.first;
            match.matchScore = similarity;
            
            // For simplicity, we're not computing detailed feature matches here
            // In a full implementation, this would identify which features match
            
            matches.push_back(match);
        }
    }
    
    // Sort by match score (descending)
    std::sort(matches.begin(), matches.end(), 
        [](const PatternMatch& a, const PatternMatch& b) {
            return a.matchScore > b.matchScore;
        });
    
    // Limit results if needed
    if (maxResults > 0 && matches.size() > static_cast<size_t>(maxResults)) {
        matches.resize(maxResults);
    }
    
    return Result<std::vector<PatternMatch>>(matches);
}

Result<PatternEvolution> PatternRecognitionService::getPatternEvolution(const std::string& patternId) {
    if (!isReady()) {
        return Result<PatternEvolution>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = evolutions_.find(patternId);
    if (it == evolutions_.end()) {
        return Result<PatternEvolution>(Error(Error::Code::NotFound, "Pattern evolution not found"));
    }
    
    return Result<PatternEvolution>(it->second);
}

Result<void> PatternRecognitionService::addEvolutionStage(const std::string& patternId, const Pattern& newStage) {
    if (!isReady()) {
        return Result<void>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto patternIt = patterns_.find(patternId);
    if (patternIt == patterns_.end()) {
        return Result<void>(Error(Error::Code::NotFound, "Pattern not found"));
    }
    
    auto evolIt = evolutions_.find(patternId);
    if (evolIt == evolutions_.end()) {
        // Create new evolution record
        PatternEvolution evolution;
        evolution.patternId = patternId;
        evolution.evolutionStages.push_back(std::make_shared<Pattern>(patternIt->second));
        evolution.stabilityHistory.push_back(patternIt->second.stability);
        evolution.coherenceHistory.push_back(patternIt->second.coherence);
        evolutions_[patternId] = evolution;
        evolIt = evolutions_.find(patternId);
    }
    
    // Create a copy with the right ID
    Pattern stageCopy = newStage;
    stageCopy.id = patternId;
    
    // Add evolution stage
    evolIt->second.evolutionStages.push_back(std::make_shared<Pattern>(stageCopy));
    evolIt->second.stabilityHistory.push_back(stageCopy.stability);
    evolIt->second.coherenceHistory.push_back(stageCopy.coherence);
    
    // Update the current pattern
    patterns_[patternId] = stageCopy;
    
    // Notify listeners
    notifyChangeListeners(patternId, stageCopy);
    
    return Result<void>();
}

Result<std::vector<PatternCluster>> PatternRecognitionService::clusterPatterns(
    const std::vector<std::string>& patternIds, 
    int numClusters) {
    
    if (!isReady()) {
        return Result<std::vector<PatternCluster>>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get patterns to cluster
    std::vector<Pattern> patternsToCluster;
    if (patternIds.empty()) {
        // Use all patterns
        for (const auto& pair : patterns_) {
            patternsToCluster.push_back(pair.second);
        }
    } else {
        // Use specified patterns
        for (const auto& id : patternIds) {
            auto it = patterns_.find(id);
            if (it != patterns_.end()) {
                patternsToCluster.push_back(it->second);
            }
        }
    }
    
    if (patternsToCluster.empty()) {
        return Result<std::vector<PatternCluster>>(Error(Error::Code::ResourceUnavailable, "No patterns available for clustering"));
    }
    
    // Simple clustering algorithm (for demonstration)
    // In a real implementation, this would use a more sophisticated approach like k-means
    std::vector<PatternCluster> clusters;
    
    // Determine number of clusters if not specified
    int k = numClusters;
    if (k <= 0) {
        // Simple heuristic - square root of number of patterns
        k = static_cast<int>(std::sqrt(patternsToCluster.size()));
        k = std::max(1, std::min(k, 10)); // Between 1 and 10 clusters
    }
    
    // For simplicity, just create random clusters in this implementation
    // A real implementation would use proper clustering algorithms
    
    // Create empty clusters
    for (int i = 0; i < k; ++i) {
        PatternCluster cluster;
        cluster.clusterId = "cluster_" + std::to_string(i+1);
        cluster.cohesion = 0.0f;
        cluster.separation = 0.0f;
        clusters.push_back(cluster);
    }
    
    // Assign patterns to clusters
    std::default_random_engine rng(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> dist(0, k-1);
    
    for (const auto& pattern : patternsToCluster) {
        int clusterIdx = dist(rng);
        clusters[clusterIdx].patternIds.push_back(pattern.id);
        
        // Update centroid (simple average of features)
        if (clusters[clusterIdx].centroidFeatures.empty()) {
            clusters[clusterIdx].centroidFeatures = pattern.features;
        } else {
            for (size_t i = 0; i < pattern.features.size() && i < clusters[clusterIdx].centroidFeatures.size(); ++i) {
                clusters[clusterIdx].centroidFeatures[i] = 
                    (clusters[clusterIdx].centroidFeatures[i] * (clusters[clusterIdx].patternIds.size() - 1) + 
                     pattern.features[i]) / clusters[clusterIdx].patternIds.size();
            }
        }
    }
    
    // Store the clusters
    clusters_ = clusters;
    
    return Result<std::vector<PatternCluster>>(clusters);
}

Result<float> PatternRecognitionService::calculateCoherence(const Pattern& pattern) {
    if (!isReady()) {
        return Result<float>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    // Simple coherence calculation based on feature distribution
    // A real implementation would use more sophisticated quantum coherence metrics
    
    if (pattern.features.empty()) {
        return Result<float>(Error(Error::Code::InvalidArgument, "Pattern has no features"));
    }
    
    // Calculate variance of features as a simple coherence metric
    float mean = 0.0f;
    for (float value : pattern.features) {
        mean += value;
    }
    mean /= pattern.features.size();
    
    float variance = 0.0f;
    for (float value : pattern.features) {
        float diff = value - mean;
        variance += diff * diff;
    }
    variance /= pattern.features.size();
    
    // Convert variance to coherence (0-1 scale, where higher variance means lower coherence)
    float coherence = 1.0f / (1.0f + variance);
    
    return Result<float>(coherence);
}

Result<float> PatternRecognitionService::calculateStability(const Pattern& pattern) {
    if (!isReady()) {
        return Result<float>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    // Check if we have evolution history for this pattern
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto evolIt = evolutions_.find(pattern.id);
    if (evolIt == evolutions_.end() || evolIt->second.stabilityHistory.size() < 2) {
        // If no history, use a default stability or calculate from features
        return Result<float>(0.75f); // Default moderate stability
    }
    
    // Calculate stability from history
    const auto& history = evolIt->second.stabilityHistory;
    float sum = 0.0f;
    for (size_t i = 1; i < history.size(); ++i) {
        sum += std::abs(history[i] - history[i-1]);
    }
    
    // Convert to stability (0-1 scale, where lower variance means higher stability)
    float avgChange = sum / (history.size() - 1);
    float stability = 1.0f / (1.0f + 5.0f * avgChange); // Scale factor 5.0 for sensitivity
    
    return Result<float>(stability);
}

int PatternRecognitionService::registerChangeListener(
    std::function<void(const std::string&, const Pattern&)> callback) {
    
    if (!isReady()) {
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    int subscriptionId = nextSubscriptionId_++;
    changeListeners_[subscriptionId] = callback;
    
    return subscriptionId;
}

Result<void> PatternRecognitionService::unregisterChangeListener(int subscriptionId) {
    if (!isReady()) {
        return Result<void>(Error(Error::Code::ResourceUnavailable, "Service not initialized"));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = changeListeners_.find(subscriptionId);
    if (it == changeListeners_.end()) {
        return Result<void>(Error(Error::Code::NotFound, "Subscription not found"));
    }
    
    changeListeners_.erase(it);
    return Result<void>();
}

std::string PatternRecognitionService::generateUniqueId() {
    std::stringstream ss;
    ss << "pattern_" << nextPatternId_++;
    return ss.str();
}

void PatternRecognitionService::notifyChangeListeners(const std::string& patternId, const Pattern& pattern) {
    for (const auto& pair : changeListeners_) {
        try {
            pair.second(patternId, pattern);
        } catch (...) {
            // Ignore exceptions from listeners
        }
    }
}

float PatternRecognitionService::computePatternSimilarity(const Pattern& pattern1, const Pattern& pattern2) {
    // Simple similarity calculation based on feature vector cosine similarity
    
    // Handle empty features
    if (pattern1.features.empty() || pattern2.features.empty()) {
        return 0.0f;
    }
    
    // Find the minimum size between the two feature vectors
    size_t minSize = std::min(pattern1.features.size(), pattern2.features.size());
    
    // Calculate dot product
    float dotProduct = 0.0f;
    for (size_t i = 0; i < minSize; ++i) {
        dotProduct += pattern1.features[i] * pattern2.features[i];
    }
    
    // Calculate magnitudes
    float mag1 = 0.0f;
    for (size_t i = 0; i < minSize; ++i) {
        mag1 += pattern1.features[i] * pattern1.features[i];
    }
    mag1 = std::sqrt(mag1);
    
    float mag2 = 0.0f;
    for (size_t i = 0; i < minSize; ++i) {
        mag2 += pattern2.features[i] * pattern2.features[i];
    }
    mag2 = std::sqrt(mag2);
    
    // Handle zero magnitudes
    if (mag1 < 1e-6f || mag2 < 1e-6f) {
        return 0.0f;
    }
    
    // Calculate cosine similarity
    float similarity = dotProduct / (mag1 * mag2);
    
    // Ensure the result is in [0, 1] range
    similarity = std::max(0.0f, std::min(1.0f, similarity));
    
    return similarity;
}

} // namespace services
} // namespace sep