#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "core/coherence_manager.h"
#include "core/types.h"

// using namespace sep::quantum;

class CoherenceManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        sep::quantum::CoherenceManager::Config config;
        config.max_patterns = 100;
        config.anomaly_threshold = 0.1f;
        config.enable_cuda = false;

        coherence_manager_ = std::make_unique<sep::quantum::CoherenceManager>(config);
    }
    
    void TearDown() override {
        coherence_manager_.reset();
    }

    std::unique_ptr<sep::quantum::CoherenceManager> coherence_manager_;

    // Helper function to create test patterns
    std::vector<sep::quantum::Pattern> createTestPatterns(int count) {
        std::vector<sep::quantum::Pattern> patterns;
        patterns.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            sep::quantum::Pattern pattern;
            pattern.id = i;
            pattern.position = static_cast<double>(i);
            pattern.coherence = 0.5;

            // Set up quantum state
            pattern.quantum_state.coherence = 0.5;
            pattern.quantum_state.stability = 0.7;
            pattern.quantum_state.entropy = 0.3;
            pattern.quantum_state.status = sep::quantum::QuantumState::Status::ACTIVE;

            patterns.push_back(std::move(pattern));
        }
        
        return patterns;
    }
};

// Test basic construction
TEST_F(CoherenceManagerTest, Construction) {
    EXPECT_NE(coherence_manager_, nullptr);
}

// Test updateCoherence with empty patterns
TEST_F(CoherenceManagerTest, UpdateCoherenceEmptyPatterns) {
    std::vector<sep::quantum::Pattern> patterns;
    auto result = coherence_manager_->updateCoherence(patterns);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.global_coherence, 0.0f);
    EXPECT_EQ(result.total_migrations, 0);
    EXPECT_TRUE(result.anomalies.empty());
    EXPECT_TRUE(result.tier_migrations.empty());
}

// Test updateCoherence with single pattern
TEST_F(CoherenceManagerTest, UpdateCoherenceSinglePattern) {
    auto patterns = createTestPatterns(1);
    auto result = coherence_manager_->updateCoherence(patterns);
    
    EXPECT_TRUE(result.success);
    EXPECT_GE(result.global_coherence, 0.0f);
    EXPECT_LE(result.global_coherence, 1.0f);
}

// Test updateCoherence with multiple patterns
TEST_F(CoherenceManagerTest, UpdateCoherenceMultiplePatterns) {
    auto patterns = createTestPatterns(10);
    auto result = coherence_manager_->updateCoherence(patterns);
    
    EXPECT_TRUE(result.success);
    EXPECT_GE(result.global_coherence, 0.0f);
    EXPECT_LE(result.global_coherence, 1.0f);
    EXPECT_EQ(result.total_migrations, 0); // No migrations expected in first update
}

// Test getMetrics
TEST_F(CoherenceManagerTest, GetMetrics) {
    auto patterns = createTestPatterns(5);
    coherence_manager_->updateCoherence(patterns);
    
    const auto& metrics = coherence_manager_->getMetrics();
    EXPECT_GE(metrics.global_coherence, 0.0f);
    EXPECT_LE(metrics.global_coherence, 1.0f);
    EXPECT_EQ(metrics.total_patterns, 5);
}

// Test getGlobalTick
TEST_F(CoherenceManagerTest, GetGlobalTick) {
    auto patterns = createTestPatterns(3);
    coherence_manager_->updateCoherence(patterns);
    
    auto tick = coherence_manager_->getGlobalTick();
    EXPECT_GT(tick, 0);
}

// Test getPatternCountByTier
TEST_F(CoherenceManagerTest, GetPatternCountByTier) {
    auto patterns = createTestPatterns(5);
    coherence_manager_->updateCoherence(patterns);
    
    auto stm_count = coherence_manager_->getPatternCountByTier(sep::memory::MemoryTierEnum::STM);
    auto mtm_count = coherence_manager_->getPatternCountByTier(sep::memory::MemoryTierEnum::MTM);
    auto ltm_count = coherence_manager_->getPatternCountByTier(sep::memory::MemoryTierEnum::LTM);
    
    EXPECT_GE(stm_count, 0);
    EXPECT_GE(mtm_count, 0);
    EXPECT_GE(ltm_count, 0);
}

// Test getTierFragmentation
TEST_F(CoherenceManagerTest, GetTierFragmentation) {
    auto patterns = createTestPatterns(5);
    coherence_manager_->updateCoherence(patterns);
    
    auto stm_frag = coherence_manager_->getTierFragmentation(sep::memory::MemoryTierEnum::STM);
    auto mtm_frag = coherence_manager_->getTierFragmentation(sep::memory::MemoryTierEnum::MTM);
    auto ltm_frag = coherence_manager_->getTierFragmentation(sep::memory::MemoryTierEnum::LTM);
    
    EXPECT_GE(stm_frag, 0.0f);
    EXPECT_LE(stm_frag, 1.0f);
    EXPECT_GE(mtm_frag, 0.0f);
    EXPECT_LE(mtm_frag, 1.0f);
    EXPECT_GE(ltm_frag, 0.0f);
    EXPECT_LE(ltm_frag, 1.0f);
}

// Test applyCoherenceDecay
TEST_F(CoherenceManagerTest, ApplyCoherenceDecay) {
    auto patterns = createTestPatterns(5);
    coherence_manager_->updateCoherence(patterns);
    
    auto initial_metrics = coherence_manager_->getMetrics();
    float initial_coherence = initial_metrics.global_coherence;
    
    coherence_manager_->applyCoherenceDecay(0.1f);
    
    auto updated_metrics = coherence_manager_->getMetrics();
    float updated_coherence = updated_metrics.global_coherence;
    
    // Coherence should decrease after decay
    EXPECT_LE(updated_coherence, initial_coherence);
}

// Test optimizeMemoryLayout
TEST_F(CoherenceManagerTest, OptimizeMemoryLayout) {
    auto patterns = createTestPatterns(5);
    coherence_manager_->updateCoherence(patterns);
    
    auto migrations = coherence_manager_->optimizeMemoryLayout();
    
    // We don't expect migrations with default settings and simple patterns
    EXPECT_TRUE(migrations.empty());
}

// Test computeEntanglementGraph
TEST_F(CoherenceManagerTest, ComputeEntanglementGraph) {
    auto patterns = createTestPatterns(3);
    auto graph = coherence_manager_->computeEntanglementGraph(patterns);
    
    EXPECT_EQ(graph.nodes.size(), 3);
    EXPECT_GE(graph.edges.size(), 0);
    EXPECT_GE(graph.total_entanglement, 0.0f);
    EXPECT_GE(graph.max_degree, 0);
    EXPECT_GE(graph.clustering_coefficient, 0.0f);
}

// Test createSnapshot and restoreFromSnapshot
TEST_F(CoherenceManagerTest, SnapshotAndRestore) {
    auto patterns = createTestPatterns(3);
    coherence_manager_->updateCoherence(patterns);
    
    auto snapshot = coherence_manager_->createSnapshot();
    EXPECT_GT(snapshot.timestamp, 0);
    EXPECT_FALSE(snapshot.pattern_states.empty());
    
    // Create a new manager and restore from snapshot
    sep::quantum::CoherenceManager::Config config;
    config.max_patterns = 100;
    config.anomaly_threshold = 0.1f;
    config.enable_cuda = false;

    auto new_manager = std::make_unique<sep::quantum::CoherenceManager>(config);
    bool restored = new_manager->restoreFromSnapshot(snapshot);
    
    EXPECT_TRUE(restored);
    
    // Check that metrics are restored
    const auto& original_metrics = coherence_manager_->getMetrics();
    const auto& restored_metrics = new_manager->getMetrics();
    
    EXPECT_EQ(original_metrics.total_patterns, restored_metrics.total_patterns);
}