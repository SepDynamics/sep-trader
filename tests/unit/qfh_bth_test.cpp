#include "core/qfh.h"
#include "core/forward_window_result.h"
#include "core/trajectory.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

class QFHTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test data or objects
    }

    void TearDown() override {
        // Clean up after each test
    }
};

// Test basic QFH event transformation
TEST_F(QFHTest, TransformRichBasicTransitions) {
    // Test null state transition (0->0)
    std::vector<uint8_t> null_bits = {0, 0, 0, 0};
    auto events = sep::quantum::transform_rich(null_bits);
    ASSERT_EQ(events.size(), 3);
    for (const auto& event : events) {
        EXPECT_EQ(event.state, sep::quantum::QFHState::NULL_STATE);
        EXPECT_EQ(event.bit_prev, 0);
        EXPECT_EQ(event.bit_curr, 0);
    }

    // Test flip transitions (0->1 and 1->0)
    std::vector<uint8_t> flip_bits = {0, 1, 0, 1};
    events = sep::quantum::transform_rich(flip_bits);
    ASSERT_EQ(events.size(), 3);
    for (const auto& event : events) {
        EXPECT_EQ(event.state, sep::quantum::QFHState::FLIP);
    }

    // Test rupture transitions (1->1)
    std::vector<uint8_t> rupture_bits = {1, 1, 1, 1};
    events = sep::quantum::transform_rich(rupture_bits);
    ASSERT_EQ(events.size(), 3);
    for (const auto& event : events) {
        EXPECT_EQ(event.state, sep::quantum::QFHState::RUPTURE);
        EXPECT_EQ(event.bit_prev, 1);
        EXPECT_EQ(event.bit_curr, 1);
    }
}

// Test mixed bit transitions
TEST_F(QFHTest, TransformRichMixedTransitions) {
    std::vector<uint8_t> mixed_bits = {0, 0, 1, 1, 0, 1};
    auto events = sep::quantum::transform_rich(mixed_bits);
    
    ASSERT_EQ(events.size(), 5);
    
    EXPECT_EQ(events[0].state, sep::quantum::QFHState::NULL_STATE);  // 0->0
    EXPECT_EQ(events[1].state, sep::quantum::QFHState::FLIP);        // 0->1
    EXPECT_EQ(events[2].state, sep::quantum::QFHState::RUPTURE);     // 1->1
    EXPECT_EQ(events[3].state, sep::quantum::QFHState::FLIP);        // 1->0
    EXPECT_EQ(events[4].state, sep::quantum::QFHState::FLIP);        // 0->1
}

// Test event aggregation
TEST_F(QFHTest, EventAggregation) {
    std::vector<sep::quantum::QFHEvent> events = {
        {0, sep::quantum::QFHState::NULL_STATE, 0, 0},
        {1, sep::quantum::QFHState::NULL_STATE, 0, 0},
        {2, sep::quantum::QFHState::FLIP, 0, 1},
        {3, sep::quantum::QFHState::FLIP, 1, 0},
        {4, sep::quantum::QFHState::RUPTURE, 1, 1},
        {5, sep::quantum::QFHState::RUPTURE, 1, 1},
        {6, sep::quantum::QFHState::RUPTURE, 1, 1}
    };
    
    auto aggregated = sep::quantum::aggregate(events);
    
    ASSERT_EQ(aggregated.size(), 4);
    
    EXPECT_EQ(aggregated[0].state, sep::quantum::QFHState::NULL_STATE);
    EXPECT_EQ(aggregated[0].count, 2);
    
    EXPECT_EQ(aggregated[1].state, sep::quantum::QFHState::FLIP);
    EXPECT_EQ(aggregated[1].count, 1);
    
    EXPECT_EQ(aggregated[2].state, sep::quantum::QFHState::FLIP);
    EXPECT_EQ(aggregated[2].count, 1);
    
    EXPECT_EQ(aggregated[3].state, sep::quantum::QFHState::RUPTURE);
    EXPECT_EQ(aggregated[3].count, 3);
}

// Test QFH processor basic functionality
TEST_F(QFHTest, QFHProcessorBasic) {
    sep::quantum::QFHProcessor processor;
    
    // First bit should not produce an event
    auto result = processor.process(0);
    EXPECT_FALSE(result.has_value());
    
    // 0->0 should produce NULL_STATE
    result = processor.process(0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), sep::quantum::QFHState::NULL_STATE);
    
    // 0->1 should produce FLIP
    result = processor.process(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), sep::quantum::QFHState::FLIP);
    
    // 1->1 should produce RUPTURE
    result = processor.process(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), sep::quantum::QFHState::RUPTURE);
    
    // Reset should clear state
    processor.reset();
    result = processor.process(1);
    EXPECT_FALSE(result.has_value());
}

// Test QFHBasedProcessor analysis
TEST_F(QFHTest, QFHBasedProcessorAnalysis) {
    sep::quantum::QFHOptions options;
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Test with a simple pattern
    std::vector<uint8_t> bits = {0, 0, 1, 1, 0, 1, 0, 0};
    auto result = processor.analyze(bits);
    
    // Check that we have the expected number of events
    EXPECT_EQ(result.events.size(), 7);
    EXPECT_EQ(result.aggregated_events.size(), 6);
    
    // Check event counts
    EXPECT_EQ(result.null_state_count, 2);  // 0->0, 0->0
    EXPECT_EQ(result.flip_count, 4);        // 0->1, 1->0, 0->1, 1->0
    EXPECT_EQ(result.rupture_count, 1);     // 1->1
    
    // Check ratios
    EXPECT_FLOAT_EQ(result.rupture_ratio, 1.0f/7.0f);
    EXPECT_FLOAT_EQ(result.flip_ratio, 4.0f/7.0f);
    
    // Check that coherence, stability and entropy are in valid range
    EXPECT_GE(result.coherence, 0.0);
    EXPECT_LE(result.coherence, 1.0);
    EXPECT_GE(result.entropy, 0.0);
    EXPECT_LE(result.entropy, 1.0);
    
    // Test with empty bits
    std::vector<uint8_t> empty_bits = {};
    auto empty_result = processor.analyze(empty_bits);
    EXPECT_EQ(empty_result.events.size(), 0);
    EXPECT_EQ(empty_result.aggregated_events.size(), 0);
}

// Test QFHBasedProcessor with collapse detection
TEST_F(QFHTest, QFHBasedProcessorCollapseDetection) {
    sep::quantum::QFHOptions options;
    options.collapse_threshold = 0.3f;  // Lower threshold for testing
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Create a pattern with high rupture ratio
    std::vector<uint8_t> high_rupture_bits;
    for (int i = 0; i < 10; i++) {
        high_rupture_bits.push_back(1);  // All 1s will create many ruptures
    }
    
    auto result = processor.analyze(high_rupture_bits);
    
    // With all 1s, we should have rupture_count = bits.size() - 1
    EXPECT_EQ(result.rupture_count, high_rupture_bits.size() - 1);
    
    // Rupture ratio should be 1.0 (all transitions are ruptures)
    EXPECT_FLOAT_EQ(result.rupture_ratio, 1.0f);
    
    // Collapse should be detected since rupture_ratio > collapse_threshold
    EXPECT_TRUE(result.collapse_detected);
}

// Test bit conversion utility
TEST_F(QFHTest, BitConversion) {
    sep::quantum::QFHOptions options;
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Test conversion of uint32_t values to bits
    std::vector<uint32_t> values = {0b10101010, 0b01010101};
    auto bits = processor.convertToBits(values);
    
    // Each uint32_t should produce 32 bits
    EXPECT_EQ(bits.size(), 64);
    
    // Check first few bits of first value (0b10101010)
    EXPECT_EQ(bits[0], 0);  // LSB of 0b10101010
    EXPECT_EQ(bits[1], 1);
    EXPECT_EQ(bits[2], 0);
    EXPECT_EQ(bits[3], 1);
    EXPECT_EQ(bits[4], 0);
    EXPECT_EQ(bits[5], 1);
    EXPECT_EQ(bits[6], 0);
    EXPECT_EQ(bits[7], 1);  // MSB of 0b10101010
}

// Test cosine similarity calculation
TEST_F(QFHTest, CosineSimilarity) {
    sep::quantum::QFHOptions options;
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Test identical vectors (should have similarity of 1.0)
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {1.0, 2.0, 3.0};
    double similarity = processor.calculateCosineSimilarity(vec1, vec2);
    EXPECT_DOUBLE_EQ(similarity, 1.0);
    
    // Test orthogonal vectors (should have similarity of 0.0)
    std::vector<double> vec3 = {1.0, 0.0};
    std::vector<double> vec4 = {0.0, 1.0};
    similarity = processor.calculateCosineSimilarity(vec3, vec4);
    EXPECT_DOUBLE_EQ(similarity, 0.0);
    
    // Test opposite vectors (should have similarity of -1.0)
    std::vector<double> vec5 = {1.0, 1.0};
    std::vector<double> vec6 = {-1.0, -1.0};
    similarity = processor.calculateCosineSimilarity(vec5, vec6);
    EXPECT_DOUBLE_EQ(similarity, -1.0);
    
    // Test with different sized vectors (should return 0.0)
    std::vector<double> vec7 = {1.0, 2.0};
    std::vector<double> vec8 = {1.0, 2.0, 3.0};
    similarity = processor.calculateCosineSimilarity(vec7, vec8);
    EXPECT_DOUBLE_EQ(similarity, 0.0);
    
    // Test with empty vectors (should return 0.0)
    std::vector<double> vec9 = {};
    std::vector<double> vec10 = {};
    similarity = processor.calculateCosineSimilarity(vec9, vec10);
    EXPECT_DOUBLE_EQ(similarity, 0.0);
}

// Test trajectory integration
TEST_F(QFHTest, TrajectoryIntegration) {
    sep::quantum::QFHOptions options;
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Test with a simple bitstream
    std::vector<uint8_t> bitstream = {0, 1, 0, 1, 0};
    
    // Test integration at the beginning of the stream
    auto damped_value = processor.integrateFutureTrajectories(bitstream, 0);
    
    // Should have a path with same length as remaining bits
    EXPECT_EQ(damped_value.path.size(), bitstream.size());
    
    // First value in path should be the current bit
    EXPECT_DOUBLE_EQ(damped_value.path[0], static_cast<double>(bitstream[0]));
    
    // Should have valid confidence value
    EXPECT_GE(damped_value.confidence, 0.0);
    EXPECT_LE(damped_value.confidence, 1.0);
    
    // Test with invalid index
    damped_value = processor.integrateFutureTrajectories(bitstream, bitstream.size());
    EXPECT_EQ(damped_value.path.size(), 0);
    EXPECT_DOUBLE_EQ(damped_value.final_value, 0.0);
    EXPECT_DOUBLE_EQ(damped_value.confidence, 0.0);
}

// Test pattern matching
TEST_F(QFHTest, PatternMatching) {
    sep::quantum::QFHOptions options;
    sep::quantum::QFHBasedProcessor processor(options);
    
    // Test with exponential decay pattern
    std::vector<double> exp_pattern;
    for (int i = 0; i < 10; i++) {
        exp_pattern.push_back(std::exp(-0.1 * i));
    }
    
    double similarity = processor.matchKnownPaths(exp_pattern);
    
    // Should return a valid similarity score
    EXPECT_GE(similarity, 0.0);
    EXPECT_LE(similarity, 1.0);
    
    // Test with too few points
    std::vector<double> small_pattern = {1.0};
    similarity = processor.matchKnownPaths(small_pattern);
    EXPECT_DOUBLE_EQ(similarity, 0.5);  // Default for insufficient data
}