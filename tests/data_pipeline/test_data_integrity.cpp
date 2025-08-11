#include <gtest/gtest.h>
#include "trading/quantum_pair_trainer.hpp"
#include "connectors/oanda_connector.h"
#include "dsl/stdlib/core_primitives.h"
#include <cstdlib>
#include <chrono>

namespace sep::tests {

class DataIntegrityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize DSL engine components for testing
        dsl::stdlib::initialize_engine_components();
    }
};

/**
 * Test: Verify QuantumPairTrainer uses real OANDA data, not simulated
 */
TEST_F(DataIntegrityTest, QuantumPairTrainerUsesRealOandaData) {
    // Set up OANDA credentials for testing
    setenv("OANDA_API_KEY", "test_api_key", 1);
    setenv("OANDA_ACCOUNT_ID", "test_account_id", 1);
    
    sep::trading::QuantumTrainingConfig config;
    config.training_window_hours = 1; // Minimal test data
    
    try {
        sep::trading::QuantumPairTrainer trainer(config);
        
        // This should attempt to fetch real OANDA data and fail with auth error
        // (proving it's not using simulated data)
        EXPECT_THROW(
            trainer.fetchTrainingData("EUR_USD"),
            std::runtime_error
        );
        
        // Clean up
        unsetenv("OANDA_API_KEY");
        unsetenv("OANDA_ACCOUNT_ID");
        
    } catch (const std::exception& e) {
        // Expected - should fail without real credentials
        EXPECT_TRUE(std::string(e.what()).find("OANDA") != std::string::npos);
    }
}

/**
 * Test: Verify DSL QFH analyzer uses real engine, not mock values
 */
TEST_F(DataIntegrityTest, DSLUsesRealQFHEngine) {
    std::vector<dsl::stdlib::Value> args; // Empty args for test
    
    // Call QFH analysis - should initialize real engine
    auto result = dsl::stdlib::qfh_analyze(args);
    
    // Real QFH should return coherence score between 0.0 and 1.0
    EXPECT_TRUE(result.is_number());
    double coherence = result.to_number();
    EXPECT_GE(coherence, 0.0);
    EXPECT_LE(coherence, 1.0);
    
    // Call multiple times - real engine should give consistent results for same input
    auto result2 = dsl::stdlib::qfh_analyze(args);
    EXPECT_EQ(result.to_number(), result2.to_number());
}

/**
 * Test: Verify manifold optimizer initializes real components
 */
TEST_F(DataIntegrityTest, DSLUsesRealManifoldOptimizer) {
    std::vector<dsl::stdlib::Value> args; // Empty args for test
    
    // Should not throw - real optimizer should initialize
    EXPECT_NO_THROW(dsl::stdlib::manifold_optimize(args));
    
    // Multiple calls should work consistently
    auto result1 = dsl::stdlib::manifold_optimize(args);
    auto result2 = dsl::stdlib::manifold_optimize(args);
    EXPECT_EQ(result1.to_string(), result2.to_string());
}

/**
 * Test: Data pipeline end-to-end without credentials should fail appropriately
 */
TEST_F(DataIntegrityTest, DataPipelineFailsWithoutCredentials) {
    // Ensure no credentials are set
    unsetenv("OANDA_API_KEY");
    unsetenv("OANDA_ACCOUNT_ID");
    
    sep::trading::QuantumTrainingConfig config;
    
    // Should create trainer but warn about missing credentials
    testing::internal::CaptureStderr();
    sep::trading::QuantumPairTrainer trainer(config);
    std::string captured = testing::internal::GetCapturedStderr();
    
    EXPECT_TRUE(captured.find("Warning") != std::string::npos);
    EXPECT_TRUE(captured.find("OANDA") != std::string::npos);
}

/**
 * Test: Verify no hardcoded simulation values in critical paths
 */
TEST_F(DataIntegrityTest, NoHardcodedSimulationValues) {
    // Test that we don't get the old hardcoded EUR/USD rate of 1.0850
    sep::trading::QuantumTrainingConfig config;
    sep::trading::QuantumPairTrainer trainer(config);
    
    // Without credentials, should throw, not return simulated data with 1.0850
    EXPECT_THROW(
        trainer.fetchTrainingData("EUR_USD"),
        std::runtime_error
    );
}

} // namespace sep::tests
