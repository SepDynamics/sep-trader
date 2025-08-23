#include "core/dynamic_pair_manager.hpp"
#include <gtest/gtest.h>

using namespace sep::trading;

TEST(DynamicPairManagerTest, RejectsOverThreshold) {
    DynamicPairManager mgr;
    ResourceAllocation alloc;
    alloc.max_hot_bytes = 100;
    alloc.max_streams = 1;
    alloc.max_batch_size = 10;
    mgr.setResourceAllocation(alloc);
    DynamicPairConfig cfg;
    cfg.symbol = "EUR_USD";
    cfg.required_hot_bytes = 200; // exceeds
    EXPECT_FALSE(mgr.hasAvailableResources(cfg));
}
