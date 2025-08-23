#include "util/memory_tier_service.h"
#include <gtest/gtest.h>

using Tier = ::sep::quantum::Pattern::MemoryTier;

TEST(MemoryTierServiceTest, AllocatePromoteFree) {
    sep::memory::MemoryTierService svc;
    auto h = svc.alloc(16, Tier::Hot);
    EXPECT_EQ(svc.stats().hot_blocks, 1u);
    svc.promote(h, Tier::Warm);
    EXPECT_EQ(svc.stats().warm_blocks, 1u);
    svc.promote(h, Tier::Cold);
    EXPECT_EQ(svc.stats().cold_blocks, 1u);
    svc.free(h);
    EXPECT_EQ(svc.stats().cold_blocks, 0u);
}

TEST(MemoryTierServiceTest, InvalidTransition) {
    sep::memory::MemoryTierService svc;
    auto h = svc.alloc(8, Tier::Hot);
    EXPECT_THROW(svc.promote(h, Tier::Hot), std::invalid_argument);
    svc.free(h);
}
