#include "core/processor.h"
#include <gtest/gtest.h>

using Pattern = sep::compat::Pattern;
using Tier = Pattern::MemoryTier;

TEST(ProcessorTierTest, DefaultReturnsHot) {
    sep::quantum::ProcessingConfig cfg;
    sep::quantum::Processor proc(cfg);
    Pattern hot; hot.id = 1; hot.tier = Tier::Hot;
    Pattern warm; warm.id = 2; warm.tier = Tier::Warm;
    proc.addPattern(hot);
    proc.addPattern(warm);
    auto def = proc.getPatterns();
    ASSERT_EQ(def.size(), 1u);
    EXPECT_EQ(def[0].id, 1u);
    auto warm_only = proc.getPatternsByTier(Tier::Warm);
    ASSERT_EQ(warm_only.size(), 1u);
    EXPECT_EQ(warm_only[0].id, 2u);
}
