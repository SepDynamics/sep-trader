#include <gtest/gtest.h>
#include "app/DataAccessService.h"

using sep::services::DataAccessService;

TEST(DataAccessServiceParse, HandlesRedisUrl) {
    auto hp = DataAccessService::parseValkeyUrl("redis://example.com:6380");
    EXPECT_EQ(hp.first, "example.com");
    EXPECT_EQ(hp.second, 6380);
}

TEST(DataAccessServiceParse, DefaultsPort) {
    auto hp = DataAccessService::parseValkeyUrl("redis://localhost");
    EXPECT_EQ(hp.first, "localhost");
    EXPECT_EQ(hp.second, 6379);
}
