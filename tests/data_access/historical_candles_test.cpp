#include <gtest/gtest.h>
#include "app/DataAccessService.h"

#include <hiredis/hiredis.h>
#include <cstdlib>

using namespace sep::services;

namespace {

void seedValkey() {
    redisContext* ctx = redisConnect("127.0.0.1", 6379);
    ASSERT_TRUE(ctx != nullptr);
    std::string key = "md:price:TEST";
    redisCommand(ctx, "DEL %s", key.c_str());
    auto add = [&](uint64_t ts){
        std::string member = std::string("{\"t\":") + std::to_string(ts) +
            ",\"o\":\"1\",\"h\":\"1\",\"l\":\"1\",\"c\":\"1\",\"v\":1}";
        redisCommand(ctx, "ZADD %s %llu %s", key.c_str(), (unsigned long long)ts, member.c_str());
    };
    add(1000);
    add(2000);
    add(3000);
    redisFree(ctx);
}

} // namespace

TEST(DataAccessServiceTest, RetrievesCandlesInRange) {
    // Start local Redis server if not already running
    std::system("redis-server --save '' --appendonly no --daemonize yes > /dev/null");

    seedValkey();

    DataAccessService svc;
    ASSERT_TRUE(svc.initialize().isOk());

    auto candles = svc.getHistoricalCandles("TEST", 1000, 2500);
    EXPECT_EQ(candles.size(), 2);
    if (candles.size() == 2) {
        EXPECT_EQ(candles[0].timestamp, 1000);
        EXPECT_EQ(candles[1].timestamp, 2000);
    }

    svc.shutdown();

    std::system("redis-cli -p 6379 shutdown > /dev/null");
}

