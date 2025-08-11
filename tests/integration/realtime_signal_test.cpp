#include <gtest/gtest.h>
#include <vector>
#include "../../_sep/testbed/mock_clock.h"
#include "../../_sep/testbed/mock_tick_stream.h"
#include "../../_sep/testbed/candle_assembler.h"
#include "../../_sep/testbed/simple_signal_detector.h"

TEST(RealtimeSignalTest, AssemblesCandlesAndGeneratesSignal) {
    MockClock clock;
    std::vector<MockTick> ticks = {
        {100.0, 0},
        {101.0, 20000},
        {102.0, 59000}
    };
    MockTickStream stream(ticks);
    SimpleSignalDetector detector;
    std::vector<sep::core::CandleData> candles;

    CandleAssembler assembler(60000, [&](const sep::core::CandleData& c){
        candles.push_back(c);
        detector.onCandle(c);
    });

    stream.run(clock, [&](const MockTick& t){
        assembler.onTick(t);
    });
    assembler.finalize();

    ASSERT_EQ(candles.size(), 1u);
    const auto& c = candles.front();
    EXPECT_DOUBLE_EQ(c.open, 100.0);
    EXPECT_DOUBLE_EQ(c.high, 102.0);
    EXPECT_DOUBLE_EQ(c.low, 100.0);
    EXPECT_DOUBLE_EQ(c.close, 102.0);
    EXPECT_EQ(c.volume, 3u);
    EXPECT_TRUE(detector.upTriggered());
}
