#include <gtest/gtest.h>
#include "app/QuantumProcessingService.h"

using namespace sep::services;

TEST(QuantumProcessingServiceGuard, MisuseBeforeInit) {
    QuantumProcessingService svc;
    EXPECT_THROW(svc.getAvailableAlgorithms(), std::logic_error);
}

TEST(QuantumProcessingServiceGuard, MisuseAfterStop) {
    QuantumProcessingService svc;
    EXPECT_TRUE(svc.initialize().isSuccess());
    EXPECT_NO_THROW(svc.getAvailableAlgorithms());
    EXPECT_TRUE(svc.shutdown().isSuccess());
    EXPECT_THROW(svc.getAvailableAlgorithms(), std::logic_error);
}

TEST(QuantumProcessingServiceGuard, IdempotentStop) {
    QuantumProcessingService svc;
    EXPECT_TRUE(svc.initialize().isSuccess());
    EXPECT_TRUE(svc.shutdown().isSuccess());
    EXPECT_TRUE(svc.shutdown().isSuccess());
}
