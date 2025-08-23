#include "core/facade.h"
#include <gtest/gtest.h>

TEST(FacadeStreamTest, RequiresExplicitStream) {
    auto& f = sep::engine::EngineFacade::getInstance();
    f.initialize();
    cudaStream_t s = reinterpret_cast<cudaStream_t>(1);
    void* p = f.dev_alloc(32, s);
    ASSERT_NE(p, nullptr);
    EXPECT_NO_THROW(f.dev_free(p, s));
    f.shutdown();
}
