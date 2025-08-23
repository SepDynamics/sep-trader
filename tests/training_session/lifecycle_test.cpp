#include <gtest/gtest.h>
#include <filesystem>
#include "../../_sep/testbed/evolution/training_session.hpp"
#include "../../_sep/testbed/evolution/stubs.hpp"

TEST(TrainingSession, LifecycleWritesArtifactsAndFreesResources) {
    sep::engine::EngineFacade engine;
    sep::quantum::QFHBasedProcessor proc;
    sep::testbed::TrainingSession session(engine, proc, 42);
    session.init();
    std::vector<int> corpus{1};
    session.run_epoch(corpus);
    auto path = std::filesystem::temp_directory_path() / "patterns.txt";
    session.finalize(path.string());
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_FALSE(session.is_initialized());
    EXPECT_TRUE(session.patterns().empty());
}
