#include <gtest/gtest.h>
#include "../../_sep/testbed/evolution/training_session.hpp"
#include "../../_sep/testbed/evolution/pattern.hpp"
#include "../../_sep/testbed/evolution/stubs.hpp"

TEST(QuantumPairTrainer, FitnessImprovesAfterEpochs) {
    sep::engine::EngineFacade engine;
    sep::quantum::QFHBasedProcessor proc;
    sep::testbed::TrainingSession session(engine, proc, 123);
    session.init();
    std::vector<int> corpus{1,0,1,0};
    session.run_epoch(corpus);
    double initial = session.total_fitness();
    for (int i = 0; i < 5; ++i) {
        session.run_epoch(corpus);
    }
    double final = session.total_fitness();
    EXPECT_GT(final, initial);
}
