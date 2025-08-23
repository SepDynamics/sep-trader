#include <gtest/gtest.h>
#include "../../_sep/testbed/evolution/evolution_trainer.hpp"
#include "../../_sep/testbed/evolution/training_session.hpp"
#include "../../_sep/testbed/evolution/stubs.hpp"

TEST(PatternEvolution, MutationsAppliedAndLineageRecorded) {
    auto& engine = sep::engine::EngineFacade::getInstance();
    sep::quantum::QFHOptions qfh_opts{};
    sep::quantum::QFHBasedProcessor proc(qfh_opts);
    sep::testbed::TrainingSession session(engine, proc);
    session.init();
    sep::testbed::Pattern p{"parent", {0b1010}, 1.0};
    session.add_pattern(p);
    sep::testbed::EvolutionTrainer trainer(session);
    auto child = trainer.mutate(p, 1.0, 1);
    auto lineage = trainer.record_lineage(child, {p.id}, "flip:p=1.0|shift:+1");
    EXPECT_NE(child.mask[0], p.mask[0]);
    ASSERT_EQ(trainer.lineage().size(), 1u);
    EXPECT_EQ(trainer.lineage()[0].parents[0], p.id);
}
