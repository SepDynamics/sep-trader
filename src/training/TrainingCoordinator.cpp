#include "training/TrainingCoordinator.h"

#include <iostream>

namespace sep {
namespace training {

TrainingCoordinator::TrainingCoordinator(const std::string& local_model_path, std::shared_ptr<sep::trading::RemoteDataManager> remote_data_manager)
    : local_model_path_(local_model_path), remote_data_manager_(remote_data_manager) {
}

void TrainingCoordinator::sync_latest_model() {
    // Implementation to be added
}

void TrainingCoordinator::distribute_new_model(const Model& model) {
    (void)model; // Mark as unused
    // Implementation to be added
}

void TrainingCoordinator::run_distributed_training() {
    // Implementation to be added
}

}
}