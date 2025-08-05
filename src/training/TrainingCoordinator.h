#pragma once

#include <string>
#include <vector>
#include <memory>

#include "trading/data/RemoteDataManager.h"
// Forward declaration for Model (will be created later)
struct Model {
    std::string id;
    std::string data;
};

namespace sep {
namespace training {

class TrainingCoordinator {
public:
    TrainingCoordinator(const std::string& local_model_path, std::shared_ptr<sep::trading::RemoteDataManager> remote_data_manager);

    void sync_latest_model();
    void distribute_new_model(const Model& model);
    void run_distributed_training();

private:
    std::string local_model_path_;
    std::shared_ptr<sep::trading::RemoteDataManager> remote_data_manager_;
};

}
}