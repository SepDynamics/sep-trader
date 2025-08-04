// SEP Remote Synchronizer Implementation
// Handles synchronization with remote trading systems

#include "remote_synchronizer.hpp"
#include <iostream>

namespace sep {
namespace training {

bool RemoteSynchronizer::sync() {
    std::cout << "Synchronizing with remote trader" << std::endl;
    return true;
}

} // namespace training
} // namespace sep
