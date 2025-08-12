#include "ServiceBase.h"

namespace sep {
namespace services {

ServiceBase::ServiceBase(const std::string& name, const std::string& version)
    : name_(name), version_(version), initialized_(false) {
}

Result<void> ServiceBase::initialize() {
    // Only initialize once
    if (initialized_) {
        return Result<void>{};
    }

    // Call derived class initialization
    auto result = onInitialize();
    if (result.isError()) {
        return result;
    }

    // Mark as initialized
    initialized_ = true;
    return Result<void>{};
}

Result<void> ServiceBase::shutdown() {
    // Only shutdown if initialized
    if (!initialized_) {
        return Result<void>{};
    }

    // Call derived class shutdown
    auto result = onShutdown();
    if (result.isError()) {
        return result;
    }

    // Mark as not initialized
    initialized_ = false;
    return Result<void>{};
}

std::string ServiceBase::getName() const {
    return name_;
}

std::string ServiceBase::getVersion() const {
    return version_;
}

bool ServiceBase::isReady() const {
    return initialized_;
}

} // namespace services
} // namespace sep