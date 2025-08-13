#pragma once

#include "app/IService.h"
#include "Result.h"
#include <atomic>
#include <string>

namespace sep {
namespace services {

/**
 * Base class for all service implementations
 * Provides common functionality for service lifecycle management
 */
class ServiceBase : public virtual IService {
public:
    ServiceBase(const std::string& name, const std::string& version);
    virtual ~ServiceBase() = default;

    // IService interface implementation
    Result<void> initialize() override;
    Result<void> shutdown() override;
    std::string getName() const override;
    std::string getVersion() const override;
    bool isReady() const override;

protected:
    /**
     * Derived classes implement this to perform service-specific initialization
     * @return Result<void> Success or error
     */
    virtual Result<void> onInitialize() = 0;

    /**
     * Derived classes implement this to perform service-specific shutdown
     * @return Result<void> Success or error
     */
    virtual Result<void> onShutdown() = 0;

private:
    std::string name_;
    std::string version_;
    std::atomic<bool> initialized_;
};

} // namespace services
} // namespace sep