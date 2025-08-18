#pragma once

#include "util/result.h"
#include <string>
#include <memory>

namespace sep {
namespace services {

/**
 * Base interface for all services in the SEP Engine
 * Provides common service functionality and lifecycle management
 */
class IService {
public:
    virtual ~IService() = default;

    /**
     * Initialize the service
     * @return Result<void> Success or error
     */
    virtual Result<void> initialize() = 0;

    /**
     * Shutdown the service
     * @return Result<void> Success or error
     */
    virtual Result<void> shutdown() = 0;

    /**
     * Get the service name
     * @return The service name
     */
    virtual std::string getName() const = 0;

    /**
     * Get the service version
     * @return The service version
     */
    virtual std::string getVersion() const = 0;

    /**
     * Check if service is initialized and ready
     * @return True if service is ready, false otherwise
     */
    virtual bool isReady() const = 0;
};

} // namespace services
} // namespace sep