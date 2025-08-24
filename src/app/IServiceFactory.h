#pragma once

#include "IService.h"
#include "IQuantumProcessingService.h"
#include "IPatternRecognitionService.h"
#include "ITradingLogicService.h"
#include "IDataAccessService.h"
#include <memory>
#include <string>
#include <map>
#include <vector>

namespace sep {
namespace services {

/**
 * Interface for the Service Factory
 * Responsible for creating and managing service instances,
 * handling dependency injection, and service lifecycle
 */
class IServiceFactory {
public:
    virtual ~IServiceFactory() = default;
    
    /**
     * Initialize the service factory
     * @param configParams Configuration parameters
     * @return Result<void> Success or error
     */
    virtual Result<void> initialize(
        const std::map<std::string, std::string>& configParams = {}) = 0;
    
    /**
     * Shutdown the service factory and all managed services
     * @return Result<void> Success or error
     */
    virtual Result<void> shutdown() = 0;
    
    /**
     * Get a service instance by type
     * @tparam ServiceType Type of service to get
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    template<typename ServiceType>
    Result<std::shared_ptr<ServiceType>> getService(
        const std::string& implementationName = "") {
        return getServiceImpl(typeid(ServiceType).name(), implementationName);
    }
    
    /**
     * Register a service instance
     * @tparam ServiceType Type of service to register
     * @param serviceInstance Service instance
     * @param implementationName Optional specific implementation name
     * @return Result<void> Success or error
     */
    template<typename ServiceType>
    Result<void> registerService(
        std::shared_ptr<ServiceType> serviceInstance,
        const std::string& implementationName = "") {
        return registerServiceImpl(
            typeid(ServiceType).name(), 
            std::static_pointer_cast<IService>(serviceInstance),
            implementationName);
    }
    
    /**
     * Create a Quantum Processing Service
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    virtual Result<std::shared_ptr<IQuantumProcessingService>> createQuantumProcessingService(
        const std::string& implementationName = "") = 0;
    
    /**
     * Create a Pattern Recognition Service
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    virtual Result<std::shared_ptr<IPatternRecognitionService>> createPatternRecognitionService(
        const std::string& implementationName = "") = 0;
    
    /**
     * Create a Trading Logic Service
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    virtual Result<std::shared_ptr<ITradingLogicService>> createTradingLogicService(
        const std::string& implementationName = "") = 0;
    
    /**
     * Create a Data Access Service
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    virtual Result<std::shared_ptr<IDataAccessService>> createDataAccessService(
        const std::string& implementationName = "") = 0;
    
    /**
     * Get available service implementations
     * @param serviceType Service type name
     * @return List of available implementation names
     */
    virtual std::vector<std::string> getAvailableImplementations(
        const std::string& serviceType) const = 0;
    
    /**
     * Check if factory is initialized
     * @return True if factory is initialized
     */
    virtual bool isInitialized() const = 0;

protected:
    /**
     * Implementation for getService template
     * @param serviceTypeName Type name of service to get
     * @param implementationName Optional specific implementation name
     * @return Result containing service instance or error
     */
    virtual Result<std::shared_ptr<IService>> getServiceImpl(
        const std::string& serviceTypeName,
        const std::string& implementationName) = 0;
    
    /**
     * Implementation for registerService template
     * @param serviceTypeName Type name of service to register
     * @param serviceInstance Service instance
     * @param implementationName Optional specific implementation name
     * @return Result<void> Success or error
     */
    virtual Result<void> registerServiceImpl(
        const std::string& serviceTypeName,
        std::shared_ptr<IService> serviceInstance,
        const std::string& implementationName) = 0;
};

} // namespace services
} // namespace sep