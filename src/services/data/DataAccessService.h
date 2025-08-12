#pragma once

#include "../include/IDataAccessService.h"
#include "../common/ServiceBase.h"
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

namespace sep {
namespace services {

/**
 * Implementation of the Data Access Service
 * Provides data storage, retrieval, and query capabilities
 */
class DataAccessService : public IDataAccessService, public ServiceBase {
public:
    DataAccessService();
    virtual ~DataAccessService();
    
    // IService interface
    bool isReady() const override;
    
    // IDataAccessService interface
    Result<std::string> storeObject(
        const std::string& collection,
        const std::map<std::string, std::any>& data,
        const std::string& id = "") override;
    
    Result<std::map<std::string, std::any>> retrieveObject(
        const std::string& collection,
        const std::string& id) override;
    
    Result<void> updateObject(
        const std::string& collection,
        const std::string& id,
        const std::map<std::string, std::any>& data) override;
    
    Result<void> deleteObject(
        const std::string& collection,
        const std::string& id) override;
    
    Result<std::vector<std::map<std::string, std::any>>> queryObjects(
        const std::string& collection,
        const std::vector<QueryFilter>& filters = {},
        const std::vector<SortSpec>& sortSpecs = {},
        int limit = 0,
        int skip = 0) override;
    
    Result<int> countObjects(
        const std::string& collection,
        const std::vector<QueryFilter>& filters = {}) override;
    
    Result<std::shared_ptr<TransactionContext>> beginTransaction() override;
    
    Result<void> executeTransaction(
        std::function<Result<void>(std::shared_ptr<TransactionContext>)> operations) override;
    
    int registerChangeListener(
        const std::string& collection,
        std::function<void(const std::string&, const std::string&)> callback) override;
    
    Result<void> unregisterChangeListener(int subscriptionId) override;
    
    Result<void> createCollection(
        const std::string& collection,
        const std::string& schema = "") override;
    
    Result<void> deleteCollection(const std::string& collection) override;
    
    Result<std::vector<std::string>> getCollections() override;
    
protected:
    // ServiceBase overrides
    Result<void> onInitialize() override;
    Result<void> onShutdown() override;
    
private:
    // Helper methods
    std::string generateUniqueId();
    void notifyChangeListeners(const std::string& collection, const std::string& objectId);
    
    // Data storage
    std::unordered_set<std::string> collections_;
    std::unordered_map<std::string, std::string> collectionSchemas_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::map<std::string, std::any>>> objectStore_;
    
    // Change listeners
    std::unordered_map<int, std::pair<std::string, std::function<void(const std::string&, const std::string&)>>> changeListeners_;
    
    // Synchronization
    std::mutex mutex_;
    
    // ID generators
    std::atomic<int> nextObjectId_{1};
    std::atomic<int> nextSubscriptionId_{1};
};

} // namespace services
} // namespace sep