#pragma once

#include "IService.h"
#include <vector>
#include <string>
#include <map>
#include <any>
#include <functional>
#include <memory>

namespace sep {
namespace services {

/**
 * Query filter for data access operations
 */
struct QueryFilter {
    std::string field;
    std::string operation;  // "eq", "gt", "lt", "contains", etc.
    std::string value;
    
    QueryFilter() {}
    QueryFilter(const std::string& f, const std::string& op, const std::string& val)
        : field(f), operation(op), value(val) {}
};

/**
 * Sort specification for queries
 */
struct SortSpec {
    std::string field;
    bool ascending;
    
    SortSpec() : ascending(true) {}
    SortSpec(const std::string& f, bool asc) : field(f), ascending(asc) {}
};

/**
 * Transaction context
 */
class TransactionContext {
public:
    virtual ~TransactionContext() = default;
    
    /**
     * Commit the transaction
     * @return Result<void> Success or error
     */
    virtual Result<void> commit() = 0;
    
    /**
     * Rollback the transaction
     * @return Result<void> Success or error
     */
    virtual Result<void> rollback() = 0;
    
    /**
     * Check if transaction is active
     * @return True if transaction is active
     */
    virtual bool isActive() const = 0;
};

/**
 * Interface for the Data Access Service
 * Provides data storage, retrieval, and query capabilities
 */
class IDataAccessService : public IService {
public:
    /**
     * Store a data object
     * @param collection Collection name
     * @param data Data to store (as key-value pairs)
     * @param id Optional explicit ID
     * @return Result containing stored object ID or error
     */
    virtual Result<std::string> storeObject(
        const std::string& collection,
        const std::map<std::string, std::any>& data,
        const std::string& id = "") = 0;
    
    /**
     * Retrieve a data object by ID
     * @param collection Collection name
     * @param id Object ID
     * @return Result containing object data or error
     */
    virtual Result<std::map<std::string, std::any>> retrieveObject(
        const std::string& collection,
        const std::string& id) = 0;
    
    /**
     * Update a data object
     * @param collection Collection name
     * @param id Object ID
     * @param data New data (partial update)
     * @return Result<void> Success or error
     */
    virtual Result<void> updateObject(
        const std::string& collection,
        const std::string& id,
        const std::map<std::string, std::any>& data) = 0;
    
    /**
     * Delete a data object
     * @param collection Collection name
     * @param id Object ID
     * @return Result<void> Success or error
     */
    virtual Result<void> deleteObject(
        const std::string& collection,
        const std::string& id) = 0;
    
    /**
     * Query objects in a collection
     * @param collection Collection name
     * @param filters Query filters
     * @param sortSpecs Sort specifications
     * @param limit Maximum number of results (0 for unlimited)
     * @param skip Number of results to skip
     * @return Result containing query results or error
     */
    virtual Result<std::vector<std::map<std::string, std::any>>> queryObjects(
        const std::string& collection,
        const std::vector<QueryFilter>& filters = {},
        const std::vector<SortSpec>& sortSpecs = {},
        int limit = 0,
        int skip = 0) = 0;
    
    /**
     * Count objects matching filters
     * @param collection Collection name
     * @param filters Query filters
     * @return Result containing count or error
     */
    virtual Result<int> countObjects(
        const std::string& collection,
        const std::vector<QueryFilter>& filters = {}) = 0;
    
    /**
     * Begin a transaction
     * @return Result containing transaction context or error
     */
    virtual Result<std::shared_ptr<TransactionContext>> beginTransaction() = 0;
    
    /**
     * Execute multiple operations in a transaction
     * @param operations Function to execute within transaction
     * @return Result<void> Success or error
     */
    virtual Result<void> executeTransaction(
        std::function<Result<void>(std::shared_ptr<TransactionContext>)> operations) = 0;
    
    /**
     * Register a data change listener
     * @param collection Collection to listen for changes
     * @param callback Function to call when changes occur
     * @return Subscription ID for the callback
     */
    virtual int registerChangeListener(
        const std::string& collection,
        std::function<void(const std::string&, const std::string&)> callback) = 0;
    
    /**
     * Unregister a data change listener
     * @param subscriptionId ID returned from registerChangeListener
     * @return Result<void> Success or error
     */
    virtual Result<void> unregisterChangeListener(int subscriptionId) = 0;
    
    /**
     * Create a new collection
     * @param collection Collection name
     * @param schema Optional schema definition
     * @return Result<void> Success or error
     */
    virtual Result<void> createCollection(
        const std::string& collection,
        const std::string& schema = "") = 0;
    
    /**
     * Delete a collection
     * @param collection Collection name
     * @return Result<void> Success or error
     */
    virtual Result<void> deleteCollection(const std::string& collection) = 0;
    
    /**
     * Get list of collections
     * @return Result containing collection names or error
     */
    virtual Result<std::vector<std::string>> getCollections() = 0;
};

} // namespace services
} // namespace sep