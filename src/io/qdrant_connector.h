#ifndef SEP_QDRANT_CONNECTOR_H
#define SEP_QDRANT_CONNECTOR_H

#include <string>
#include <vector>
#include <cstdint>
#include "util/nlohmann_json_safe.h"

namespace sep {
namespace connectors {

// Result codes for operations
enum class SEPResult {
    SUCCESS,
    NETWORK_ERROR,
    INVALID_ARGUMENT,
    NOT_FOUND,
    ALREADY_EXISTS,
    PROCESSING_ERROR,
    NOT_INITIALIZED
};

// Search result structure
struct SearchResult {
    uint64_t id;
    float score;
    nlohmann::json payload;
};

class QdrantConnector {
public:
    /**
     * Constructor
     * 
     * @param host Hostname or IP address of the Qdrant server
     * @param port HTTP port for the Qdrant REST API
     */
    QdrantConnector(const std::string& host = "localhost", int port = 6333);
    
    /**
     * Destructor
     */
    ~QdrantConnector();
    
    /**
     * Initialize the connector
     * 
     * @return SEPResult status code
     */
    SEPResult initialize();
    
    /**
     * Check if Qdrant server is healthy
     * 
     * @return true if server is healthy, false otherwise
     */
    bool isHealthy();
    
    /**
     * Create a new collection
     * 
     * @param collection_name Name of the collection
     * @param vector_size Dimensionality of vectors
     * @param distance_type Distance metric type ("cosine", "euclid", or "dot")
     * @return SEPResult status code
     */
    SEPResult createCollection(const std::string& collection_name, 
                               size_t vector_size, 
                               const std::string& distance_type = "cosine");
    
    /**
     * Delete a collection
     * 
     * @param collection_name Name of the collection
     * @return SEPResult status code
     */
    SEPResult deleteCollection(const std::string& collection_name);
    
    /**
     * Check if a collection exists
     * 
     * @param collection_name Name of the collection
     * @return true if collection exists, false otherwise
     */
    bool collectionExists(const std::string& collection_name);
    
    /**
     * Insert or update a single vector
     * 
     * @param collection_name Name of the collection
     * @param id ID of the vector
     * @param vector Vector data
     * @param payload Optional metadata payload
     * @return SEPResult status code
     */
    SEPResult upsertVector(const std::string& collection_name, 
                           uint64_t id, 
                           const std::vector<float>& vector, 
                           const nlohmann::json& payload = nlohmann::json());
    
    /**
     * Insert or update multiple vectors in batch
     * 
     * @param collection_name Name of the collection
     * @param ids Vector of IDs
     * @param vectors Vector of vector data
     * @param payloads Vector of metadata payloads (optional)
     * @return SEPResult status code
     */
    SEPResult upsertVectors(const std::string& collection_name, 
                            const std::vector<uint64_t>& ids, 
                            const std::vector<std::vector<float>>& vectors, 
                            const std::vector<nlohmann::json>& payloads = {});
    
    /**
     * Delete a vector
     * 
     * @param collection_name Name of the collection
     * @param id ID of the vector to delete
     * @return SEPResult status code
     */
    SEPResult deleteVector(const std::string& collection_name, uint64_t id);
    
    /**
     * Search for similar vectors
     * 
     * @param collection_name Name of the collection
     * @param query_vector Query vector
     * @param limit Maximum number of results
     * @param score_threshold Minimum similarity score threshold
     * @return Vector of search results
     */
    std::vector<SearchResult> searchVectors(const std::string& collection_name, 
                                           const std::vector<float>& query_vector, 
                                           size_t limit = 10, 
                                           float score_threshold = 0.0);

private:
    std::string m_host;
    int m_port;
    bool m_initialized;
    
    // Helper methods for HTTP communication
    std::string buildUrl(const std::string& endpoint);
    SEPResult sendRequest(const std::string& endpoint, 
                         const std::string& method, 
                         const nlohmann::json& data, 
                         nlohmann::json& response);
};

} // namespace connectors
} // namespace sep

#endif // SEP_QDRANT_CONNECTOR_H