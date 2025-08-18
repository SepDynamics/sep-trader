#include "util/nlohmann_json_safe.h"
#include "io/qdrant_connector.h"
#include <curl/curl.h>
#include <sstream>
#include <iostream>

namespace sep {
namespace connectors {

// Callback function for curl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}

QdrantConnector::QdrantConnector(const std::string& host, int port)
    : m_host(host), m_port(port), m_initialized(false) {
}

QdrantConnector::~QdrantConnector() {
    // Cleanup curl if initialized
    if (m_initialized) {
        curl_global_cleanup();
    }
}

SEPResult QdrantConnector::initialize() {
    // Initialize libcurl
    CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (res != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl: " << curl_easy_strerror(res) << std::endl;
        return SEPResult::NETWORK_ERROR;
    }
    
    m_initialized = true;
    return SEPResult::SUCCESS;
}

bool QdrantConnector::isHealthy() {
    if (!m_initialized) {
        return false;
    }
    
    nlohmann::json response;
    SEPResult result = sendRequest("healthz", "GET", nlohmann::json(), response);
    
    if (result != SEPResult::SUCCESS) {
        return false;
    }
    
    // Check if response contains a status field with value "ok"
    return response.contains("status") && response["status"] == "ok";
}

SEPResult QdrantConnector::createCollection(const std::string& collection_name, 
                                           size_t vector_size, 
                                           const std::string& distance_type) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    if (collection_name.empty() || vector_size == 0) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    // Check if collection already exists
    if (collectionExists(collection_name)) {
        return SEPResult::ALREADY_EXISTS;
    }
    
    // Build request body
    nlohmann::json request = {
        {"vectors", {
            {"size", vector_size},
            {"distance", distance_type}
        }},
        {"optimizers_config", {
            {"default_segment_number", 2}
        }},
        {"replication_factor", 1}
    };
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name, "PUT", request, response);
    
    if (result != SEPResult::SUCCESS) {
        return result;
    }
    
    // Check if operation was successful
    if (response.contains("result") && response["result"] == true) {
        return SEPResult::SUCCESS;
    }
    
    return SEPResult::PROCESSING_ERROR;
}

SEPResult QdrantConnector::deleteCollection(const std::string& collection_name) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    if (collection_name.empty()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    // Check if collection exists
    if (!collectionExists(collection_name)) {
        return SEPResult::NOT_FOUND;
    }
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name, "DELETE", nlohmann::json(), response);
    
    if (result != SEPResult::SUCCESS) {
        return result;
    }
    
    // Check if operation was successful
    if (response.contains("result") && response["result"] == true) {
        return SEPResult::SUCCESS;
    }
    
    return SEPResult::PROCESSING_ERROR;
}

bool QdrantConnector::collectionExists(const std::string& collection_name) {
    if (!m_initialized || collection_name.empty()) {
        return false;
    }
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name, "GET", nlohmann::json(), response);
    
    if (result != SEPResult::SUCCESS) {
        return false;
    }
    
    // Check if response indicates collection exists
    return response.contains("result") && !response["result"].is_null();
}

SEPResult QdrantConnector::upsertVector(const std::string& collection_name, 
                                       uint64_t id, 
                                       const std::vector<float>& vector, 
                                       const nlohmann::json& payload) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    if (collection_name.empty() || vector.empty()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    // Check if collection exists
    if (!collectionExists(collection_name)) {
        return SEPResult::NOT_FOUND;
    }
    
    // Build request body
    nlohmann::json point = {
        {"id", id},
        {"vector", vector}
    };
    
    if (!payload.is_null()) {
        point["payload"] = payload;
    }
    
    nlohmann::json request = {
        {"points", {point}}
    };
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name + "/points", "PUT", request, response);
    
    if (result != SEPResult::SUCCESS) {
        return result;
    }
    
    // Check if operation was successful
    if (response.contains("result") && response["result"]["status"] == "completed") {
        return SEPResult::SUCCESS;
    }
    
    return SEPResult::PROCESSING_ERROR;
}

SEPResult QdrantConnector::upsertVectors(const std::string& collection_name, 
                                        const std::vector<uint64_t>& ids, 
                                        const std::vector<std::vector<float>>& vectors, 
                                        const std::vector<nlohmann::json>& payloads) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    if (collection_name.empty() || ids.empty() || vectors.empty()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    if (ids.size() != vectors.size()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    if (!payloads.empty() && payloads.size() != ids.size()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    // Check if collection exists
    if (!collectionExists(collection_name)) {
        return SEPResult::NOT_FOUND;
    }
    
    // Build request body
    nlohmann::json points = nlohmann::json::array();
    
    for (size_t i = 0; i < ids.size(); ++i) {
        nlohmann::json point = {
            {"id", ids[i]},
            {"vector", vectors[i]}
        };
        
        if (!payloads.empty() && !payloads[i].is_null()) {
            point["payload"] = payloads[i];
        }
        
        points.push_back(point);
    }
    
    nlohmann::json request = {
        {"points", points}
    };
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name + "/points", "PUT", request, response);
    
    if (result != SEPResult::SUCCESS) {
        return result;
    }
    
    // Check if operation was successful
    if (response.contains("result") && response["result"]["status"] == "completed") {
        return SEPResult::SUCCESS;
    }
    
    return SEPResult::PROCESSING_ERROR;
}

SEPResult QdrantConnector::deleteVector(const std::string& collection_name, uint64_t id) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    if (collection_name.empty()) {
        return SEPResult::INVALID_ARGUMENT;
    }
    
    // Check if collection exists
    if (!collectionExists(collection_name)) {
        return SEPResult::NOT_FOUND;
    }
    
    // Build request body
    nlohmann::json request = {
        {"points", {id}}
    };
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name + "/points/delete", "POST", request, response);
    
    if (result != SEPResult::SUCCESS) {
        return result;
    }
    
    // Check if operation was successful
    if (response.contains("result") && response["result"]["status"] == "completed") {
        return SEPResult::SUCCESS;
    }
    
    return SEPResult::PROCESSING_ERROR;
}

std::vector<SearchResult> QdrantConnector::searchVectors(const std::string& collection_name, 
                                                       const std::vector<float>& query_vector, 
                                                       size_t limit, 
                                                       float score_threshold) {
    std::vector<SearchResult> results;
    
    if (!m_initialized || collection_name.empty() || query_vector.empty()) {
        return results;
    }
    
    // Check if collection exists
    if (!collectionExists(collection_name)) {
        return results;
    }
    
    // Build request body
    nlohmann::json request = {
        {"vector", query_vector},
        {"limit", limit}
    };
    
    if (score_threshold > 0.0f) {
        request["score_threshold"] = score_threshold;
    }
    
    nlohmann::json response;
    SEPResult result = sendRequest("collections/" + collection_name + "/points/search", "POST", request, response);
    
    if (result != SEPResult::SUCCESS) {
        return results;
    }
    
    // Parse search results
    if (response.contains("result") && response["result"].is_array()) {
        for (const auto& item : response["result"]) {
            if (item.contains("id") && item.contains("score")) {
                SearchResult search_result;
                search_result.id = item["id"];
                search_result.score = item["score"];
                
                if (item.contains("payload")) {
                    search_result.payload = item["payload"];
                }
                
                results.push_back(search_result);
            }
        }
    }
    
    return results;
}

std::string QdrantConnector::buildUrl(const std::string& endpoint) {
    std::stringstream ss;
    ss << "http://" << m_host << ":" << m_port << "/" << endpoint;
    return ss.str();
}

SEPResult QdrantConnector::sendRequest(const std::string& endpoint, 
                                      const std::string& method, 
                                      const nlohmann::json& data, 
                                      nlohmann::json& response) {
    if (!m_initialized) {
        return SEPResult::NOT_INITIALIZED;
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        return SEPResult::NETWORK_ERROR;
    }
    
    // Set URL
    std::string url = buildUrl(endpoint);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    
    // Set request method
    if (method == "GET") {
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    } else if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
    } else if (method == "PUT") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
    } else if (method == "DELETE") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    
    // Set request headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Set request data
    std::string request_data;
    if (!data.is_null()) {
        request_data = data.dump();
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_data.c_str());
    }
    
    // Set callback function to receive response
    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Clean up
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        return SEPResult::NETWORK_ERROR;
    }
    
    // Parse response
    try {
        response = nlohmann::json::parse(response_string);
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse response: " << e.what() << std::endl;
        std::cerr << "Response: " << response_string << std::endl;
        return SEPResult::PROCESSING_ERROR;
    }
    
    return SEPResult::SUCCESS;
}

} // namespace connectors
} // namespace sep