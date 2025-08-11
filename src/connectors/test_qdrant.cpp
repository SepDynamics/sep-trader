#include "qdrant_connector.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

using namespace sep::connectors;

// Helper function to generate random vector of specified dimension
std::vector<float> generateRandomVector(size_t dimension) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> vec(dimension);
    for (auto& v : vec) {
        v = dist(gen);
    }
    return vec;
}

// Helper function to normalize vector (for cosine similarity)
void normalizeVector(std::vector<float>& vec) {
    float sum = 0.0f;
    for (const auto& v : vec) {
        sum += v * v;
    }
    float norm = std::sqrt(sum);
    if (norm > 0) {
        for (auto& v : vec) {
            v /= norm;
        }
    }
}

// Helper function to print vector (truncated if long)
void printVector(const std::vector<float>& vec, size_t max_items = 5) {
    std::cout << "[";
    for (size_t i = 0; i < std::min(vec.size(), max_items); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < std::min(vec.size(), max_items) - 1) {
            std::cout << ", ";
        }
    }
    if (vec.size() > max_items) {
        std::cout << ", ... (" << vec.size() - max_items << " more)";
    }
    std::cout << "]" << std::endl;
}

// Helper function to create a timestamp-based collection name
std::string getUniqueCollectionName() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "test_collection_" << time;
    return ss.str();
}

int main() {
    std::cout << "=== Qdrant Connector Test ===\n" << std::endl;
    
    // Initialize connector
    std::cout << "Initializing Qdrant connector..." << std::endl;
    QdrantConnector connector("localhost", 6333);
    auto init_result = connector.initialize();
    
    if (init_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to initialize connector: " << static_cast<int>(init_result) << std::endl;
        return 1;
    }
    
    // Check if Qdrant server is healthy
    std::cout << "Checking server health..." << std::endl;
    if (!connector.isHealthy()) {
        std::cerr << "Qdrant server is not healthy. Please ensure it's running." << std::endl;
        return 1;
    }
    std::cout << "Qdrant server is healthy!" << std::endl;
    
    // Create a unique collection name to avoid conflicts
    const std::string collection_name = getUniqueCollectionName();
    const size_t vector_dim = 128;
    std::cout << "Creating collection: " << collection_name << " with dimension " << vector_dim << std::endl;
    
    // Create collection
    auto create_result = connector.createCollection(collection_name, vector_dim, "cosine");
    if (create_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to create collection: " << static_cast<int>(create_result) << std::endl;
        return 1;
    }
    
    // Verify collection exists
    std::cout << "Verifying collection exists..." << std::endl;
    bool exists = connector.collectionExists(collection_name);
    assert(exists && "Collection should exist after creation");
    std::cout << "Collection exists: " << (exists ? "yes" : "no") << std::endl;
    
    // Generate test vectors
    std::cout << "\nGenerating test vectors..." << std::endl;
    const size_t num_vectors = 10;
    std::vector<std::vector<float>> test_vectors;
    std::vector<uint64_t> vector_ids;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = generateRandomVector(vector_dim);
        normalizeVector(vec); // Normalize for cosine similarity
        test_vectors.push_back(vec);
        vector_ids.push_back(i + 1); // IDs start from 1
        
        std::cout << "Vector " << (i + 1) << ": ";
        printVector(vec);
    }
    
    // Test single vector upsert
    std::cout << "\nTesting single vector upsert..." << std::endl;
    nlohmann::json metadata = {
        {"pattern_id", "test_pattern_1"},
        {"timestamp", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
        {"source", "test_script"},
        {"coherence", 0.85},
        {"stability", 0.92}
    };
    
    auto upsert_result = connector.upsertVector(collection_name, vector_ids[0], test_vectors[0], metadata);
    if (upsert_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to upsert vector: " << static_cast<int>(upsert_result) << std::endl;
        return 1;
    }
    std::cout << "Successfully upserted vector with ID: " << vector_ids[0] << std::endl;
    
    // Test batch vector upsert
    std::cout << "\nTesting batch vector upsert..." << std::endl;
    std::vector<nlohmann::json> batch_metadata;
    for (size_t i = 1; i < num_vectors; ++i) {
        batch_metadata.push_back({
            {"pattern_id", "test_pattern_" + std::to_string(i + 1)},
            {"timestamp", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
            {"source", "test_script"},
            {"coherence", 0.75 + static_cast<float>(i) / 100.0f},
            {"stability", 0.82 + static_cast<float>(i) / 100.0f}
        });
    }
    
    // Skip the first vector which was already inserted
    std::vector<uint64_t> batch_ids(vector_ids.begin() + 1, vector_ids.end());
    std::vector<std::vector<float>> batch_vectors(test_vectors.begin() + 1, test_vectors.end());
    
    auto batch_result = connector.upsertVectors(collection_name, batch_ids, batch_vectors, batch_metadata);
    if (batch_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to batch upsert vectors: " << static_cast<int>(batch_result) << std::endl;
        return 1;
    }
    std::cout << "Successfully upserted " << batch_ids.size() << " vectors in batch mode" << std::endl;
    
    // Test vector search
    std::cout << "\nTesting vector search..." << std::endl;
    // Create a query vector similar to one of our test vectors but slightly modified
    auto query_vector = test_vectors[3];
    // Slightly modify the vector to make it similar but not identical
    for (auto& v : query_vector) {
        v += static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
    }
    normalizeVector(query_vector);
    
    std::cout << "Query vector: ";
    printVector(query_vector);
    
    std::cout << "Searching for similar vectors..." << std::endl;
    auto search_results = connector.searchVectors(collection_name, query_vector, 5, 0.7);
    
    if (search_results.empty()) {
        std::cout << "No search results found" << std::endl;
    } else {
        std::cout << "Found " << search_results.size() << " similar vectors:" << std::endl;
        for (const auto& result : search_results) {
            std::cout << "ID: " << result.id << ", Score: " << std::fixed << std::setprecision(4) << result.score << std::endl;
            std::cout << "Metadata: " << result.payload.dump(2) << std::endl;
        }
    }
    
    // Test vector deletion
    std::cout << "\nTesting vector deletion..." << std::endl;
    auto delete_result = connector.deleteVector(collection_name, vector_ids[0]);
    if (delete_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to delete vector: " << static_cast<int>(delete_result) << std::endl;
        return 1;
    }
    std::cout << "Successfully deleted vector with ID: " << vector_ids[0] << std::endl;
    
    // Verify vector deletion by searching for it
    auto verify_results = connector.searchVectors(collection_name, test_vectors[0], 1, 0.99);
    if (verify_results.empty() || verify_results[0].id != vector_ids[0]) {
        std::cout << "Vector deletion verified: Vector " << vector_ids[0] << " not found" << std::endl;
    } else {
        std::cerr << "Vector deletion failed: Vector " << vector_ids[0] << " still exists" << std::endl;
    }
    
    // Clean up - delete collection
    std::cout << "\nCleaning up - deleting collection..." << std::endl;
    auto delete_coll_result = connector.deleteCollection(collection_name);
    if (delete_coll_result != SEPResult::SUCCESS) {
        std::cerr << "Failed to delete collection: " << static_cast<int>(delete_coll_result) << std::endl;
        return 1;
    }
    
    // Verify collection deletion
    exists = connector.collectionExists(collection_name);
    std::cout << "Collection exists after deletion: " << (exists ? "yes (error)" : "no (success)") << std::endl;
    assert(!exists && "Collection should not exist after deletion");
    
    std::cout << "\n=== Qdrant Connector Test Completed Successfully ===\n" << std::endl;
    return 0;
}