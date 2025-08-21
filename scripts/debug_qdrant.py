#!/usr/bin/env python3
"""
Debug script to inspect Qdrant collection and test embeddings.
"""

import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

# Qdrant configuration
QDRANT_URL = "https://14078592-0e60-4aa2-8e5b-eba64bdb8eb4.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOlt7ImNvbGxlY3Rpb24iOiJzZXAtdHJhZGVyIiwiYWNjZXNzIjoicncifV19.noTToqFWXRNZ4YyTTw4dnJe5lWq7e2PfUTNvrQqncmw"
COLLECTION_NAME = "sep-trader"

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"Collection info: {collection_info}")
        print(f"Vector size: {list(collection_info.config.params.vectors.values())[0].size}")
        print(f"Distance: {list(collection_info.config.params.vectors.values())[0].distance}")
        
        # Count total points
        count_result = client.count(collection_name=COLLECTION_NAME)
        print(f"Total points in collection: {count_result.count}")
        
        # Get some sample points
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=True  # Get vectors too for debugging
        )
        
        print(f"\nSample points ({len(scroll_result[0])}):")
        for point in scroll_result[0]:
            print(f"ID: {point.id}")
            print(f"Payload keys: {list(point.payload.keys())}")
            if 'file_path' in point.payload:
                print(f"File: {point.payload['file_path']}")
            if 'content' in point.payload:
                print(f"Content preview: {point.payload['content'][:100]}...")
            if point.vector:
                print(f"Vector dimensions: {len(point.vector)}")
                print(f"Vector sample: {point.vector[:3]}...")  # First 3 elements
            print("-" * 40)
            
        # Test a simple search with basic embedding
        print("\nTesting search with simple query...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Test multiple queries of varying complexity
        test_queries = [
            "CUDA quantum trading system",
            "C++ source code",
            "trading algorithm",
            "quantum processing",
            "build system",
            "cmake"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            query_embedding = model.encode(query).tolist()
            
            # Pad to 3072 like the upload script does
            padded_embedding = []
            for i in range(3072):
                padded_embedding.append(query_embedding[i % len(query_embedding)])
            
            # Search
            search_result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=padded_embedding,
                limit=3,
                with_payload=True,
                score_threshold=0.1  # Lower threshold to see if any matches exist
            )
            
            print(f"Results: {len(search_result)} matches")
            if search_result:
                for hit in search_result:
                    print(f"Score: {hit.score:.4f}")
                    print(f"File: {hit.payload.get('file_path', 'Unknown')}")
                    print(f"Category: {hit.payload.get('category', 'Unknown')}")
                    print(f"Content preview: {hit.payload.get('content', '')[:100]}...")
            else:
                print("No results found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()