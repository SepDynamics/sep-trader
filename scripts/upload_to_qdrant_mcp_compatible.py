#!/usr/bin/env python3
"""
Upload SEP project files to Qdrant vector database using MCP-compatible embeddings.
This version uses standard 768-dimensional embeddings compatible with the MCP server.
"""

import os
import json
import asyncio
from pathlib import Path
import tempfile
from typing import List, Dict, Any
import hashlib
import mimetypes
import fnmatch

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import requests
except ImportError:
    print("Missing dependencies. Install with: pip install qdrant-client sentence-transformers requests")
    exit(1)

# Qdrant configuration from MCP settings
QDRANT_URL = "https://14078592-0e60-4aa2-8e5b-eba64bdb8eb4.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOlt7ImNvbGxlY3Rpb24iOiJzZXAtdHJhZGVyIiwiYWNjZXNzIjoicncifV19.noTToqFWXRNZ4YyTTw4dnJe5lWq7e2PfUTNvrQqncmw"
COLLECTION_NAME = "sep-trader-mcp"

class QdrantMCPUploader:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        # Use standard embedding model with 768 dimensions (MCP server compatible)
        print("Loading MCP-compatible embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # 768 dimensions
        self.embedding_cache = {}
        
    def get_embedding(self, text: str) -> List[float]:
        """Get standard 768-dimensional embeddings compatible with MCP server"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Generate standard 768-dimensional embedding (no padding)
        embedding = self.embedding_model.encode(text).tolist()
        
        self.embedding_cache[text] = embedding
        return embedding

    def should_include_file(self, filepath: Path, project_root: Path) -> bool:
        """Simple filtering: include src/, docs/, and important root files"""
        try:
            rel_path = filepath.relative_to(project_root)
        except ValueError:
            return False
            
        path_str = str(rel_path).replace('\\', '/')
        
        # Always include everything in src/ and docs/
        if path_str.startswith('src/') or path_str.startswith('docs/'):
            return True
            
        # Exclude these directories entirely
        exclude_dirs = [
            'qdrant_storage/', 'cache/', 'build/', 'output/', '__pycache__/',
            '.git/', 'node_modules/', '.vscode/', '.idea/', 'temp/', 'tmp/',
            'third_party/', 'extern/', 'vcpkg/', 'libs/', 'bin/', '.github/',
            'models/', 'assets/', 'public/', 'week_optimization/', 'training_results/',
            'validation/', 'docs_archive/', 'testing/', '_sep/', 'build_minimal/'
        ]
        
        for exclude_dir in exclude_dirs:
            if path_str.startswith(exclude_dir):
                return False
        
        # For root-level files, include important ones
        if '/' not in path_str:
            important_files = [
                'CMakeLists.txt', 'README.md', 'Dockerfile', 'build.sh', 'build.bat',
                'install.sh', 'install.bat', 'docker-compose.yml', 'LICENSE',
                'requirements.txt', 'package.json', 'nginx.conf', 'TODO.md',
                'AGENT.md', 'SYSTEM_STATUS_REPORT.md', 'SEP_Engine_White_Paper.md'
            ]
            
            file_extensions = ['.py', '.sh', '.bat', '.json', '.yml', '.yaml', '.md', '.txt']
            
            return (filepath.name in important_files or
                    any(filepath.name.endswith(ext) for ext in file_extensions))
        
        # Include config/ and scripts/ directories
        if path_str.startswith('config/') or path_str.startswith('scripts/'):
            return filepath.suffix in ['.py', '.sh', '.json', '.yml', '.yaml', '.env', '.cfg', '.ini']
            
        return False

    def chunk_text(self, text: str, max_chunk_size: int = 8000) -> List[str]:
        """Split text into manageable chunks"""
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += '\n' + line if current_chunk else line
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def process_file(self, filepath: Path, project_root: Path) -> List[Dict[str, Any]]:
        """Process a single file and return point data"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []

        if not content.strip():
            return []

        relative_path = str(filepath.relative_to(project_root))
        file_type = filepath.suffix or 'text'
        
        # Determine category based on path
        category = "source"
        if "docs/" in relative_path:
            category = "documentation"
        elif "config/" in relative_path:
            category = "configuration"
        elif "tests/" in relative_path:
            category = "test"
        elif "scripts/" in relative_path:
            category = "script"
        elif ".md" in relative_path.lower():
            category = "documentation"

        chunks = self.chunk_text(content)
        points = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # Create point ID
            point_id = hashlib.md5(f"{relative_path}_{i}".encode()).hexdigest()
            
            # Get MCP-compatible embedding
            embedding = self.get_embedding(chunk)
            
            # Create metadata - store full content for better retrieval
            metadata = {
                "file_path": relative_path,
                "file_name": filepath.name,
                "file_type": file_type,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk,  # Store full content for better retrieval
                "size": len(chunk)
            }
            
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": metadata
            })
            
        return points

    async def upload_files(self, project_root: Path):
        """Upload all relevant files to Qdrant with MCP-compatible embeddings"""
        print(f"Scanning project at: {project_root}")
        
        # Create new collection with 768 dimensions for MCP compatibility
        try:
            # Delete old collection if it exists
            try:
                self.client.delete_collection(COLLECTION_NAME)
                print(f"Deleted existing collection: {COLLECTION_NAME}")
            except:
                pass
            
            print(f"Creating MCP-compatible collection with 768 dimensions: {COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Error with collection: {e}")
            return

        # Collect files to process
        files_to_process = []
        
        # Prioritize important directories
        priority_dirs = ['src/', 'docs/', 'config/', 'scripts/']
        
        for priority_dir in priority_dirs:
            priority_path = project_root / priority_dir
            if priority_path.exists():
                for filepath in priority_path.rglob("*"):
                    if filepath.is_file() and self.should_include_file(filepath, project_root):
                        files_to_process.append(filepath)

        # Add other files
        for filepath in project_root.rglob("*"):
            if (filepath.is_file() and
                self.should_include_file(filepath, project_root) and
                filepath not in files_to_process):
                files_to_process.append(filepath)

        print(f"Found {len(files_to_process)} files to process")
        
        # Process files in batches
        batch_size = 50
        total_points = 0
        
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            batch_points = []
            
            for filepath in batch_files:
                print(f"Processing: {filepath.relative_to(project_root)}")
                points = self.process_file(filepath, project_root)
                batch_points.extend(points)
            
            if batch_points:
                try:
                    # Convert to PointStruct for upload
                    qdrant_points = [
                        PointStruct(
                            id=point["id"],
                            vector=point["vector"],
                            payload=point["payload"]
                        )
                        for point in batch_points
                    ]
                    
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=qdrant_points
                    )
                    
                    total_points += len(batch_points)
                    print(f"Uploaded batch {i//batch_size + 1}, total points: {total_points}")
                    
                except Exception as e:
                    print(f"Error uploading batch: {e}")
        
        print(f"MCP-compatible upload complete! Total points uploaded: {total_points}")
        print(f"Collection name: {COLLECTION_NAME}")

def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    
    print("SEP Project MCP-Compatible Qdrant Uploader")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print("Using 768-dimensional embeddings for MCP server compatibility")
    print()
    
    uploader = QdrantMCPUploader()
    asyncio.run(uploader.upload_files(project_root))

if __name__ == "__main__":
    main()