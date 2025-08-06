from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import warnings
import numpy as np

# Suppress insecure connection warnings
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

load_dotenv()

def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all metadata values to Qdrant-compatible formats"""
    cleaned = {}
    for key, value in metadata.items():
        # Handle None values
        if value is None:
            cleaned[key] = None
            continue
            
        # Convert numpy numbers to Python native types
        if isinstance(value, np.number):
            cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
            
        # Convert numeric strings to actual numbers
        elif isinstance(value, str):
            if value.replace('.', '', 1).isdigit():
                cleaned[key] = float(value) if '.' in value else int(value)
            else:
                cleaned[key] = value
                
        # Keep other basic types as-is
        elif isinstance(value, (int, float, str, bool)):
            cleaned[key] = value
            
        # Convert everything else to string
        else:
            cleaned[key] = str(value)
    
    return cleaned

class QdrantVectorStore:
    def __init__(self, collection_name: str = "ecommerce_faq"):
        """
        Initialize Qdrant vector store client.
        
        Args:
            collection_name: Name of the collection to use/create
        """
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,  # Disable GRPC to avoid number conversion issues
            timeout=10.0
        )
        self.collection_name = collection_name
    
    def collection_exists(self) -> bool:
        """Check if collection already exists"""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except:
            return False
    
    def create_collection(self, vector_size: int = 1536) -> None:
        """
        Create or recreate a collection with optimized settings.
        
        Args:
            vector_size: Dimension of the vectors (1536 for text-embedding-ada-002)
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=20000
            )
        )
    
    def upload_data(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Upload documents with embeddings to Qdrant.
        
        Args:
            texts: List of document texts
            embeddings: List of corresponding embeddings
            metadatas: List of metadata dictionaries
            batch_size: Number of points to upload at once
        """
        # Clean all metadata first
        cleaned_metadatas = [clean_metadata(md) for md in metadatas]
        
        # Prepare payloads
        payloads = [
            {
                "text": text,
                **metadata
            }
            for text, metadata in zip(texts, cleaned_metadatas)
        ]
        
        # Upload in batches
        for i in range(0, len(embeddings), batch_size):
            batch_points = [
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload=payload
                )
                for idx, (embedding, payload) in enumerate(
                    zip(embeddings[i:i+batch_size], payloads[i:i+batch_size]),
                    start=i
                )
            ]
            
            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            except Exception as e:
                print(f"Error uploading batch {i//batch_size}: {str(e)}")
                print("Problematic batch sample:", batch_points[0] if batch_points else "Empty")
                raise
    
    def hybrid_search(
        self, 
        query: str, 
        query_embedding: List[float], 
        category_filter: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword) on Qdrant.
        """
        # Build filter if category is specified
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter)
                )
            ]
        ) if category_filter else None
        
        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_text=query,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            filter=search_filter
        )
        
        return [
            {
                "text": result.payload["text"],
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score,
                "id": result.id
            }
            for result in results
        ]
    
    def semantic_search(
        self, 
        query_embedding: List[float],
        query: Optional[str] = None,
        threshold: float = 0.7,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with similarity threshold.
        
        Args:
            query_embedding: Vector embedding of the query
            query: Original search query (optional, for debugging)
            threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of filtered and formatted results
        """
        # Perform initial vector search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit*2,  # Get more results than needed for filtering
            with_payload=True,
            with_vectors=False
        )
        
        # Filter by similarity score and format results
        filtered_results = []
        for hit in results:
            if hit.score >= threshold:
                filtered_results.append({
                    "text": hit.payload["text"],
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                    "score": hit.score,
                    "id": hit.id
                })
                
                # Stop when we reach the limit
                if len(filtered_results) >= limit:
                    break
        
        return filtered_results

    def search(
        self,
        query_embedding: List[float],
        query: Optional[str] = None,
        mode: str = "semantic",  # "semantic" or "hybrid"
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface that routes to appropriate method.
        
        Args:
            query_embedding: Vector embedding (required for both modes)
            query: Text query (required for hybrid search)
            mode: Search mode ("semantic" or "hybrid")
            **kwargs: Additional arguments for the search methods
            
        Returns:
            List of search results
        """
        if mode == "semantic":
            return self.semantic_search(query_embedding=query_embedding, query=query, **kwargs)
        elif mode == "hybrid":
            if query is None:
                raise ValueError("Query is required for hybrid search")
            return self.hybrid_search(query=query, query_embedding=query_embedding, **kwargs)
        else:
            raise ValueError(f"Invalid search mode: {mode}. Use 'semantic' or 'hybrid'")