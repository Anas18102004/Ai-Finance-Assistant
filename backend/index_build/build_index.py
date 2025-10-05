import json
import asyncio
import logging
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from services.embeddings import embedding_service
from config import config

logger = logging.getLogger(__name__)

class ChromaIndexBuilder:
    """Build and manage Chroma vector database index."""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or config.CHROMA_DB_PATH
        self.client = None
        self.collection = None
        self.collection_name = "financial_transactions"
        
    def initialize_client(self):
        """Initialize Chroma client with persistence."""
        # Close any existing client first to avoid conflicts
        if self.client:
            try:
                self.client._server.close()
            except:
                pass
            self.client = None
            
        # Create a new client instance
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            logger.warning(f"Error creating Chroma client: {e}")
            # Try again with a clean instance
            import shutil
            try:
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except Exception as e2:
                logger.error(f"Failed to create Chroma client after cleanup: {e2}")
                raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if self.client:
            try:
                self.client._server.close()
            except:
                pass
    
    def load_transactions(self, file_path: str = None) -> List[Dict[str, Any]]:
        """Load transactions from JSON file."""
        file_path = file_path or config.DATA_FILE_PATH
        with open(file_path, 'r', encoding='utf-8') as f:
            transactions = json.load(f)
        
        logger.info(f"Loaded {len(transactions)} transactions from {file_path}")
        return transactions
    
    async def build_index(self, transactions: List[Dict[str, Any]] = None):
        """Build the complete Chroma index with embeddings."""
        if not self.client:
            self.initialize_client()
        
        if transactions is None:
            transactions = self.load_transactions()
        
        if not transactions:
            logger.warning("No transactions to index")
            return
        
        logger.info(f"Building index for {len(transactions)} transactions...")
        
        # Generate embeddings for all transactions
        embeddings = await embedding_service.embed_transactions(transactions)
        
        # Prepare data for Chroma
        ids = [txn["id"] for txn in transactions]
        documents = [embedding_service.format_transaction_text(txn) for txn in transactions]
        metadatas = []
        
        for txn in transactions:
            metadata = {
                "userId": txn["userId"],
                "date": txn["date"],
                "category": txn["category"],
                "amount": float(txn["amount"]),
                "type": txn["type"],
                "description": txn["description"],
                "balance": float(txn["balance"])
            }
            metadatas.append(metadata)
        
        # Clear existing data and add new data
        try:
            # Delete all documents in the collection by using where clause that matches everything
            self.collection.delete(where={"userId": {"$exists": True}})
            logger.info("Cleared existing collection data")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")
        
        # Add documents in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {end_idx}/{len(transactions)} transactions")
        
        logger.info(f"Successfully built index with {len(transactions)} transactions")
        
        # Verify the index
        count = self.collection.count()
        logger.info(f"Index verification: {count} documents in collection")
    
    async def rebuild_index(self):
        """Rebuild the entire index from scratch."""
        logger.info("Starting index rebuild...")
        
        # Load fresh data
        transactions = self.load_transactions()
        
        # Build index
        await self.build_index(transactions)
        
        logger.info("Index rebuild completed")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        
        # Get sample of metadata to understand data distribution
        sample = self.collection.get(limit=10, include=["metadatas"])
        
        users = set()
        categories = set()
        
        for metadata in sample["metadatas"]:
            users.add(metadata["userId"])
            categories.add(metadata["category"])
        
        return {
            "total_documents": count,
            "sample_users": list(users),
            "sample_categories": list(categories),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

# Global index builder instance
index_builder = ChromaIndexBuilder()

async def main():
    """Main function to build the index."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize embedding service
    await embedding_service.initialize()
    
    # Build index
    await index_builder.rebuild_index()
    
    # Print stats
    stats = index_builder.get_collection_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
