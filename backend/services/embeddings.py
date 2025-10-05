import asyncio
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """HuggingFace embeddings service using configurable model."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        
    async def initialize(self):
        """Initialize the embedding model asynchronously."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            logger.info("Embedding model loaded successfully")
    
    def format_transaction_text(self, transaction: Dict[str, Any]) -> str:
        """Format transaction as text for embedding."""
        return (
            f"{transaction['type']} of ₹{transaction['amount']} on {transaction['date']} "
            f"for {transaction['description']} under {transaction['category']}"
        )
    
    async def embed_transactions(self, transactions: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for a list of transactions in batches."""
        if not self.model:
            await self.initialize()
        
        # Format transactions as text
        texts = [self.format_transaction_text(txn) for txn in transactions]
        
        # Process in batches for better performance
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(batch_texts, convert_to_numpy=True)
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
            
            if i % (self.batch_size * 5) == 0:  # Log progress every 5 batches
                logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} transactions")
        
        return all_embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        if not self.model:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode([query], convert_to_numpy=True)
        )
        
        return embedding[0].tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not self.model:
            await self.initialize()
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(batch_texts, convert_to_numpy=True)
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings

# Global embedding service instance
embedding_service = EmbeddingService()
