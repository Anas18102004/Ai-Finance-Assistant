import asyncio
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from services.embeddings import embedding_service
from nodes.query_parser import QueryIntent

logger = logging.getLogger(__name__)

class TransactionRetriever:
    """Retrieve transactions from Chroma vector database with filtering."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "financial_transactions"
        self.query_cache = {}  # Simple in-memory cache for repeated queries
        
    def initialize_client(self):
        """Initialize Chroma client."""
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Connected to collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to connect to collection: {e}")
                raise
    
    async def retrieve(self, query_intent: QueryIntent, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve transactions based on query intent and filters."""
        if not self.client:
            self.initialize_client()
        
        # Check cache first
        cache_key = f"{query_text}_{hash(str(query_intent.dict()))}"
        if cache_key in self.query_cache:
            logger.info("Retrieved from cache")
            return self.query_cache[cache_key]
        
        # Generate query embedding
        query_embedding = await embedding_service.embed_query(query_text)
        
        # Build where clause for filtering
        where_clause = self._build_where_clause(query_intent)
        
        try:
            # Perform vector search with filters
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(query_intent.top_k * 2, 100),  # Get more results for post-filtering
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process and format results
            transactions = self._format_results(results, query_intent)
            
            # Cache the results
            if len(self.query_cache) > 100:  # Simple cache eviction
                self.query_cache.clear()
            self.query_cache[cache_key] = transactions
            
            logger.info(f"Retrieved {len(transactions)} transactions")
            return transactions
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _build_where_clause(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """Build Chroma where clause from query intent."""
        where_clause = {}
        
        # User filter (always present)
        if "userId" in query_intent.filters:
            where_clause["userId"] = query_intent.filters["userId"]
        
        # Date range filter
        if query_intent.time_range:
            if "start_date" in query_intent.time_range:
                where_clause["date"] = {"$gte": query_intent.time_range["start_date"]}
            if "end_date" in query_intent.time_range:
                if "date" in where_clause:
                    where_clause["date"]["$lte"] = query_intent.time_range["end_date"]
                else:
                    where_clause["date"] = {"$lte": query_intent.time_range["end_date"]}
        
        # Category filter
        if query_intent.categories:
            if len(query_intent.categories) == 1:
                where_clause["category"] = query_intent.categories[0]
            else:
                where_clause["category"] = {"$in": query_intent.categories}
        
        # Amount range filter
        if query_intent.amount_range:
            if "min_amount" in query_intent.amount_range:
                where_clause["amount"] = {"$gte": query_intent.amount_range["min_amount"]}
            if "max_amount" in query_intent.amount_range:
                if "amount" in where_clause:
                    where_clause["amount"]["$lte"] = query_intent.amount_range["max_amount"]
                else:
                    where_clause["amount"] = {"$lte": query_intent.amount_range["max_amount"]}
        
        return where_clause
    
    def _format_results(self, results: Dict[str, Any], query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """Format Chroma results into transaction objects."""
        if not results["ids"] or not results["ids"][0]:
            return []
        
        transactions = []
        
        for i, (doc_id, metadata, document, distance) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        )):
            transaction = {
                "id": doc_id,
                "userId": metadata["userId"],
                "date": metadata["date"],
                "description": metadata["description"],
                "amount": metadata["amount"],
                "type": metadata["type"],
                "category": metadata["category"],
                "balance": metadata["balance"],
                "similarity_score": 1 - distance,  # Convert distance to similarity
                "document_text": document
            }
            transactions.append(transaction)
        
        # Apply additional filtering and sorting based on intent
        transactions = self._post_process_results(transactions, query_intent)
        
        return transactions[:query_intent.top_k]
    
    def _post_process_results(self, transactions: List[Dict[str, Any]], query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """Post-process results based on query intent."""
        
        if query_intent.intent == "top_expenses":
            # Sort by amount (descending) for expenses only
            expense_transactions = [t for t in transactions if t["type"] == "debit"]
            return sorted(expense_transactions, key=lambda x: x["amount"], reverse=True)
        
        elif query_intent.intent == "sum_spent":
            # For sum queries, we might want to return all matching transactions
            return transactions
        
        elif query_intent.intent == "compare":
            # Sort by amount for comparison
            return sorted(transactions, key=lambda x: x["amount"], reverse=True)
        
        else:
            # For general and filter queries, sort by similarity score
            return sorted(transactions, key=lambda x: x["similarity_score"], reverse=True)
    
    async def get_aggregated_data(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """Get aggregated data for sum and comparison queries."""
        if not self.client:
            self.initialize_client()
        
        # Build where clause
        where_clause = self._build_where_clause(query_intent)
        
        try:
            # Get all matching transactions (up to a reasonable limit)
            results = self.collection.get(
                where=where_clause,
                include=["metadatas"],
                limit=10000  # Reasonable limit for aggregation
            )
            
            if not results["metadatas"]:
                return {"total_amount": 0, "transaction_count": 0, "categories": {}}
            
            # Calculate aggregations
            total_amount = 0
            transaction_count = len(results["metadatas"])
            categories = {}
            
            for metadata in results["metadatas"]:
                amount = metadata["amount"]
                category = metadata["category"]
                transaction_type = metadata["type"]
                
                # For expenses, add to total
                if transaction_type == "debit":
                    total_amount += amount
                
                # Count by category
                if category not in categories:
                    categories[category] = {"count": 0, "total": 0}
                
                categories[category]["count"] += 1
                if transaction_type == "debit":
                    categories[category]["total"] += amount
            
            # Sort categories by total amount
            sorted_categories = dict(sorted(
                categories.items(),
                key=lambda x: x[1]["total"],
                reverse=True
            ))
            
            return {
                "total_amount": total_amount,
                "transaction_count": transaction_count,
                "categories": sorted_categories,
                "filters_applied": where_clause
            }
            
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            return {"total_amount": 0, "transaction_count": 0, "categories": {}}
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")

# Global retriever instance
retriever = TransactionRetriever()
