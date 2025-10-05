"""
RAG Agent - Handles knowledge queries using ChromaDB transaction data with sentence-transformers.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from config import config
from nodes.retriever import retriever
from services.embeddings import embedding_service

logger = logging.getLogger(__name__)

class RAGAgent:
    """Agent for handling knowledge queries using ChromaDB with sentence-transformers/all-MiniLM-L6-v2."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.model_name = config.GEMINI_MODEL
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        if api_key:
            self.initialize(api_key)
    
    def initialize(self, api_key: str):
        """Initialize the Gemini model."""
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.api_key = api_key
        logger.info("RAG Agent initialized")
    
    async def answer_knowledge_query(self, query: str, topic: str = None, user_id: str = None) -> Dict[str, Any]:
        """Answer knowledge-based queries using RAG (Retrieval-Augmented Generation) approach."""
        try:
            logger.info(f"RAG Agent: Starting RAG process for query: '{query}'")
            
            # STEP 1: RETRIEVAL - Get top 5-8 most relevant documents
            logger.info("RAG Agent: Step 1 - Retrieving relevant documents from ChromaDB")
            relevant_docs = await self._retrieve_relevant_data(query, user_id)
            
            if not relevant_docs:
                logger.warning("RAG Agent: No relevant documents found in ChromaDB")
                return {
                    "success": True,
                    "source": "chromadb_no_results",
                    "retrieved_docs": 0
                }
            
            logger.info(f"RAG Agent: Step 1 Complete - Retrieved {len(relevant_docs)} relevant documents")
            
            if not self.model:
                # Return basic info if no model available for generation
                return {
                    "success": True,
                    "answer": f"Found {len(relevant_docs)} relevant transactions. Enable Gemini for detailed analysis.",
                    "source": "chromadb_basic",
                    "retrieved_docs": len(relevant_docs)
                }
            
            # STEP 2: GENERATION - Use Gemini to analyze retrieved documents
            logger.info("RAG Agent: Step 2 - Generating response using retrieved documents")
            prompt = self._build_knowledge_prompt(query, relevant_docs, topic)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            answer = response.text.strip()
            logger.info("RAG Agent: Step 2 Complete - Generated knowledge-rich response")
            
            return {
                "success": True,
                "answer": answer,
                "source": "rag_chromadb_gemini",
                "topic": topic,
                "retrieved_docs": len(relevant_docs),
                "rag_process": {
                    "retrieval": "completed",
                    "generation": "completed",
                    "documents_used": len(relevant_docs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG process: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error during the RAG process while searching your transaction data."
            }
    
    async def _retrieve_relevant_data(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve top 5-8 most relevant documents using sentence-transformers/all-MiniLM-L6-v2."""
        try:
            logger.info(f"RAG Agent: Starting vector similarity search with {self.embedding_model}")
            logger.info(f"RAG Agent: Query: '{query}' for user: {user_id}")
            
            # Step 1: Generate query embedding using sentence-transformers
            logger.info("RAG Agent: Generating query embedding using sentence-transformers/all-MiniLM-L6-v2")
            query_embedding = await self.embedding_service.embed_query(query)
            
            if not query_embedding:
                logger.error("RAG Agent: Failed to generate query embedding")
                return []
            
            logger.info(f"RAG Agent: Generated query embedding (dim: {len(query_embedding)})")
            
            # Step 2: Perform vector similarity search in ChromaDB
            logger.info("RAG Agent: Performing vector similarity search in ChromaDB")
            
            # Get the ChromaDB collection
            if not self.retriever.client:
                self.retriever.initialize_client()
            
            collection = self.retriever.collection
            
            # Build where clause for user filtering
            where_clause = {}
            if user_id:
                where_clause = {"userId": {"$eq": user_id}}
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=8,  # Top 5-8 most relevant documents
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Step 3: Process and format results
            relevant_docs = []
            if results["ids"] and results["ids"][0]:
                for i, (doc_id, metadata, document, distance) in enumerate(zip(
                    results["ids"][0],
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0]
                )):
                    # Convert metadata to transaction format
                    transaction = {
                        "id": doc_id,
                        "description": metadata.get("description", "Unknown"),
                        "amount": float(metadata.get("amount", 0)),
                        "date": metadata.get("date", ""),
                        "category": metadata.get("category", "Unknown"),
                        "type": metadata.get("type", "Unknown"),
                        "userId": metadata.get("userId", ""),
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "embedding_distance": distance
                    }
                    relevant_docs.append(transaction)
            
            logger.info(f"RAG Agent: Retrieved {len(relevant_docs)} documents using sentence-transformers")
            
            # Log top 3 retrieved documents with similarity scores
            for i, doc in enumerate(relevant_docs[:3], 1):
                desc = doc.get('description', 'Unknown')[:30]
                amount = doc.get('amount', 0)
                category = doc.get('category', 'Unknown')
                similarity = doc.get('similarity_score', 0)
                logger.info(f"  Doc {i}: ₹{amount} - {desc}... ({category}) [similarity: {similarity:.3f}]")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval with sentence-transformers: {e}")
            return []
    
    def _build_knowledge_prompt(self, query: str, transactions: List[Dict[str, Any]], topic: str = None) -> str:
        """Build prompt for knowledge query using transaction data."""
        
        # Prepare transaction context
        transaction_context = ""
        if transactions:
            transaction_context = f"\nUSER'S TRANSACTION DATA ({len(transactions)} transactions):\n"
            
            # Show sample transactions with similarity scores
            for i, txn in enumerate(transactions[:5]):  # Show first 5 transactions
                date = txn.get('date', 'Unknown')
                desc = txn.get('description', 'Unknown')
                amount = txn.get('amount', 0)
                category = txn.get('category', 'Unknown')
                similarity = txn.get('similarity_score', 0)
                transaction_context += f"- {date}: ₹{amount} - {desc} ({category}) [similarity: {similarity:.3f}]\n"
            
            if len(transactions) > 5:
                transaction_context += f"... and {len(transactions) - 5} more transactions\n"
            
            # Add category summary
            categories = {}
            total_amount = 0
            for txn in transactions:
                cat = txn.get('category', 'Unknown')
                amount = txn.get('amount', 0)
                categories[cat] = categories.get(cat, 0) + amount
                total_amount += amount
            
            transaction_context += f"\nCATEGORY SUMMARY:\n"
            for cat, amount in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (amount / total_amount * 100) if total_amount > 0 else 0
                transaction_context += f"- {cat}: ₹{amount:,.0f} ({percentage:.1f}%)\n"
        
        return f"""You are a RAG (Retrieval-Augmented Generation) Agent. You have retrieved the most relevant documents from the user's transaction database using sentence-transformers/all-MiniLM-L6-v2 embeddings for vector similarity search.

USER QUERY: "{query}"
TOPIC: "{topic}"

RETRIEVED DOCUMENTS (Top {len(transactions)} most relevant using sentence-transformers):
{transaction_context}

RAG PROCESS:
1. ✅ RETRIEVAL COMPLETED: Used sentence-transformers/all-MiniLM-L6-v2 to find {len(transactions)} most semantically relevant documents
2. 🔄 GENERATION: Now analyze these retrieved documents to answer the user's query

EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2
SIMILARITY SEARCH: Vector embeddings matched query semantics to transaction descriptions

INSTRUCTIONS:
1. Base your answer ONLY on the retrieved documents above
2. Synthesize insights and patterns from the retrieved transaction data
3. If the query asks about something not in the retrieved documents, say so clearly
4. Provide specific examples from the retrieved documents with similarity scores
5. Generate a comprehensive answer using the RAG approach

RESPONSE FORMAT:
Provide a detailed, knowledge-rich response that combines information from the retrieved documents to answer the user's query. Use specific amounts, dates, and categories from the retrieved data.

Generate your RAG-based response:"""

# Global instance
rag_agent = RAGAgent()
