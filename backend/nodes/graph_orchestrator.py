import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from nodes.query_parser import query_parser, QueryIntent
from nodes.retriever import retriever
from nodes.summarizer import summarizer

logger = logging.getLogger(__name__)

class GraphState:
    """State object that flows through the graph nodes."""
    
    def __init__(self, user_id: str, query: str, **kwargs):
        self.user_id = user_id
        self.query = query
        self.query_intent: Optional[QueryIntent] = None
        self.retrieved_transactions: List[Dict[str, Any]] = []
        self.aggregated_data: Optional[Dict[str, Any]] = None
        self.summary: str = ""
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = kwargs
        self.start_time = time.time()
        
    def add_timing(self, step: str):
        """Add timing information."""
        if "timings" not in self.metadata:
            self.metadata["timings"] = {}
        self.metadata["timings"][step] = time.time() - self.start_time

class FinancialAssistantGraph:
    """LangGraph-style orchestrator for the financial assistant."""
    
    def __init__(self):
        self.nodes = {
            "parse_query": self.parse_query_node,
            "retrieve": self.retrieve_node,
            "summarize": self.summarize_node
        }
        
    async def run(
        self, 
        user_id: str, 
        query: str, 
        summarize: bool = True,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete graph workflow."""
        
        # Initialize state
        state = GraphState(
            user_id=user_id,
            query=query,
            summarize=summarize,
            top_k=top_k,
            filters=filters or {},
            gemini_api_key=gemini_api_key
        )
        
        try:
            # Execute nodes in sequence
            state = await self.parse_query_node(state)
            if state.error:
                return self._build_error_response(state)
            
            state = await self.retrieve_node(state)
            if state.error:
                return self._build_error_response(state)
            
            if summarize and gemini_api_key:
                state = await self.summarize_node(state)
            
            # Build final response
            return self._build_success_response(state)
            
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            state.error = str(e)
            return self._build_error_response(state)
    
    async def parse_query_node(self, state: GraphState) -> GraphState:
        """Parse the user query and extract intent."""
        try:
            logger.info(f"Parsing query: {state.query}")
            
            # Parse the query
            query_intent = query_parser.parse_query(state.query, state.user_id)
            
            # Override top_k if provided in metadata
            if "top_k" in state.metadata:
                query_intent.top_k = state.metadata["top_k"]
            
            # Apply additional filters if provided
            if "filters" in state.metadata and state.metadata["filters"]:
                query_intent.filters.update(state.metadata["filters"])
            
            state.query_intent = query_intent
            state.add_timing("parse_query")
            
            logger.info(f"Parsed intent: {query_intent.intent}, filters: {query_intent.filters}")
            
        except Exception as e:
            logger.error(f"Error in parse_query_node: {e}")
            state.error = f"Query parsing failed: {str(e)}"
        
        return state
    
    async def retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve relevant transactions."""
        try:
            if not state.query_intent:
                state.error = "No query intent available"
                return state
            
            logger.info(f"Retrieving transactions for intent: {state.query_intent.intent}")
            
            # For sum_spent queries, get aggregated data
            if state.query_intent.intent == "sum_spent":
                state.aggregated_data = await retriever.get_aggregated_data(state.query_intent)
                state.add_timing("retrieve_aggregated")
                
                # Also get some sample transactions for context
                state.retrieved_transactions = await retriever.retrieve(
                    state.query_intent, 
                    state.query
                )
            else:
                # Regular retrieval
                state.retrieved_transactions = await retriever.retrieve(
                    state.query_intent, 
                    state.query
                )
            
            state.add_timing("retrieve")
            
            logger.info(f"Retrieved {len(state.retrieved_transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error in retrieve_node: {e}")
            state.error = f"Retrieval failed: {str(e)}"
        
        return state
    
    async def summarize_node(self, state: GraphState) -> GraphState:
        """Generate summary using Gemini."""
        try:
            if not state.metadata.get("gemini_api_key"):
                state.error = "Gemini API key required for summarization"
                return state
            
            # Initialize summarizer if needed
            if not summarizer.model:
                summarizer.initialize(state.metadata["gemini_api_key"])
            
            logger.info("Generating summary with Gemini")
            
            # Generate summary
            state.summary = await summarizer.summarize_transactions(
                state.retrieved_transactions,
                state.query_intent,
                state.aggregated_data,
                state.query
            )
            
            state.add_timing("summarize")
            
            logger.info("Summary generated successfully")
            
        except Exception as e:
            logger.error(f"Error in summarize_node: {e}")
            state.error = f"Summarization failed: {str(e)}"
        
        return state
    
    def _build_success_response(self, state: GraphState) -> Dict[str, Any]:
        """Build successful response."""
        response = {
            "status": "success",
            "query": state.query,
            "user_id": state.user_id,
            "intent": state.query_intent.intent if state.query_intent else "unknown",
            "retrieved": state.retrieved_transactions,
            "summary": state.summary,
            "metadata": {
                "total_retrieved": len(state.retrieved_transactions),
                "query_intent": state.query_intent.dict() if state.query_intent else None,
                "timings": state.metadata.get("timings", {}),
                "total_time_ms": (time.time() - state.start_time) * 1000
            }
        }
        
        # Add aggregated data if available
        if state.aggregated_data:
            response["aggregated_data"] = state.aggregated_data
        
        return response
    
    def _build_error_response(self, state: GraphState) -> Dict[str, Any]:
        """Build error response."""
        return {
            "status": "error",
            "query": state.query,
            "user_id": state.user_id,
            "error": state.error,
            "metadata": {
                "timings": state.metadata.get("timings", {}),
                "total_time_ms": (time.time() - state.start_time) * 1000
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        try:
            # Check retriever
            retriever.initialize_client()
            stats = retriever.collection.count() if retriever.collection else 0
            health["components"]["retriever"] = {
                "status": "healthy",
                "documents_indexed": stats
            }
        except Exception as e:
            health["components"]["retriever"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        try:
            # Check embedding service
            await embedding_service.initialize()
            health["components"]["embeddings"] = {
                "status": "healthy",
                "model": embedding_service.model_name
            }
        except Exception as e:
            health["components"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        # Check summarizer (just check if it can be initialized)
        health["components"]["summarizer"] = {
            "status": "ready",
            "model": summarizer.model_name,
            "note": "Requires API key for operation"
        }
        
        return health

# Global graph orchestrator instance
graph = FinancialAssistantGraph()

# Import embedding service here to avoid circular imports
from services.embeddings import embedding_service
