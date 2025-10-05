import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our modules
from config import config, print_startup_info, get_gemini_api_key
from agents.orchestrator import orchestrator
from index_build.build_index import index_builder
from generate_data import generate_synthetic_transactions, save_transactions_to_file

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    user_id: str
    query: str
    summarize: bool = True
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    status: str
    query: str
    user_id: str
    intent: Optional[str] = None
    retrieved: List[Dict[str, Any]] = []
    summary: Optional[str] = None
    aggregated_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class GenerateDataResponse(BaseModel):
    status: str
    message: str
    transaction_count: int

class IndexBuildResponse(BaseModel):
    status: str
    message: str
    documents_indexed: int

# Initialize FastAPI app
app = FastAPI(
    title="AI Financial Assistant",
    description="High-performance AI Financial Assistant with HuggingFace embeddings, Chroma vector DB, and Gemini summarizer",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
app_state = {
    "initialized": False,
    "startup_time": None
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print_startup_info()
    
    try:
        # Initialize embedding service
        from services.embeddings import embedding_service
        await embedding_service.initialize()
        
        # Initialize retriever
        from nodes.retriever import retriever
        retriever.initialize_client()
        
        # Initialize orchestrator with Gemini API key
        gemini_api_key = get_gemini_api_key()
        if gemini_api_key:
            orchestrator.initialize(gemini_api_key)
            logger.info("Orchestrator initialized with Gemini API key")
        else:
            logger.warning("No Gemini API key found - some features may be limited")
        
        app_state["initialized"] = True
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state["initialized"] = False

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Financial Assistant API",
        "version": "1.0.0",
        "status": "running" if app_state["initialized"] else "initializing",
        "endpoints": {
            "query": "/query",
            "generate": "/generate",
            "index_build": "/index/build",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy" if app_state["initialized"] else "initializing",
            "service": "AI Financial Assistant",
            "architecture": "Multi-Agent (Intent → Data/RAG → Synthesizer)",
            "agents": {
                "intent_agent": "active",
                "data_agent": "active", 
                "rag_agent": "active",
                "synthesizer_agent": "active"
            },
            "memory_stats": orchestrator.get_memory_stats()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def query_transactions(request: QueryRequest):
    """Query transactions using natural language with new agent architecture."""
    if not app_state["initialized"]:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Processing query: {request.query} for user: {request.user_id}")
        
        # Process query using the orchestrator
        result = await orchestrator.process_query(
            user_id=request.user_id,
            query=request.query
        )
        
        # Convert orchestrator result to QueryResponse format
        query_response = {
            "status": result.get("status", "success"),
            "query": result.get("query", request.query),
            "user_id": result.get("user_id", request.user_id),
            "intent": result.get("intent", "unknown"),
            "retrieved": [],  # Not used in new architecture
            "summary": result.get("response", ""),
            "metadata": {
                "total_retrieved": 0,
                "query_intent": {"intent": result.get("intent", "unknown")},
                "timings": {},
                "total_time_ms": result.get("response_time_ms", 0),
                "agent_path": result.get("agent_path", "unknown"),
                "conversational": result.get("metadata", {}).get("conversational", False),
                "rag_used": result.get("metadata", {}).get("rag_used", False),
                "data_retrieved": result.get("metadata", {}).get("data_retrieved", False)
            }
        }
        
        # Add data summary if available
        if "data_summary" in result:
            query_response["metadata"]["data_summary"] = result["data_summary"]
        
        if "knowledge_summary" in result:
            query_response["metadata"]["knowledge_summary"] = result["knowledge_summary"]
        
        return QueryResponse(**query_response)
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateDataResponse)
async def generate_data(
    background_tasks: BackgroundTasks,
    gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key")
):
    """Generate new synthetic transaction data using Faker + optional Gemini AI."""
    try:
        logger.info("Generating synthetic transaction data with Faker + Gemini...")
        
        # Use header API key or fall back to environment
        api_key = gemini_api_key or get_gemini_api_key()
        
        # Generate transactions with optional Gemini API key
        transactions = generate_synthetic_transactions(api_key)
        
        # Save to file
        save_transactions_to_file(transactions, config.DATA_FILE_PATH)
        
        logger.info(f"Generated {len(transactions)} transactions")
        
        ai_used = "Faker + Gemini AI" if api_key else "Faker + realistic companies"
        
        return GenerateDataResponse(
            status="success",
            message=f"Successfully generated {len(transactions)} transactions using {ai_used}",
            transaction_count=len(transactions)
        )
        
    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/build", response_model=IndexBuildResponse)
async def build_index():
    """Build or rebuild the Chroma vector index."""
    try:
        logger.info("Building vector index...")
        
        # Rebuild the index
        await index_builder.rebuild_index()
        
        # Get stats
        stats = index_builder.get_collection_stats()
        document_count = stats.get("total_documents", 0)
        
        logger.info(f"Index built with {document_count} documents")
        
        return IndexBuildResponse(
            status="success",
            message=f"Successfully built index with {document_count} documents",
            documents_indexed=document_count
        )
        
    except Exception as e:
        logger.error(f"Index building error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        from nodes.retriever import retriever
        
        # Initialize if needed
        if not retriever.client:
            retriever.initialize_client()
        
        stats = index_builder.get_collection_stats()
        
        return {
            "status": "success",
            "database_stats": stats,
            "cache_size": len(retriever.query_cache),
            "initialized": app_state["initialized"]
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache."""
    try:
        from nodes.retriever import retriever
        retriever.clear_cache()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example queries endpoint for testing
@app.get("/examples")
async def get_example_queries():
    """Get example queries for testing."""
    return {
        "example_queries": [
            {
                "query": "Show me my top 5 expenses this month",
                "description": "Get top expenses with time filter"
            },
            {
                "query": "How much did I spend on food in September?",
                "description": "Sum spending by category and time"
            },
            {
                "query": "Find transactions above ₹5000",
                "description": "Filter by amount range"
            },
            {
                "query": "Show my entertainment expenses",
                "description": "Filter by category"
            },
            {
                "query": "What are my recent shopping transactions?",
                "description": "General search with category"
            }
        ],
        "users": [
            "user_001",
            "user_002", 
            "user_003"
        ],
        "note": "Make sure to generate data and build index first"
    }

@app.get("/memory/stats")
async def get_memory_stats():
    """Get conversation memory statistics."""
    try:
        stats = orchestrator.get_memory_stats()
        return {
            "status": "success",
            "memory_stats": stats,
            "architecture": "Multi-Agent System"
        }
    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server with config settings
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )
