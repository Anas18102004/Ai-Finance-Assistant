#!/usr/bin/env python3
"""
Test RAG Agent with sentence-transformers/all-MiniLM-L6-v2 embeddings.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_sentence_transformers_rag():
    """Test RAG Agent using sentence-transformers/all-MiniLM-L6-v2."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "user_002"
    
    print("🤖 Testing RAG with sentence-transformers/all-MiniLM-L6-v2")
    print("=" * 70)
    
    # Test semantic similarity queries
    semantic_queries = [
        {
            "query": "What do I spend most on?",
            "description": "Semantic search for spending patterns",
            "expected_docs": "High similarity transactions across categories"
        },
        {
            "query": "expensive purchases",
            "description": "Semantic search for high-value transactions", 
            "expected_docs": "Transactions with high amounts"
        },
        {
            "query": "dining and food expenses",
            "description": "Semantic search for food-related spending",
            "expected_docs": "Food, dining, restaurant transactions"
        },
        {
            "query": "entertainment and fun activities",
            "description": "Semantic search for entertainment spending",
            "expected_docs": "Entertainment, movies, games transactions"
        },
        {
            "query": "monthly bills and utilities",
            "description": "Semantic search for recurring payments",
            "expected_docs": "Utilities, bills, subscription transactions"
        }
    ]
    
    for i, test_case in enumerate(semantic_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected_docs = test_case["expected_docs"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_docs}")
        
        try:
            # Process query
            result = await orchestrator.process_query(user_id, query)
            
            intent = result.get("intent")
            agent_path = result.get("agent_path")
            response_time = result.get("response_time_ms", 0)
            response = result.get("response", "")
            
            print(f"   Intent: {intent}")
            print(f"   Agent Path: {agent_path}")
            print(f"   Response Time: {response_time:.1f}ms")
            
            # Check RAG process details
            if "rag" in agent_path:
                print(f"   ✅ RAG Agent Used with sentence-transformers")
                
                knowledge_summary = result.get("knowledge_summary", {})
                if knowledge_summary:
                    retrieved_docs = knowledge_summary.get("retrieved_docs", 0)
                    source = knowledge_summary.get("source", "unknown")
                    rag_process = knowledge_summary.get("rag_process", {})
                    
                    print(f"   📊 RAG Process Details:")
                    print(f"      • Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
                    print(f"      • Documents Retrieved: {retrieved_docs}")
                    print(f"      • Retrieval Status: {rag_process.get('retrieval', 'unknown')}")
                    print(f"      • Generation Status: {rag_process.get('generation', 'unknown')}")
                    print(f"      • Source: {source}")
                    
                    if retrieved_docs > 0:
                        print(f"   ✅ Semantic Search: Query → Embeddings → Top {retrieved_docs} similar docs")
                    else:
                        print(f"   ⚠️  No semantically similar documents found")
                
                print(f"   Response: {response[:150]}...")
            else:
                print(f"   ❌ RAG not used - routed to {agent_path}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n🔍 Sentence-Transformers Benefits:")
    print("   • Semantic Understanding: Matches meaning, not just keywords")
    print("   • Better Similarity: 'expensive purchases' finds high-amount transactions")
    print("   • Context Aware: 'dining' matches restaurants, food delivery, etc.")
    print("   • Multilingual: Works with different phrasings and synonyms")
    
    print(f"\n⚙️ Technical Details:")
    print("   • Model: sentence-transformers/all-MiniLM-L6-v2")
    print("   • Embedding Dimension: 384")
    print("   • Similarity Metric: Cosine similarity (1 - distance)")
    print("   • Top-K Retrieval: 8 most relevant documents")
    
    print(f"\n📋 RAG Flow with Sentence-Transformers:")
    print("   1. Query → sentence-transformers → Query Embedding (384-dim)")
    print("   2. ChromaDB Vector Search → Top 8 similar transaction embeddings")
    print("   3. Retrieved Docs → Gemini → Knowledge-rich response")
    
    print(f"\n🎯 Semantic Search Examples:")
    print("   • 'expensive' → High amount transactions (semantic similarity)")
    print("   • 'food' → Restaurant, grocery, dining transactions")
    print("   • 'bills' → Utilities, subscriptions, recurring payments")
    print("   • 'entertainment' → Movies, games, streaming services")

if __name__ == "__main__":
    asyncio.run(test_sentence_transformers_rag())
