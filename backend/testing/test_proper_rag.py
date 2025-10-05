#!/usr/bin/env python3
"""
Test the proper RAG (Retrieval-Augmented Generation) flow.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_proper_rag():
    """Test the proper RAG flow: Retrieve → Generate."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "user_002"
    
    print("🔍 Testing Proper RAG (Retrieval-Augmented Generation) Flow")
    print("=" * 70)
    
    # Test knowledge queries that should use proper RAG
    rag_queries = [
        {
            "query": "What do I spend most on?",
            "description": "Pattern analysis requiring document retrieval"
        },
        {
            "query": "How is my spending behavior?", 
            "description": "Behavioral analysis from transaction history"
        },
        {
            "query": "Any unusual patterns in my transactions?",
            "description": "Anomaly detection requiring context"
        },
        {
            "query": "What insights can you give about my finances?",
            "description": "General insights requiring comprehensive analysis"
        }
    ]
    
    for i, test_case in enumerate(rag_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        
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
            
            # Check if RAG was used properly
            if "rag" in agent_path:
                print(f"   ✅ RAG Agent Used")
                
                # Show RAG process details
                knowledge_summary = result.get("knowledge_summary", {})
                if knowledge_summary:
                    retrieved_docs = knowledge_summary.get("retrieved_docs", 0)
                    source = knowledge_summary.get("source", "unknown")
                    rag_process = knowledge_summary.get("rag_process", {})
                    
                    print(f"   📊 RAG Process:")
                    print(f"      • Step 1 (Retrieval): Retrieved {retrieved_docs} most relevant documents")
                    print(f"      • Step 2 (Generation): {rag_process.get('generation', 'unknown')} using {source}")
                    
                    if retrieved_docs > 0:
                        print(f"   ✅ Proper RAG Flow: Retrieve ({retrieved_docs} docs) → Generate")
                    else:
                        print(f"   ⚠️  No documents retrieved")
                
                print(f"   Response: {response[:120]}...")
            else:
                print(f"   ❌ RAG not used - routed to {agent_path}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n🔄 Proper RAG Flow Explanation:")
    print("   1️⃣ RETRIEVAL: Query → ChromaDB → Vector Similarity → Top 5-8 docs")
    print("   2️⃣ GENERATION: Retrieved docs → Gemini → Knowledge-rich response")
    
    print(f"\n🎯 RAG Benefits:")
    print("   • Only retrieves MOST RELEVANT documents (not all data)")
    print("   • Uses vector similarity for intelligent document selection")
    print("   • Generates responses based on retrieved context")
    print("   • Provides specific examples from relevant transactions")
    
    print(f"\n📋 RAG vs Data Agent:")
    print("   • RAG Agent: 'What do I spend most on?' → Retrieve relevant docs → Analyze patterns")
    print("   • Data Agent: 'Top 5 expenses' → Structured query → Calculate results")
    
    print(f"\n✅ Expected RAG Behavior:")
    print("   • Knowledge queries trigger vector similarity search")
    print("   • Top 5-8 most relevant documents retrieved")
    print("   • Gemini analyzes retrieved docs to generate insights")
    print("   • Response includes specific examples from retrieved data")

if __name__ == "__main__":
    asyncio.run(test_proper_rag())
