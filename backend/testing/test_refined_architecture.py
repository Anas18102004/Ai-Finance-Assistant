#!/usr/bin/env python3
"""
Test the refined multi-agent architecture with optimized prompts.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_refined_architecture():
    """Test the refined multi-agent architecture."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "user_002"
    
    print("🚀 Testing Refined Multi-Agent Architecture")
    print("=" * 65)
    
    # Test cases following the refined prompt examples
    test_cases = [
        # Simple responses
        {
            "query": "Hi",
            "expected_intent": "simple_response",
            "expected_path": "intent → synthesizer",
            "description": "Basic greeting"
        },
        {
            "query": "Hi, how are you?", 
            "expected_intent": "simple_response",
            "expected_path": "intent → synthesizer",
            "description": "Conversational greeting"
        },
        
        # Data queries with structured plans
        {
            "query": "Top 3 food expenses last month",
            "expected_intent": "data_query", 
            "expected_path": "intent → data → synthesizer",
            "description": "Structured data query with filters"
        },
        {
            "query": "my last month spendings on food",
            "expected_intent": "data_query",
            "expected_path": "intent → data → synthesizer", 
            "description": "Natural language data query"
        },
        {
            "query": "How much did I spend on food?",
            "expected_intent": "data_query",
            "expected_path": "intent → data → synthesizer",
            "description": "Sum calculation query"
        },
        
        # Knowledge queries using ChromaDB
        {
            "query": "What do I spend most on?",
            "expected_intent": "knowledge_query",
            "expected_path": "intent → rag → synthesizer",
            "description": "Pattern analysis from transaction data"
        },
        {
            "query": "What is GST?",
            "expected_intent": "knowledge_query", 
            "expected_path": "intent → rag → synthesizer",
            "description": "General financial knowledge"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_intent = test_case["expected_intent"]
        expected_path = test_case["expected_path"]
        description = test_case["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_intent} via {expected_path}")
        
        try:
            # Process query
            result = await orchestrator.process_query(user_id, query)
            
            actual_intent = result.get("intent")
            agent_path = result.get("agent_path")
            response_time = result.get("response_time_ms", 0)
            response = result.get("response", "")
            
            print(f"   Actual: {actual_intent} via {agent_path}")
            print(f"   Response Time: {response_time:.1f}ms")
            print(f"   Response: {response[:100]}...")
            
            # Validation
            intent_match = actual_intent == expected_intent
            path_match = expected_path in agent_path
            
            if intent_match and path_match:
                print(f"   ✅ PASS - Correct intent and routing")
            else:
                print(f"   ❌ FAIL - Intent: {intent_match}, Path: {path_match}")
                
            # Show additional metadata
            metadata = result.get("metadata", {})
            if metadata.get("data_summary"):
                data_summary = metadata["data_summary"]
                print(f"   📊 Data: {data_summary.get('operation')} on {data_summary.get('transaction_count', 0)} transactions")
            
            if metadata.get("knowledge_summary"):
                knowledge_summary = metadata["knowledge_summary"]
                print(f"   🧠 Knowledge: {knowledge_summary.get('source')} with {knowledge_summary.get('transaction_count', 0)} transactions")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n🎯 Architecture Benefits:")
    print("   • Intent Agent: Smart classification with structured plans")
    print("   • Data Agent: Machine-readable output, no explanations")
    print("   • RAG Agent: ChromaDB transaction data as knowledge base")
    print("   • Synthesizer Agent: Polished human-friendly responses")
    
    print(f"\n📈 Performance Expectations:")
    print("   • Simple responses: ~50-200ms (direct routing)")
    print("   • Data queries: ~1-2s (structured execution)")
    print("   • Knowledge queries: ~1-3s (ChromaDB + analysis)")
    
    print(f"\n✅ Key Improvements:")
    print("   • Faster simple queries (skip unnecessary processing)")
    print("   • More accurate data queries (structured plans)")
    print("   • Personalized knowledge (user's transaction data)")
    print("   • Consistent responses (single synthesizer)")

if __name__ == "__main__":
    asyncio.run(test_refined_architecture())
