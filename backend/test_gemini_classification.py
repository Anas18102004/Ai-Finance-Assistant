#!/usr/bin/env python3
"""
Test script to demonstrate Gemini-based query classification.
"""

import asyncio
import os
from nodes.graph_orchestrator import graph

async def test_gemini_classification():
    """Test the Gemini-based query classification system."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    user_id = "test_user"
    
    print("🤖 Testing Gemini-Based Query Classification")
    print("=" * 60)
    
    # Test queries - mix of conversational and financial
    test_queries = [
        "hello",
        "what's my name", 
        "thanks",
        "show me my expenses",
        "how much did I spend on food?",
        "what can you do?",
        "find transactions above 5000",
        "good morning",
        "bye"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            # Run the query
            result = await graph.run(
                user_id=user_id,
                query=query,
                summarize=True,
                gemini_api_key=gemini_api_key
            )
            
            classification = result['metadata'].get('gemini_classification', 'N/A')
            rag_used = result['metadata'].get('rag_used', 'N/A')
            response_time = result['metadata'].get('total_time_ms', 0)
            
            print(f"   Classification: {classification}")
            print(f"   RAG Used: {rag_used}")
            print(f"   Response Time: {response_time:.1f}ms")
            print(f"   Response: {result['summary'][:100]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n📊 Expected Results:")
    print("   • Greetings/personal questions → SIMPLE (no RAG)")
    print("   • Financial queries → RAG (with transaction data)")
    print("   • Fast responses for SIMPLE queries")

if __name__ == "__main__":
    asyncio.run(test_gemini_classification())
