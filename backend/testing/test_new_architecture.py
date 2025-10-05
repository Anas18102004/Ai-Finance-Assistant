#!/usr/bin/env python3
"""
Test the new multi-agent architecture.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_new_architecture():
    """Test the new multi-agent architecture."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "test_user"
    
    print("🤖 Testing New Multi-Agent Architecture")
    print("=" * 60)
    
    # Test queries for each intent type
    test_queries = [
        # Simple responses
        ("hello", "simple_response"),
        ("what can you do", "simple_response"),
        ("thanks", "simple_response"),
        
        # Data queries
        ("show my top 5 expenses last month", "data_query"),
        ("how much did I spend on food", "data_query"),
        ("my monthly spending summary", "data_query"),
        
        # Knowledge queries
        ("what is GST", "knowledge_query"),
        ("explain EMI calculation", "knowledge_query"),
        ("investment tips", "knowledge_query")
    ]
    
    for i, (query, expected_intent) in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print(f"   Expected Intent: {expected_intent}")
        
        try:
            # Process query
            result = await orchestrator.process_query(user_id, query)
            
            actual_intent = result.get("intent")
            agent_path = result.get("agent_path")
            response_time = result.get("response_time_ms", 0)
            response = result.get("response", "")
            
            print(f"   Actual Intent: {actual_intent}")
            print(f"   Agent Path: {agent_path}")
            print(f"   Response Time: {response_time:.1f}ms")
            print(f"   Response: {response[:100]}...")
            
            # Check if intent matches expectation
            if actual_intent == expected_intent:
                print(f"   ✅ Intent classification correct")
            else:
                print(f"   ❌ Intent mismatch (expected {expected_intent})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show memory stats
    print(f"\n📊 Memory Stats:")
    stats = orchestrator.get_memory_stats()
    print(f"   Total users: {stats['total_users']}")
    print(f"   Memory size per user: {stats['memory_size_per_user']}")
    print(f"   Users with memory: {len(stats['users_with_memory'])}")
    
    print(f"\n🎯 Architecture Benefits:")
    print("   • Simple queries skip unnecessary processing → Fast responses")
    print("   • Data queries use structured execution → Accurate results")
    print("   • Knowledge queries leverage RAG → Rich information")
    print("   • All responses synthesized consistently → Natural language")

if __name__ == "__main__":
    asyncio.run(test_new_architecture())
