#!/usr/bin/env python3
"""
Quick test script for the multi-agent system.
Run this to verify the complete workflow is working.
"""

import asyncio
import os
from agents.orchestrator import orchestrator

async def quick_test():
    """Quick test of the agent workflow."""
    
    print("🧪 Quick Agent Workflow Test")
    print("="*50)
    
    # Check for Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("✅ Gemini API key found")
    
    # Initialize system
    try:
        orchestrator.initialize(gemini_api_key)
        print("✅ Multi-agent system initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test queries
    test_queries = [
        ("hello", "simple_response"),
        ("top 3 expenses", "data_query"), 
        ("what do I spend most on", "knowledge_query")
    ]
    
    user_id = "user_002"
    
    for query, expected_intent in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        try:
            result = await orchestrator.process_query(user_id, query)
            
            intent = result.get("intent")
            path = result.get("agent_path")
            time_ms = result.get("response_time_ms", 0)
            response = result.get("response", "")
            
            print(f"   Intent: {intent} {'✅' if intent == expected_intent else '❌'}")
            print(f"   Path: {path}")
            print(f"   Time: {time_ms:.1f}ms")
            print(f"   Response: {response[:80]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Quick test complete!")
    print(f"Run 'python testing/test_complete_workflow.py' for detailed testing")

if __name__ == "__main__":
    asyncio.run(quick_test())
