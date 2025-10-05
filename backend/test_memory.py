#!/usr/bin/env python3
"""
Test script to demonstrate the conversation memory system.
"""

import asyncio
from nodes.graph_orchestrator import graph

async def test_memory():
    """Test the conversation memory system."""
    user_id = "test_user"
    
    print("🧠 Testing Conversation Memory System")
    print("=" * 50)
    
    # Test conversations
    test_queries = [
        "hello",
        "what's my name", 
        "hi again",
        "thanks",
        "show me my expenses"  # This will be a financial query
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        # Run the query
        result = await graph.run(
            user_id=user_id,
            query=query,
            summarize=False  # Skip summarization for this test
        )
        
        print(f"   Intent: {result['intent']}")
        print(f"   Response: {result['summary'][:100]}...")
        print(f"   Memory size: {result['metadata'].get('memory_size', 0)}")
        
        if result['metadata'].get('conversation_context'):
            print(f"   Context: {result['metadata']['conversation_context']}")
    
    # Show final memory stats
    print(f"\n📊 Final Memory Stats:")
    stats = graph.get_memory_stats()
    print(f"   Total users: {stats['total_users']}")
    print(f"   Memory size per user: {stats['memory_size_per_user']}")
    print(f"   Users with memory: {stats['users_with_memory']}")

if __name__ == "__main__":
    asyncio.run(test_memory())
