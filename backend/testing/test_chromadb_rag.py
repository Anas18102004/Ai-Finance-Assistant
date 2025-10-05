#!/usr/bin/env python3
"""
Test the RAG agent with ChromaDB transaction data.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_chromadb_rag():
    """Test RAG agent using ChromaDB transaction data."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "user_002"  # Use a user with transaction data
    
    print("🗄️ Testing RAG Agent with ChromaDB Transaction Data")
    print("=" * 65)
    
    # Test knowledge queries that should use transaction data
    knowledge_queries = [
        "What do I spend most on?",
        "How is my spending pattern?", 
        "Any unusual transactions in my data?",
        "What are my financial habits?",
        "Show me insights from my transaction history",
        "What patterns do you see in my spending?"
    ]
    
    for i, query in enumerate(knowledge_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
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
            print(f"   Response: {response[:150]}...")
            
            # Check if it used RAG
            if "rag" in agent_path:
                print(f"   ✅ Used RAG with ChromaDB")
                
                # Show knowledge summary if available
                knowledge_summary = result.get("knowledge_summary", {})
                if knowledge_summary:
                    transaction_count = knowledge_summary.get("transaction_count", 0)
                    source = knowledge_summary.get("source", "unknown")
                    print(f"   📊 Analyzed {transaction_count} transactions from {source}")
            else:
                print(f"   ❌ Did not use RAG")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 RAG Agent Benefits:")
    print("   • Uses actual user transaction data from ChromaDB")
    print("   • Provides personalized insights based on spending history")
    print("   • Analyzes patterns and trends in user's financial behavior")
    print("   • Gives data-driven answers with specific amounts and categories")
    
    print(f"\n📋 Example Use Cases:")
    print("   • 'What do I spend most on?' → Category analysis with percentages")
    print("   • 'Any unusual transactions?' → Identifies outliers in spending")
    print("   • 'How is my spending pattern?' → Behavioral analysis over time")

if __name__ == "__main__":
    asyncio.run(test_chromadb_rag())
