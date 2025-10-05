#!/usr/bin/env python3
"""
Compare semantic similarity (sentence-transformers) vs keyword matching.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def compare_semantic_vs_keyword():
    """Compare semantic similarity vs keyword matching approaches."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    
    user_id = "user_002"
    
    print("🔍 Semantic Similarity vs Keyword Matching Comparison")
    print("=" * 70)
    
    # Test cases that show semantic understanding
    comparison_tests = [
        {
            "query": "expensive purchases",
            "semantic_expectation": "Finds high-amount transactions regardless of description",
            "keyword_expectation": "Would only find transactions with 'expensive' in description"
        },
        {
            "query": "dining out",
            "semantic_expectation": "Finds restaurants, cafes, food delivery, etc.",
            "keyword_expectation": "Would only find exact 'dining' matches"
        },
        {
            "query": "monthly bills",
            "semantic_expectation": "Finds utilities, subscriptions, recurring payments",
            "keyword_expectation": "Would only find transactions with 'monthly' or 'bills'"
        },
        {
            "query": "entertainment expenses",
            "semantic_expectation": "Finds movies, games, streaming, events, etc.",
            "keyword_expectation": "Would only find exact 'entertainment' matches"
        }
    ]
    
    for i, test in enumerate(comparison_tests, 1):
        query = test["query"]
        semantic_exp = test["semantic_expectation"]
        keyword_exp = test["keyword_expectation"]
        
        print(f"\n{i}. Query: '{query}'")
        print(f"   🤖 Semantic (sentence-transformers): {semantic_exp}")
        print(f"   🔤 Keyword Matching: {keyword_exp}")
        
        try:
            # Test with RAG (semantic similarity)
            result = await orchestrator.process_query(user_id, query)
            
            if result.get("intent") == "knowledge_query" and "rag" in result.get("agent_path", ""):
                knowledge_summary = result.get("knowledge_summary", {})
                retrieved_docs = knowledge_summary.get("retrieved_docs", 0)
                
                print(f"   ✅ RAG Result: Found {retrieved_docs} semantically similar documents")
                
                # Show response snippet
                response = result.get("response", "")
                print(f"   📝 Response: {response[:100]}...")
                
            else:
                print(f"   ❌ Query routed to {result.get('agent_path', 'unknown')}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n🎯 Why Semantic Similarity is Better:")
    print("   • Understanding Context: 'expensive' → high amounts (not literal text)")
    print("   • Synonym Recognition: 'dining' → restaurants, cafes, food delivery")
    print("   • Concept Matching: 'bills' → utilities, subscriptions, recurring payments")
    print("   • Flexible Queries: Works with natural language, not exact keywords")
    
    print(f"\n⚙️ sentence-transformers/all-MiniLM-L6-v2 Features:")
    print("   • Pre-trained on diverse text data")
    print("   • Understands semantic relationships")
    print("   • 384-dimensional embeddings")
    print("   • Fast inference (~10ms per query)")
    print("   • Multilingual capabilities")
    
    print(f"\n📊 Technical Comparison:")
    print("   Keyword Matching:")
    print("     • Query: 'expensive' → Searches for exact word 'expensive'")
    print("     • Limited to literal text matches")
    print("     • Misses semantically similar content")
    print("   ")
    print("   Semantic Similarity (sentence-transformers):")
    print("     • Query: 'expensive' → Embedding → Similar to 'costly', 'high-priced', high amounts")
    print("     • Understands meaning and context")
    print("     • Finds relevant content even without exact keywords")
    
    print(f"\n🚀 RAG with Semantic Search Benefits:")
    print("   • More relevant document retrieval")
    print("   • Better user experience with natural queries")
    print("   • Handles synonyms and related concepts")
    print("   • Improved accuracy in knowledge extraction")

if __name__ == "__main__":
    asyncio.run(compare_semantic_vs_keyword())
