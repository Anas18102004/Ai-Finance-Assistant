#!/usr/bin/env python3
"""
Test script for the AI Financial Assistant system.
This script tests all components without requiring the full API server.
"""

import asyncio
import json
import logging
import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_synthetic_transactions, save_transactions_to_file
from services.embeddings import embedding_service
from index_build.build_index import index_builder
from nodes.query_parser import query_parser
from nodes.retriever import retriever
from nodes.graph_orchestrator import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_generation():
    """Test synthetic data generation."""
    print("\n🧪 Testing Data Generation...")
    
    try:
        transactions = generate_synthetic_transactions()
        
        # Verify data quality
        assert len(transactions) >= 400, f"Expected at least 400 transactions, got {len(transactions)}"
        
        users = set(t["userId"] for t in transactions)
        assert len(users) >= 3, f"Expected at least 3 users, got {len(users)}"
        
        # Check data structure
        required_fields = ["id", "userId", "date", "description", "amount", "type", "category", "balance"]
        for field in required_fields:
            assert field in transactions[0], f"Missing field: {field}"
        
        # Save test data
        save_transactions_to_file(transactions, "data/transactions.json")
        
        print(f"✅ Generated {len(transactions)} transactions for {len(users)} users")
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

async def test_embeddings():
    """Test embedding service."""
    print("\n🧪 Testing Embedding Service...")
    
    try:
        # Initialize service
        await embedding_service.initialize()
        
        # Test single embedding
        test_text = "debit of ₹500 on 2024-01-15 for Coffee shop under Food & Dining"
        embedding = await embedding_service.embed_query(test_text)
        
        assert len(embedding) == 384, f"Expected 384-dim embedding, got {len(embedding)}"
        assert isinstance(embedding[0], float), "Embedding should contain floats"
        
        # Test batch embeddings
        test_transactions = [
            {"type": "debit", "amount": 500, "date": "2024-01-15", "description": "Coffee", "category": "Food"},
            {"type": "credit", "amount": 50000, "date": "2024-01-01", "description": "Salary", "category": "Income"}
        ]
        
        embeddings = await embedding_service.embed_transactions(test_transactions)
        assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
        
        print(f"✅ Embeddings working - dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

async def test_index_building():
    """Test Chroma index building."""
    print("\n🧪 Testing Index Building...")
    
    try:
        # Build index
        await index_builder.rebuild_index()
        
        # Verify index
        stats = index_builder.get_collection_stats()
        
        assert stats["total_documents"] > 0, "Index should contain documents"
        assert "sample_users" in stats, "Stats should include user info"
        
        print(f"✅ Index built with {stats['total_documents']} documents")
        return True
        
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        return False

async def test_query_parsing():
    """Test query parser."""
    print("\n🧪 Testing Query Parser...")
    
    try:
        test_queries = [
            ("Show me my top 5 expenses", "top_expenses"),
            ("How much did I spend on food?", "sum_spent"),
            ("Find transactions above ₹1000", "filter"),
            ("Compare my spending", "compare"),
            ("What did I buy yesterday?", "general")
        ]
        
        for query, expected_intent in test_queries:
            parsed = query_parser.parse_query(query, "user_001")
            assert parsed.intent == expected_intent, f"Expected {expected_intent}, got {parsed.intent}"
        
        # Test filter extraction
        parsed = query_parser.parse_query("Show food expenses above ₹500 in September", "user_001")
        assert "food" in parsed.categories or "Food" in str(parsed.categories)
        assert parsed.amount_range is not None
        
        print("✅ Query parsing working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Query parsing failed: {e}")
        return False

async def test_retrieval():
    """Test transaction retrieval."""
    print("\n🧪 Testing Retrieval...")
    
    try:
        # Initialize retriever
        retriever.initialize_client()
        
        # Test query
        query_intent = query_parser.parse_query("Show me my top expenses", "user_001")
        results = await retriever.retrieve(query_intent, "top expenses")
        
        assert isinstance(results, list), "Results should be a list"
        
        if results:
            # Check result structure
            result = results[0]
            required_fields = ["id", "userId", "amount", "category", "similarity_score"]
            for field in required_fields:
                assert field in result, f"Missing field in result: {field}"
        
        print(f"✅ Retrieved {len(results)} transactions")
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False

async def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\n🧪 Testing End-to-End Workflow...")
    
    try:
        test_queries = [
            "Show me my top 3 expenses",
            "How much did I spend on food?",
            "Find transactions above ₹2000"
        ]
        
        for query in test_queries:
            start_time = time.time()
            
            result = await graph.run(
                user_id="user_001",
                query=query,
                summarize=False,  # Skip Gemini for testing
                top_k=5
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            assert result["status"] == "success", f"Query failed: {result.get('error')}"
            assert "retrieved" in result, "Result should contain retrieved transactions"
            
            print(f"   • '{query}' - {latency:.1f}ms - {len(result['retrieved'])} results")
        
        print("✅ End-to-end workflow working")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

async def run_performance_test():
    """Run performance benchmarks."""
    print("\n⚡ Performance Testing...")
    
    try:
        queries = [
            "Show me my expenses",
            "Food transactions",
            "Top spending categories",
            "Transactions above ₹1000",
            "September expenses"
        ]
        
        total_time = 0
        successful_queries = 0
        
        for query in queries:
            start_time = time.time()
            
            result = await graph.run(
                user_id="user_001",
                query=query,
                summarize=False,
                top_k=10
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            total_time += latency
            
            if result["status"] == "success":
                successful_queries += 1
            
            print(f"   • {latency:.1f}ms - '{query}'")
        
        avg_latency = total_time / len(queries)
        success_rate = (successful_queries / len(queries)) * 100
        
        print(f"\n📊 Performance Results:")
        print(f"   • Average latency: {avg_latency:.1f}ms")
        print(f"   • Success rate: {success_rate:.1f}%")
        print(f"   • Target: <500ms ({'✅' if avg_latency < 500 else '❌'})")
        
        return avg_latency < 500 and success_rate > 90
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 AI Financial Assistant - System Tests")
    print("=" * 50)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Embeddings", test_embeddings),
        ("Index Building", test_index_building),
        ("Query Parsing", test_query_parsing),
        ("Retrieval", test_retrieval),
        ("End-to-End", test_end_to_end),
        ("Performance", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n🚀 Next steps:")
        print("   1. Get a Gemini API key from https://makersuite.google.com/app/apikey")
        print("   2. Run: python setup_and_run.py")
        print("   3. Test API at http://localhost:8000/docs")
    else:
        print("❌ Some tests failed. Please check the logs and fix issues.")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
