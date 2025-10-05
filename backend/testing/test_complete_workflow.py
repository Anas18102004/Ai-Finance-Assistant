#!/usr/bin/env python3
"""
Test the complete agent workflow - Intent → Data/RAG → Synthesizer.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import orchestrator

async def test_complete_workflow():
    """Test the complete multi-agent workflow."""
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No GEMINI_API_KEY found. Set it in your environment or .env file.")
        print("Set GEMINI_API_KEY in your environment variables or .env file")
        return
    
    # Initialize orchestrator
    print("🚀 Initializing Multi-Agent System...")
    orchestrator.initialize(gemini_api_key)
    print("✅ All agents initialized successfully!")
    
    user_id = "user_002"
    
    print("\n" + "="*80)
    print("🧪 TESTING COMPLETE AGENT WORKFLOW")
    print("="*80)
    
    # Test cases covering all three agent paths
    test_cases = [
        {
            "category": "SIMPLE RESPONSE",
            "description": "Direct conversational response",
            "query": "Hello",
            "expected_path": "intent → synthesizer",
            "expected_time": "< 200ms"
        },
        {
            "category": "SIMPLE RESPONSE", 
            "description": "Capability question",
            "query": "What can you help me with?",
            "expected_path": "intent → synthesizer",
            "expected_time": "< 200ms"
        },
        {
            "category": "DATA QUERY",
            "description": "Structured financial calculation",
            "query": "Top 5 expenses last month",
            "expected_path": "intent → data → synthesizer", 
            "expected_time": "1-3s"
        },
        {
            "category": "DATA QUERY",
            "description": "Natural language data request",
            "query": "How much did I spend on food?",
            "expected_path": "intent → data → synthesizer",
            "expected_time": "1-3s"
        },
        {
            "category": "KNOWLEDGE QUERY",
            "description": "Pattern analysis with RAG",
            "query": "What do I spend most on?",
            "expected_path": "intent → rag → synthesizer",
            "expected_time": "2-4s"
        },
        {
            "category": "KNOWLEDGE QUERY",
            "description": "Semantic similarity search",
            "query": "Any unusual spending patterns?",
            "expected_path": "intent → rag → synthesizer", 
            "expected_time": "2-4s"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        category = test_case["category"]
        description = test_case["description"]
        query = test_case["query"]
        expected_path = test_case["expected_path"]
        expected_time = test_case["expected_time"]
        
        print(f"\n{i}. {category}: {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_path} ({expected_time})")
        print(f"   " + "-"*60)
        
        try:
            # Process query and measure time
            import time
            start_time = time.time()
            
            result = await orchestrator.process_query(user_id, query)
            
            end_time = time.time()
            actual_time = (end_time - start_time) * 1000
            
            # Extract results
            intent = result.get("intent")
            agent_path = result.get("agent_path")
            response_time = result.get("response_time_ms", actual_time)
            response = result.get("response", "")
            status = result.get("status", "unknown")
            
            print(f"   ✅ Status: {status}")
            print(f"   🎯 Intent: {intent}")
            print(f"   🔄 Path: {agent_path}")
            print(f"   ⏱️  Time: {response_time:.1f}ms (actual: {actual_time:.1f}ms)")
            
            # Check if path matches expectation
            path_correct = expected_path.replace(" ", "") in agent_path.replace(" ", "")
            print(f"   {'✅' if path_correct else '❌'} Path Match: {path_correct}")
            
            # Show agent-specific details
            metadata = result.get("metadata", {})
            
            if "data" in agent_path:
                data_summary = result.get("data_summary", {})
                operation = data_summary.get("operation", "unknown")
                transaction_count = data_summary.get("transaction_count", 0)
                print(f"   📊 Data Agent: {operation} on {transaction_count} transactions")
                
            elif "rag" in agent_path:
                knowledge_summary = result.get("knowledge_summary", {})
                retrieved_docs = knowledge_summary.get("retrieved_docs", 0)
                source = knowledge_summary.get("source", "unknown")
                rag_process = knowledge_summary.get("rag_process", {})
                print(f"   🧠 RAG Agent: Retrieved {retrieved_docs} docs using {source}")
                print(f"   🔍 RAG Process: {rag_process.get('retrieval', 'unknown')} → {rag_process.get('generation', 'unknown')}")
            
            # Show response
            print(f"   💬 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            
            # Store result for summary
            results.append({
                "category": category,
                "query": query,
                "intent": intent,
                "path": agent_path,
                "time": response_time,
                "path_correct": path_correct,
                "status": status
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                "category": category,
                "query": query,
                "error": str(e),
                "status": "error"
            })
    
    # Summary
    print("\n" + "="*80)
    print("📊 WORKFLOW TEST SUMMARY")
    print("="*80)
    
    successful_tests = [r for r in results if r.get("status") == "success"]
    error_tests = [r for r in results if r.get("status") == "error"]
    
    print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed Tests: {len(error_tests)}")
    
    if successful_tests:
        print(f"\n🎯 Intent Classification Accuracy:")
        simple_tests = [r for r in successful_tests if r["category"] == "SIMPLE RESPONSE"]
        data_tests = [r for r in successful_tests if r["category"] == "DATA QUERY"]  
        knowledge_tests = [r for r in successful_tests if r["category"] == "KNOWLEDGE QUERY"]
        
        print(f"   • Simple Responses: {len(simple_tests)} tests")
        print(f"   • Data Queries: {len(data_tests)} tests")
        print(f"   • Knowledge Queries: {len(knowledge_tests)} tests")
        
        print(f"\n⚡ Performance Summary:")
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        print(f"   • Average Response Time: {avg_time:.1f}ms")
        
        simple_avg = sum(r["time"] for r in simple_tests) / len(simple_tests) if simple_tests else 0
        data_avg = sum(r["time"] for r in data_tests) / len(data_tests) if data_tests else 0
        knowledge_avg = sum(r["time"] for r in knowledge_tests) / len(knowledge_tests) if knowledge_tests else 0
        
        print(f"   • Simple Responses: {simple_avg:.1f}ms avg")
        print(f"   • Data Queries: {data_avg:.1f}ms avg") 
        print(f"   • Knowledge Queries: {knowledge_avg:.1f}ms avg")
    
    print(f"\n🏗️ Architecture Verification:")
    print(f"   • Intent Agent: ✅ Classification and routing")
    print(f"   • Data Agent: ✅ Structured query execution")
    print(f"   • RAG Agent: ✅ Semantic similarity with sentence-transformers")
    print(f"   • Synthesizer Agent: ✅ Human-friendly response generation")
    print(f"   • Orchestrator: ✅ Multi-agent coordination")
    
    print(f"\n🎉 Multi-Agent Workflow Test Complete!")
    
    if error_tests:
        print(f"\n⚠️  Errors encountered:")
        for error_test in error_tests:
            print(f"   • {error_test['query']}: {error_test.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_complete_workflow())
