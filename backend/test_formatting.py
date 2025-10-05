#!/usr/bin/env python3
"""
Test the new beautiful response formatting.
"""

import asyncio
import os
from agents.orchestrator import orchestrator

async def test_formatting():
    """Test the beautiful response formatting."""
    
    print("✨ Testing Beautiful Response Formatting")
    print("="*60)
    
    # Get Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found!")
        return
    
    # Initialize orchestrator
    orchestrator.initialize(gemini_api_key)
    print("✅ Multi-agent system initialized")
    
    user_id = "user_001"
    
    # Test queries that should produce beautifully formatted responses
    test_queries = [
        {
            "query": "my top 3 expenses",
            "expected_format": "Numbered list with bold amounts and descriptions"
        },
        {
            "query": "how much did I spend on food",
            "expected_format": "Total with bullet point breakdown"
        },
        {
            "query": "monthly summary",
            "expected_format": "Structured summary with categories"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected_format = test["expected_format"]
        
        print(f"\n{i}. Query: '{query}'")
        print(f"   Expected: {expected_format}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_query(user_id, query)
            
            response = result.get("response", "")
            response_time = result.get("response_time_ms", 0)
            intent = result.get("intent")
            agent_path = result.get("agent_path")
            
            print(f"   Intent: {intent}")
            print(f"   Path: {agent_path}")
            print(f"   Response Time: {response_time:.1f}ms")
            print(f"   \n📝 Formatted Response:")
            print(f"   {'-' * 40}")
            
            # Display the response with proper formatting
            for line in response.split('\n'):
                if line.strip():
                    print(f"   {line}")
                else:
                    print()
            
            print(f"   {'-' * 40}")
            
            # Check formatting elements
            formatting_checks = {
                "Bold amounts": "**₹" in response,
                "Bold descriptions": "**" in response and "₹" not in response.split("**")[1] if "**" in response else False,
                "Numbered lists": any(f"{j}." in response for j in range(1, 6)),
                "Bullet points": "•" in response,
                "No response time in text": "Response time:" not in response and "ms" not in response
            }
            
            print(f"\n   ✅ Formatting Analysis:")
            for check, passed in formatting_checks.items():
                status = "✅" if passed else "❌"
                print(f"      {status} {check}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Formatting Standards:")
    print("   • **Bold** amounts: **₹16,969**")
    print("   • **Bold** descriptions: **Room rent**")
    print("   • Numbered lists: 1. 2. 3.")
    print("   • Bullet points: • for details")
    print("   • Clean structure with proper spacing")
    print("   • Response time shown separately (not in text)")
    
    print(f"\n✨ Perfect Response Example:")
    print(f"   Here are your top 3 expenses:")
    print(f"   ")
    print(f"   1. **₹16,969** for **Room rent** on 2025-09-21")
    print(f"   2. **₹13,084** for **Ola Booking** on 2025-09-29")
    print(f"   3. **₹4,899** for **Hotstar Payment** on 2025-10-02")
    print(f"   ")
    print(f"   [Response time shown separately: ⚡ 1234ms]")

if __name__ == "__main__":
    asyncio.run(test_formatting())
