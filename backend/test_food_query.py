#!/usr/bin/env python3
"""
Test how 'my monthly expense on food' query is processed.
"""

import asyncio
from nodes.query_parser import query_parser
from nodes.graph_orchestrator import graph

async def test_food_query():
    """Test food expense query processing."""
    
    print("🍽️ Testing: 'my monthly expense on food'")
    print("=" * 50)
    
    # Step 1: Query parsing
    query = "my monthly expense on food"
    user_id = "user_002"
    
    print("Step 1: Query Parsing")
    query_intent = query_parser.parse_query(query, user_id)
    print(f"  Intent: {query_intent.intent}")
    print(f"  Categories: {query_intent.categories}")
    print(f"  Filters: {query_intent.filters}")
    print(f"  Time Range: {query_intent.time_range}")
    
    # Step 2: Show expected ChromaDB filter
    print(f"\nStep 2: ChromaDB Filter (Expected)")
    if query_intent.categories and query_intent.time_range:
        expected_filter = {
            "$and": [
                {"userId": {"$eq": user_id}},
                {"category": {"$in": query_intent.categories}},
                {"date": {"$gte": query_intent.time_range.get("start_date")}},
                {"date": {"$lte": query_intent.time_range.get("end_date")}}
            ]
        }
        print(f"  Filter: {expected_filter}")
    
    # Step 3: Sample matching transaction
    print(f"\nStep 3: Sample Matching Transaction")
    sample_transaction = {
        "id": "txn_000123",
        "userId": "user_002",
        "date": "2025-10-15", 
        "description": "Restaurant dinner",
        "amount": 850,
        "type": "Debit",
        "category": "Food",  # This should match
        "balance": 1082716
    }
    print(f"  Transaction: {sample_transaction}")
    
    # Step 4: Expected Gemini response format
    print(f"\nStep 4: Expected Gemini Response")
    expected_response = """Based on your transactions, you spent ₹2,340 on food this month across 8 transactions. Your food expenses included:

• Restaurant meals: ₹1,200 (3 transactions)
• Groceries: ₹890 (4 transactions) 
• Food delivery: ₹250 (1 transaction)

This represents about 12% of your total monthly spending.

Would you like to see a breakdown by specific food categories or compare with last month?"""
    
    print(f"  Response: {expected_response}")

if __name__ == "__main__":
    asyncio.run(test_food_query())
