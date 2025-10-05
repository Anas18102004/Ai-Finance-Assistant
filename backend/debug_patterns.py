#!/usr/bin/env python3
"""
Debug script to test pattern matching.
"""

from nodes.query_parser import query_parser

# Test queries
test_queries = [
    "hello",
    "my name", 
    "what's my name",
    "whats my name",
    "show me expenses",
    "hi"
]

print("🔍 Testing Pattern Matching")
print("=" * 40)

for query in test_queries:
    intent = query_parser.parse_query(query, "test_user")
    print(f"'{query}' -> {intent.intent}")

print("\n" + "=" * 40)
print("Expected:")
print("'hello' -> greeting")
print("'my name' -> personal_info") 
print("'what's my name' -> personal_info")
print("'show me expenses' -> general or filter")
