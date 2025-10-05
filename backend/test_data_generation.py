#!/usr/bin/env python3
"""
Test the new Faker + Gemini data generation system.
"""

import json
from generate_data import generate_synthetic_transactions, save_transactions_to_file

def test_faker_generation():
    """Test data generation with Faker only."""
    print("🧪 Testing Faker-only data generation...")
    
    transactions = generate_synthetic_transactions(gemini_api_key=None)
    
    print(f"\n📊 Generated {len(transactions)} transactions")
    
    # Show sample transactions
    print("\n📋 Sample Transactions (Faker + realistic companies):")
    for i, txn in enumerate(transactions[:5], 1):
        print(f"   {i}. {txn['id']} - {txn['type']} of ₹{txn['amount']:,}")
        print(f"      📝 {txn['description']}")
        print(f"      🏷️ {txn['category']} | 📅 {txn['date']}")
        print()
    
    # Verify data structure matches assignment requirements
    sample = transactions[0]
    required_fields = ["id", "userId", "date", "description", "amount", "type", "category", "balance"]
    
    print("✅ Data Structure Validation:")
    for field in required_fields:
        if field in sample:
            print(f"   ✓ {field}: {sample[field]}")
        else:
            print(f"   ❌ Missing field: {field}")
    
    # Check categories match assignment
    assignment_categories = ["Food", "Shopping", "Rent", "Salary", "Utilities", "Entertainment", "Travel", "Others"]
    found_categories = set(t["category"] for t in transactions)
    
    print(f"\n📂 Categories Found: {sorted(found_categories)}")
    print(f"📋 Assignment Categories: {assignment_categories}")
    
    missing = set(assignment_categories) - found_categories
    if missing:
        print(f"⚠️ Missing categories: {missing}")
    else:
        print("✅ All assignment categories present")
    
    return transactions

def test_with_gemini():
    """Test data generation with Gemini AI."""
    print("\n🤖 Testing with Gemini AI...")
    
    api_key = input("🔑 Enter Gemini API key (press Enter to skip): ").strip()
    
    if not api_key:
        print("⏭️ Skipping Gemini test")
        return None
    
    print("🚀 Generating with Gemini AI...")
    transactions = generate_synthetic_transactions(gemini_api_key=api_key)
    
    print(f"\n📊 Generated {len(transactions)} transactions with AI")
    
    # Show AI-generated descriptions
    print("\n🤖 Sample AI-Generated Descriptions:")
    ai_descriptions = [t for t in transactions if "UPI-" not in t["description"] and "Purchase" not in t["description"]][:3]
    
    for i, txn in enumerate(ai_descriptions, 1):
        print(f"   {i}. {txn['description']} (₹{txn['amount']:,} - {txn['category']})")
    
    return transactions

def main():
    """Main test function."""
    print("🧪 AI Financial Assistant - Data Generation Test")
    print("=" * 60)
    print("Testing: Faker (Python) + Gemini AI")
    print()
    
    # Test 1: Faker only
    faker_transactions = test_faker_generation()
    
    # Test 2: Gemini AI (optional)
    gemini_transactions = test_with_gemini()
    
    # Save test data
    final_transactions = gemini_transactions if gemini_transactions else faker_transactions
    save_transactions_to_file(final_transactions, "data/transactions.json")
    
    print("\n🎉 Data generation testing complete!")
    print(f"📁 Check data/transactions.json for {len(final_transactions)} transactions")
    
    # Show exact format matching assignment
    sample = final_transactions[0]
    print(f"\n📋 Sample Transaction (Assignment Format):")
    print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
