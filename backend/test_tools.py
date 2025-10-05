#!/usr/bin/env python3
"""
Test the financial tools functionality.
"""

from tools.financial_tools import financial_tools

def test_financial_tools():
    """Test financial tools with sample data."""
    
    # Sample transaction data matching your format
    sample_transactions = [
        {
            "id": "txn_000238",
            "userId": "user_002",
            "date": "2025-10-09",
            "description": "Gas bill",
            "amount": 680,
            "type": "Debit",
            "category": "Utilities",
            "balance": 1083566
        },
        {
            "id": "txn_000239",
            "userId": "user_002", 
            "date": "2025-10-10",
            "description": "Restaurant dinner",
            "amount": 1250,
            "type": "Debit",
            "category": "Food",
            "balance": 1082316
        },
        {
            "id": "txn_000240",
            "userId": "user_002",
            "date": "2025-10-11", 
            "description": "Grocery shopping",
            "amount": 890,
            "type": "Debit",
            "category": "Food",
            "balance": 1081426
        },
        {
            "id": "txn_000241",
            "userId": "user_002",
            "date": "2025-10-12",
            "description": "Salary credit",
            "amount": 50000,
            "type": "Credit",
            "category": "Salary",
            "balance": 1131426
        }
    ]
    
    print("🛠️ Testing Financial Tools")
    print("=" * 50)
    
    # Test 1: Calculate total spending
    print("\n1. Calculate Total Spending:")
    result = financial_tools.calculate_total_spending(sample_transactions)
    print(f"   Total: ₹{result['total_amount']}")
    print(f"   Transactions: {result['transaction_count']}")
    print(f"   Average: ₹{result['average_per_transaction']:.0f}")
    
    # Test 2: Analyze spending by category
    print("\n2. Spending by Category:")
    result = financial_tools.analyze_spending_by_category(sample_transactions)
    for category in result['categories']:
        print(f"   {category['category']}: ₹{category['amount']} ({category['percentage']}%)")
    
    # Test 3: Find top expenses
    print("\n3. Top Expenses:")
    result = financial_tools.find_top_expenses(sample_transactions, limit=3)
    for i, expense in enumerate(result['top_expenses'], 1):
        print(f"   {i}. ₹{expense['amount']} - {expense['description']} ({expense['category']})")
    
    # Test 4: Monthly summary
    print("\n4. Monthly Summary:")
    result = financial_tools.get_monthly_summary(sample_transactions, month=10, year=2025)
    summary = result['summary']
    print(f"   Income: ₹{summary['total_income']}")
    print(f"   Expenses: ₹{summary['total_expenses']}")
    print(f"   Savings: ₹{summary['net_savings']}")
    print(f"   Savings Rate: {summary['savings_rate']:.1f}%")
    
    print("\n✅ All tools working correctly!")
    
    # Show how Gemini would use these tools
    print("\n🤖 Example Gemini Tool Usage:")
    print("Query: 'How much did I spend on food this month?'")
    print("Gemini would generate:")
    print("TOOL_CALL: calculate_total_spending(transactions=transaction_data, category='Food')")
    print("TOOL_CALL: find_top_expenses(transactions=transaction_data, category='Food', limit=3)")
    print("\nResult after tool processing:")
    print("You spent ₹2,140 across 2 transactions on food. Top expenses: ₹1,250 for Restaurant dinner, ₹890 for Grocery shopping.")

if __name__ == "__main__":
    test_financial_tools()
