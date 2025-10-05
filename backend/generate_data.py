import json
import random
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from faker import Faker
import google.generativeai as genai
from config import config

# Initialize Faker with Indian locale for realistic Indian data
fake = Faker('en_IN')

class GeminiDataGenerator:
    """Generate realistic transaction descriptions using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.fallback_descriptions = {
            "Food": ["Restaurant meal", "Coffee shop", "Food delivery", "Swiggy order", "Zomato delivery"],
            "Shopping": ["Online purchase", "Amazon order", "Flipkart shopping", "Mall purchase", "Clothing store"],
            "Rent": ["Monthly rent", "House rent", "Apartment rent", "Room rent"],
            "Salary": ["Monthly salary", "Salary credit", "Payroll deposit", "Income credit"],
            "Utilities": ["Electricity bill", "Internet bill", "Mobile recharge", "Water bill", "Gas bill"],
            "Entertainment": ["Movie ticket", "Netflix subscription", "Gaming", "Concert ticket", "Event booking"],
            "Travel": ["Uber ride", "Ola cab", "Bus ticket", "Metro card", "Flight booking", "Hotel booking"],
            "Others": ["ATM withdrawal", "Bank transfer", "Online payment", "UPI payment", "Card payment"]
        }
    
    def initialize_gemini(self, api_key: str):
        """Initialize Gemini model."""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            self.api_key = api_key
            print("✅ Gemini initialized for realistic data generation")
            return True
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")
            print("📝 Will use Faker + predefined descriptions instead")
            return False
    
    def generate_description(self, category: str, amount: float, transaction_type: str) -> str:
        """Generate realistic transaction description using Gemini or fallback."""
        
        if self.model and self.api_key:
            try:
                prompt = f"""Generate a realistic Indian transaction description for:
Category: {category}
Amount: ₹{amount:.0f}
Type: {transaction_type}

Requirements:
- Use real Indian company/service names (Swiggy, Zomato, Amazon, Flipkart, Ola, Uber, etc.)
- Make it sound like a real bank statement entry
- Keep it concise (2-4 words)
- Use UPI/payment terminology when appropriate

Examples:
- "UPI-Swiggy Food Delivery"
- "Amazon Shopping"
- "Ola Cab Ride"
- "Salary Credit HDFC"

Generate only the description, no extra text:"""

                response = self.model.generate_content(prompt)
                description = response.text.strip()
                
                # Clean up the response
                if description and len(description) < 50:
                    return description
                    
            except Exception as e:
                print(f"⚠️ Gemini generation failed: {e}")
        
        # Fallback to predefined descriptions
        category_key = category.split()[0] if category else "Others"
        if category_key in self.fallback_descriptions:
            return fake.random_element(self.fallback_descriptions[category_key])
        else:
            return fake.random_element(self.fallback_descriptions["Others"])

def generate_synthetic_transactions(gemini_api_key: Optional[str] = None) -> List[Dict]:
    """Generate 200-300 synthetic transactions per user using Faker and Gemini."""
    
    print("🤖 Generating synthetic transaction data with Faker + Gemini...")
    
    # Initialize Gemini generator
    gemini_gen = GeminiDataGenerator()
    use_gemini = False
    
    if gemini_api_key:
        use_gemini = gemini_gen.initialize_gemini(gemini_api_key)
    
    # Categories matching assignment requirements
    categories = [
        "Food", "Shopping", "Rent", "Salary", "Utilities", 
        "Entertainment", "Travel", "Others"
    ]
    
    # Generate realistic users using Faker
    users = []
    for i in range(config.NUM_USERS):
        users.append({
            "id": f"user_{i+1:03d}",
            "name": fake.name()
        })
    
    all_transactions = []
    transaction_id = 1
    
    for user in users:
        user_id = user["id"]
        user_name = user["name"]
        
        # Generate transactions per user from config
        num_transactions = random.randint(config.TRANSACTIONS_PER_USER_MIN, config.TRANSACTIONS_PER_USER_MAX)
        current_balance = fake.random_int(min=50000, max=200000)  # Starting balance using Faker
        
        print(f"📊 Generating {num_transactions} transactions for {user_name} ({user_id})")
        
        # Generate transactions for the last 6 months using Faker dates
        for i in range(num_transactions):
            # Use Faker to generate realistic dates
            transaction_date = fake.date_between(start_date='-6M', end_date='today')
            
            # Determine transaction type (70% debit, 30% credit)
            is_credit = fake.boolean(chance_of_getting_true=30)
            
            if is_credit:
                # Credit transactions (income)
                if fake.boolean(chance_of_getting_true=80):  # 80% salary
                    category = "Salary"
                    amount = fake.random_int(min=25000, max=80000)
                else:  # 20% other income
                    category = fake.random_element(["Others"])
                    amount = fake.random_int(min=1000, max=15000)
                
                transaction_type = "Credit"
                current_balance += amount
            else:
                # Debit transactions (expenses)
                category = fake.random_element(categories[:-1])  # Exclude Salary from expenses
                
                # Realistic amount ranges based on category using Faker
                if category == "Food":
                    amount = fake.random_int(min=50, max=2000)
                elif category == "Shopping":
                    amount = fake.random_int(min=200, max=8000)
                elif category == "Rent":
                    amount = fake.random_int(min=8000, max=25000)
                elif category == "Utilities":
                    amount = fake.random_int(min=500, max=3000)
                elif category == "Entertainment":
                    amount = fake.random_int(min=200, max=5000)
                elif category == "Travel":
                    amount = fake.random_int(min=500, max=15000)
                else:  # Others
                    amount = fake.random_int(min=100, max=5000)
                
                transaction_type = "Debit"
                current_balance -= amount
            
            # Generate description using Gemini or Faker fallback
            if use_gemini and random.random() < 0.3:  # Use Gemini for 30% of transactions
                description = gemini_gen.generate_description(category, amount, transaction_type)
            else:
                # Use Faker + predefined realistic descriptions
                category_key = category
                if category_key in gemini_gen.fallback_descriptions:
                    base_desc = fake.random_element(gemini_gen.fallback_descriptions[category_key])
                    
                    # Add realistic Indian company names using Faker
                    if category == "Food":
                        companies = ["Swiggy", "Zomato", "McDonald's", "KFC", "Domino's"]
                        description = f"UPI-{fake.random_element(companies)}"
                    elif category == "Shopping":
                        companies = ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa"]
                        description = f"{fake.random_element(companies)} Purchase"
                    elif category == "Travel":
                        companies = ["Ola", "Uber", "IRCTC", "MakeMyTrip", "Goibibo"]
                        description = f"{fake.random_element(companies)} Booking"
                    elif category == "Entertainment":
                        companies = ["BookMyShow", "Netflix", "Amazon Prime", "Hotstar", "Spotify"]
                        description = f"{fake.random_element(companies)} Payment"
                    else:
                        description = base_desc
                else:
                    description = fake.random_element(gemini_gen.fallback_descriptions["Others"])
            
            transaction = {
                "id": f"txn_{transaction_id:06d}",
                "userId": user_id,
                "date": transaction_date.strftime("%Y-%m-%d"),
                "description": description,
                "amount": amount,
                "type": transaction_type,
                "category": category,
                "balance": max(0, current_balance)  # Ensure balance doesn't go negative
            }
            
            all_transactions.append(transaction)
            transaction_id += 1
            
            # Progress indicator
            if i % 50 == 0:
                print(f"   Generated {i}/{num_transactions} transactions...")
    
    # Sort transactions by date using Faker's sorting
    all_transactions.sort(key=lambda x: x["date"])
    
    print(f"✅ Generated {len(all_transactions)} total transactions using Faker + Gemini")
    return all_transactions

def save_transactions_to_file(transactions: List[Dict], filename: str = "data/transactions.json"):
    """Save transactions to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {len(transactions)} transactions to {filename}")
    
    # Print summary
    users = set(t["userId"] for t in transactions)
    for user_id in users:
        user_transactions = [t for t in transactions if t["userId"] == user_id]
        print(f"   👤 {user_id}: {len(user_transactions)} transactions")
    
    # Print category breakdown
    categories = {}
    for txn in transactions:
        cat = txn["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 Category Distribution:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {category}: {count} transactions")

def generate_with_gemini_api_key():
    """Interactive function to generate data with optional Gemini API key."""
    print("🤖 AI-Powered Transaction Data Generator")
    print("=" * 50)
    print("Using: Faker (Python) + Gemini AI")
    print()
    
    # Ask for Gemini API key
    gemini_key = input("🔑 Enter Gemini API key (optional, press Enter to skip): ").strip()
    
    if not gemini_key:
        print("📝 Will use Faker + realistic company names (no AI generation)")
        gemini_key = None
    else:
        print("🚀 Will use Gemini AI for realistic transaction descriptions")
    
    print()
    
    # Generate transactions
    transactions = generate_synthetic_transactions(gemini_key)
    
    # Save to file
    save_transactions_to_file(transactions)
    
    # Show sample transactions
    print(f"\n📋 Sample Generated Transactions:")
    for i, txn in enumerate(transactions[:5], 1):
        print(f"   {i}. {txn['type']} of ₹{txn['amount']} - {txn['description']} ({txn['category']})")
    
    print(f"\n✅ Data generation complete! Check data/transactions.json")
    return transactions

if __name__ == "__main__":
    generate_with_gemini_api_key()
