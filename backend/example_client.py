#!/usr/bin/env python3
"""
Example client for the AI Financial Assistant API.
Demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class FinancialAssistantClient:
    """Client for interacting with the AI Financial Assistant API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", gemini_api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.gemini_api_key = gemini_api_key
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        if gemini_api_key:
            self.session.headers.update({
                "X-Gemini-API-Key": gemini_api_key
            })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def query_transactions(
        self, 
        user_id: str, 
        query: str, 
        summarize: bool = True, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query transactions using natural language."""
        
        payload = {
            "user_id": user_id,
            "query": query,
            "summarize": summarize,
            "top_k": top_k,
            "filters": filters or {}
        }
        
        try:
            response = self.session.post(f"{self.base_url}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate new synthetic transaction data."""
        try:
            response = self.session.post(f"{self.base_url}/generate")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def build_index(self) -> Dict[str, Any]:
        """Build or rebuild the vector index."""
        try:
            response = self.session.post(f"{self.base_url}/index/build")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_examples(self) -> Dict[str, Any]:
        """Get example queries."""
        try:
            response = self.session.get(f"{self.base_url}/examples")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

def demo_queries(client: FinancialAssistantClient):
    """Run demo queries to showcase the system."""
    
    print("🎯 Running Demo Queries")
    print("=" * 40)
    
    # Test queries
    demo_queries_list = [
        {
            "user_id": "user_001",
            "query": "Show me my top 5 expenses this month",
            "description": "Top expenses with time filter"
        },
        {
            "user_id": "user_001", 
            "query": "How much did I spend on food?",
            "description": "Sum spending by category"
        },
        {
            "user_id": "user_002",
            "query": "Find transactions above ₹5000",
            "description": "Filter by amount range"
        },
        {
            "user_id": "user_002",
            "query": "Show my entertainment expenses",
            "description": "Filter by category"
        },
        {
            "user_id": "user_003",
            "query": "What are my recent shopping transactions?",
            "description": "General search with category"
        }
    ]
    
    for i, demo_query in enumerate(demo_queries_list, 1):
        print(f"\n{i}. {demo_query['description']}")
        print(f"   Query: '{demo_query['query']}'")
        print(f"   User: {demo_query['user_id']}")
        
        start_time = time.time()
        
        # Query with summarization if API key available
        result = client.query_transactions(
            user_id=demo_query["user_id"],
            query=demo_query["query"],
            summarize=bool(client.gemini_api_key),
            top_k=5
        )
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        if result.get("status") == "success":
            print(f"   ✅ Success ({latency:.1f}ms)")
            print(f"   📊 Found: {len(result.get('retrieved', []))} transactions")
            print(f"   🎯 Intent: {result.get('intent', 'unknown')}")
            
            if result.get("summary"):
                print(f"   💡 Summary: {result['summary'][:100]}...")
            
            # Show top transaction
            if result.get("retrieved"):
                top_txn = result["retrieved"][0]
                print(f"   🔝 Top result: ₹{top_txn['amount']:.0f} - {top_txn['description']}")
        
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        time.sleep(0.5)  # Small delay between queries

def setup_system(client: FinancialAssistantClient):
    """Setup the system by generating data and building index."""
    
    print("🔧 Setting up system...")
    
    # Check if system is already set up
    stats = client.get_stats()
    if stats.get("status") == "success" and stats.get("database_stats", {}).get("total_documents", 0) > 0:
        print("✅ System already set up")
        return True
    
    # Generate data
    print("📊 Generating synthetic data...")
    result = client.generate_data()
    if result.get("status") != "success":
        print(f"❌ Data generation failed: {result.get('error')}")
        return False
    
    print(f"✅ Generated {result.get('transaction_count', 0)} transactions")
    
    # Build index
    print("🔍 Building vector index...")
    result = client.build_index()
    if result.get("status") != "success":
        print(f"❌ Index building failed: {result.get('error')}")
        return False
    
    print(f"✅ Indexed {result.get('documents_indexed', 0)} documents")
    return True

def main():
    """Main demo function."""
    
    print("🤖 AI Financial Assistant - Demo Client")
    print("=" * 50)
    
    # Get Gemini API key from user (optional)
    gemini_api_key = input("\n🔑 Enter Gemini API key (optional, press Enter to skip): ").strip()
    if not gemini_api_key:
        gemini_api_key = None
        print("⚠️  Summarization will be disabled without API key")
    
    # Initialize client
    client = FinancialAssistantClient(gemini_api_key=gemini_api_key)
    
    # Health check
    print("\n🏥 Checking API health...")
    health = client.health_check()
    
    if health.get("status") != "healthy":
        print("❌ API is not healthy. Make sure the server is running:")
        print("   python setup_and_run.py")
        return
    
    print("✅ API is healthy")
    
    # Setup system
    if not setup_system(client):
        print("❌ System setup failed")
        return
    
    # Get system stats
    print("\n📊 System Statistics:")
    stats = client.get_stats()
    if stats.get("status") == "success":
        db_stats = stats.get("database_stats", {})
        print(f"   • Documents indexed: {db_stats.get('total_documents', 0)}")
        print(f"   • Cache size: {stats.get('cache_size', 0)}")
        print(f"   • Sample users: {', '.join(db_stats.get('sample_users', []))}")
    
    # Run demo queries
    demo_queries(client)
    
    # Interactive mode
    print("\n🎮 Interactive Mode")
    print("Enter queries or 'quit' to exit:")
    
    while True:
        try:
            query = input("\n💬 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            user_id = input("👤 User ID (default: user_001): ").strip() or "user_001"
            
            result = client.query_transactions(
                user_id=user_id,
                query=query,
                summarize=bool(client.gemini_api_key)
            )
            
            if result.get("status") == "success":
                print(f"\n✅ Found {len(result.get('retrieved', []))} transactions")
                
                if result.get("summary"):
                    print(f"💡 {result['summary']}")
                else:
                    # Show top results without summary
                    for i, txn in enumerate(result.get("retrieved", [])[:3], 1):
                        print(f"   {i}. ₹{txn['amount']:.0f} - {txn['description']} ({txn['category']})")
            else:
                print(f"❌ Error: {result.get('error')}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for using AI Financial Assistant!")

if __name__ == "__main__":
    main()
