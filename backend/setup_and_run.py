#!/usr/bin/env python3
"""
Setup and run script for the AI Financial Assistant.
This script will generate data, build the index, and start the API server.
"""

import asyncio
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_synthetic_transactions, save_transactions_to_file
from index_build.build_index import index_builder
from services.embeddings import embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_system():
    """Setup the complete system."""
    
    print("🚀 Setting up AI Financial Assistant...")
    print("=" * 50)
    
    try:
        # Step 1: Generate synthetic data
        print("\n📊 Step 1: Generating synthetic transaction data...")
        if not os.path.exists("data/transactions.json") or os.path.getsize("data/transactions.json") < 100:
            # Ask for Gemini API key for data generation
            gemini_key = input("🔑 Enter Gemini API key for AI-generated descriptions (optional, press Enter to skip): ").strip()
            if not gemini_key:
                gemini_key = None
                print("📝 Using Faker + realistic company names")
            
            transactions = generate_synthetic_transactions(gemini_key)
            save_transactions_to_file(transactions)
            print(f"✅ Generated {len(transactions)} transactions")
        else:
            print("✅ Transaction data already exists")
        
        # Step 2: Initialize embedding service
        print("\n🤖 Step 2: Initializing embedding service...")
        await embedding_service.initialize()
        print("✅ Embedding service initialized")
        
        # Step 3: Build vector index
        print("\n🔍 Step 3: Building vector index...")
        await index_builder.rebuild_index()
        stats = index_builder.get_collection_stats()
        print(f"✅ Index built with {stats['total_documents']} documents")
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 System Summary:")
        print(f"   • Documents indexed: {stats['total_documents']}")
        print(f"   • Sample users: {', '.join(stats['sample_users'])}")
        print(f"   • Sample categories: {', '.join(stats['sample_categories'])}")
        
        print("\n🌐 Starting API server...")
        print("   • API will be available at: http://localhost:8000")
        print("   • Interactive docs at: http://localhost:8000/docs")
        print("   • Health check at: http://localhost:8000/health")
        
        print("\n🔑 API Usage:")
        print("   • For summarization, include Gemini API key in header:")
        print("     X-Gemini-API-Key: your_api_key_here")
        
        print("\n💡 Example queries:")
        print("   • 'Show me my top 5 expenses this month'")
        print("   • 'How much did I spend on food in September?'")
        print("   • 'Find transactions above ₹5000'")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return False

def run_api_server():
    """Run the API server."""
    import uvicorn
    from api.app import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

async def main():
    """Main function."""
    success = await setup_system()
    
    if success:
        print("\n" + "=" * 50)
        print("🚀 Starting API server...")
        run_api_server()
    else:
        print("\n❌ Setup failed. Please check the logs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
