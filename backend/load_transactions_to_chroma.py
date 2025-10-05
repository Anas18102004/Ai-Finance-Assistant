import json
import os
import chromadb
from typing import List, Dict, Any
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_transactions_to_chroma():
    """
    Load transaction data from JSON file into ChromaDB cloud instance.
    """
    # Check if transaction data exists
    data_path = os.path.join(os.path.dirname(__file__), "data", "transactions.json")
    
    if not os.path.exists(data_path):
        print("❌ Transaction data not found. Please run generate_data.py first.")
        return False
    
    try:
        # Load transaction data
        with open(data_path, 'r') as f:
            transactions = json.load(f)
        
        print(f"✅ Loaded {len(transactions)} transactions from {data_path}")
        
        # Initialize ChromaDB client using environment variables
        client = chromadb.CloudClient(
            api_key=os.getenv('CHROMA_CLOUD_API_KEY'),
            tenant=os.getenv('CHROMA_CLOUD_TENANT'),
            database=os.getenv('CHROMA_CLOUD_DATABASE')
        )
        
        print("✅ Connected to ChromaDB cloud")
        
        # Create or get collection
        collection_name = "transactions"
        try:
            collection = client.get_collection(name=collection_name)
            print(f"✅ Using existing collection: {collection_name}")
        except Exception:
            collection = client.create_collection(name=collection_name)
            print(f"✅ Created new collection: {collection_name}")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for transaction in transactions:
            # Create document text from transaction description and category
            document = f"{transaction['description']} - {transaction['category']}"
            documents.append(document)
            
            # Store all transaction data as metadata
            metadatas.append(transaction)
            
            # Generate unique ID for each transaction
            ids.append(str(uuid.uuid4()))
        
        # Add data to collection in batches to avoid potential size limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"✅ Added batch {i//batch_size + 1} ({end_idx - i} transactions)")
        
        print(f"✅ Successfully added {len(documents)} transactions to ChromaDB")
        return True
        
    except Exception as e:
        print(f"❌ Error loading transactions to ChromaDB: {e}")
        return False

if __name__ == "__main__":
    load_transactions_to_chroma()