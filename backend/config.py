"""
Configuration management for AI Financial Assistant.
Loads settings from environment variables and .env file.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the AI Financial Assistant."""
    
    # Gemini API Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    DATA_FILE_PATH: str = os.getenv("DATA_FILE_PATH", "data/transactions.json")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    
    # Performance Configuration
    QUERY_CACHE_SIZE: int = int(os.getenv("QUERY_CACHE_SIZE", "100"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_RETRIEVAL_RESULTS", "100"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Data Generation Configuration
    TRANSACTIONS_PER_USER_MIN: int = int(os.getenv("TRANSACTIONS_PER_USER_MIN", "150"))
    TRANSACTIONS_PER_USER_MAX: int = int(os.getenv("TRANSACTIONS_PER_USER_MAX", "200"))
    NUM_USERS: int = int(os.getenv("NUM_USERS", "3"))
    USE_GEMINI_FOR_DATA_GENERATION: bool = os.getenv("USE_GEMINI_FOR_DATA_GENERATION", "false").lower() == "true"
    
    @classmethod
    def get_gemini_api_key(cls) -> Optional[str]:
        """Get Gemini API key from environment or return None."""
        return cls.GEMINI_API_KEY if cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here" else None
    
    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if Gemini API key is properly configured."""
        api_key = cls.get_gemini_api_key()
        return api_key is not None and len(api_key) > 10
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("🔧 AI Financial Assistant Configuration:")
        print(f"   • Gemini Model: {cls.GEMINI_MODEL}")
        print(f"   • Gemini Configured: {'✅' if cls.is_gemini_configured() else '❌'}")
        print(f"   • Chroma DB Path: {cls.CHROMA_DB_PATH}")
        print(f"   • Data File: {cls.DATA_FILE_PATH}")
        print(f"   • API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"   • Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"   • Batch Size: {cls.EMBEDDING_BATCH_SIZE}")
        print(f"   • Cache Size: {cls.QUERY_CACHE_SIZE}")
        print(f"   • Log Level: {cls.LOG_LEVEL}")

# Global config instance
config = Config()

# Convenience functions
def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key."""
    return config.get_gemini_api_key()

def is_gemini_configured() -> bool:
    """Check if Gemini is configured."""
    return config.is_gemini_configured()

def print_startup_info():
    """Print startup configuration info."""
    print("🚀 Starting AI Financial Assistant...")
    config.print_config()
    
    if not is_gemini_configured():
        print("\n⚠️  Gemini API key not configured!")
        print("   • Set GEMINI_API_KEY in .env file")
        print("   • Or pass API key in request headers")
        print("   • Get key from: https://makersuite.google.com/app/apikey")
