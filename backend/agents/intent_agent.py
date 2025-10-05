"""
Intent Agent - Classifies queries and generates structured plans.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from config import config

logger = logging.getLogger(__name__)

class IntentAgent:
    """Agent for query classification and plan generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.model_name = config.GEMINI_MODEL
        
        if api_key:
            self.initialize(api_key)
    
    def initialize(self, api_key: str):
        """Initialize the Gemini model."""
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.api_key = api_key
        logger.info("Intent Agent initialized")
    
    async def classify_and_plan(self, query: str, user_id: str) -> Dict[str, Any]:
        """Classify query and generate structured plan if needed."""
        if not self.model:
            raise ValueError("Intent Agent not initialized")
        
        try:
            prompt = self._build_classification_prompt(query)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            result = self._parse_response(response.text.strip(), query, user_id)
            logger.info(f"Query classified as: {result.get('intent')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            # Fallback to simple response
            return {
                "intent": "simple_response",
                "error": str(e)
            }
    
    def _build_classification_prompt(self, query: str) -> str:
        """Build the classification prompt."""
        return f"""You are a highly intelligent finance assistant responsible for understanding user queries. Your task is to classify each query and, if needed, create a structured query plan for data queries.

QUERY: "{query}"

Rules:
1. Classify the query into one of three intents:
   - simple_response → queries that are general greetings, small talk, or basic info requests.
   - data_query → queries that require structured calculations (top N, totals, specific amounts, filtering).
   - knowledge_query → queries that ask for patterns, insights, analysis, or understanding from transaction data.

2. If the intent is data_query, output a structured JSON query plan including:
   - operation (sum, average, top_n, count, compare_max)
   - filters (category, date_range, type)
   - sort_by (optional)
   - order (optional)
   - n (if top_n)

3. If the intent is knowledge_query, output the topic.

Output Format:
{{
  "intent": "<intent_name>",
  "plan": {{ ... }}, # Only for data_query
  "topic": "<topic>" # Only for knowledge_query
}}

Examples:

User: "Top 3 food expenses last month"
Output:
{{
  "intent": "data_query",
  "plan": {{
    "operation": "top_n",
    "n": 3,
    "filters": {{
      "category": "Food",
      "date_range": "last_month",
      "type": "Debit"
    }},
    "sort_by": "amount",
    "order": "desc"
  }}
}}

User: "What do I spend most on?"
Output:
{{
  "intent": "knowledge_query",
  "topic": "spending_patterns"
}}

User: "Any unusual spending patterns?"
Output:
{{
  "intent": "knowledge_query",
  "topic": "spending_analysis"
}}

User: "Hi, how are you?"
Output:
{{
  "intent": "simple_response"
}}

Respond with ONLY the JSON, no explanation."""

    def _parse_response(self, response_text: str, original_query: str, user_id: str) -> Dict[str, Any]:
        """Parse the Gemini response into structured format."""
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            result = json.loads(json_text)
            
            # Add metadata
            result["original_query"] = original_query
            result["user_id"] = user_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Process date ranges for data queries
            if result.get("intent") == "data_query" and "filters" in result:
                result["filters"] = self._process_date_range(result["filters"])
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Fallback classification based on keywords
            return self._fallback_classification(original_query, user_id)
    
    def _process_date_range(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Process date range filters into actual dates."""
        if "date_range" not in filters:
            return filters
        
        date_range = filters["date_range"]
        now = datetime.now()
        
        if date_range == "last_month":
            # Previous calendar month
            first_day_current = now.replace(day=1)
            last_day_previous = first_day_current - timedelta(days=1)
            first_day_previous = last_day_previous.replace(day=1)
            
            filters["start_date"] = first_day_previous.strftime("%Y-%m-%d")
            filters["end_date"] = last_day_previous.strftime("%Y-%m-%d")
            
        elif date_range == "this_month":
            first_day = now.replace(day=1)
            filters["start_date"] = first_day.strftime("%Y-%m-%d")
            filters["end_date"] = now.strftime("%Y-%m-%d")
            
        elif date_range == "last_week":
            week_ago = now - timedelta(days=7)
            filters["start_date"] = week_ago.strftime("%Y-%m-%d")
            filters["end_date"] = now.strftime("%Y-%m-%d")
            
        elif date_range == "this_week":
            # Monday of current week
            days_since_monday = now.weekday()
            monday = now - timedelta(days=days_since_monday)
            filters["start_date"] = monday.strftime("%Y-%m-%d")
            filters["end_date"] = now.strftime("%Y-%m-%d")
            
        elif date_range == "last_year":
            last_year = now.year - 1
            filters["start_date"] = f"{last_year}-01-01"
            filters["end_date"] = f"{last_year}-12-31"
        
        return filters
    
    def _fallback_classification(self, query: str, user_id: str) -> Dict[str, Any]:
        """Fallback classification when JSON parsing fails."""
        query_lower = query.lower().strip()
        
        # Simple response keywords
        simple_keywords = [
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", 
            "bye", "goodbye", "help", "what can you do", "who am i",
            "what's my name", "how are you"
        ]
        
        if any(keyword in query_lower for keyword in simple_keywords):
            return {
                "intent": "simple_response",
                "original_query": query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        
        # Knowledge query keywords (patterns and insights)
        knowledge_keywords = [
            "what do i spend", "spending pattern", "financial pattern", 
            "unusual transaction", "spending habit", "financial behavior",
            "insights", "analysis", "trends", "patterns"
        ]
        
        if any(keyword in query_lower for keyword in knowledge_keywords):
            return {
                "intent": "knowledge_query",
                "topic": "spending_patterns",
                "original_query": query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        
        # Default to data query for financial terms
        return {
            "intent": "data_query",
            "operation": "filter",
            "filters": {"type": "Debit"},
            "parameters": {"n": 10, "sort_by": "date", "order": "desc"},
            "original_query": query,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

# Global instance
intent_agent = IntentAgent()
