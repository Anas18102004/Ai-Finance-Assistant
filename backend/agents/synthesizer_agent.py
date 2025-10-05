"""
Synthesizer Agent - Converts raw data into human-friendly responses.
You are a finance assistant responsible for generating human-friendly, conversational responses.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from config import config

logger = logging.getLogger(__name__)

class SynthesizerAgent:
    """Agent for converting raw data into conversational responses."""
    
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
        logger.info("Synthesizer Agent initialized")
    
    async def synthesize_response(self, intent: str, query: str, data: Any = None, 
                                 context: Dict[str, Any] = None) -> str:
        """Synthesize final response based on intent and data."""
        try:
            if intent == "simple_response":
                return await self._handle_simple_response(query, context)
            
            elif intent == "data_query":
                return await self._handle_data_response(query, data, context)
            
            elif intent == "knowledge_query":
                return await self._handle_knowledge_response(query, data, context)
            
            else:
                return "I'm not sure how to help with that. Could you please rephrase your question?"
                
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return "I encountered an error while processing your request. Please try again."
    
    async def _handle_simple_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """Handle simple conversational responses."""
        query_lower = query.lower().strip()
        
        # Check if user is returning (has conversation history)
        is_returning_user = context and context.get("conversation_context")
        
        # Greetings
        if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
            if is_returning_user:
                return "Hello again! Great to have you back! What financial question can I help you with today?"
            else:
                return "Hello! I'm your AI financial assistant. I can help you analyze your expenses, track spending patterns, and answer questions about your transactions. What would you like to explore?"
        
        elif "good morning" in query_lower:
            if is_returning_user:
                return "Good morning! Welcome back! Ready to dive into your financial data again?"
            else:
                return "Good morning! I'm your AI financial assistant. How can I help you with your finances today?"
        
        elif "good afternoon" in query_lower:
            return "Good afternoon! What financial insights can I help you with?"
        
        elif "good evening" in query_lower:
            return "Good evening! Ready to review your financial data? What can I help you with?"
        
        # Personal questions
        elif any(personal in query_lower for personal in ["my name", "who am i", "what's my name"]):
            return "I don't have access to your personal information like your name. I only work with your financial transaction data to help you analyze spending patterns, track expenses, and provide financial insights. What would you like to know about your finances?"
        
        # Capability questions
        elif any(capability in query_lower for capability in ["what can you do", "help", "capabilities"]):
            return """I can help you with various financial tasks:

📊 **Transaction Analysis**
• Find your top expenses and spending patterns
• Calculate totals for specific categories or time periods
• Compare spending across different months

📈 **Financial Insights**
• Analyze spending by category with percentages
• Track spending trends over time
• Identify unusual or large transactions

💰 **Budget & Planning**
• Monthly financial summaries
• Savings rate calculations
• Budget analysis and recommendations

💡 **Financial Knowledge**
• Explain financial terms (GST, EMI, investments, etc.)
• Provide budgeting tips and advice

Just ask me things like "show my top 5 expenses last month" or "how much did I spend on food this month"!"""
        
        # Thanks/conversational
        elif any(thanks in query_lower for thanks in ["thanks", "thank you"]):
            return "You're welcome! How can I help you with your finances today?"
        
        elif query_lower in ["ok", "okay", "cool", "nice", "great", "awesome"]:
            return "Great! What would you like to know about your finances?"
        
        elif any(bye in query_lower for bye in ["bye", "goodbye", "see you"]):
            return "Goodbye! Feel free to ask me about your finances anytime."
        
        # Default simple response following the refined prompt pattern
        elif query_lower == "hi":
            return "Hi! How can I help you with your finances today?"
        
        else:
            return "I'm here to help with your financial questions! What would you like to know?"
    
    async def _handle_data_response(self, query: str, data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> str:
        """Handle responses for data queries."""
        if not data or not data.get("success"):
            error_msg = data.get("message", "No data found") if data else "No data available"
            return f"{error_msg}. Try adjusting your search terms or date range."
        
        operation = data.get("operation", "unknown")
        result_data = data.get("data", {})
        
        if not self.model:
            return self._format_data_fallback(operation, result_data, query)
        
        # Use Gemini to create natural response
        prompt = self._build_data_prompt(query, operation, result_data, data)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating data response: {e}")
            return self._format_data_fallback(operation, result_data, query)
    
    async def _handle_knowledge_response(self, query: str, data: Dict[str, Any], 
                                        context: Dict[str, Any] = None) -> str:
        """Handle responses for knowledge queries."""
        if not data or not data.get("success"):
            return "I don't have specific information about that topic. Could you try asking about common financial terms like GST, EMI, investments, or budgeting?"
        
        return data.get("answer", "I couldn't find information about that topic.")
    
    def _build_data_prompt(self, query: str, operation: str, result_data: Dict[str, Any], 
                          full_data: Dict[str, Any]) -> str:
        """Build prompt for data response synthesis."""
        return f"""You are a finance assistant responsible for generating beautifully formatted, professional responses.

Input:
- Original user query: "{query}"
- Structured results from Data Agent

FORMATTING RULES:
1. Use **bold** for amounts (e.g., **₹16,969**)
2. Use **bold** for important items/descriptions
3. Use numbered lists for top N items (1. 2. 3.)
4. Use bullet points (•) for additional details
5. Use emojis sparingly and appropriately (💰 📊 📈)
6. Keep it professional and well-structured
7. DO NOT include response time in the text - it will be shown separately

DATA FROM DATA AGENT:
Operation: {operation}
Transaction Count: {full_data.get('transaction_count', 0)}
Results: {result_data}

PERFECT FORMATTING EXAMPLES:

User: "Top 3 expenses"
Perfect Response:
"Here are your top 3 expenses:

1. **₹16,969** for **Room rent** on 2025-09-21
2. **₹13,084** for **Ola Booking** on 2025-09-29  
3. **₹4,899** for **Hotstar Payment** on 2025-10-02"

User: "How much did I spend on food?"
Perfect Response:
"Your total food spending: **₹8,450**

• **Restaurant meals**: ₹5,200 (3 transactions)
• **Groceries**: ₹2,100 (2 transactions)  
• **Food delivery**: ₹1,150 (4 transactions)"

User: "Monthly summary"
Perfect Response:
"📊 **Monthly Financial Summary**

**Total Expenses**: ₹45,230
**Total Income**: ₹65,000
**Net Savings**: ₹19,770

**Top Categories:**
• **Food & Dining**: ₹12,500 (28%)
• **Transportation**: ₹8,900 (20%)
• **Entertainment**: ₹6,200 (14%)"

Generate a perfectly formatted, professional response:"""
    
    def _format_data_fallback(self, operation: str, result_data: Dict[str, Any], query: str) -> str:
        """Fallback data formatting when Gemini is not available."""
        if operation == "top_n":
            transactions = result_data.get("transactions", [])
            if not transactions:
                return "No transactions found matching your criteria."
            
            response = f"Here are your top {len(transactions)} transactions:\n\n"
            for i, txn in enumerate(transactions[:5], 1):
                amount = txn.get("amount", 0)
                desc = txn.get("description", "Unknown")
                date = txn.get("date", "")
                response += f"{i}. **₹{amount:,.0f}** for **{desc}** on {date}\n"
            
            return response
        
        elif operation == "total":
            total = result_data.get("total_amount", 0)
            count = result_data.get("transaction_count", 0)
            return f"**Total spending**: ₹{total:,.0f} across {count} transactions"
        
        elif operation == "category_analysis":
            categories = result_data.get("categories", [])[:5]
            if not categories:
                return "No category data available."
            
            response = "📊 **Spending by Category**\n\n"
            for cat in categories:
                name = cat.get("category", "Unknown")
                amount = cat.get("amount", 0)
                percentage = cat.get("percentage", 0)
                response += f"• **{name}**: ₹{amount:,.0f} ({percentage}%)\n"
            
            return response
        
        else:
            return f"Found **{result_data.get('total_found', 0)}** transactions matching your criteria"

# Global instance
synthesizer_agent = SynthesizerAgent()
