import asyncio
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from nodes.query_parser import QueryIntent

logger = logging.getLogger(__name__)

class GeminiSummarizer:
    """Summarize transaction data using Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.model_name = "gemini-pro"
        
    def initialize(self, api_key: str):
        """Initialize Gemini with API key."""
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.api_key = api_key
        logger.info("Gemini model initialized")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for Gemini."""
        return """You are a financial assistant AI that analyzes transaction data and provides clear, accurate summaries.

RULES:
1. ONLY use data provided in the user's message - never hallucinate or make up information
2. Always use ₹ symbol for amounts and round to nearest rupee
3. Provide specific numbers and totals when available
4. Summarize spending patterns, top categories, and notable transactions
5. Be concise but informative
6. End with ONE relevant follow-up question
7. If no data is provided, clearly state that no transactions were found

FORMAT:
- Start with a brief overview
- List key insights with specific amounts
- Mention top spending categories
- Highlight any notable patterns or large transactions
- End with a follow-up question

EXAMPLE:
"Based on your transactions, you spent ₹45,230 across 23 transactions. Your top spending categories were Food & Dining (₹18,500), Transportation (₹12,200), and Shopping (₹8,900). Notable expenses include ₹5,000 for electronics and ₹3,200 for dining. 

Would you like to see a breakdown of your food expenses for this period?"
"""
    
    async def summarize_transactions(
        self, 
        transactions: List[Dict[str, Any]], 
        query_intent: QueryIntent,
        aggregated_data: Optional[Dict[str, Any]] = None,
        original_query: str = ""
    ) -> str:
        """Generate a summary of transactions using Gemini."""
        
        if not self.model:
            return "Error: Gemini model not initialized. Please provide API key."
        
        if not transactions and not aggregated_data:
            return "No transactions found matching your criteria. Try adjusting your search terms or date range."
        
        # Build the prompt based on query intent
        prompt = self._build_prompt(transactions, query_intent, aggregated_data, original_query)
        
        try:
            # Generate response using Gemini
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            summary = response.text.strip()
            logger.info("Generated summary using Gemini")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with Gemini: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _build_prompt(
        self, 
        transactions: List[Dict[str, Any]], 
        query_intent: QueryIntent,
        aggregated_data: Optional[Dict[str, Any]],
        original_query: str
    ) -> str:
        """Build the prompt for Gemini based on query intent."""
        
        system_prompt = self._build_system_prompt()
        
        # Build context based on intent
        if query_intent.intent == "sum_spent" and aggregated_data:
            context = self._build_aggregation_context(aggregated_data, query_intent)
        else:
            context = self._build_transaction_context(transactions, query_intent)
        
        prompt = f"""{system_prompt}

USER QUERY: "{original_query}"
QUERY INTENT: {query_intent.intent}

TRANSACTION DATA:
{context}

Please provide a clear, accurate summary based on this data."""
        
        return prompt
    
    def _build_transaction_context(self, transactions: List[Dict[str, Any]], query_intent: QueryIntent) -> str:
        """Build context from individual transactions."""
        if not transactions:
            return "No transactions found."
        
        context = f"Found {len(transactions)} transactions:\n\n"
        
        # Add transaction details
        for i, txn in enumerate(transactions[:10], 1):  # Limit to top 10 for context
            context += f"{i}. {txn['date']} - {txn['type'].title()} of ₹{txn['amount']:.0f} "
            context += f"for {txn['description']} ({txn['category']})\n"
        
        if len(transactions) > 10:
            context += f"\n... and {len(transactions) - 10} more transactions\n"
        
        # Add summary statistics
        total_amount = sum(t['amount'] for t in transactions if t['type'] == 'debit')
        credit_amount = sum(t['amount'] for t in transactions if t['type'] == 'credit')
        
        context += f"\nSUMMARY:\n"
        context += f"- Total expenses: ₹{total_amount:.0f}\n"
        if credit_amount > 0:
            context += f"- Total income: ₹{credit_amount:.0f}\n"
        
        # Category breakdown
        categories = {}
        for txn in transactions:
            if txn['type'] == 'debit':  # Only count expenses
                cat = txn['category']
                categories[cat] = categories.get(cat, 0) + txn['amount']
        
        if categories:
            context += f"- Top categories: "
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            cat_strs = [f"{cat} (₹{amount:.0f})" for cat, amount in sorted_cats]
            context += ", ".join(cat_strs)
        
        return context
    
    def _build_aggregation_context(self, aggregated_data: Dict[str, Any], query_intent: QueryIntent) -> str:
        """Build context from aggregated data."""
        context = f"AGGREGATED DATA:\n"
        context += f"- Total amount spent: ₹{aggregated_data['total_amount']:.0f}\n"
        context += f"- Number of transactions: {aggregated_data['transaction_count']}\n"
        
        if aggregated_data['categories']:
            context += f"\nCATEGORY BREAKDOWN:\n"
            for category, data in list(aggregated_data['categories'].items())[:5]:
                context += f"- {category}: ₹{data['total']:.0f} ({data['count']} transactions)\n"
        
        if 'filters_applied' in aggregated_data and aggregated_data['filters_applied']:
            context += f"\nFILTERS APPLIED: {aggregated_data['filters_applied']}\n"
        
        return context
    
    def _generate_follow_up_question(self, query_intent: QueryIntent, transactions: List[Dict[str, Any]]) -> str:
        """Generate a relevant follow-up question."""
        
        if query_intent.intent == "top_expenses":
            return "Would you like to see expenses for a specific category or time period?"
        elif query_intent.intent == "sum_spent":
            return "Would you like to compare this with your spending in a different period?"
        elif query_intent.intent == "filter":
            return "Would you like to see more details about any specific transactions?"
        elif query_intent.intent == "compare":
            return "Would you like to analyze trends in your spending patterns?"
        else:
            # Generate based on available data
            if transactions:
                categories = list(set(t['category'] for t in transactions))
                if len(categories) > 1:
                    return f"Would you like to focus on {categories[0]} or {categories[1]} expenses?"
                else:
                    return "Would you like to see transactions from a different time period?"
            else:
                return "Would you like to try a different search or time period?"

# Global summarizer instance
summarizer = GeminiSummarizer()
