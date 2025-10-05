import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from nodes.query_parser import QueryIntent
from config import config
from tools.financial_tools import financial_tools

logger = logging.getLogger(__name__)

class GeminiSummarizer:
    """Summarize transaction data using Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.model_name = config.GEMINI_MODEL
        
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
        tool_descriptions = financial_tools.get_tool_descriptions()
        tools_text = "\n".join([f"- {name}: {desc}" for name, desc in tool_descriptions.items()])
        
        return f"""You are a financial assistant AI that analyzes transaction data and provides clear, accurate summaries.

AVAILABLE TOOLS:
{tools_text}

RULES:
1. Use the provided tools to perform accurate calculations and analysis
2. Always use ₹ symbol for amounts and round to nearest rupee
3. Provide specific numbers and totals from tool results
4. Summarize spending patterns, top categories, and notable transactions
5. Be concise but informative
6. End with ONE relevant follow-up question
7. If no data is provided, clearly state that no transactions were found

TOOL USAGE:
When you need to perform calculations or analysis, use the format:
TOOL_CALL: tool_name(param1=value1, param2=value2)

EXAMPLE TOOL USAGE:
TOOL_CALL: calculate_total_spending(transactions=transaction_data, category="Food")
TOOL_CALL: analyze_spending_by_category(transactions=transaction_data)

FORMAT:
- Start with a brief overview using tool results
- List key insights with specific amounts from calculations
- Mention top spending categories with percentages
- Highlight any notable patterns or large transactions
- End with a follow-up question

EXAMPLE:
"Based on your transactions, you spent ₹45,230 across 23 transactions. Your top spending categories were Food & Dining (₹18,500, 41%), Transportation (₹12,200, 27%), and Shopping (₹8,900, 20%). Notable expenses include ₹5,000 for electronics and ₹3,200 for dining. 

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
            
            # Process tool calls if present
            summary = await self._process_tool_calls(summary, transactions)
            
            logger.info("Generated summary using Gemini with tools")
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
    
    async def _process_tool_calls(self, summary: str, transactions: List[Dict[str, Any]]) -> str:
        """Process tool calls in the summary and replace with actual results."""
        import re
        
        # Find all tool calls in the format: TOOL_CALL: tool_name(params)
        tool_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        tool_calls = re.findall(tool_pattern, summary)
        
        for tool_name, params_str in tool_calls:
            try:
                # Parse parameters
                params = self._parse_tool_params(params_str, transactions)
                
                # Execute tool
                if tool_name in financial_tools.tools:
                    result = financial_tools.tools[tool_name](**params)
                    
                    # Replace tool call with result
                    tool_call_text = f"TOOL_CALL: {tool_name}({params_str})"
                    result_text = self._format_tool_result(tool_name, result)
                    summary = summary.replace(tool_call_text, result_text)
                    
                    logger.info(f"Executed tool {tool_name} with result: {result}")
                else:
                    logger.warning(f"Unknown tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                # Replace with error message
                tool_call_text = f"TOOL_CALL: {tool_name}({params_str})"
                summary = summary.replace(tool_call_text, f"[Tool error: {str(e)}]")
        
        return summary
    
    def _parse_tool_params(self, params_str: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse tool parameters from string."""
        params = {}
        
        # Handle transactions parameter specially
        if "transactions=" in params_str:
            params["transactions"] = transactions
            params_str = re.sub(r'transactions=\w+', '', params_str)
        
        # Parse other parameters
        param_pattern = r'(\w+)=([^,]+)'
        matches = re.findall(param_pattern, params_str)
        
        for key, value in matches:
            key = key.strip()
            value = value.strip().strip('"\'')
            
            # Try to convert to appropriate type
            if value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '').isdigit():
                params[key] = float(value)
            elif value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
        
        return params
    
    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool result for inclusion in summary."""
        if "error" in result:
            return f"[Error in {tool_name}: {result['error']}]"
        
        if tool_name == "calculate_total_spending":
            return f"₹{result.get('total_amount', 0):,.0f} across {result.get('transaction_count', 0)} transactions"
        
        elif tool_name == "analyze_spending_by_category":
            categories = result.get('categories', [])[:3]  # Top 3
            category_text = ", ".join([f"{cat['category']} (₹{cat['amount']:,.0f}, {cat['percentage']}%)" for cat in categories])
            return f"Top categories: {category_text}"
        
        elif tool_name == "find_top_expenses":
            expenses = result.get('top_expenses', [])[:3]  # Top 3
            expense_text = ", ".join([f"₹{exp['amount']:,.0f} for {exp['description']}" for exp in expenses])
            return f"Top expenses: {expense_text}"
        
        else:
            # Generic formatting
            return str(result)
    
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
