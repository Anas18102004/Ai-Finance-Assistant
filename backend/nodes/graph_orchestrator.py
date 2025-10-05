import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
from nodes.query_parser import query_parser, QueryIntent
from nodes.retriever import retriever
from nodes.summarizer import summarizer

logger = logging.getLogger(__name__)

class GraphState:
    """State object that flows through the graph nodes."""
    
    def __init__(self, user_id: str, query: str, **kwargs):
        self.user_id = user_id
        self.query = query
        self.query_intent: Optional[QueryIntent] = None
        self.retrieved_transactions: List[Dict[str, Any]] = []
        self.aggregated_data: Optional[Dict[str, Any]] = None
        self.summary: str = ""
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = kwargs
        self.start_time = time.time()
        
    def add_timing(self, step: str):
        """Add timing information."""
        if "timings" not in self.metadata:
            self.metadata["timings"] = {}
        self.metadata["timings"][step] = time.time() - self.start_time

class FinancialAssistantGraph:
    """LangGraph-style orchestrator for the financial assistant."""
    
    def __init__(self):
        self.nodes = {
            "parse_query": self.parse_query_node,
            "retrieve": self.retrieve_node,
            "summarize": self.summarize_node
        }
        # Simple memory system - stores last 5 conversations per user
        self.conversation_memory: Dict[str, deque] = {}
        self.max_memory_size = 5
    
    def _get_user_memory(self, user_id: str) -> deque:
        """Get or create memory deque for a user."""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = deque(maxlen=self.max_memory_size)
        return self.conversation_memory[user_id]
    
    def _add_to_memory(self, user_id: str, query: str, response: str, intent: str):
        """Add a conversation to user's memory."""
        memory = self._get_user_memory(user_id)
        memory.append({
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "intent": intent
        })
    
    def _get_context_from_memory(self, user_id: str) -> str:
        """Get context from recent conversations."""
        memory = self._get_user_memory(user_id)
        if not memory:
            return ""
        
        context_parts = []
        for conv in list(memory)[-3:]:  # Last 3 conversations for context
            context_parts.append(f"Previous: '{conv['query']}' -> {conv['intent']}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    async def _classify_query_with_gemini(self, state: GraphState, gemini_api_key: str) -> bool:
        """Use Gemini to classify if query needs RAG retrieval."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            classification_prompt = f"""You are a financial assistant AI. Analyze this user query and respond appropriately.

QUERY: "{state.query}"

If the query asks about specific transactions, expenses, spending data, financial analysis, or requires access to transaction history, respond with: "RAG_NEEDED"

If the query is conversational (greetings, personal questions, capabilities, thanks, etc.) that doesn't need transaction data, respond directly with a helpful answer.

Examples:
- "hello" → "Hello! I'm your AI financial assistant. I can help analyze your expenses, track spending patterns, and answer questions about your transactions. What would you like to explore?"
- "what's my name" → "I don't have access to your personal information. I only work with your financial transaction data to help analyze spending patterns and expenses. What would you like to know about your finances?"
- "show me expenses" → "RAG_NEEDED"

Response:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(classification_prompt)
            )
            
            response_text = response.text.strip()
            logger.info(f"Gemini response for '{state.query}': {response_text[:100]}...")
            
            if "RAG_NEEDED" in response_text.upper():
                return True
            else:
                # Store the direct response for use
                state.metadata["gemini_direct_response"] = response_text
                return False
            
        except Exception as e:
            logger.error(f"Error in Gemini classification: {e}")
            # Default to RAG if classification fails
            return True
    
    async def _generate_simple_response(self, state: GraphState, gemini_api_key: str, context: str) -> str:
        """Generate simple response using Gemini without RAG."""
        # Use the direct response from classification if available
        if "gemini_direct_response" in state.metadata:
            return state.metadata["gemini_direct_response"]
        
        # Fallback to generating a new response (shouldn't happen with new approach)
        return "I'm your AI financial assistant. How can I help you with your financial data today?"
    
    def _classify_basic_intent(self, query: str) -> str:
        """Basic intent classification for fallback when no Gemini API key."""
        query_lower = query.lower().strip()
        
        # Direct checks for common queries
        if query_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
            return "greeting"
        
        if query_lower in ["my name", "whats my name", "what's my name", "what is my name", "who am i"]:
            return "personal_info"
            
        if query_lower in ["thanks", "thank you", "ok", "okay", "bye", "goodbye"]:
            return "conversational"
            
        # If it contains financial keywords, assume it needs RAG
        financial_keywords = ["spend", "expense", "transaction", "money", "budget", "cost", "price", "amount"]
        if any(keyword in query_lower for keyword in financial_keywords):
            return "financial"
            
        return "conversational"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_users": len(self.conversation_memory),
            "memory_size_per_user": self.max_memory_size,
            "users_with_memory": [user_id for user_id, memory in self.conversation_memory.items() if len(memory) > 0]
        }
        
    async def run(
        self, 
        user_id: str, 
        query: str, 
        summarize: bool = True,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete graph workflow."""
        
        # Initialize state
        state = GraphState(
            user_id=user_id,
            query=query,
            summarize=summarize,
            top_k=top_k,
            filters=filters or {},
            gemini_api_key=gemini_api_key
        )
        
        try:
            # Get conversation context
            context = self._get_context_from_memory(user_id)
            state.metadata["conversation_context"] = context
            
            # Use Gemini to classify query and decide if RAG is needed
            if gemini_api_key:
                needs_rag = await self._classify_query_with_gemini(state, gemini_api_key)
                
                if not needs_rag:
                    # Handle with simple Gemini response (no RAG)
                    state.summary = await self._generate_simple_response(state, gemini_api_key, context)
                    response = self._build_conversational_response(state)
                    response["metadata"]["gemini_classification"] = "SIMPLE"
                    response["metadata"]["rag_used"] = False
                    self._add_to_memory(user_id, query, state.summary, "conversational")
                    return response
            else:
                # Fallback to basic pattern matching when no Gemini API key
                basic_intent = self._classify_basic_intent(query)
                if basic_intent in ["greeting", "conversational", "personal_info"]:
                    state.summary = self._generate_conversational_response(basic_intent, query, context)
                    response = self._build_conversational_response(state)
                    self._add_to_memory(user_id, query, state.summary, basic_intent)
                    return response
            
            # Execute nodes in sequence for RAG-based queries
            state = await self.parse_query_node(state)
            if state.error:
                response = self._build_error_response(state)
                self._add_to_memory(user_id, query, response.get("error", ""), "error")
                return response
            
            state = await self.retrieve_node(state)
            if state.error:
                response = self._build_error_response(state)
                self._add_to_memory(user_id, query, response.get("error", ""), "error")
                return response
            
            if summarize and gemini_api_key:
                state = await self.summarize_node(state)
            
            # Build final response
            response = self._build_success_response(state)
            if gemini_api_key:
                response["metadata"]["gemini_classification"] = "RAG"
                response["metadata"]["rag_used"] = True
            response_text = state.summary or f"Found {len(state.retrieved_transactions)} transactions"
            self._add_to_memory(user_id, query, response_text, state.query_intent.intent if state.query_intent else "unknown")
            return response
            
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            state.error = str(e)
            return self._build_error_response(state)
    
    async def parse_query_node(self, state: GraphState) -> GraphState:
        """Parse the user query and extract intent."""
        try:
            logger.info(f"Parsing query: {state.query}")
            
            # Parse the query
            query_intent = query_parser.parse_query(state.query, state.user_id)
            
            # Override top_k if provided in metadata
            if "top_k" in state.metadata:
                query_intent.top_k = state.metadata["top_k"]
            
            # Apply additional filters if provided
            if "filters" in state.metadata and state.metadata["filters"]:
                query_intent.filters.update(state.metadata["filters"])
            
            state.query_intent = query_intent
            state.add_timing("parse_query")
            
            logger.info(f"Parsed intent: {query_intent.intent}, filters: {query_intent.filters}")
            
        except Exception as e:
            logger.error(f"Error in parse_query_node: {e}")
            state.error = f"Query parsing failed: {str(e)}"
        
        return state
    
    async def retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve relevant transactions."""
        try:
            if not state.query_intent:
                state.error = "No query intent available"
                return state
            
            logger.info(f"Retrieving transactions for intent: {state.query_intent.intent}")
            
            # For sum_spent queries, get aggregated data
            if state.query_intent.intent == "sum_spent":
                state.aggregated_data = await retriever.get_aggregated_data(state.query_intent)
                state.add_timing("retrieve_aggregated")
                
                # Also get some sample transactions for context
                state.retrieved_transactions = await retriever.retrieve(
                    state.query_intent, 
                    state.query
                )
            else:
                # Regular retrieval
                state.retrieved_transactions = await retriever.retrieve(
                    state.query_intent, 
                    state.query
                )
            
            state.add_timing("retrieve")
            
            logger.info(f"Retrieved {len(state.retrieved_transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error in retrieve_node: {e}")
            state.error = f"Retrieval failed: {str(e)}"
        
        return state
    
    async def summarize_node(self, state: GraphState) -> GraphState:
        """Generate summary using Gemini."""
        try:
            if not state.metadata.get("gemini_api_key"):
                state.error = "Gemini API key required for summarization"
                return state
            
            # Initialize summarizer if needed
            if not summarizer.model:
                summarizer.initialize(state.metadata["gemini_api_key"])
            
            logger.info("Generating summary with Gemini")
            
            # Generate summary
            state.summary = await summarizer.summarize_transactions(
                state.retrieved_transactions,
                state.query_intent,
                state.aggregated_data,
                state.query
            )
            
            state.add_timing("summarize")
            
            logger.info("Summary generated successfully")
            
        except Exception as e:
            logger.error(f"Error in summarize_node: {e}")
            state.error = f"Summarization failed: {str(e)}"
        
        return state
    
    def _generate_conversational_response(self, intent: str, query: str, context: str = "") -> str:
        """Generate appropriate response for greetings and conversational queries."""
        query_lower = query.lower().strip()
        
        # Check if user is returning (has context)
        is_returning_user = bool(context)
        
        if intent == "greeting":
            if is_returning_user:
                # Returning user greetings
                if "morning" in query_lower:
                    return "Good morning! Welcome back! Ready to dive into your financial data again? What would you like to explore today?"
                elif "afternoon" in query_lower:
                    return "Good afternoon! Nice to see you again! What financial insights can I help you with this time?"
                elif "evening" in query_lower:
                    return "Good evening! Back for more financial analysis? What can I help you discover today?"
                elif any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
                    return "Hello again! Great to have you back! What financial question can I help you with today?"
                else:
                    return "Welcome back! What would you like to explore about your finances today?"
            else:
                # First-time user greetings
                if "morning" in query_lower:
                    return "Good morning! I'm your AI financial assistant. I can help you analyze your expenses, track spending patterns, and answer questions about your transactions. What would you like to explore?"
                elif "afternoon" in query_lower:
                    return "Good afternoon! I'm here to help you analyze your expenses and financial data. What would you like to know?"
                elif "evening" in query_lower:
                    return "Good evening! Ready to dive into your financial insights? What can I help you with?"
                elif any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
                    return "Hello! I'm your AI financial assistant. I can help you analyze your expenses, track spending patterns, and answer questions about your transactions. What would you like to explore?"
                elif "how are you" in query_lower:
                    return "I'm doing great, thank you! I'm here and ready to help you with your financial analysis. How can I assist you today?"
                elif "what's up" in query_lower or "whats up" in query_lower:
                    return "Not much, just ready to help you understand your finances better! What would you like to know about your spending or transactions?"
                else:
                    return "Hello! I'm your AI financial assistant. I can help you analyze expenses, track budgets, and provide insights about your financial data. What can I help you with?"
        
        elif intent == "personal_info":
            return "I don't have access to your personal information like your name, age, or contact details. I only work with your financial transaction data to help you analyze spending patterns, track expenses, and provide financial insights. What would you like to know about your finances?"
        
        elif intent == "conversational":
            if any(thanks in query_lower for thanks in ["thanks", "thank you"]):
                return "You're welcome! Feel free to ask me anything about your finances - expenses, budgets, spending patterns, or specific transactions."
            elif query_lower in ["ok", "okay", "cool", "nice", "great", "awesome"]:
                return "Great! Is there anything specific about your finances you'd like to explore? I can help with expense analysis, budget tracking, or transaction searches."
            elif query_lower in ["yes", "sure", "alright"]:
                return "Perfect! What financial question can I help you with? You can ask about spending patterns, specific transactions, or budget analysis."
            elif query_lower in ["no", "maybe"]:
                return "No problem! I'm here whenever you need help with your financial data. Just ask about expenses, transactions, or any financial insights you'd like."
            elif any(bye in query_lower for bye in ["bye", "goodbye", "see you", "talk later"]):
                return "Goodbye! Feel free to come back anytime you need help with your financial analysis. Have a great day!"
            elif "who are you" in query_lower:
                return "I'm your AI financial assistant! I can analyze your transaction data, help you understand spending patterns, track expenses by category, and provide insights about your financial habits. What would you like to know?"
            elif "what can you do" in query_lower:
                return "I can help you with many financial tasks:\n• Analyze your spending patterns\n• Find specific transactions\n• Calculate totals by category or time period\n• Track your top expenses\n• Compare spending across different periods\n• Provide budget insights\n\nJust ask me something like 'Show me my food expenses' or 'How much did I spend last month?'"
            elif query_lower == "help":
                return "I'm here to help with your financial data! You can ask me things like:\n• 'Show me my top 5 expenses'\n• 'How much did I spend on food?'\n• 'Find transactions above ₹5000'\n• 'What's my spending this month?'\n\nWhat would you like to explore?"
            elif any(name_q in query_lower for name_q in ["what's my name", "what is my name", "whats my name", "my name"]):
                return "I don't have access to your personal information like your name. I only work with your financial transaction data to help you analyze spending patterns and expenses. What would you like to know about your finances?"
            elif "do you know my name" in query_lower:
                return "No, I don't have access to your personal details. I focus on helping you analyze your financial data - transactions, expenses, and spending patterns. What financial insights can I help you with?"
            elif "who am i" in query_lower:
                return "I can see you as a user with financial transaction data, but I don't have access to your personal information. I'm here to help analyze your spending patterns and financial habits. What would you like to explore about your finances?"
            elif any(personal in query_lower for personal in ["my age", "my birthday", "my email", "my phone"]):
                return "I don't have access to your personal information. I only work with your financial transaction data to provide spending insights and expense analysis. What would you like to know about your transactions?"
            elif any(about_me in query_lower for about_me in ["tell me about myself", "tell me about me", "what do you know about me"]):
                return "I only know about your financial transaction patterns - your spending habits, expense categories, and transaction history. I don't have access to personal information. Would you like me to analyze your spending patterns or find specific transactions?"
            elif any(profile in query_lower for profile in ["my profile", "my details", "my info"]):
                return "I don't have access to your personal profile or details. I only work with your financial transaction data to help with expense analysis and spending insights. What financial question can I help you with?"
            else:
                return "I understand! Is there anything about your finances you'd like to explore? I can help analyze expenses, find transactions, or provide spending insights."
        
        return "I'm your AI financial assistant. How can I help you with your financial data today?"
    
    def _build_conversational_response(self, state: GraphState) -> Dict[str, Any]:
        """Build response for conversational queries without RAG retrieval."""
        user_memory = self._get_user_memory(state.user_id)
        return {
            "status": "success",
            "query": state.query,
            "user_id": state.user_id,
            "intent": state.query_intent.intent if state.query_intent else "conversational",
            "retrieved": [],  # No documents retrieved for conversational queries
            "summary": state.summary,
            "metadata": {
                "total_retrieved": 0,
                "query_intent": state.query_intent.dict() if state.query_intent else None,
                "timings": state.metadata.get("timings", {}),
                "total_time_ms": (time.time() - state.start_time) * 1000,
                "conversational": True,  # Flag to indicate this was a conversational response
                "conversation_context": state.metadata.get("conversation_context", ""),
                "memory_size": len(user_memory)
            }
        }
    
    def _build_success_response(self, state: GraphState) -> Dict[str, Any]:
        """Build successful response."""
        user_memory = self._get_user_memory(state.user_id)
        response = {
            "status": "success",
            "query": state.query,
            "user_id": state.user_id,
            "intent": state.query_intent.intent if state.query_intent else "unknown",
            "retrieved": state.retrieved_transactions,
            "summary": state.summary,
            "metadata": {
                "total_retrieved": len(state.retrieved_transactions),
                "query_intent": state.query_intent.dict() if state.query_intent else None,
                "timings": state.metadata.get("timings", {}),
                "total_time_ms": (time.time() - state.start_time) * 1000,
                "conversation_context": state.metadata.get("conversation_context", ""),
                "memory_size": len(user_memory)
            }
        }
        
        # Add aggregated data if available
        if state.aggregated_data:
            response["aggregated_data"] = state.aggregated_data
        
        return response
    
    def _build_error_response(self, state: GraphState) -> Dict[str, Any]:
        """Build error response."""
        return {
            "status": "error",
            "query": state.query,
            "user_id": state.user_id,
            "error": state.error,
            "metadata": {
                "timings": state.metadata.get("timings", {}),
                "total_time_ms": (time.time() - state.start_time) * 1000
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        try:
            # Check retriever
            retriever.initialize_client()
            stats = retriever.collection.count() if retriever.collection else 0
            health["components"]["retriever"] = {
                "status": "healthy",
                "documents_indexed": stats
            }
        except Exception as e:
            health["components"]["retriever"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        try:
            # Check embedding service
            await embedding_service.initialize()
            health["components"]["embeddings"] = {
                "status": "healthy",
                "model": embedding_service.model_name
            }
        except Exception as e:
            health["components"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        # Check summarizer (just check if it can be initialized)
        health["components"]["summarizer"] = {
            "status": "ready",
            "model": summarizer.model_name,
            "note": "Requires API key for operation"
        }
        
        return health

# Global graph orchestrator instance
graph = FinancialAssistantGraph()

# Import embedding service here to avoid circular imports
from services.embeddings import embedding_service
