"""
Orchestrator - Routes queries between agents based on intent.
"""

import logging
import time
from typing import Dict, Any, Optional
from agents.intent_agent import intent_agent
from agents.data_agent import data_agent
from agents.rag_agent import rag_agent
from agents.synthesizer_agent import synthesizer_agent

logger = logging.getLogger(__name__)

class Orchestrator:
    """Main orchestrator that routes queries between specialized agents."""
    
    def __init__(self):
        self.intent_agent = intent_agent
        self.data_agent = data_agent
        self.rag_agent = rag_agent
        self.synthesizer_agent = synthesizer_agent
        
        # Simple conversation memory (last 5 interactions per user)
        self.conversation_memory = {}
        self.max_memory_size = 5
    
    def initialize(self, gemini_api_key: str):
        """Initialize all agents with API key."""
        self.intent_agent.initialize(gemini_api_key)
        self.rag_agent.initialize(gemini_api_key)
        self.synthesizer_agent.initialize(gemini_api_key)
        logger.info("Orchestrator initialized with all agents")
    
    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a user query through the agent pipeline."""
        start_time = time.time()
        
        try:
            # Get conversation context
            context = self._get_conversation_context(user_id)
            
            # Step 1: Intent Classification and Planning
            logger.info(f"Step 1: Classifying query for user {user_id}")
            intent_result = await self.intent_agent.classify_and_plan(query, user_id)
            
            intent = intent_result.get("intent")
            logger.info(f"Query classified as: {intent}")
            
            # Step 2: Route to appropriate agent
            if intent == "simple_response":
                # Direct to synthesizer
                logger.info("Step 2: Routing to synthesizer (simple response)")
                response = await self.synthesizer_agent.synthesize_response(
                    intent="simple_response",
                    query=query,
                    context={"conversation_context": context}
                )
                
                result = {
                    "status": "success",
                    "response": response,
                    "intent": intent,
                    "agent_path": "intent → synthesizer",
                    "metadata": {
                        "conversational": True,
                        "rag_used": False,
                        "data_retrieved": False
                    }
                }
            
            elif intent == "data_query":
                # Route to data agent then synthesizer
                logger.info("Step 2: Routing to data agent")
                data_result = await self.data_agent.execute_query(intent_result)
                
                logger.info("Step 3: Routing to synthesizer (data response)")
                response = await self.synthesizer_agent.synthesize_response(
                    intent="data_query",
                    query=query,
                    data=data_result,
                    context={"conversation_context": context}
                )
                
                result = {
                    "status": "success",
                    "response": response,
                    "intent": intent,
                    "agent_path": "intent → data → synthesizer",
                    "data_summary": {
                        "operation": data_result.get("operation"),
                        "transaction_count": data_result.get("transaction_count", 0),
                        "success": data_result.get("success", False)
                    },
                    "metadata": {
                        "conversational": False,
                        "rag_used": False,
                        "data_retrieved": True
                    }
                }
            
            elif intent == "knowledge_query":
                # Route to RAG agent then synthesizer
                logger.info("Step 2: Routing to RAG agent")
                topic = intent_result.get("topic", "general")
                rag_result = await self.rag_agent.answer_knowledge_query(query, topic, user_id)
                
                logger.info("Step 3: Routing to synthesizer (knowledge response)")
                response = await self.synthesizer_agent.synthesize_response(
                    intent="knowledge_query",
                    query=query,
                    data=rag_result,
                    context={"conversation_context": context}
                )
                
                result = {
                    "status": "success",
                    "response": response,
                    "intent": intent,
                    "agent_path": "intent → rag → synthesizer",
                    "knowledge_summary": {
                        "topic": topic,
                        "source": rag_result.get("source"),
                        "success": rag_result.get("success", False),
                        "retrieved_docs": rag_result.get("retrieved_docs", 0),
                        "rag_process": rag_result.get("rag_process", {})
                    },
                    "metadata": {
                        "conversational": False,
                        "rag_used": True,
                        "data_retrieved": False,
                        "rag_retrieval_count": rag_result.get("retrieved_docs", 0)
                    }
                }
            
            else:
                # Fallback
                response = "I'm not sure how to help with that. Could you please rephrase your question?"
                result = {
                    "status": "success",
                    "response": response,
                    "intent": "unknown",
                    "agent_path": "fallback",
                    "metadata": {
                        "conversational": True,
                        "rag_used": False,
                        "data_retrieved": False
                    }
                }
            
            # Add timing and user info
            total_time = (time.time() - start_time) * 1000
            result.update({
                "user_id": user_id,
                "query": query,
                "response_time_ms": round(total_time, 2),
                "timestamp": time.time()
            })
            
            # Store in conversation memory
            self._add_to_memory(user_id, query, response, intent)
            
            logger.info(f"Query processed successfully in {total_time:.1f}ms via {result['agent_path']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_time = (time.time() - start_time) * 1000
            
            return {
                "status": "error",
                "response": "I encountered an error while processing your request. Please try again.",
                "error": str(e),
                "user_id": user_id,
                "query": query,
                "response_time_ms": round(error_time, 2),
                "agent_path": "error",
                "metadata": {
                    "conversational": True,
                    "rag_used": False,
                    "data_retrieved": False
                }
            }
    
    def _get_conversation_context(self, user_id: str) -> str:
        """Get recent conversation context for user."""
        if user_id not in self.conversation_memory:
            return ""
        
        memory = self.conversation_memory[user_id]
        if not memory:
            return ""
        
        # Get last 3 interactions for context
        context_parts = []
        for conv in list(memory)[-3:]:
            context_parts.append(f"Previous: '{conv['query']}' -> {conv['intent']}")
        
        return " | ".join(context_parts)
    
    def _add_to_memory(self, user_id: str, query: str, response: str, intent: str):
        """Add interaction to conversation memory."""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        memory = self.conversation_memory[user_id]
        
        # Add new interaction
        memory.append({
            "timestamp": time.time(),
            "query": query,
            "response": response[:100] + "..." if len(response) > 100 else response,
            "intent": intent
        })
        
        # Keep only last N interactions
        if len(memory) > self.max_memory_size:
            memory.pop(0)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics."""
        return {
            "total_users": len(self.conversation_memory),
            "memory_size_per_user": self.max_memory_size,
            "users_with_memory": [
                user_id for user_id, memory in self.conversation_memory.items() 
                if len(memory) > 0
            ]
        }

# Global instance
orchestrator = Orchestrator()
