"""
Data Agent - Executes structured queries on transaction data.
You are a Data Agent that executes structured financial queries on a user's transaction data stored in JSON format. 
You DO NOT generate explanations, only process the data and return results.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from tools.financial_tools import financial_tools
from nodes.retriever import retriever

logger = logging.getLogger(__name__)

class DataAgent:
    """Agent for executing structured data queries. Returns machine-readable output only."""
    
    def __init__(self):
        self.retriever = retriever
        self.tools = financial_tools
    
    async def execute_query(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a structured query plan."""
        try:
            operation = plan.get("operation", "filter")
            filters = plan.get("filters", {})
            parameters = plan.get("parameters", {})
            user_id = plan.get("user_id")
            
            logger.info(f"Executing {operation} query for user {user_id}")
            
            # Get transactions based on filters
            transactions = await self._get_filtered_transactions(filters, user_id)
            
            if not transactions:
                return {
                    "success": False,
                    "data": [],
                    "message": "No transactions found matching your criteria",
                    "operation": operation
                }
            
            # Execute specific operation
            result = await self._execute_operation(operation, transactions, parameters, filters)
            
            return {
                "success": True,
                "data": result,
                "operation": operation,
                "transaction_count": len(transactions),
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": plan.get("operation", "unknown")
            }
    
    async def _get_filtered_transactions(self, filters: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """Get transactions based on filters."""
        try:
            # Build query for retriever
            query_filters = {"userId": user_id}
            
            # Add filters
            if "category" in filters:
                query_filters["category"] = filters["category"]
            
            if "type" in filters:
                query_filters["type"] = filters["type"]
            
            if "amount_range" in filters:
                amount_range = filters["amount_range"]
                if "min" in amount_range:
                    query_filters["amount_min"] = amount_range["min"]
                if "max" in amount_range:
                    query_filters["amount_max"] = amount_range["max"]
            
            # Date range
            time_range = None
            if "start_date" in filters and "end_date" in filters:
                time_range = {
                    "start_date": filters["start_date"],
                    "end_date": filters["end_date"]
                }
            
            # Create a mock query intent for retriever
            from nodes.query_parser import QueryIntent
            query_intent = QueryIntent(
                intent="filter",
                filters=query_filters,
                time_range=time_range,
                top_k=1000  # Get many transactions for analysis
            )
            
            # Get transactions
            transactions = await self.retriever.retrieve(query_intent, "data_query")
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting filtered transactions: {e}")
            return []
    
    async def _execute_operation(self, operation: str, transactions: List[Dict[str, Any]], 
                                parameters: Dict[str, Any], filters: Dict[str, Any]) -> Any:
        """Execute specific operation on transactions."""
        
        if operation == "top_n":
            n = parameters.get("n", 5)
            sort_by = parameters.get("sort_by", "amount")
            order = parameters.get("order", "desc")
            
            # Sort transactions
            reverse = (order == "desc")
            if sort_by == "amount":
                sorted_txns = sorted(transactions, key=lambda x: x.get("amount", 0), reverse=reverse)
            elif sort_by == "date":
                sorted_txns = sorted(transactions, key=lambda x: x.get("date", ""), reverse=reverse)
            else:
                sorted_txns = transactions
            
            return {
                "transactions": sorted_txns[:n],
                "total_found": len(transactions),
                "showing": min(n, len(transactions))
            }
        
        elif operation == "total":
            category = filters.get("category")
            start_date = filters.get("start_date")
            end_date = filters.get("end_date")
            
            result = self.tools.calculate_total_spending(
                transactions=transactions,
                start_date=start_date,
                end_date=end_date,
                category=category
            )
            return result
        
        elif operation == "category_analysis":
            period = f"{filters.get('start_date', '')} to {filters.get('end_date', '')}"
            result = self.tools.analyze_spending_by_category(
                transactions=transactions,
                period=period
            )
            return result
        
        elif operation == "compare_periods":
            # This would need two sets of transactions
            # For now, return current period analysis
            result = self.tools.analyze_spending_by_category(transactions)
            return result
        
        elif operation == "trends":
            group_by = parameters.get("group_by", "month")
            result = self.tools.analyze_spending_trends(
                transactions=transactions,
                group_by=group_by
            )
            return result
        
        elif operation == "budget_analysis":
            # Would need budget limits from user preferences
            # For now, return category analysis
            result = self.tools.analyze_spending_by_category(transactions)
            return result
        
        elif operation == "summary":
            # Get current month/year from filters or use current
            now = datetime.now()
            month = now.month
            year = now.year
            
            if "start_date" in filters:
                try:
                    start_date = datetime.strptime(filters["start_date"], "%Y-%m-%d")
                    month = start_date.month
                    year = start_date.year
                except:
                    pass
            
            result = self.tools.get_monthly_summary(
                transactions=transactions,
                month=month,
                year=year
            )
            return result
        
        else:  # Default filter operation
            n = parameters.get("n", 10)
            sort_by = parameters.get("sort_by", "date")
            order = parameters.get("order", "desc")
            
            # Sort and limit
            reverse = (order == "desc")
            if sort_by == "amount":
                sorted_txns = sorted(transactions, key=lambda x: x.get("amount", 0), reverse=reverse)
            elif sort_by == "date":
                sorted_txns = sorted(transactions, key=lambda x: x.get("date", ""), reverse=reverse)
            else:
                sorted_txns = transactions
            
            return {
                "transactions": sorted_txns[:n],
                "total_found": len(transactions)
            }

# Global instance
data_agent = DataAgent()
