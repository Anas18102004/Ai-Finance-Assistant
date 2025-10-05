"""
Financial analysis tools for the AI assistant.
These tools provide structured data analysis capabilities.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FinancialTools:
    """Collection of financial analysis tools."""
    
    def __init__(self):
        self.tools = {
            "calculate_total_spending": self.calculate_total_spending,
            "analyze_spending_by_category": self.analyze_spending_by_category,
            "compare_periods": self.compare_periods,
            "find_top_expenses": self.find_top_expenses,
            "calculate_average_spending": self.calculate_average_spending,
            "analyze_spending_trends": self.analyze_spending_trends,
            "get_budget_analysis": self.get_budget_analysis,
            "find_unusual_transactions": self.find_unusual_transactions,
            "calculate_savings_rate": self.calculate_savings_rate,
            "get_monthly_summary": self.get_monthly_summary
        }
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools for Gemini."""
        return {
            "calculate_total_spending": "Calculate total spending for a specific period or category. Parameters: transactions (list), start_date (optional), end_date (optional), category (optional)",
            "analyze_spending_by_category": "Break down spending by categories with percentages. Parameters: transactions (list), period (optional)",
            "compare_periods": "Compare spending between two time periods. Parameters: transactions (list), period1_start, period1_end, period2_start, period2_end",
            "find_top_expenses": "Find the highest individual transactions. Parameters: transactions (list), limit (default 5), category (optional)",
            "calculate_average_spending": "Calculate average daily/weekly/monthly spending. Parameters: transactions (list), period_type ('daily'|'weekly'|'monthly')",
            "analyze_spending_trends": "Analyze spending trends over time. Parameters: transactions (list), group_by ('day'|'week'|'month')",
            "get_budget_analysis": "Analyze spending against budget limits. Parameters: transactions (list), budget_limits (dict)",
            "find_unusual_transactions": "Find transactions that are unusually high or low. Parameters: transactions (list), threshold_multiplier (default 2.0)",
            "calculate_savings_rate": "Calculate savings rate from income and expenses. Parameters: transactions (list), period (optional)",
            "get_monthly_summary": "Get comprehensive monthly financial summary. Parameters: transactions (list), month (optional), year (optional)"
        }
    
    def calculate_total_spending(self, transactions: List[Dict], start_date: str = None, end_date: str = None, category: str = None) -> Dict[str, Any]:
        """Calculate total spending with optional filters."""
        try:
            filtered_transactions = self._filter_transactions(transactions, start_date, end_date, category, transaction_type="Debit")
            
            total = sum(t.get('amount', 0) for t in filtered_transactions)
            count = len(filtered_transactions)
            
            result = {
                "total_amount": total,
                "transaction_count": count,
                "average_per_transaction": total / count if count > 0 else 0,
                "period": f"{start_date} to {end_date}" if start_date and end_date else "All time",
                "category": category or "All categories"
            }
            
            logger.info(f"Calculated total spending: ₹{total} across {count} transactions")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating total spending: {e}")
            return {"error": str(e)}
    
    def analyze_spending_by_category(self, transactions: List[Dict], period: str = None) -> Dict[str, Any]:
        """Analyze spending breakdown by category."""
        try:
            filtered_transactions = self._filter_transactions(transactions, transaction_type="Debit")
            
            category_totals = defaultdict(float)
            category_counts = defaultdict(int)
            
            for txn in filtered_transactions:
                category = txn.get('category', 'Unknown')
                amount = txn.get('amount', 0)
                category_totals[category] += amount
                category_counts[category] += 1
            
            total_spending = sum(category_totals.values())
            
            categories = []
            for category, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = (amount / total_spending * 100) if total_spending > 0 else 0
                categories.append({
                    "category": category,
                    "amount": amount,
                    "percentage": round(percentage, 1),
                    "transaction_count": category_counts[category],
                    "average_per_transaction": amount / category_counts[category]
                })
            
            return {
                "total_spending": total_spending,
                "categories": categories,
                "period": period or "All time"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spending by category: {e}")
            return {"error": str(e)}
    
    def compare_periods(self, transactions: List[Dict], period1_start: str, period1_end: str, period2_start: str, period2_end: str) -> Dict[str, Any]:
        """Compare spending between two periods."""
        try:
            period1_txns = self._filter_transactions(transactions, period1_start, period1_end, transaction_type="Debit")
            period2_txns = self._filter_transactions(transactions, period2_start, period2_end, transaction_type="Debit")
            
            period1_total = sum(t.get('amount', 0) for t in period1_txns)
            period2_total = sum(t.get('amount', 0) for t in period2_txns)
            
            change = period2_total - period1_total
            change_percentage = (change / period1_total * 100) if period1_total > 0 else 0
            
            return {
                "period1": {
                    "start": period1_start,
                    "end": period1_end,
                    "total": period1_total,
                    "transaction_count": len(period1_txns)
                },
                "period2": {
                    "start": period2_start,
                    "end": period2_end,
                    "total": period2_total,
                    "transaction_count": len(period2_txns)
                },
                "comparison": {
                    "absolute_change": change,
                    "percentage_change": round(change_percentage, 1),
                    "trend": "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing periods: {e}")
            return {"error": str(e)}
    
    def find_top_expenses(self, transactions: List[Dict], limit: int = 5, category: str = None) -> Dict[str, Any]:
        """Find the highest individual transactions."""
        try:
            filtered_transactions = self._filter_transactions(transactions, category=category, transaction_type="Debit")
            
            # Sort by amount descending
            top_transactions = sorted(filtered_transactions, key=lambda x: x.get('amount', 0), reverse=True)[:limit]
            
            return {
                "top_expenses": [
                    {
                        "description": txn.get('description', 'Unknown'),
                        "amount": txn.get('amount', 0),
                        "date": txn.get('date', ''),
                        "category": txn.get('category', 'Unknown')
                    }
                    for txn in top_transactions
                ],
                "total_of_top_expenses": sum(t.get('amount', 0) for t in top_transactions),
                "category_filter": category or "All categories"
            }
            
        except Exception as e:
            logger.error(f"Error finding top expenses: {e}")
            return {"error": str(e)}
    
    def calculate_average_spending(self, transactions: List[Dict], period_type: str = "monthly") -> Dict[str, Any]:
        """Calculate average spending per period."""
        try:
            filtered_transactions = self._filter_transactions(transactions, transaction_type="Debit")
            
            if not filtered_transactions:
                return {"error": "No transactions found"}
            
            # Get date range
            dates = [datetime.strptime(t.get('date', ''), '%Y-%m-%d') for t in filtered_transactions if t.get('date')]
            if not dates:
                return {"error": "No valid dates found"}
            
            start_date = min(dates)
            end_date = max(dates)
            total_amount = sum(t.get('amount', 0) for t in filtered_transactions)
            
            if period_type == "daily":
                days = (end_date - start_date).days + 1
                average = total_amount / days if days > 0 else 0
                period_count = days
            elif period_type == "weekly":
                weeks = ((end_date - start_date).days + 1) / 7
                average = total_amount / weeks if weeks > 0 else 0
                period_count = round(weeks, 1)
            else:  # monthly
                months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1
                average = total_amount / months if months > 0 else 0
                period_count = months
            
            return {
                "average_spending": round(average, 2),
                "period_type": period_type,
                "total_amount": total_amount,
                "period_count": period_count,
                "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating average spending: {e}")
            return {"error": str(e)}
    
    def get_monthly_summary(self, transactions: List[Dict], month: int = None, year: int = None) -> Dict[str, Any]:
        """Get comprehensive monthly financial summary."""
        try:
            # Use current month if not specified
            if month is None or year is None:
                now = datetime.now()
                month = month or now.month
                year = year or now.year
            
            # Filter transactions for the specific month
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            month_transactions = self._filter_transactions(transactions, start_date, end_date)
            
            # Calculate totals
            expenses = [t for t in month_transactions if t.get('type') == 'Debit']
            income = [t for t in month_transactions if t.get('type') == 'Credit']
            
            total_expenses = sum(t.get('amount', 0) for t in expenses)
            total_income = sum(t.get('amount', 0) for t in income)
            net_savings = total_income - total_expenses
            
            # Category breakdown
            category_analysis = self.analyze_spending_by_category(expenses)
            
            # Top expenses
            top_expenses = self.find_top_expenses(expenses, limit=3)
            
            return {
                "month": month,
                "year": year,
                "summary": {
                    "total_income": total_income,
                    "total_expenses": total_expenses,
                    "net_savings": net_savings,
                    "savings_rate": (net_savings / total_income * 100) if total_income > 0 else 0,
                    "transaction_count": len(month_transactions)
                },
                "category_breakdown": category_analysis.get('categories', []),
                "top_expenses": top_expenses.get('top_expenses', [])
            }
            
        except Exception as e:
            logger.error(f"Error generating monthly summary: {e}")
            return {"error": str(e)}
    
    def analyze_spending_trends(self, transactions: List[Dict], group_by: str = "month") -> Dict[str, Any]:
        """Analyze spending trends over time."""
        try:
            filtered_transactions = self._filter_transactions(transactions, transaction_type="Debit")
            
            if not filtered_transactions:
                return {"error": "No transactions found"}
            
            # Group transactions by time period
            grouped_data = defaultdict(float)
            
            for txn in filtered_transactions:
                date_str = txn.get('date', '')
                if not date_str:
                    continue
                    
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if group_by == "day":
                        key = date_obj.strftime('%Y-%m-%d')
                    elif group_by == "week":
                        # Get week start (Monday)
                        week_start = date_obj - timedelta(days=date_obj.weekday())
                        key = week_start.strftime('%Y-%m-%d')
                    else:  # month
                        key = date_obj.strftime('%Y-%m')
                    
                    grouped_data[key] += txn.get('amount', 0)
                    
                except ValueError:
                    continue
            
            # Convert to sorted list
            trends = []
            for period, amount in sorted(grouped_data.items()):
                trends.append({
                    "period": period,
                    "amount": amount
                })
            
            return {
                "trends": trends,
                "group_by": group_by,
                "total_periods": len(trends)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spending trends: {e}")
            return {"error": str(e)}
    
    def get_budget_analysis(self, transactions: List[Dict], budget_limits: Dict[str, float]) -> Dict[str, Any]:
        """Analyze spending against budget limits."""
        try:
            # Get category spending
            category_analysis = self.analyze_spending_by_category(transactions)
            categories = category_analysis.get('categories', [])
            
            budget_analysis = []
            total_budget = sum(budget_limits.values())
            total_spent = sum(cat['amount'] for cat in categories)
            
            for category_name, budget_limit in budget_limits.items():
                # Find actual spending for this category
                actual_spending = 0
                for cat in categories:
                    if cat['category'].lower() == category_name.lower():
                        actual_spending = cat['amount']
                        break
                
                over_budget = actual_spending > budget_limit
                remaining = budget_limit - actual_spending
                percentage_used = (actual_spending / budget_limit * 100) if budget_limit > 0 else 0
                
                budget_analysis.append({
                    "category": category_name,
                    "budget_limit": budget_limit,
                    "actual_spending": actual_spending,
                    "remaining": remaining,
                    "percentage_used": round(percentage_used, 1),
                    "over_budget": over_budget,
                    "variance": actual_spending - budget_limit
                })
            
            return {
                "budget_analysis": budget_analysis,
                "total_budget": total_budget,
                "total_spent": total_spent,
                "overall_remaining": total_budget - total_spent,
                "overall_percentage_used": (total_spent / total_budget * 100) if total_budget > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing budget: {e}")
            return {"error": str(e)}
    
    def find_unusual_transactions(self, transactions: List[Dict], threshold_multiplier: float = 2.0) -> Dict[str, Any]:
        """Find transactions that are unusually high or low."""
        try:
            filtered_transactions = self._filter_transactions(transactions, transaction_type="Debit")
            
            if len(filtered_transactions) < 3:
                return {"error": "Need at least 3 transactions for analysis"}
            
            amounts = [t.get('amount', 0) for t in filtered_transactions]
            avg_amount = sum(amounts) / len(amounts)
            
            # Calculate standard deviation
            variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            threshold_high = avg_amount + (threshold_multiplier * std_dev)
            threshold_low = max(0, avg_amount - (threshold_multiplier * std_dev))
            
            unusual_transactions = []
            for txn in filtered_transactions:
                amount = txn.get('amount', 0)
                if amount > threshold_high or amount < threshold_low:
                    unusual_transactions.append({
                        "description": txn.get('description', 'Unknown'),
                        "amount": amount,
                        "date": txn.get('date', ''),
                        "category": txn.get('category', 'Unknown'),
                        "type": "high" if amount > threshold_high else "low",
                        "deviation_from_avg": amount - avg_amount
                    })
            
            return {
                "unusual_transactions": unusual_transactions,
                "average_amount": round(avg_amount, 2),
                "threshold_high": round(threshold_high, 2),
                "threshold_low": round(threshold_low, 2),
                "total_unusual": len(unusual_transactions)
            }
            
        except Exception as e:
            logger.error(f"Error finding unusual transactions: {e}")
            return {"error": str(e)}
    
    def calculate_savings_rate(self, transactions: List[Dict], period: str = None) -> Dict[str, Any]:
        """Calculate savings rate from income and expenses."""
        try:
            filtered_transactions = transactions
            
            # Separate income and expenses
            income_transactions = [t for t in filtered_transactions if t.get('type') == 'Credit']
            expense_transactions = [t for t in filtered_transactions if t.get('type') == 'Debit']
            
            total_income = sum(t.get('amount', 0) for t in income_transactions)
            total_expenses = sum(t.get('amount', 0) for t in expense_transactions)
            
            net_savings = total_income - total_expenses
            savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
            
            return {
                "total_income": total_income,
                "total_expenses": total_expenses,
                "net_savings": net_savings,
                "savings_rate": round(savings_rate, 2),
                "income_transactions": len(income_transactions),
                "expense_transactions": len(expense_transactions),
                "period": period or "All time"
            }
            
        except Exception as e:
            logger.error(f"Error calculating savings rate: {e}")
            return {"error": str(e)}
    
    def _filter_transactions(self, transactions: List[Dict], start_date: str = None, end_date: str = None, 
                           category: str = None, transaction_type: str = None) -> List[Dict]:
        """Filter transactions based on criteria."""
        filtered = transactions
        
        if start_date:
            filtered = [t for t in filtered if t.get('date', '') >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t.get('date', '') < end_date]
        
        if category:
            filtered = [t for t in filtered if t.get('category', '').lower() == category.lower()]
        
        if transaction_type:
            filtered = [t for t in filtered if t.get('type', '') == transaction_type]
        
        return filtered

# Global instance
financial_tools = FinancialTools()
