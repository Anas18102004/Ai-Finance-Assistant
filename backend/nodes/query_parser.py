import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

class QueryIntent(BaseModel):
    """Parsed query intent and parameters."""
    intent: str  # ["greeting", "conversational", "personal_info", "top_expenses", "sum_spent", "filter", "compare", "general"]
    filters: Dict[str, Any] = {}
    top_k: int = 5
    time_range: Optional[Dict[str, str]] = None
    categories: List[str] = []
    amount_range: Optional[Dict[str, float]] = None
    keywords: List[str] = []

class QueryParser:
    """Parse natural language queries and extract intent and filters."""
    
    def __init__(self):
        self.intent_patterns = {
            "greeting": [
                r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)$",
                r"^(hi|hello|hey)\s*[!.]*$",
                r"^how\s+are\s+you\??$",
                r"^what'?s\s+up\??$",
                r"^good\s+(morning|afternoon|evening|day)\s*[!.]*$"
            ],
            "conversational": [
                r"^(thanks?|thank\s+you|ok|okay|cool|nice|great|awesome)$",
                r"^(yes|no|maybe|sure|alright)\s*[!.]*$",
                r"^(bye|goodbye|see\s+you|talk\s+later)$",
                r"^who\s+are\s+you\??$",
                r"^what\s+can\s+you\s+do\??$",
                r"^help\s*[!.]*$",
                r"^who\s+am\s+i\??$",
                r"^tell\s+me\s+about\s+(myself|me)\??$",
                r"^what\s+do\s+you\s+know\s+about\s+me\??$",
                r"^can\s+you\s+see\s+my\s+(profile|details|info)\??$",
                r"^show\s+me\s+my\s+(profile|info|details)\??$"
            ],
            "personal_info": [
                r"what'?s\s+my\s+name",
                r"what\s+is\s+my\s+name", 
                r"whats\s+my\s+name",
                r"^my\s+name$",  # Just "my name"
                r"my\s+name\s+is",
                r"do\s+you\s+know\s+my\s+name",
                r"what'?s\s+my\s+(age|birthday|email|phone|address)",
                r"do\s+you\s+have\s+my\s+(name|age|info|details)",
                r"tell\s+me\s+my\s+name"
            ],
            "top_expenses": [
                r"top\s+\d*\s*expenses?", r"highest\s+spending", r"most\s+expensive",
                r"biggest\s+transactions?", r"largest\s+amounts?"
            ],
            "sum_spent": [
                r"total\s+spent", r"how\s+much.*spent", r"sum\s+of", r"total\s+amount",
                r"spent.*total", r"overall\s+spending"
            ],
            "filter": [
                r"transactions?\s+in", r"expenses?\s+for", r"spending\s+on",
                r"show.*transactions?", r"find.*expenses?"
            ],
            "compare": [
                r"compare", r"vs", r"versus", r"difference\s+between",
                r"more\s+than", r"less\s+than"
            ]
        }
        
        self.month_patterns = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "aug": "08", "sep": "09",
            "oct": "10", "nov": "11", "dec": "12"
        }
        
        self.categories = [
            "food", "dining", "restaurant", "groceries", "transportation", 
            "shopping", "entertainment", "bills", "utilities", "gas", 
            "electricity", "water", "healthcare", "education", "travel",
            "fuel", "investment", "salary", "freelance", "gift", "rent"
        ]
    
    def parse_query(self, query: str, user_id: str) -> QueryIntent:
        """Parse a natural language query and extract intent and filters."""
        query_lower = query.lower()
        
        # Determine intent
        intent = self._classify_intent(query_lower)
        
        # Extract filters
        filters = {"userId": user_id}  # Always filter by user
        
        # Extract time range
        time_range = self._extract_time_range(query_lower)
        if time_range:
            filters.update(time_range)
        
        # Extract categories
        categories = self._extract_categories(query_lower)
        
        # Extract amount range
        amount_range = self._extract_amount_range(query_lower)
        
        # Extract top_k
        top_k = self._extract_top_k(query_lower)
        
        # Extract keywords for general search
        keywords = self._extract_keywords(query_lower)
        
        return QueryIntent(
            intent=intent,
            filters=filters,
            top_k=top_k,
            time_range=time_range,
            categories=categories,
            amount_range=amount_range,
            keywords=keywords
        )
    
    def _classify_intent(self, query: str) -> str:
        """Classify the query intent."""
        query_lower = query.lower().strip()
        
        # Direct checks for common personal queries (highest priority)
        if query_lower in ["my name", "whats my name", "what's my name", "what is my name"]:
            return "personal_info"
        
        # First check for greetings, conversational, and personal info queries (highest priority)
        for intent in ["greeting", "conversational", "personal_info"]:
            if intent in self.intent_patterns:
                for pattern in self.intent_patterns[intent]:
                    if re.search(pattern, query, re.IGNORECASE):
                        return intent
        
        # Check for "How much did I spend" pattern for sum_spent intent
        if re.search(r"how\s+much.*spend", query, re.IGNORECASE):
            return "sum_spent"
            
        if re.search(r"find\s+transactions", query, re.IGNORECASE):
            return "filter"
            
        # Check other financial intents
        for intent, patterns in self.intent_patterns.items():
            if intent not in ["greeting", "conversational", "personal_info", "filter"]:  # Skip already checked intents
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        return intent
        
        return "general"
    
    def _extract_time_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract time range from query."""
        current_year = datetime.now().year
        
        # Check for specific months
        for month_name, month_num in self.month_patterns.items():
            if month_name in query:
                # Extract year if mentioned, otherwise use current year
                year_match = re.search(r'\b(20\d{2})\b', query)
                year = year_match.group(1) if year_match else str(current_year)
                
                start_date = f"{year}-{month_num}-01"
                
                # Calculate end date (last day of month)
                if month_num in ["01", "03", "05", "07", "08", "10", "12"]:
                    end_day = "31"
                elif month_num in ["04", "06", "09", "11"]:
                    end_day = "30"
                else:  # February
                    end_day = "29" if int(year) % 4 == 0 else "28"
                
                end_date = f"{year}-{month_num}-{end_day}"
                
                return {"start_date": start_date, "end_date": end_date}
        
        # Check for relative time periods
        if "last week" in query:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
        
        if "last month" in query:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
        
        if "this year" in query:
            return {
                "start_date": f"{current_year}-01-01",
                "end_date": f"{current_year}-12-31"
            }
        
        return None
    
    def _extract_categories(self, query: str) -> List[str]:
        """Extract categories from query."""
        found_categories = []
        
        for category in self.categories:
            if category in query:
                found_categories.append(category)
        
        # Map common terms to categories (based on actual data format)
        category_mappings = {
            "food": ["Food", "Food & Dining"],
            "dining": ["Food", "Food & Dining"], 
            "restaurant": ["Food", "Food & Dining"],
            "groceries": ["Food", "Groceries"],
            "transport": ["Transportation"],
            "transportation": ["Transportation"],
            "shopping": ["Shopping"],
            "entertainment": ["Entertainment"],
            "bills": ["Utilities", "Bills & Utilities"],
            "utilities": ["Utilities", "Bills & Utilities"],
            "gas": ["Utilities"],
            "electricity": ["Utilities"],
            "water": ["Utilities"],
            "healthcare": ["Healthcare"],
            "education": ["Education"],
            "travel": ["Travel"],
            "fuel": ["Fuel", "Transportation"],
            "investment": ["Investment"],
            "salary": ["Salary"],
            "freelance": ["Freelance"],
            "gift": ["Gift"],
            "rent": ["Rent"],
            "others": ["Others"]
        }
        
        mapped_categories = []
        for cat in found_categories:
            if cat in category_mappings:
                mapped_categories.extend(category_mappings[cat])
        
        return mapped_categories
    
    def _extract_amount_range(self, query: str) -> Optional[Dict[str, float]]:
        """Extract amount range from query."""
        # Look for patterns like "above 1000", "more than 500", "under 2000"
        
        # Above/more than patterns
        above_match = re.search(r'(?:above|more\s+than|greater\s+than|over)\s+(?:₹\s*)?(\d+(?:,\d+)*)', query)
        if above_match:
            amount = float(above_match.group(1).replace(',', ''))
            return {"min_amount": amount}
        
        # Below/less than patterns
        below_match = re.search(r'(?:below|less\s+than|under|fewer\s+than)\s+(?:₹\s*)?(\d+(?:,\d+)*)', query)
        if below_match:
            amount = float(below_match.group(1).replace(',', ''))
            return {"max_amount": amount}
        
        # Between pattern
        between_match = re.search(r'between\s+(?:₹\s*)?(\d+(?:,\d+)*)\s+(?:and|to)\s+(?:₹\s*)?(\d+(?:,\d+)*)', query)
        if between_match:
            min_amount = float(between_match.group(1).replace(',', ''))
            max_amount = float(between_match.group(2).replace(',', ''))
            return {"min_amount": min_amount, "max_amount": max_amount}
        
        return None
    
    def _extract_top_k(self, query: str) -> int:
        """Extract top_k from query."""
        # Look for patterns like "top 10", "first 5", "show 3"
        top_match = re.search(r'(?:top|first|show|last)\s+(\d+)', query)
        if top_match:
            return int(top_match.group(1))
        
        return 5  # Default
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords for general search."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "show", "find",
            "get", "give", "tell", "me", "my", "i", "you", "what", "where", "when",
            "how", "why", "which", "who", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "can", "will", "just"
        }
        
        # Split query into words and filter
        words = re.findall(r'\b\w+\b', query)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

# Global query parser instance
query_parser = QueryParser()
