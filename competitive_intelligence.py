import numpy as np
import pandas as pd
import requests
import json
import os
import logging
import time
import re
import random
import hashlib
import asyncio
import aiohttp
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
# from ratelimiter import RateLimiter  # Removed this import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("competitive_intelligence")

# Custom RateLimiter implementation compatible with Python 3.11+
class RateLimiter:
    """Simple rate limiter that limits the number of calls in a time period."""
    
    def __init__(self, max_calls: int, period: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self._lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
    
    def __enter__(self):
        """Context manager entry that blocks if rate limit is exceeded."""
        self._clear_old_calls()
        while len(self.calls) >= self.max_calls:
            # Sleep until oldest call expires
            sleep_time = self.period - (time.monotonic() - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._clear_old_calls()
        self.calls.append(time.monotonic())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self._lock:
            await self._lock.acquire()
        self._clear_old_calls()
        while len(self.calls) >= self.max_calls:
            # Sleep until oldest call expires
            sleep_time = self.period - (time.monotonic() - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self._clear_old_calls()
        self.calls.append(time.monotonic())
        if self._lock:
            self._lock.release()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False
    
    def _clear_old_calls(self):
        """Remove calls older than the period."""
        now = time.monotonic()
        self.calls = [call for call in self.calls if now - call <= self.period]

@dataclass
class Competitor:
    """Data class for competitor information."""
    name: str
    url: str
    founded_year: int = 0
    estimated_employees: int = 0
    funding_rounds: int = 0
    total_funding: float = 0.0
    market_share: float = 0.0
    growth_rate: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    description: str = ""
    ceo: str = ""
    headquarters: str = ""
    last_updated: datetime = None
    tech_stack: List[str] = field(default_factory=list)
    social_mentions: int = 0
    sentiment_score: float = 0.0
    recent_news: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
        if self.tech_stack is None:
            self.tech_stack = []
        if self.recent_news is None:
            self.recent_news = []
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


class ApiClient:
    """Base class for API clients with rate limiting and error handling."""
    
    def __init__(self, api_key: str, base_url: str, rate_limit: int = 1, period: int = 1):
        """
        Initialize API client with rate limiting.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoints
            rate_limit: Number of requests allowed in period
            period: Time period in seconds for rate limiting
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(max_calls=rate_limit, period=period)
        
        # Set up common headers
        self.session.headers.update({
            'User-Agent': 'Competitive-Intelligence/1.0',
            'Content-Type': 'application/json'
        })
    
    async def get_async(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make asynchronous GET request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            with self.rate_limiter:
                try:
                    async with session.get(url, params=params, headers=self._get_headers()) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            return await self.get_async(endpoint, params)
                        else:
                            logger.error(f"API error: {response.status} - {await response.text()}")
                            return {"error": f"API error: {response.status}"}
                except Exception as e:
                    logger.error(f"Request error: {str(e)}")
                    return {"error": str(e)}
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make synchronous GET request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        with self.rate_limiter:
            try:
                response = self.session.get(url, params=params, headers=self._get_headers())
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    return self.get(endpoint, params)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return {"error": f"API error: {response.status_code}"}
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                return {"error": str(e)}
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        with self.rate_limiter:
            try:
                response = self.session.post(url, json=data, headers=self._get_headers())
                
                if response.status_code == 200 or response.status_code == 201:
                    return response.json()
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    return self.post(endpoint, data)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return {"error": f"API error: {response.status_code}"}
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                return {"error": str(e)}
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API request. Override in subclasses if needed."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }


class CrunchbaseClient(ApiClient):
    """Client for the Crunchbase API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.crunchbase.com/api/v4",
            rate_limit=10,  # Crunchbase rate limits
            period=60
        )
    
    def search_companies(self, query: str, sector: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for companies matching query and sector.
        
        Args:
            query: Search query (company name or keywords)
            sector: Industry sector to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of company data dictionaries
        """
        search_params = {
            "field_ids": [
                "identifier", 
                "name", 
                "short_description", 
                "website_url", 
                "linkedin", 
                "facebook", 
                "twitter", 
                "logo_url", 
                "location_identifiers", 
                "founded_on", 
                "closed_on", 
                "num_employees_enum", 
                "operating_status", 
                "categories", 
                "category_groups", 
                "funding_total", 
                "funding_stage", 
                "last_funding_type"
            ],
            "query": [
                {"type": "predicate", "field_id": "name", "operator_id": "contains", "values": [query]}
            ],
            "order": [
                {"field_id": "rank_org", "sort": "desc"}
            ],
            "limit": limit
        }
        
        # Add sector filter if provided
        if sector:
            search_params["query"].append(
                {"type": "predicate", "field_id": "category_groups", "operator_id": "includes", "values": [sector]}
            )
        
        # Make the API call
        response = self.post("searches/organizations", search_params)
        
        if "error" in response:
            logger.error(f"Crunchbase search error: {response['error']}")
            return []
        
        # Extract and process entities
        entities = response.get("entities", [])
        results = []
        
        for entity in entities:
            properties = entity.get("properties", {})
            
            company_data = {
                "name": properties.get("name", ""),
                "url": properties.get("website_url", ""),
                "description": properties.get("short_description", ""),
                "founded_year": self._extract_year(properties.get("founded_on", "")),
                "total_funding": properties.get("funding_total", {}).get("value_usd", 0),
                "funding_rounds": 0,  # Would require another API call to get detailed funding rounds
                "estimated_employees": self._parse_employee_range(properties.get("num_employees_enum", "")),
                "headquarters": self._extract_hq(properties.get("location_identifiers", [])),
                "categories": [cat.get("value", "") for cat in properties.get("categories", [])]
            }
            
            results.append(company_data)
        
        return results
    
    def get_company_details(self, company_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific company."""
        response = self.get(f"entities/organizations/{company_id}")
        
        if "error" in response:
            logger.error(f"Crunchbase company details error: {response['error']}")
            return {}
        
        # Process and return the company details
        properties = response.get("properties", {})
        
        return {
            "name": properties.get("name", ""),
            "description": properties.get("short_description", ""),
            "long_description": properties.get("long_description", ""),
            "founded_year": self._extract_year(properties.get("founded_on", "")),
            "headquarters": self._extract_hq(properties.get("location_identifiers", [])),
            "ceo": self._get_ceo(response.get("cards", {}).get("executives", {}).get("items", [])),
            "total_funding": properties.get("funding_total", {}).get("value_usd", 0),
            "last_funding_date": properties.get("last_funding_at", ""),
            "last_funding_type": properties.get("last_funding_type", ""),
            "funding_stage": properties.get("funding_stage", ""),
            "investors": self._extract_investors(response.get("cards", {}).get("investors", {}).get("items", [])),
            "acquisitions": self._extract_acquisitions(response.get("cards", {}).get("acquiree_acquisitions", {}).get("items", [])),
            "categories": [cat.get("value", "") for cat in properties.get("categories", [])],
            "competitors": self._extract_competitors(response.get("cards", {}).get("competitors", {}).get("items", []))
        }
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string."""
        if not date_str:
            return 0
        
        try:
            return int(date_str.split("-")[0])
        except (ValueError, IndexError):
            return 0
    
    def _parse_employee_range(self, range_str: str) -> int:
        """Parse employee range string to estimate employee count."""
        if not range_str:
            return 0
        
        # Convert ranges like "1-10", "11-50", "51-100" to estimates
        ranges = {
            "1-10": 5,
            "11-50": 30,
            "51-100": 75,
            "101-250": 175,
            "251-500": 375,
            "501-1000": 750,
            "1001-5000": 3000,
            "5001-10000": 7500,
            "10001+": 15000
        }
        
        return ranges.get(range_str, 0)
    
    def _extract_hq(self, locations: List[Dict[str, Any]]) -> str:
        """Extract headquarters location from location identifiers."""
        if not locations:
            return ""
        
        # Look for HQ or first location
        hq_location = next((loc for loc in locations if loc.get("location_type", "") == "headquarters"), None)
        
        if not hq_location:
            hq_location = locations[0] if locations else {}
        
        return hq_location.get("value", "")
    
    def _get_ceo(self, executives: List[Dict[str, Any]]) -> str:
        """Extract CEO name from executives list."""
        ceo = next((exec for exec in executives if "CEO" in exec.get("title", "")), None)
        return ceo.get("name", "") if ceo else ""
    
    def _extract_investors(self, investors: List[Dict[str, Any]]) -> List[str]:
        """Extract investor names from investors list."""
        return [investor.get("name", "") for investor in investors]
    
    def _extract_acquisitions(self, acquisitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract acquisition details from acquisitions list."""
        result = []
        for acq in acquisitions:
            result.append({
                "acquired_by": acq.get("acquirer_name", ""),
                "price": acq.get("price", {}).get("value_usd", 0),
                "announced_on": acq.get("announced_on", "")
            })
        return result
    
    def _extract_competitors(self, competitors: List[Dict[str, Any]]) -> List[str]:
        """Extract competitor names from competitors list."""
        return [comp.get("name", "") for comp in competitors]


class ClearbitClient(ApiClient):
    """Client for the Clearbit API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://company.clearbit.com/v2",
            rate_limit=30,
            period=60
        )
    
    def find_company(self, domain: str) -> Dict[str, Any]:
        """
        Find company by domain.
        
        Args:
            domain: Company website domain
            
        Returns:
            Company information dictionary
        """
        response = self.get("companies/find", {"domain": domain})
        
        if "error" in response:
            logger.error(f"Clearbit find company error: {response['error']}")
            return {}
        
        return response
    
    def find_similar_companies(self, domain: str) -> List[Dict[str, Any]]:
        """
        Find similar companies to the given domain.
        
        Args:
            domain: Company website domain
            
        Returns:
            List of similar companies
        """
        # First get the company details to extract industry and tags
        company = self.find_company(domain)
        
        if not company or "error" in company:
            return []
        
        # Extract industry and tags for similarity search
        industry = company.get("category", {}).get("industry", "")
        tags = company.get("tags", [])
        
        # Use these to find similar companies
        similar_companies = []
        
        # If we have an industry, search by that
        if industry:
            industry_companies = self._search_by_industry(industry)
            similar_companies.extend(industry_companies)
        
        # If we have tags, search by those too
        if tags:
            top_tags = tags[:3]  # Use top 3 tags
            for tag in top_tags:
                tag_companies = self._search_by_tag(tag)
                similar_companies.extend(tag_companies)
        
        # Remove duplicates and the original company
        unique_companies = []
        seen_domains = set()
        
        for comp in similar_companies:
            comp_domain = comp.get("domain", "")
            if comp_domain and comp_domain != domain and comp_domain not in seen_domains:
                seen_domains.add(comp_domain)
                unique_companies.append(comp)
        
        return unique_companies[:10]  # Return top 10 similar companies
    
    def _search_by_industry(self, industry: str) -> List[Dict[str, Any]]:
        """Search companies by industry."""
        # This would be a real API call in production
        # Here we'll mock it with an empty response since Clearbit doesn't directly support this
        logger.info(f"Searching companies by industry: {industry}")
        return []
    
    def _search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Search companies by tag."""
        # This would be a real API call in production
        # Here we'll mock it with an empty response since Clearbit doesn't directly support this
        logger.info(f"Searching companies by tag: {tag}")
        return []


class NewsApiClient(ApiClient):
    """Client for the News API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://newsapi.org/v2",
            rate_limit=100,  # Free tier limit
            period=86400  # Per day
        )
    
    def get_company_news(self, company_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent news articles about a company.
        
        Args:
            company_name: Name of the company
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Make API request
        response = self.get(
            "everything", 
            {
                "q": f"\"{company_name}\"",  # Exact match
                "from": start_date_str,
                "to": end_date_str,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 10
            }
        )
        
        if "error" in response:
            logger.error(f"News API error: {response['error']}")
            return []
        
        # Process and return articles
        articles = response.get("articles", [])
        
        results = []
        for article in articles:
            results.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "description": article.get("description", "")
            })
        
        return results
    
    def get_industry_news(self, industry: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent news articles about an industry.
        
        Args:
            industry: Industry name
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Make API request
        response = self.get(
            "everything", 
            {
                "q": f"\"{industry}\" AND (market OR trend OR growth OR industry OR sector)",
                "from": start_date_str,
                "to": end_date_str,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 10
            }
        )
        
        if "error" in response:
            logger.error(f"News API error: {response['error']}")
            return []
        
        # Process and return articles
        articles = response.get("articles", [])
        
        results = []
        for article in articles:
            results.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "description": article.get("description", "")
            })
        
        return results
    
    def _get_headers(self) -> Dict[str, str]:
        """Override parent method for News API specific headers."""
        return {
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }


class AlphaVantageClient(ApiClient):
    """Client for the Alpha Vantage API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://www.alphavantage.co",
            rate_limit=5,  # Free tier limit
            period=60
        )
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock data for a company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock data dictionary
        """
        response = self.get("query", {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        })
        
        if "error" in response or "Error Message" in response:
            logger.error(f"Alpha Vantage error: {response.get('error') or response.get('Error Message')}")
            return {}
        
        global_quote = response.get("Global Quote", {})
        
        if not global_quote:
            return {}
        
        return {
            "symbol": global_quote.get("01. symbol", ""),
            "price": float(global_quote.get("05. price", 0)),
            "change": float(global_quote.get("09. change", 0)),
            "change_percent": global_quote.get("10. change percent", "").replace("%", ""),
            "volume": int(global_quote.get("06. volume", 0)),
            "latest_trading_day": global_quote.get("07. latest trading day", "")
        }
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Get sector performance metrics.
        
        Returns:
            Sector performance data dictionary
        """
        response = self.get("query", {
            "function": "SECTOR",
            "apikey": self.api_key
        })
        
        if "error" in response:
            logger.error(f"Alpha Vantage error: {response['error']}")
            return {}
        
        # Extract sector performance data
        return {
            "rank": response.get("Rank A: Real-Time Performance", {}),
            "one_day": response.get("Rank B: 1 Day Performance", {}),
            "five_day": response.get("Rank C: 5 Day Performance", {}),
            "one_month": response.get("Rank D: 1 Month Performance", {}),
            "three_month": response.get("Rank E: 3 Month Performance", {}),
            "year_to_date": response.get("Rank F: Year-to-Date (YTD) Performance", {}),
            "one_year": response.get("Rank G: 1 Year Performance", {}),
            "three_year": response.get("Rank H: 3 Year Performance", {}),
            "five_year": response.get("Rank I: 5 Year Performance", {}),
            "ten_year": response.get("Rank J: 10 Year Performance", {})
        }


class GitHubClient(ApiClient):
    """Client for the GitHub API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.github.com",
            rate_limit=60,  # Authenticated rate limit
            period=3600  # Per hour
        )
    
    def get_organization_repositories(self, org_name: str) -> List[Dict[str, Any]]:
        """
        Get repositories for an organization.
        
        Args:
            org_name: Organization name
            
        Returns:
            List of repository data dictionaries
        """
        response = self.get(f"orgs/{org_name}/repos", {
            "sort": "updated",
            "per_page": 10
        })
        
        if isinstance(response, dict) and "error" in response:
            logger.error(f"GitHub API error: {response['error']}")
            return []
        
        results = []
        for repo in response:
            results.append({
                "name": repo.get("name", ""),
                "description": repo.get("description", ""),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "updated_at": repo.get("updated_at", ""),
                "url": repo.get("html_url", "")
            })
        
        return results
    
    def search_repositories(self, keyword: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Search repositories by keyword and language.
        
        Args:
            keyword: Search keyword
            language: Programming language filter
            
        Returns:
            List of repository data dictionaries
        """
        query = keyword
        if language:
            query += f" language:{language}"
        
        response = self.get("search/repositories", {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 10
        })
        
        if "error" in response:
            logger.error(f"GitHub API error: {response['error']}")
            return []
        
        items = response.get("items", [])
        results = []
        
        for repo in items:
            results.append({
                "name": repo.get("name", ""),
                "owner": repo.get("owner", {}).get("login", ""),
                "description": repo.get("description", ""),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "updated_at": repo.get("updated_at", ""),
                "url": repo.get("html_url", "")
            })
        
        return results


class CompetitiveIntelligence:
    """
    Advanced competitive intelligence platform for startup analysis.
    
    This module provides comprehensive competitive analysis capabilities:
    - Competitor identification and profiling
    - Competitive positioning analysis
    - Market trends detection and forecasting
    - Competitive advantage (moat) analysis
    - Strategic opportunity identification
    
    It integrates with multiple data sources to provide real-time, accurate intelligence.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None, cache_dir: str = ".cache", cache_ttl: int = 86400):
        """
        Initialize the competitive intelligence platform.
        
        Args:
            api_keys: Dictionary of API keys for various data sources
                Keys: 'crunchbase', 'clearbit', 'news_api', 'alpha_vantage', 'github'
            cache_dir: Directory to store cached data
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.api_keys = api_keys or {}
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize API clients
        self.api_clients = self._initialize_api_clients()
        
        # Determine if we're in demo mode
        self.demo_mode = len(self.api_clients) == 0
        
        if self.demo_mode:
            logger.warning("No API keys provided. Running in demo mode with limited functionality.")
        else:
            logger.info(f"Initialized with {len(self.api_clients)} API clients")
            for client_name in self.api_clients.keys():
                logger.info(f"  - {client_name} client active")

    def _initialize_api_clients(self) -> Dict[str, Any]:
        """Initialize API clients based on provided API keys."""
        clients = {}
        
        # Crunchbase client
        if 'crunchbase' in self.api_keys and self.api_keys['crunchbase']:
            try:
                clients['crunchbase'] = CrunchbaseClient(self.api_keys['crunchbase'])
                logger.info("Crunchbase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Crunchbase client: {e}")
        
        # Clearbit client
        if 'clearbit' in self.api_keys and self.api_keys['clearbit']:
            try:
                clients['clearbit'] = ClearbitClient(self.api_keys['clearbit'])
                logger.info("Clearbit client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Clearbit client: {e}")
        
        # News API client
        if 'news_api' in self.api_keys and self.api_keys['news_api']:
            try:
                clients['news_api'] = NewsApiClient(self.api_keys['news_api'])
                logger.info("News API client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize News API client: {e}")
        
        # Alpha Vantage client
        if 'alpha_vantage' in self.api_keys and self.api_keys['alpha_vantage']:
            try:
                clients['alpha_vantage'] = AlphaVantageClient(self.api_keys['alpha_vantage'])
                logger.info("Alpha Vantage client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage client: {e}")
        
        # GitHub client
        if 'github' in self.api_keys and self.api_keys['github']:
            try:
                clients['github'] = GitHubClient(self.api_keys['github'])
                logger.info("GitHub client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GitHub client: {e}")
        
        return clients
        
    def _get_from_cache(self, key, default=None):
        """
        Get a value from the cache using the given key.
        
        Args:
            key (str): The cache key
            default: Default value to return if key not found
            
        Returns:
            The cached value or the default value
        """
        # Sanitize the key first
        safe_key = self._sanitize_cache_key(key)
        
        # Check if we have a cache attribute
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        # Return value from cache or default
        return self._cache.get(safe_key, default)

    def _add_to_cache(self, key, value):
        """
        Add a value to the cache.
        
        Args:
            key (str): The cache key
            value: The value to cache
        """
        # Sanitize the key first
        safe_key = self._sanitize_cache_key(key)
        
        # Ensure cache exists
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        # Add to cache
        self._cache[safe_key] = value
        
    def _save_to_cache(self, key, value):
        """Alias for _add_to_cache for backward compatibility."""
        return self._add_to_cache(key, value)

    def _sanitize_cache_key(self, key):
        """
        Sanitize a string to be used as a cache key.
        
        Args:
            key (str): The string to sanitize
            
        Returns:
            str: A sanitized version of the string suitable for use as a cache key
        """
        if not key or not isinstance(key, str):
            return "default_key"
        
        # Replace any non-alphanumeric characters with underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', key.lower())
        
        # Ensure the key isn't too long
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        # Ensure the key isn't empty
        if not sanitized:
            sanitized = "empty_key"
        
        return sanitized

    def get_competitors(self, company_name: str, sector: str, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Identify and profile competitors for a given company.
        
        Args:
            company_name: Name of the company to find competitors for
            sector: Industry sector (e.g., 'fintech', 'saas')
            keywords: Optional list of keywords to refine the search
            
        Returns:
            List of competitor profiles with detailed information
        """
        logger.info(f"Finding competitors for {company_name} in {sector} sector")
        
        # Prepare keywords if provided
        keywords_str = "_".join(keywords) if keywords else ""
        
        # Create cache key
        cache_key = f"competitors_{company_name}_{sector}_{keywords_str}"
        cache_key = self._sanitize_cache_key(cache_key)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached competitor data for {company_name}")
            return cached_data
        
        # If in demo mode or API calls fail, generate synthetic competitors
        if self.demo_mode:
            logger.info(f"Using synthetic competitor data for {company_name} in {sector}")
            competitors = self._fetch_competitors_demo(company_name, sector, keywords)
        else:
            # Try to get real competitor data from APIs
            try:
                competitors = self._fetch_competitors_from_apis(company_name, sector, keywords)
                
                # If we didn't find any competitors, fall back to demo mode
                if not competitors:
                    logger.warning(f"No competitors found via APIs for {company_name}. Using demo data.")
                    competitors = self._fetch_competitors_demo(company_name, sector, keywords)
            except Exception as e:
                logger.error(f"Error fetching competitors from APIs: {str(e)}")
                logger.info("Falling back to demo mode for competitor data")
                competitors = self._fetch_competitors_demo(company_name, sector, keywords)
        
        # Enrich competitor data with additional information
        enriched_competitors = self._enrich_competitor_profiles(competitors, sector)
        
        # Convert to dictionary format for JSON serialization
        competitor_dicts = [comp.to_dict() for comp in enriched_competitors]
        
        # Save to cache
        try:
            self._save_to_cache(cache_key, competitor_dicts)
        except Exception as e:
            logger.warning(f"Failed to cache competitor data: {e}")
        
        return competitor_dicts

    def _fetch_competitors_from_apis(self, company_name: str, sector: str, keywords: Optional[List[str]] = None) -> List[Competitor]:
        """Fetch competitor data from external APIs."""
        competitors = []
        tasks = []
        
        # Extract domain from company name (simple heuristic)
        domain = self._extract_domain_from_name(company_name)
        
        # Use Crunchbase API if available
        if 'crunchbase' in self.api_clients:
            try:
                crunchbase_client = self.api_clients['crunchbase']
                logger.info(f"Searching for competitors in Crunchbase: {company_name} ({sector})")
                
                # Search for competitors in the same sector
                # We use company name and sector for the search to find direct competitors
                search_results = crunchbase_client.search_companies(company_name, sector)
                
                if not search_results and keywords:
                    # Try with keywords if the first search failed
                    for keyword in keywords:
                        logger.info(f"Trying Crunchbase search with keyword: {keyword}")
                        keyword_results = crunchbase_client.search_companies(keyword, sector)
                        search_results.extend(keyword_results)
                
                # Convert results to Competitor objects
                for result in search_results:
                    # Skip if it's the company we're analyzing
                    if self._is_same_company(result.get("name", ""), company_name):
                        continue
                    
                    competitor = Competitor(
                        name=result.get("name", ""),
                        url=result.get("url", ""),
                        founded_year=result.get("founded_year", 0),
                        estimated_employees=result.get("estimated_employees", 0),
                        total_funding=result.get("total_funding", 0),
                        description=result.get("description", ""),
                        headquarters=result.get("headquarters", "")
                    )
                    competitors.append(competitor)
                
                logger.info(f"Found {len(competitors)} competitors from Crunchbase")
            except Exception as e:
                logger.error(f"Error fetching competitors from Crunchbase: {e}")
        
        # Use Clearbit API if available
        if 'clearbit' in self.api_clients and domain:
            try:
                clearbit_client = self.api_clients['clearbit']
                logger.info(f"Searching for similar companies in Clearbit: {domain}")
                
                # Find similar companies based on the company's domain
                similar_companies = clearbit_client.find_similar_companies(domain)
                
                # Convert results to Competitor objects
                for company in similar_companies:
                    # Skip if it's already in our list
                    existing_names = {comp.name.lower() for comp in competitors}
                    if company.get("name", "").lower() in existing_names:
                        continue
                    
                    founded_year = 0
                    if company.get("foundedDate"):
                        try:
                            founded_year = int(company.get("foundedDate", "").split("-")[0])
                        except (ValueError, IndexError):
                            pass
                    
                    competitor = Competitor(
                        name=company.get("name", ""),
                        url=company.get("domain", ""),
                        founded_year=founded_year,
                        estimated_employees=company.get("metrics", {}).get("employees", 0),
                        description=company.get("description", ""),
                        headquarters=self._format_clearbit_location(company.get("location", {}))
                    )
                    competitors.append(competitor)
                
                logger.info(f"Found {len(similar_companies)} competitors from Clearbit")
            except Exception as e:
                logger.error(f"Error fetching competitors from Clearbit: {e}")
        
        # Fetch news and stock data for the competitors we found
        # This is done in parallel to speed up the process
        if competitors:
            self._enrich_with_external_data(competitors)
        
        # Sort by total funding (highest first) as a proxy for importance
        competitors.sort(key=lambda x: x.total_funding, reverse=True)
        
        # Generate market share estimates for the competitors
        self._estimate_market_shares(competitors)
        
        return competitors[:10]  # Return top 10 competitors

    def _extract_domain_from_name(self, company_name: str) -> str:
        """Extract a likely domain from company name for API searches."""
        # Remove common legal suffixes
        name = re.sub(r'\b(Inc|LLC|Ltd|Corp|Corporation|Company)\b\.?', '', company_name)
        # Remove spaces, lowercase and add .com
        domain = name.strip().lower().replace(' ', '')
        return f"{domain}.com"

    def _is_same_company(self, name1: str, name2: str) -> bool:
        """Check if two company names likely refer to the same company."""
        # Normalize names
        def normalize(name):
            # Remove common legal suffixes
            name = re.sub(r'\b(Inc|LLC|Ltd|Corp|Corporation|Company)\b\.?', '', name)
            # Remove special characters
            name = re.sub(r'[^\w\s]', '', name)
            # Convert to lowercase and strip whitespace
            return name.strip().lower()
        
        return normalize(name1) == normalize(name2)

    def _format_clearbit_location(self, location: Dict[str, Any]) -> str:
        """Format location data from Clearbit."""
        parts = []
        
        if location.get("city"):
            parts.append(location.get("city"))
        
        if location.get("state"):
            parts.append(location.get("state"))
        
        if location.get("country"):
            parts.append(location.get("country"))
        
        return ", ".join(parts)

    def _enrich_with_external_data(self, competitors: List[Competitor]) -> None:
        """Enrich competitor profiles with data from news, stock, and other sources."""
        # This is computationally expensive, so we'll do it in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(competitors))) as executor:
            # Create tasks for each competitor
            future_to_competitor = {}
            
            for comp in competitors:
                future = executor.submit(self._enrich_single_competitor, comp)
                future_to_competitor[future] = comp
            
            # Collect results
            for future in as_completed(future_to_competitor):
                comp = future_to_competitor[future]
                try:
                    # The enriched data is applied directly to the competitor object
                    future.result()
                except Exception as e:
                    logger.error(f"Error enriching competitor {comp.name}: {e}")
    
    def _enrich_single_competitor(self, competitor: Competitor) -> None:
        """Enrich a single competitor with external data."""
        # Get news articles
        if 'news_api' in self.api_clients:
            try:
                news_client = self.api_clients['news_api']
                news_articles = news_client.get_company_news(competitor.name, days=30)
                
                if news_articles:
                    competitor.recent_news = news_articles[:5]  # Keep top 5 articles
                    competitor.social_mentions = len(news_articles)
                    
                    # Simple sentiment analysis based on presence of positive/negative words
                    sentiment_score = self._calculate_sentiment(news_articles)
                    competitor.sentiment_score = sentiment_score
            except Exception as e:
                logger.error(f"Error fetching news for {competitor.name}: {e}")
        
        # Get GitHub data if available
        if 'github' in self.api_clients:
            try:
                # Extract organization name from URL or company name
                org_name = self._extract_github_org(competitor)
                
                if org_name:
                    github_client = self.api_clients['github']
                    repos = github_client.get_organization_repositories(org_name)
                    
                    if repos:
                        # Extract tech stack from repository languages
                        tech_stack = set()
                        for repo in repos:
                            if repo.get("language") and repo.get("language") not in tech_stack:
                                tech_stack.add(repo.get("language"))
                        
                        competitor.tech_stack = list(tech_stack)
            except Exception as e:
                logger.error(f"Error fetching GitHub data for {competitor.name}: {e}")
        
        # Get stock data if available
        if 'alpha_vantage' in self.api_clients:
            try:
                # Try to get stock symbol (this would be more sophisticated in production)
                # For now we'll just use a simple heuristic based on company name
                symbol = self._guess_stock_symbol(competitor.name)
                
                if symbol:
                    alpha_client = self.api_clients['alpha_vantage']
                    stock_data = alpha_client.get_stock_data(symbol)
                    
                    if stock_data and "price" in stock_data:
                        # Use this to estimate growth rate if not already set
                        if not competitor.growth_rate and "change_percent" in stock_data:
                            try:
                                change_pct = float(stock_data["change_percent"])
                                # Annualize daily change (rough estimate)
                                competitor.growth_rate = change_pct * 252 / 100
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.error(f"Error fetching stock data for {competitor.name}: {e}")

    def _extract_github_org(self, competitor: Competitor) -> str:
        """Extract GitHub organization name from competitor data."""
        # First check if URL contains github.com
        if "github.com" in competitor.url:
            # Extract org name from GitHub URL
            match = re.search(r'github\.com/([^/]+)', competitor.url)
            if match:
                return match.group(1)
        
        # Otherwise, try a simple conversion of company name
        # This is a very simple heuristic - in production, you'd want to be more sophisticated
        org_name = competitor.name.lower().replace(' ', '-').replace('.', '-')
        return org_name

    def _guess_stock_symbol(self, company_name: str) -> str:
        """Guess stock symbol based on company name."""
        # This is a very simple heuristic - in production, you'd want to use
        # a proper lookup service or database
        # Strip common corporate endings
        name = re.sub(r'\b(Inc|LLC|Ltd|Corp|Corporation|Company)\b\.?', '', company_name)
        
        # Generate a likely ticker - take first letter of each word, uppercase
        words = name.split()
        if len(words) >= 2:
            ticker = ''.join(word[0] for word in words if word)
        else:
            # For single word names, take first 3-4 letters
            ticker = name[:min(4, len(name))]
        
        return ticker.upper()

    def _calculate_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score based on news articles (-100 to +100)."""
        if not articles:
            return 0.0
        
        # Simple keyword-based sentiment analysis
        # In production, you'd use a proper NLP sentiment analysis API
        positive_keywords = [
            "growth", "profit", "launch", "success", "innovation", "partnership", 
            "expand", "increase", "positive", "award", "achievement", "milestone"
        ]
        
        negative_keywords = [
            "lawsuit", "decline", "loss", "failure", "layoff", "cut", "down", 
            "bankrupt", "scandal", "investigation", "problem", "issue", "crisis"
        ]
        
        positive_count = 0
        negative_count = 0
        
        for article in articles:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            content = title + " " + description
            
            for word in positive_keywords:
                if word in content:
                    positive_count += 1
            
            for word in negative_keywords:
                if word in content:
                    negative_count += 1
        
        # Calculate score between -100 and 100
        if positive_count + negative_count == 0:
            return 0.0
            
        return 100 * (positive_count - negative_count) / (positive_count + negative_count)

    def _estimate_market_shares(self, competitors: List[Competitor]) -> None:
        """Estimate market shares for competitors based on available data."""
        # This is a simplified model - in production, you'd use more sophisticated methods
        # Here we'll use a combination of employee count, funding, and age
        
        # Calculate a composite "size score" for each competitor
        for comp in competitors:
            # Employee factor (0-100)
            employee_factor = min(100, comp.estimated_employees / 100)
            
            # Funding factor (0-100)
            funding_factor = min(100, comp.total_funding / 10000000)  # $10M = 100
            
            # Age factor (0-100)
            current_year = datetime.now().year
            age = current_year - comp.founded_year if comp.founded_year > 0 else 5
            age_factor = min(100, age * 5)  # 20 years = 100
            
            # Composite score with weights
            size_score = (employee_factor * 0.4) + (funding_factor * 0.4) + (age_factor * 0.2)
            
            # Store raw score for later normalization
            comp.size_score = size_score
        
        # Get total score to calculate relative market shares
        total_score = sum(comp.size_score for comp in competitors) or 1.0  # Avoid division by zero
        
        # Normalize market shares
        for comp in competitors:
            # Calculate normalized market share (0-1)
            comp.market_share = comp.size_score / total_score
            
            # Adjust to make more realistic (sigmoid-like transformation)
            # This gives us a more realistic distribution with a few leaders and a long tail
            comp.market_share = 0.5 * (comp.market_share ** 0.5)
            
            # Estimate growth rate if not already set
            if not comp.growth_rate:
                # Inverse correlation with size (smaller companies often grow faster)
                inverse_size = 1 - (comp.size_score / 100)
                comp.growth_rate = 0.05 + (inverse_size * 0.25)  # Range 0.05 to 0.30 (5% to 30%)

    def _fetch_competitors_demo(self, company_name: str, sector: str, keywords: Optional[List[str]] = None) -> List[Competitor]:
        """Generate synthetic competitors for demo mode."""
        # Define sector-specific competitor sets
        sector_competitors = {
            "fintech": [
                ("Square", "https://squareup.com", 15000000000, 0.12, 12.5),
                ("Stripe", "https://stripe.com", 25000000000, 0.18, 15.3),
                ("Plaid", "https://plaid.com", 5000000000, 0.05, 22.1),
                ("Robinhood", "https://robinhood.com", 10000000000, 0.08, 18.7),
                ("Chime", "https://chime.com", 8000000000, 0.06, 25.4)
            ],
            "saas": [
                ("Salesforce", "https://salesforce.com", 30000000000, 0.15, 9.8),
                ("HubSpot", "https://hubspot.com", 8000000000, 0.07, 14.2),
                ("Zoom", "https://zoom.us", 15000000000, 0.10, 16.5),
                ("Slack", "https://slack.com", 10000000000, 0.08, 12.3),
                ("Shopify", "https://shopify.com", 20000000000, 0.12, 18.4)
            ],
            "ecommerce": [
                ("Amazon", "https://amazon.com", 50000000000, 0.35, 8.5),
                ("Shopify", "https://shopify.com", 20000000000, 0.12, 18.4),
                ("BigCommerce", "https://bigcommerce.com", 3000000000, 0.04, 15.2),
                ("WooCommerce", "https://woocommerce.com", 5000000000, 0.08, 10.8),
                ("Magento", "https://magento.com", 4000000000, 0.06, 9.3)
            ],
            "biotech": [
                ("Moderna", "https://moderna.com", 25000000000, 0.12, 18.7),
                ("BioNTech", "https://biontech.com", 20000000000, 0.10, 22.3),
                ("Illumina", "https://illumina.com", 15000000000, 0.08, 14.5),
                ("Regeneron", "https://regeneron.com", 30000000000, 0.15, 11.8),
                ("CRISPR Therapeutics", "https://crisprtx.com", 10000000000, 0.06, 25.4)
            ],
            "ai": [
                ("OpenAI", "https://openai.com", 15000000000, 0.15, 35.2),
                ("Anthropic", "https://anthropic.com", 8000000000, 0.08, 42.5),
                ("Cohere", "https://cohere.ai", 3000000000, 0.04, 38.7),
                ("Hugging Face", "https://huggingface.co", 2000000000, 0.03, 30.4),
                ("Stability AI", "https://stability.ai", 1500000000, 0.02, 45.2)
            ],
            "marketplace": [
                ("Airbnb", "https://airbnb.com", 30000000000, 0.18, 14.2),
                ("DoorDash", "https://doordash.com", 20000000000, 0.12, 16.8),
                ("Uber", "https://uber.com", 40000000000, 0.25, 10.5),
                ("Instacart", "https://instacart.com", 15000000000, 0.10, 18.3),
                ("Fiverr", "https://fiverr.com", 5000000000, 0.04, 15.7)
            ],
            "crypto": [
                ("Coinbase", "https://coinbase.com", 10000000000, 0.15, 22.8),
                ("Binance", "https://binance.com", 15000000000, 0.22, 26.4),
                ("Kraken", "https://kraken.com", 5000000000, 0.08, 18.5),
                ("Uniswap", "https://uniswap.org", 3000000000, 0.05, 30.2),
                ("Aave", "https://aave.com", 2000000000, 0.03, 28.7)
            ]
        }
        
        # Default competitors if sector not recognized
        default_competitors = [
            ("Competitor A", "https://competitora.com", 5000000000, 0.07, 15.2),
            ("Competitor B", "https://competitorb.com", 8000000000, 0.10, 12.8),
            ("Competitor C", "https://competitorc.com", 3000000000, 0.05, 18.5),
            ("Competitor D", "https://competitord.com", 7000000000, 0.08, 14.3),
            ("Competitor E", "https://competitore.com", 4000000000, 0.06, 16.7)
        ]
        
        # Choose the appropriate competitor set
        competitor_set = sector_competitors.get(sector.lower(), default_competitors)
        
        # Create competitor objects with realistic values
        competitors = []
        for i, (name, url, valuation, market_share, growth_rate) in enumerate(competitor_set):
            # Skip if it's the company we're analyzing
            if self._is_same_company(name, company_name):
                continue
                
            # Generate values with some randomization
            found_year = random.randint(2005, 2020)
            employees = int(valuation / random.randint(500000, 2000000))  # Rough estimate based on valuation
            funding_rounds = random.randint(2, 6)
            total_funding = valuation * random.uniform(0.1, 0.3)  # Funding as % of valuation
            
            # Generate strengths and weaknesses relevant to the sector
            strengths, weaknesses = self._generate_strengths_weaknesses(sector, i)
            
            # Create headquarters location
            major_us_cities = ["San Francisco, CA", "New York, NY", "Boston, MA", "Austin, TX", 
                              "Seattle, WA", "Los Angeles, CA", "Chicago, IL", "Miami, FL"]
            hq = random.choice(major_us_cities)
            
            # Create CEO name
            ceo = f"{random.choice(['John', 'Sarah', 'Michael', 'Emily', 'David', 'Anna', 'Robert', 'Lisa'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson'])}"
            
            # Create competitor description
            description = self._generate_competitor_description(name, sector, found_year, valuation, growth_rate)
            
            # Create sample tech stack
            tech_stack = self._generate_tech_stack(sector)
            
            # Create recent news
            recent_news = self._generate_news_articles(name, sector)
            
            # Create and add competitor
            competitor = Competitor(
                name=name,
                url=url,
                founded_year=found_year,
                estimated_employees=employees,
                funding_rounds=funding_rounds,
                total_funding=total_funding,
                market_share=market_share,
                growth_rate=growth_rate / 100.0,  # Convert percentage to decimal
                strengths=strengths,
                weaknesses=weaknesses,
                description=description,
                ceo=ceo,
                headquarters=hq,
                last_updated=datetime.now(),
                tech_stack=tech_stack,
                social_mentions=random.randint(10, 100),
                sentiment_score=random.uniform(-30, 70),
                recent_news=recent_news
            )
            competitors.append(competitor)
        
        return competitors
        
    def _generate_strengths_weaknesses(self, sector: str, index: int) -> Tuple[List[str], List[str]]:
        """Generate relevant strengths and weaknesses based on sector."""
        # Base strengths and weaknesses that apply to all sectors
        common_strengths = [
            "Strong brand recognition",
            "Robust product ecosystem",
            "Efficient customer acquisition",
            "Solid industry partnerships",
            "Scalable business model",
            "High customer retention",
            "Experienced leadership team"
        ]
        
        common_weaknesses = [
            "High customer acquisition costs",
            "Limited geographic presence",
            "Regulatory challenges",
            "Reliance on key partnerships",
            "Competitive pricing pressure",
            "Technical debt in platform",
            "Talent retention issues"
        ]
        
        # Sector-specific strengths
        sector_strengths = {
            "fintech": [
                "Advanced risk management",
                "Regulatory compliance expertise",
                "Low-cost banking infrastructure",
                "Innovative payment solutions",
                "Automated underwriting capabilities"
            ],
            "saas": [
                "High-margin business model",
                "Predictable recurring revenue",
                "Enterprise customer base",
                "Advanced AI capabilities",
                "Extensive API ecosystem"
            ],
            "ecommerce": [
                "Efficient supply chain",
                "High conversion rate optimization",
                "Superior mobile experience",
                "Effective inventory management",
                "Strong logistics network"
            ],
            "biotech": [
                "Robust IP portfolio",
                "Strong R&D pipeline",
                "Notable clinical trial results",
                "Strategic pharma partnerships",
                "Diversified therapeutic areas"
            ],
            "ai": [
                "Cutting-edge research team",
                "Proprietary training datasets",
                "Efficient model architecture",
                "Hardware optimization",
                "Strong academic partnerships"
            ],
            "marketplace": [
                "Network effect advantages",
                "High marketplace liquidity",
                "Low take rate with high volume",
                "Strong verification systems",
                "Effective trust & safety measures"
            ],
            "crypto": [
                "Robust security infrastructure",
                "Regulatory compliance focus",
                "Advanced trading capabilities",
                "Strong liquidity partnerships",
                "Innovative staking products"
            ]
        }
        
        # Sector-specific weaknesses
        sector_weaknesses = {
            "fintech": [
                "Regulatory scrutiny",
                "High compliance costs",
                "Credit risk exposure",
                "Dependence on banking partners",
                "Fraud management challenges"
            ],
            "saas": [
                "High R&D costs",
                "Enterprise sales cycle length",
                "Integration complexity",
                "API dependency risks",
                "Churn in SMB segment"
            ],
            "ecommerce": [
                "Thin profit margins",
                "Expensive fulfillment costs",
                "Return rate management",
                "Amazon competition",
                "Seasonal revenue fluctuations"
            ],
            "biotech": [
                "Long regulatory approval timelines",
                "High clinical trial failure rates",
                "Significant capital requirements",
                "Patent cliff risks",
                "Manufacturing complexity"
            ],
            "ai": [
                "Computational cost scaling",
                "Data privacy challenges",
                "Algorithm transparency issues",
                "Regulatory uncertainty",
                "Technical talent acquisition"
            ],
            "marketplace": [
                "Disintermediation risk",
                "Quality control challenges",
                "Supply-demand balance issues",
                "Geographic expansion costs",
                "Regulatory classification risks"
            ],
            "crypto": [
                "Regulatory uncertainty",
                "Market volatility impact",
                "Security breach risks",
                "Banking relationship challenges",
                "Public perception issues"
            ]
        }
        
        # Select strengths (2 common + 2 sector-specific)
        strengths = random.sample(common_strengths, 2)
        specific_strengths = sector_strengths.get(sector, common_strengths)
        strengths.extend(random.sample(specific_strengths, min(2, len(specific_strengths))))
        
        # Select weaknesses (2 common + 1 sector-specific)
        weaknesses = random.sample(common_weaknesses, 2)
        specific_weaknesses = sector_weaknesses.get(sector, common_weaknesses)
        weaknesses.extend(random.sample(specific_weaknesses, min(1, len(specific_weaknesses))))
        
        return strengths, weaknesses
        
    def _generate_competitor_description(self, name: str, sector: str, founded_year: int, valuation: float, growth_rate: float) -> str:
        """Generate a realistic company description."""
        sector_descriptions = {
            "fintech": f"{name} is a leading financial technology company founded in {founded_year} that provides innovative payment, banking, and investment solutions. With a valuation of ${valuation/1e9:.1f}B and {growth_rate:.1f}% annual growth, they're disrupting traditional financial services through technology-driven approaches.",
            "saas": f"Founded in {founded_year}, {name} is a cloud-based software company offering solutions for business productivity, analytics, and automation. Valued at ${valuation/1e9:.1f}B with {growth_rate:.1f}% growth rate, they serve thousands of enterprise customers globally.",
            "ecommerce": f"{name} is an e-commerce platform established in {founded_year} that enables online retail and digital marketplace solutions. With a ${valuation/1e9:.1f}B valuation and {growth_rate:.1f}% year-over-year growth, they're transforming how products are bought and sold online.",
            "biotech": f"Established in {founded_year}, {name} is a biotechnology company focused on developing novel therapeutics and diagnostic technologies. Their ${valuation/1e9:.1f}B valuation and {growth_rate:.1f}% growth rate reflect their strong pipeline and innovative approach to healthcare.",
            "ai": f"{name} is an artificial intelligence company founded in {founded_year} that specializes in machine learning technologies and AI applications. Valued at ${valuation/1e9:.1f}B with impressive {growth_rate:.1f}% growth, they're at the forefront of AI innovation.",
            "marketplace": f"Since {founded_year}, {name} has operated a digital marketplace connecting buyers and sellers across various sectors. Their platform has achieved a ${valuation/1e9:.1f}B valuation with {growth_rate:.1f}% growth by efficiently solving market friction and enabling transactions.",
            "crypto": f"{name} is a cryptocurrency and blockchain company founded in {founded_year} that provides trading, custody, and blockchain infrastructure services. With a ${valuation/1e9:.1f}B valuation and {growth_rate:.1f}% growth rate, they're pioneering digital asset adoption."
        }
        
        return sector_descriptions.get(sector, f"{name} is a technology company founded in {founded_year} with a valuation of ${valuation/1e9:.1f}B and {growth_rate:.1f}% annual growth rate, focused on innovative solutions in the {sector} space.")

    def _generate_tech_stack(self, sector: str) -> List[str]:
        """Generate a realistic tech stack based on sector."""
        # Base technologies used across sectors
        base_tech = ["AWS", "Kubernetes", "Docker", "React", "Python"]
        
        # Sector-specific technologies
        sector_tech = {
            "fintech": ["Node.js", "PostgreSQL", "Redis", "Kafka", "Stripe API", "Plaid API"],
            "saas": ["GraphQL", "TypeScript", "Django", "MongoDB", "ElasticSearch", "Redis"],
            "ecommerce": ["Node.js", "MySQL", "Algolia", "Shopify API", "Stripe", "Redux"],
            "biotech": ["R", "Python", "TensorFlow", "Postgres", "Docker", "NumPy", "SciPy"],
            "ai": ["PyTorch", "TensorFlow", "CUDA", "Jupyter", "JAX", "Hugging Face", "ONNX"],
            "marketplace": ["Node.js", "PostgreSQL", "ElasticSearch", "Redis", "React Native"],
            "crypto": ["Rust", "Solidity", "Go", "Redis", "PostgreSQL", "Web3.js"]
        }
        
        # Get sector-specific tech stack or use generic one
        specific_tech = sector_tech.get(sector, ["Node.js", "MySQL", "Redis", "MongoDB"])
        
        # Combine base and specific technologies
        all_tech = base_tech + specific_tech
        
        # Return a random selection
        return random.sample(all_tech, min(len(all_tech), random.randint(5, 8)))

    def _generate_news_articles(self, company_name: str, sector: str) -> List[Dict[str, Any]]:
        """Generate synthetic news articles about a company."""
        # News article templates
        article_templates = [
            {"title": "{company} Announces New {product_type} for {market}", 
             "source": "TechCrunch",
             "sentiment": "positive"},
            
            {"title": "{company} Reports Q{quarter} Earnings: Revenue {revenue_direction} {percent}%", 
             "source": "Reuters",
             "sentiment": "variable"},
            
            {"title": "{company} Partners with {partner} to Expand {area} Capabilities", 
             "source": "Business Wire",
             "sentiment": "positive"},
            
            {"title": "{company} Raises ${amount}M in Series {series} Funding Round", 
             "source": "VentureBeat",
             "sentiment": "positive"},
            
            {"title": "{company} Named to {publication}'s {year} {list_type} List", 
             "source": "PRNewswire",
             "sentiment": "positive"},
            
            {"title": "{company} Expands to {region}, Plans to Add {number} Jobs", 
             "source": "Forbes",
             "sentiment": "positive"},
            
            {"title": "{company} CEO Discusses {topic} Challenges in {publication} Interview", 
             "source": "Bloomberg",
             "sentiment": "neutral"},
            
            {"title": "Industry Analysis: How {company} is Disrupting {industry}", 
             "source": "Harvard Business Review",
             "sentiment": "positive"},
            
            {"title": "{company} Faces {challenge_type} Challenges as {problem} Concerns Grow", 
             "source": "Wall Street Journal",
             "sentiment": "negative"},
            
            {"title": "{company} Acquires {acquired_company} to Strengthen {capability} Offering", 
             "source": "CNBC",
             "sentiment": "positive"}
        ]
        
        # Fill-in values by sector
        sector_values = {
            "fintech": {
                "product_type": ["Payment Platform", "Banking API", "Lending Service", "Investment Tool"],
                "market": ["Small Businesses", "Enterprise Clients", "Consumers", "Financial Institutions"],
                "area": ["Banking", "Payments", "Lending", "Compliance", "Risk Management"],
                "topic": ["Regulatory", "Fintech Innovation", "Banking Disruption", "Financial Inclusion"],
                "challenge_type": ["Regulatory", "Security", "Compliance", "Market"],
                "problem": ["Data Privacy", "Competition", "Regulatory Scrutiny", "Market Volatility"],
                "capability": ["Payments", "Risk Assessment", "Banking", "Financial Analysis"]
            },
            "saas": {
                "product_type": ["Enterprise Suite", "Analytics Platform", "Collaboration Tool", "Automation Solution"],
                "market": ["Enterprise", "SMB", "Agencies", "Healthcare", "Education"],
                "area": ["Data Processing", "Workflow", "Collaboration", "Analytics", "Automation"],
                "topic": ["Remote Work", "Enterprise Adoption", "AI Integration", "Data Security"],
                "challenge_type": ["Scaling", "Integration", "Competition", "Talent"],
                "problem": ["Churn", "Feature Bloat", "Technical Debt", "Market Saturation"],
                "capability": ["Analytics", "Automation", "Integration", "Customer Success"]
            },
            "ai": {
                "product_type": ["Language Model", "Vision System", "Prediction Engine", "AI Platform"],
                "market": ["Enterprise", "Healthcare", "Financial Services", "Retail"],
                "area": ["Natural Language", "Computer Vision", "Predictive Analytics", "Automation"],
                "topic": ["Model Explainability", "AI Ethics", "Model Training", "Inference Optimization"],
                "challenge_type": ["Ethical", "Technical", "Regulatory", "Adoption"],
                "problem": ["Bias", "Data Privacy", "Computational Cost", "Regulatory Uncertainty"],
                "capability": ["NLP", "Computer Vision", "Predictive Analytics", "Synthetic Data"]
            }
        }
        
        # Default values if sector not found
        default_values = {
            "product_type": ["Platform", "Solution", "Service", "Tool"],
            "market": ["Enterprise", "Consumers", "Small Business", "Global Market"],
            "area": ["Core", "Product", "Service", "Technology"],
            "topic": ["Industry", "Market", "Technology", "Strategy"],
            "challenge_type": ["Market", "Technical", "Competitive", "Operational"],
            "problem": ["Growth", "Competition", "Implementation", "Adoption"],
            "capability": ["Core", "Strategic", "Technical", "Market"]
        }
        
        # Get sector-specific values or defaults
        values = sector_values.get(sector, default_values)
        
        # Generate 3-5 random articles
        num_articles = random.randint(3, 5)
        articles = []
        
        for i in range(num_articles):
            # Select a random template
            template = random.choice(article_templates)
            
            # Current date minus random days (within last 60 days)
            days_ago = random.randint(1, 60)
            article_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Format template
            article = {
                "title": template["title"].format(
                    company=company_name,
                    product_type=random.choice(values["product_type"]),
                    market=random.choice(values["market"]),
                    quarter=random.randint(1, 4),
                    revenue_direction="Up" if random.random() > 0.3 else "Down",
                    percent=random.randint(5, 30),
                    partner=f"{random.choice(['Major', 'Leading', 'Global', 'Industry'])} {random.choice(values['market'])} Provider",
                    area=random.choice(values["area"]),
                    amount=random.randint(10, 200),
                    series=random.choice(["A", "B", "C", "D"]),
                    publication=random.choice(["Forbes", "Fortune", "Fast Company", "Inc", "Wired"]),
                    year=datetime.now().year,
                    list_type=random.choice(["Top Companies", "Disruptors", "Innovators", "Growth Leaders"]),
                    region=random.choice(["Europe", "Asia", "Latin America", "APAC", "EMEA"]),
                    number=random.randint(50, 500),
                    topic=random.choice(values["topic"]),
                    industry=sector.capitalize(),
                    challenge_type=random.choice(values["challenge_type"]),
                    problem=random.choice(values["problem"]),
                    acquired_company=f"{random.choice(['Innovative', 'Leading', 'Emerging'])} {sector.capitalize()} Startup",
                    capability=random.choice(values["capability"])
                ),
                "source": template["source"],
                "published_at": article_date,
                "url": f"https://example.com/news/{sector}/{company_name.lower().replace(' ', '-')}-{random.randint(1000, 9999)}",
                "description": "This is a mock news article for demonstration purposes."
            }
            
            articles.append(article)
        
        # Sort by date (newest first)
        articles.sort(key=lambda x: x["published_at"], reverse=True)
        
        return articles

    def _enrich_competitor_profiles(self, competitors: List[Competitor], sector: str) -> List[Competitor]:
        """
        Enrich competitor profiles with additional data such as:
        - SWOT analysis
        - Growth metrics
        - Technology stack
        - Product comparison
        """
        # Apply sector-specific enrichment
        for comp in competitors:
            if sector.lower() == 'fintech':
                self._enrich_fintech_competitor(comp)
            elif sector.lower() == 'saas':
                self._enrich_saas_competitor(comp)
            elif sector.lower() == 'ecommerce':
                self._enrich_ecommerce_competitor(comp)
            elif sector.lower() == 'ai':
                self._enrich_ai_competitor(comp)
            else:
                self._enrich_generic_competitor(comp, sector)
                
        return competitors

    def _enrich_fintech_competitor(self, competitor: Competitor) -> None:
        """Add fintech-specific enrichment to competitor profile."""
        # Check if we need to add any missing strengths/weaknesses
        if len(competitor.strengths) < 3:
            fintech_strengths = [
                "Advanced risk management systems",
                "Regulatory compliance expertise",
                "Low-cost banking infrastructure",
                "Innovative payment solutions"
            ]
            missing = 3 - len(competitor.strengths)
            competitor.strengths.extend(random.sample(fintech_strengths, missing))
        
        if len(competitor.weaknesses) < 2:
            fintech_weaknesses = [
                "Regulatory uncertainty",
                "High compliance costs",
                "Banking partnership dependencies",
                "Fraud management challenges"
            ]
            missing = 2 - len(competitor.weaknesses)
            competitor.weaknesses.extend(random.sample(fintech_weaknesses, missing))
        
        # Add fintech-specific tech stack if missing
        if not competitor.tech_stack:
            competitor.tech_stack = random.sample([
                "Node.js", "Python", "Golang", "PostgreSQL", "AWS", 
                "Kafka", "Redis", "Kubernetes", "Stripe API", "Docker"
            ], 5)

    def _enrich_saas_competitor(self, competitor: Competitor) -> None:
        """Add SaaS-specific enrichment to competitor profile."""
        # Similar implementation as for fintech_competitor
        # Add SaaS-specific metrics and attributes
        if len(competitor.strengths) < 3:
            saas_strengths = [
                "High-margin business model",
                "Predictable recurring revenue",
                "Enterprise customer base",
                "Extensive API ecosystem"
            ]
            missing = 3 - len(competitor.strengths)
            competitor.strengths.extend(random.sample(saas_strengths, missing))
            
        if len(competitor.weaknesses) < 2:
            saas_weaknesses = [
                "Long enterprise sales cycles",
                "Integration complexity",
                "Churn in SMB segment",
                "Feature creep and complexity"
            ]
            missing = 2 - len(competitor.weaknesses)
            competitor.weaknesses.extend(random.sample(saas_weaknesses, missing))

    def _enrich_ecommerce_competitor(self, competitor: Competitor) -> None:
        """Add e-commerce-specific enrichment to competitor profile."""
        # Add e-commerce specific strengths/weaknesses
        if len(competitor.strengths) < 3:
            ecommerce_strengths = [
                "Efficient supply chain management",
                "High conversion rate optimization",
                "Superior mobile shopping experience",
                "Effective inventory management"
            ]
            missing = 3 - len(competitor.strengths)
            competitor.strengths.extend(random.sample(ecommerce_strengths, missing))
            
        if len(competitor.weaknesses) < 2:
            ecommerce_weaknesses = [
                "Thin profit margins",
                "High customer acquisition costs",
                "Return management challenges",
                "Warehousing and logistics complexity"
            ]
            missing = 2 - len(competitor.weaknesses)
            competitor.weaknesses.extend(random.sample(ecommerce_weaknesses, missing))

    def _enrich_ai_competitor(self, competitor: Competitor) -> None:
        """Add AI-specific enrichment to competitor profile."""
        # Add AI-specific attributes
        if len(competitor.strengths) < 3:
            ai_strengths = [
                "Cutting-edge research team",
                "Proprietary training datasets",
                "Efficient model architecture",
                "Strong academic partnerships"
            ]
            missing = 3 - len(competitor.strengths)
            competitor.strengths.extend(random.sample(ai_strengths, missing))
            
        if len(competitor.weaknesses) < 2:
            ai_weaknesses = [
                "High computational costs",
                "Data privacy challenges",
                "Explainability limitations",
                "Talent acquisition difficulties"
            ]
            missing = 2 - len(competitor.weaknesses)
            competitor.weaknesses.extend(random.sample(ai_weaknesses, missing))
            
        # Add AI-specific tech stack if missing
        if not competitor.tech_stack:
            competitor.tech_stack = random.sample([
                "PyTorch", "TensorFlow", "Python", "CUDA", "JAX", 
                "Kubernetes", "AWS", "Docker", "Hugging Face", "Redis"
            ], 6)

    def _enrich_generic_competitor(self, competitor: Competitor, sector: str) -> None:
        """Add generic enrichment to competitor profile."""
        # Ensure minimum strengths and weaknesses
        if len(competitor.strengths) < 3:
            generic_strengths = [
                "Strong brand recognition",
                "Robust product ecosystem",
                "Efficient customer acquisition",
                "Solid industry partnerships"
            ]
            missing = 3 - len(competitor.strengths)
            competitor.strengths.extend(random.sample(generic_strengths, missing))
            
        if len(competitor.weaknesses) < 2:
            generic_weaknesses = [
                "Limited geographic presence",
                "Competitive pricing pressure",
                "Technical debt challenges",
                "Talent retention issues"
            ]
            missing = 2 - len(competitor.weaknesses)
            competitor.weaknesses.extend(random.sample(generic_weaknesses, missing))

    def competitive_positioning_analysis(self, company_data: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform detailed competitive positioning analysis.
        
        Args:
            company_data: Company data including metrics and performance indicators
            competitors: List of competitor profiles
            
        Returns:
            Comprehensive competitive positioning analysis
        """
        logger.info(f"Performing competitive positioning analysis for {company_data.get('name', 'company')}")
        
        # Check cache first
        cache_key = f"positioning_{company_data.get('name', 'company')}_{company_data.get('sector', '')}"
        cache_key = self._sanitize_cache_key(cache_key)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached positioning data for {company_data.get('name', 'company')}")
            return cached_data
        
        # Convert competitor dictionaries to Competitor objects if needed
        competitor_objects = []
        for comp in competitors:
            if isinstance(comp, dict):
                # Create Competitor object from dict
                competitor_objects.append(Competitor(
                    name=comp.get("name", ""),
                    url=comp.get("url", ""),
                    founded_year=comp.get("founded_year", 0),
                    estimated_employees=comp.get("estimated_employees", 0),
                    funding_rounds=comp.get("funding_rounds", 0),
                    total_funding=comp.get("total_funding", 0),
                    market_share=comp.get("market_share", 0),
                    growth_rate=comp.get("growth_rate", 0),
                    strengths=comp.get("strengths", []),
                    weaknesses=comp.get("weaknesses", [])
                ))
            else:
                competitor_objects.append(comp)
        
        # Get sector from company_data
        sector = company_data.get("sector", "").lower()
        
        # Define dimensions based on sector
        dimensions = self._get_sector_dimensions(sector)
        
        # Score company across dimensions
        company_scores = self._score_company_dimensions(company_data, dimensions)
        
        # Score competitors across dimensions
        competitor_scores = {}
        for comp in competitor_objects:
            competitor_scores[comp.name] = self._score_competitor_dimensions(comp, dimensions, sector)
        
        # Calculate average scores for benchmarking
        avg_scores = {}
        for dim in dimensions:
            scores = [scores[dim] for comp_name, scores in competitor_scores.items()]
            avg_scores[dim] = sum(scores) / len(scores) if scores else 0
        
        # Calculate market position
        company_total = sum(company_scores.values())
        avg_total = sum(avg_scores.values())
        
        # Calculate competitive advantages and disadvantages
        advantages = []
        disadvantages = []
        
        for dim in dimensions:
            company_score = company_scores[dim]
            avg_score = avg_scores[dim]
            
            if company_score > avg_score + 10:
                advantages.append({
                    'dimension': dim,
                    'score_difference': company_score - avg_score,
                    'description': f"Major advantage in {dim.lower()}"
                })
            elif company_score > avg_score + 5:
                advantages.append({
                    'dimension': dim,
                    'score_difference': company_score - avg_score,
                    'description': f"Advantage in {dim.lower()}"
                })
            elif company_score < avg_score - 10:
                disadvantages.append({
                    'dimension': dim,
                    'score_difference': avg_score - company_score,
                    'description': f"Major weakness in {dim.lower()}"
                })
            elif company_score < avg_score - 5:
                disadvantages.append({
                    'dimension': dim,
                    'score_difference': avg_score - company_score,
                    'description': f"Weakness in {dim.lower()}"
                })
        
        # Determine market position
        if company_total > avg_total + 20:
            position = "Market Leader"
        elif company_total > avg_total + 10:
            position = "Strong Competitor"
        elif company_total > avg_total - 10:
            position = "Average Competitor"
        elif company_total > avg_total - 20:
            position = "Challenger"
        else:
            position = "Market Laggard"
        
        # Calculate opportunity and threat scores
        opportunity_scores = self._calculate_opportunity_scores(company_data, competitor_objects, dimensions)
        threat_scores = self._calculate_threat_scores(company_data, competitor_objects, dimensions)
        
        # Generate top strategic recommendations
        recommendations = self._generate_strategic_recommendations(
            company_data, 
            advantages, 
            disadvantages,
            opportunity_scores,
            threat_scores
        )
        
        # Create final analysis
        positioning_analysis = {
            'dimensions': dimensions,
            'company_scores': company_scores,
            'competitor_scores': competitor_scores,
            'average_scores': avg_scores,
            'advantages': advantages,
            'disadvantages': disadvantages,
            'position': position,
            'overall_company_score': company_total,
            'overall_average_score': avg_total,
            'opportunity_scores': opportunity_scores,
            'threat_scores': threat_scores,
            'recommendations': recommendations,
            'company_position': self._create_position_map(company_scores, dimensions),
            'competitor_positions': {comp.name: self._create_position_map(scores, dimensions) 
                                   for comp, scores in zip(competitor_objects, competitor_scores.values())}
        }
        
        # Save to cache
        self._save_to_cache(cache_key, positioning_analysis)
        
        return positioning_analysis

    def _create_position_map(self, scores: Dict[str, float], dimensions: List[str]) -> Dict[str, float]:
        """Create a position map for visualization from dimension scores."""
        return {dim: scores.get(dim, 50) for dim in dimensions}

    def _get_sector_dimensions(self, sector: str) -> List[str]:
        """Get relevant competitive dimensions for a specific sector."""
        # Define base dimensions
        base_dimensions = ["Product", "Technology", "Market Reach", "Pricing", "Customer Experience"]
        
        # Add sector-specific dimensions
        if sector == "fintech":
            return base_dimensions + ["Regulatory Compliance", "Security", "Financial Inclusion"]
        elif sector == "saas":
            return base_dimensions + ["API Ecosystem", "Integration Capabilities", "Enterprise Readiness"]
        elif sector == "ecommerce":
            return base_dimensions + ["Logistics & Fulfillment", "Omnichannel", "Customer Loyalty"]
        elif sector == "biotech":
            return base_dimensions + ["R&D Pipeline", "Intellectual Property", "Regulatory Approval"]
        elif sector == "ai":
            return base_dimensions + ["Model Performance", "Data Advantage", "Research Capability"]
        else:
            return base_dimensions

    def _score_company_dimensions(self, company_data: Dict[str, Any], dimensions: List[str]) -> Dict[str, float]:
        """Score a company across competitive dimensions using company data."""
        scores = {}
        
        # Scoring for Product dimension
        if "Product" in dimensions:
            product_maturity = company_data.get("product_maturity_score", 50) / 10
            innovation = company_data.get("technical_innovation_score", 50) / 10
            product_score = (product_maturity + innovation) / 2 * 10
            scores["Product"] = product_score
        
        # Scoring for Technology dimension
        if "Technology" in dimensions:
            tech_debt = 10 - (company_data.get("technical_debt_score", 50) / 10)
            scalability = company_data.get("scalability_score", 50) / 10
            security = company_data.get("product_security_score", 75) / 10
            tech_score = (tech_debt + scalability + security) / 3 * 10
            scores["Technology"] = tech_score
        
        # Scoring for Market Reach dimension
        if "Market Reach" in dimensions:
            market_share = min(10, company_data.get("market_share", 0) * 100)
            category_leadership = company_data.get("category_leadership_score", 50) / 10
            user_growth = min(10, company_data.get("user_growth_rate", 10) / 10)
            market_score = (market_share + category_leadership + user_growth) / 3 * 10
            scores["Market Reach"] = market_score
        
        # Scoring for Pricing dimension
        if "Pricing" in dimensions:
            ltv_cac = min(10, company_data.get("ltv_cac_ratio", 0) * 2)
            gross_margin = company_data.get("gross_margin_percent", 70) / 10
            pricing_score = (ltv_cac + gross_margin) / 2 * 10
            scores["Pricing"] = pricing_score
        
        # Scoring for Customer Experience dimension
        if "Customer Experience" in dimensions:
            nps = (company_data.get("nps_score", 0) + 100) / 20
            support = company_data.get("support_ticket_sla_percent", 0) / 10
            cx_score = (nps + support) / 2 * 10
            scores["Customer Experience"] = cx_score
        
        # Scoring for API Ecosystem dimension (SaaS-specific)
        if "API Ecosystem" in dimensions:
            api_count = min(10, company_data.get("api_integrations_count", 0) / 5)
            api_score = api_count * 10
            scores["API Ecosystem"] = api_score
        
        # Scoring for Model Performance dimension (AI-specific)
        if "Model Performance" in dimensions:
            model_score = company_data.get("technical_innovation_score", 50)
            scores["Model Performance"] = model_score
        
        # Add scores for other dimensions based on available data
        for dim in dimensions:
            if dim not in scores:
                # Default score for dimensions we don't have explicit data for
                scores[dim] = 50.0
        
        return scores

    def _score_competitor_dimensions(self, competitor: Competitor, dimensions: List[str], sector: str) -> Dict[str, float]:
        """Score a competitor across competitive dimensions."""
        scores = {}
        
        # Use available competitor data to score dimensions
        strengths_lower = [s.lower() for s in competitor.strengths]
        weaknesses_lower = [w.lower() for w in competitor.weaknesses]
        
        # Analyze competitor strengths and weaknesses for scoring
        for dim in dimensions:
            dim_lower = dim.lower()
            base_score = 50.0  # Default base score
            
            # Adjust score based on strength/weakness match to dimension
            for strength in strengths_lower:
                if any(k in strength for k in dim_lower.split()):
                    base_score += 15.0
                    break
            
            for weakness in weaknesses_lower:
                if any(k in weakness for k in dim_lower.split()):
                    base_score -= 15.0
                    break
            
            # Special case for market reach dimension - use market share
            if dim == "Market Reach" and competitor.market_share > 0:
                base_score = 40.0 + (competitor.market_share * 300)  # Scale market share to score
            
            # Special case for Growth dimension - use growth rate
            if dim == "Growth" and competitor.growth_rate > 0:
                base_score = 40.0 + (competitor.growth_rate * 150)  # Scale growth rate to score
            
            # Clamp scores to valid range
            scores[dim] = max(10.0, min(100.0, base_score))
        
        return scores

    def _calculate_opportunity_scores(self, company_data: Dict[str, Any], competitors: List[Competitor], dimensions: List[str]) -> Dict[str, float]:
        """Calculate opportunity scores for each dimension based on competitive analysis."""
        opportunity_scores = {}
        
        # Get company scores
        company_scores = self._score_company_dimensions(company_data, dimensions)
        
        # Calculate average competitor scores
        avg_competitor_scores = {}
        for dim in dimensions:
            scores = []
            for comp in competitors:
                comp_scores = self._score_competitor_dimensions(comp, dimensions, company_data.get("sector", ""))
                scores.append(comp_scores[dim])
            avg_competitor_scores[dim] = sum(scores) / len(scores) if scores else 50.0
        
        # Calculate opportunity scores
        for dim in dimensions:
            # Opportunity is higher when competitors score low and company has potential to excel
            # Base opportunity on inverse of competitor score
            inverse_competitor_score = 100.0 - avg_competitor_scores[dim]
            
            # Adjust opportunity based on company's current position
            # Higher opportunity when company is already somewhat strong but not dominant
            position_adjustment = 0.0
            if company_scores[dim] > 70.0:
                position_adjustment = -10.0  # Already strong, less opportunity
            elif company_scores[dim] > 40.0 and company_scores[dim] < 70.0:
                position_adjustment = 10.0  # Sweet spot for opportunity
            
            # Adjust for market dynamics - faster growing markets have more opportunity
            market_growth_adjustment = min(20.0, company_data.get("market_growth_rate", 10) / 2)
            
            opportunity_scores[dim] = min(100.0, max(0.0, 
                (inverse_competitor_score * 0.6) + position_adjustment + market_growth_adjustment
            ))

    def _calculate_threat_scores(self, company_data: Dict[str, Any], competitors: List[Dict[str, Any]], dimensions: List[str]) -> Dict[str, float]:
        """
        Calculate threat scores for each dimension based on competitor positions.
        
        Args:
            company_data: Data for the company being analyzed
            competitors: List of competitor data dictionaries
            dimensions: List of dimensions to calculate threats for
            
        Returns:
            Dictionary mapping dimensions to threat scores (0-100)
        """
        try:
            threat_scores = {}
            company_scores = self._score_company_dimensions(company_data, dimensions)
            
            # Convert competitor dicts to Competitor objects if needed
            competitor_objects = []
            for comp in competitors:
                if isinstance(comp, dict):
                    # Create minimal Competitor object from dictionary
                    competitor = Competitor(
                        name=comp.get("name", "Unknown"),
                        url=comp.get("url", ""),
                        market_share=comp.get("market_share", 0),
                        growth_rate=comp.get("growth_rate", 0)
                    )
                    competitor_objects.append(competitor)
                elif hasattr(comp, 'name'):
                    # It's already a Competitor object
                    competitor_objects.append(comp)
            
            # Calculate threat for each dimension
            for dimension in dimensions:
                # Get the company's score for this dimension
                company_score = company_scores.get(dimension, 50)
                
                # Collect all competitors' scores for this dimension
                comp_scores = []
                for comp in competitor_objects:
                    # Score each competitor
                    comp_scores_dict = self._score_competitor_dimensions(comp, [dimension], company_data.get("sector", "saas"))
                    if dimension in comp_scores_dict:
                        comp_scores.append(comp_scores_dict[dimension])
                
                if not comp_scores:
                    threat_scores[dimension] = 25  # Low threat if no competitors
                    continue
                
                # Calculate threat based on:
                # 1. How many competitors score higher than the company
                # 2. By how much the top competitors exceed the company
                
                # Count competitors scoring higher
                higher_scores = [s for s in comp_scores if s > company_score]
                higher_count = len(higher_scores)
                
                # Calculate average advantage of top competitors
                top_advantage = 0
                if higher_scores:
                    top_scores = sorted(higher_scores, reverse=True)[:3]  # Top 3 competitors
                    top_advantage = sum(s - company_score for s in top_scores) / len(top_scores)
                
                # Calculate threat score (0-100)
                # Weight: 60% for count of better competitors, 40% for their advantage
                count_factor = min(1.0, higher_count / max(1, len(competitors))) * 60
                advantage_factor = min(1.0, top_advantage / 50) * 40  # Scale advantage to 0-40
                
                threat_score = count_factor + advantage_factor
                threat_scores[dimension] = threat_score
            
            return threat_scores
        except Exception as e:
            logger.error(f"Error calculating threat scores: {e}")
            # Return default low threat scores
            return {dim: 25 for dim in dimensions}

    def _generate_strategic_recommendations(self, company_data, advantages, disadvantages, opportunity_scores, threat_scores):
        """
        Generate strategic recommendations based on competitive analysis results.
        
        Args:
            company_data (dict): Company data and trends
            advantages (list): List of company advantages
            disadvantages (list): List of company disadvantages
            opportunity_scores (dict): Market opportunity scores
            threat_scores (dict): Competitive threat scores
            
        Returns:
            list: Strategic recommendations for the company
        """
        current_year = datetime.now().year
        recommendations = []
        
        # Calculate average scores for opportunities and threats
        avg_opportunity_score = sum(opportunity_scores.values()) / max(1, len(opportunity_scores))
        avg_threat_score = sum(threat_scores.values()) / max(1, len(threat_scores))
        
        # Get company strengths from advantages
        strength_score = sum(adv.get('score_difference', 0) for adv in advantages) if advantages else 0
        
        # Core recommendation based on position and scores
        if strength_score > 20 and avg_opportunity_score > 65:
            recommendations.append("Pursue aggressive growth strategy leveraging identified market opportunities")
        elif strength_score > 10 and avg_opportunity_score > 50:
            recommendations.append("Focus on moderate expansion while addressing key competitive weaknesses")
        elif avg_threat_score > 70:
            recommendations.append("Prioritize defensive strategy to counter immediate competitive threats")
        else:
            recommendations.append("Adopt balanced approach focusing on strengthening core advantages while exploring new opportunities")
        
        # Add specific recommendations based on advantages and disadvantages
        if len(advantages) >= 2:
            # Leverage top advantages
            top_advantage = advantages[0].get('description', '').replace("Major advantage in ", "").replace("Advantage in ", "")
            recommendations.append(f"Leverage {top_advantage} as a key differentiator in marketing and business development")
        
        if len(disadvantages) >= 2:
            # Address top disadvantages
            top_disadvantage = disadvantages[0].get('description', '').replace("Major weakness in ", "").replace("Weakness in ", "")
            recommendations.append(f"Prioritize addressing {top_disadvantage} to improve competitive position")
        
        # Industry-specific recommendations
        sector = company_data.get('sector', '').lower()
        if sector == 'fintech':
            recommendations.append("Strengthen compliance protocols to stay ahead of regulatory changes")
        elif sector == 'saas':
            recommendations.append("Enhance API capabilities and integration ecosystem to build platform advantages")
        elif sector == 'ecommerce':
            recommendations.append("Invest in supply chain optimization and last-mile delivery improvements")
        elif sector == 'ai':
            recommendations.append("Focus on proprietary data acquisition to strengthen AI model performance")
        
        # Opportunity-based recommendations
        if opportunity_scores:
            top_opportunity = max(opportunity_scores.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Capitalize on {top_opportunity} opportunity with targeted initiatives")
        
        # Threat-based recommendations
        if threat_scores:
            top_threat = max(threat_scores.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Develop mitigation strategies for {top_threat} threats")
        
        # Time-based recommendations
        recommendations.append(f"Develop a competitive response plan for {current_year+1} based on emerging market trends")
        
        # Focus on top 5 recommendations
        return recommendations[:5]

    def market_trends_analysis(self, sector: str) -> Dict[str, Any]:
        """
        Analyze market trends for a specific sector.
        
        Args:
            sector: Industry sector to analyze
            
        Returns:
            Dictionary containing market trend analysis
        """
        logger.info(f"Analyzing market trends for {sector}")
        
        # Check cache first
        cache_key = f"market_trends_{sector.lower()}"
        cache_key = self._sanitize_cache_key(cache_key)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached market trends for {sector}")
            return cached_data
        
        # Initialize return structure
        trend_analysis = {
            "sector": sector,
            "growth_rate": 0.0,
            "trends": [],
            "technology_shifts": [],
            "regulations": [],
            "opportunity_areas": [],
            "threat_areas": [],
            "forecast": {}
        }
        
        # If in demo mode or no API keys, generate synthetic data
        if self.demo_mode:
            trend_analysis = self._generate_synthetic_trends(sector)
        else:
            # Fetch real trend data from APIs
            api_trend_data = self._fetch_market_trends_from_apis(sector)
            
            if api_trend_data:
                trend_analysis.update(api_trend_data)
        
        # Normalize sector-specific data
        trend_analysis = self._normalize_sector_trends(trend_analysis, sector)
        
        # Calculate 5-year forecast
        trend_analysis["forecast"] = self._generate_market_forecast(trend_analysis)
        
        # Save to cache
        self._save_to_cache(cache_key, trend_analysis)
        
        return trend_analysis
    
    def _fetch_market_trends_from_apis(self, sector: str) -> Dict[str, Any]:
        """Fetch market trend data from external APIs."""
        trends_data = {
            "trends": [],
            "technology_shifts": [],
            "regulations": [],
            "growth_rate": 0.0
        }
        
        try:
            # Try to get data from news API for recent trend mentions
            if "news_api" in self.api_clients:
                news_client = self.api_clients["news_api"]
                
                # Get news articles related to the sector
                news_data = news_client.get("everything", {
                    "q": f"{sector} market trends",
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 20
                })
                
                if "articles" in news_data:
                    # Extract trends from article titles and descriptions
                    trends = self._extract_trends_from_news(news_data["articles"])
                    trends_data["trends"].extend(trends)
            
            # Try to get market growth data from financial API
            if "alpha_vantage" in self.api_clients:
                finance_client = self.api_clients["alpha_vantage"]
                
                # Get sector performance
                sector_data = finance_client.get("sector", {})
                
                if "Rank A: Real-Time Performance" in sector_data:
                    # Convert performance string to float
                    sector_match = self._map_sector_to_alpha_vantage(sector)
                    if sector_match in sector_data["Rank A: Real-Time Performance"]:
                        performance_str = sector_data["Rank A: Real-Time Performance"][sector_match]
                        try:
                            # Remove % and convert to float
                            performance = float(performance_str.strip('%'))
                            # Annualize the daily performance (rough approximation)
                            trends_data["growth_rate"] = performance * 252 / 100
                        except ValueError:
                            pass
            
            # If we didn't get growth rate, assign a reasonable value based on sector
            if trends_data["growth_rate"] == 0.0:
                trends_data["growth_rate"] = self._get_default_growth_rate(sector)
                
        except Exception as e:
            logger.error(f"Error fetching market trends: {str(e)}")
        
        return trends_data
    
    def _extract_trends_from_news(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract market trends from news articles."""
        trends = []
        trend_keywords = {
            "growing demand for", "increasing adoption of", "trend towards", 
            "shift to", "rise of", "emergence of", "expansion in",
            "decline in", "reduction of", "shrinking market for"
        }
        
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = title + " " + description
            content_lower = content.lower()
            
            # Look for trend phrases
            for keyword in trend_keywords:
                if keyword in content_lower:
                    # Find the position of the keyword
                    pos = content_lower.find(keyword)
                    # Extract the complete phrase (up to 100 chars after keyword)
                    phrase_end = min(pos + 100, len(content))
                    
                    # Find a good end point (period, comma, etc.)
                    for i in range(pos + len(keyword), phrase_end):
                        if i < len(content) and content[i] in ['.', ',', ';', ':', '!', '?']:
                            phrase_end = i
                            break
                    
                    trend_phrase = content[pos:phrase_end].strip()
                    
                    # Clean up and format the trend
                    trend_phrase = re.sub(r'\s+', ' ', trend_phrase)
                    if len(trend_phrase) > 10 and trend_phrase not in trends:
                        # Capitalize first letter
                        trends.append(trend_phrase[0].upper() + trend_phrase[1:])
        
        # If we found fewer than 3 trends, add some generic ones
        if len(trends) < 3:
            trends.extend(self._get_generic_trends(sector)[:3 - len(trends)])
        
        # Return up to 8 trends
        return trends[:8]
    
    def _map_sector_to_alpha_vantage(self, sector: str) -> str:
        """Map startup sector to Alpha Vantage sector classification."""
        sector_map = {
            "saas": "Technology",
            "fintech": "Financial",
            "ai": "Technology",
            "healthcare": "Healthcare",
            "ecommerce": "Consumer Discretionary",
            "retail": "Consumer Discretionary",
            "biotech": "Healthcare",
            "energy": "Energy",
            "manufacturing": "Industrials",
            "media": "Communication Services",
            "education": "Consumer Discretionary",
            "food": "Consumer Staples",
            "real estate": "Real Estate",
            "transportation": "Industrials",
        }
        return sector_map.get(sector.lower(), "Technology")
    
    def _get_default_growth_rate(self, sector: str) -> float:
        """Get a default market growth rate for a sector based on industry averages."""
        growth_rates = {
            "saas": 18.0,
            "fintech": 24.0,
            "ai": 35.0,
            "healthcare": 8.5,
            "ecommerce": 14.0,
            "retail": 4.5,
            "biotech": 12.0,
            "energy": 3.0,
            "manufacturing": 4.0,
            "media": 10.0,
            "education": 8.0,
            "food": 3.5,
            "real estate": 5.0,
            "transportation": 6.0,
        }
        return growth_rates.get(sector.lower(), 10.0)
    
    def _get_generic_trends(self, sector: str) -> List[str]:
        """Get generic trends for a sector when specific ones can't be found."""
        generic_trends = {
            "saas": [
                "Migration to cloud-native architectures",
                "Increasing adoption of API-first development",
                "Growing demand for vertical-specific SaaS solutions",
                "Rise of product-led growth strategies",
                "Shift towards usage-based pricing models"
            ],
            "fintech": [
                "Expansion of embedded finance in non-financial apps",
                "Growing adoption of blockchain for payments infrastructure",
                "Increasing focus on financial inclusion",
                "Rise of buy-now-pay-later services",
                "Shift towards open banking ecosystems"
            ],
            "ai": [
                "Increasing deployment of AI in enterprise workflows",
                "Growing adoption of large language models",
                "Rise of AI for content generation and personalization",
                "Shift towards explainable AI for regulated industries",
                "Growing demand for domain-specific AI solutions"
            ],
            "healthcare": [
                "Expansion of telehealth and remote patient monitoring",
                "Increasing adoption of AI for diagnostics",
                "Growing focus on mental health technology",
                "Rise of value-based care models",
                "Shift towards personalized medicine"
            ],
            "ecommerce": [
                "Growing demand for social commerce integration",
                "Increasing focus on sustainable shipping options",
                "Rise of augmented reality for product visualization",
                "Shift towards headless commerce architectures",
                "Expansion of quick-commerce (ultra-fast delivery)"
            ]
        }
        
        return generic_trends.get(sector.lower(), [
            "Digital transformation across business operations",
            "Increasing focus on sustainability initiatives",
            "Growing adoption of data-driven decision making",
            "Rise of automation for operational efficiency",
            "Shift towards flexible and remote work environments"
        ])
    
    def _normalize_sector_trends(self, trend_analysis: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Apply sector-specific adjustments to trend analysis."""
        sector = sector.lower()
        
        # Add sector-specific technology shifts
        if not trend_analysis.get("technology_shifts"):
            if sector == "saas":
                trend_analysis["technology_shifts"] = [
                    "Shift from monolithic to microservices architecture",
                    "Adoption of serverless computing",
                    "Integration of AI-powered features"
                ]
            elif sector == "fintech":
                trend_analysis["technology_shifts"] = [
                    "Blockchain and distributed ledger technology",
                    "Advanced data analytics for risk assessment",
                    "Biometric authentication methods"
                ]
            elif sector == "ai":
                trend_analysis["technology_shifts"] = [
                    "Transition from supervised to unsupervised learning",
                    "Edge AI deployment",
                    "Federated learning for privacy"
                ]
            elif sector == "healthcare":
                trend_analysis["technology_shifts"] = [
                    "IoT medical devices and wearables",
                    "AI-assisted diagnostics",
                    "Secure health data exchanges"
                ]
            else:
                trend_analysis["technology_shifts"] = [
                    "Increasing cloud adoption",
                    "Mobile-first development",
                    "Automation of manual processes"
                ]
        
        # Add sector-specific regulations
        if not trend_analysis.get("regulations"):
            if sector == "fintech":
                trend_analysis["regulations"] = [
                    "Open Banking regulations",
                    "Anti-money laundering compliance",
                    "Consumer financial protection"
                ]
            elif sector == "healthcare":
                trend_analysis["regulations"] = [
                    "Patient data privacy regulations",
                    "Telehealth reimbursement policies",
                    "Medical device approval processes"
                ]
            elif sector == "ai":
                trend_analysis["regulations"] = [
                    "AI ethics guidelines",
                    "Algorithmic transparency requirements",
                    "Data usage restrictions"
                ]
            else:
                trend_analysis["regulations"] = [
                    "Data privacy regulations",
                    "Industry-specific compliance standards",
                    "Consumer protection rules"
                ]
        
        # Add opportunity and threat areas based on sector
        if not trend_analysis.get("opportunity_areas"):
            trend_analysis["opportunity_areas"] = self._get_sector_opportunities(sector)
        
        if not trend_analysis.get("threat_areas"):
            trend_analysis["threat_areas"] = self._get_sector_threats(sector)
        
        return trend_analysis
    
    def _get_sector_opportunities(self, sector: str) -> List[str]:
        """Get sector-specific opportunity areas."""
        opportunities = {
            "saas": [
                "Integration with enterprise AI platforms",
                "Expansion into industry-specific vertical solutions",
                "SMB market penetration with simplified products"
            ],
            "fintech": [
                "Embedded finance partnerships with non-financial platforms",
                "Financial inclusion products for underserved markets",
                "Enterprise blockchain infrastructure"
            ],
            "ai": [
                "Domain-specific AI models for specialized industries",
                "AI governance and explainable AI tools",
                "Edge AI deployment for real-time applications"
            ],
            "healthcare": [
                "Remote patient monitoring platforms",
                "Mental health technology solutions",
                "Value-based care enablement tools"
            ],
            "ecommerce": [
                "Social commerce integration",
                "Sustainable logistics solutions",
                "Augmented reality shopping experiences"
            ]
        }
        
        return opportunities.get(sector.lower(), [
            "Digital transformation solutions",
            "Data analytics and business intelligence",
            "Workflow automation platforms"
        ])
    
    def _get_sector_threats(self, sector: str) -> List[str]:
        """Get sector-specific threat areas."""
        threats = {
            "saas": [
                "Increasing competition from integrated platform offerings",
                "Open-source alternatives gaining enterprise adoption",
                "Pricing pressure from larger vendors"
            ],
            "fintech": [
                "Regulatory scrutiny and compliance costs",
                "Big tech companies entering financial services",
                "Rising customer acquisition costs"
            ],
            "ai": [
                "Algorithm bias concerns limiting adoption",
                "Open-source models challenging commercial offerings",
                "Integration complexity slowing enterprise deployment"
            ],
            "healthcare": [
                "Lengthy regulatory approval processes",
                "Security and privacy concerns limiting adoption",
                "Resistance from established healthcare providers"
            ],
            "ecommerce": [
                "Rising digital advertising costs",
                "Competition from marketplace giants",
                "Supply chain disruptions"
            ]
        }
        
        return threats.get(sector.lower(), [
            "Increasing customer acquisition costs",
            "Talent shortage for specialized roles",
            "Rapid technology changes requiring constant adaptation"
        ])
    
    def _generate_market_forecast(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a 5-year market forecast based on trend analysis."""
        current_growth = trend_analysis.get("growth_rate", 10.0)
        
        # Get current year for forecast timeline
        current_year = datetime.now().year
        
        # Simple growth model with gradual tapering
        growth_rates = [
            current_growth,
            current_growth * 0.95,
            current_growth * 0.90,
            current_growth * 0.85,
            current_growth * 0.80
        ]
        
        # Calculate compound growth
        market_size = 100  # Index value starting at 100
        market_sizes = [market_size]
        
        for rate in growth_rates:
            market_size *= (1 + rate/100)
            market_sizes.append(market_size)
        
        return {
            "years": list(range(current_year, current_year + 6)),
            "growth_rates": growth_rates,
            "market_size_index": market_sizes
        }
    
    def _generate_synthetic_trends(self, sector: str) -> Dict[str, Any]:
        """Generate synthetic market trend data for demo mode."""
        # Get default growth rate for the sector
        growth_rate = self._get_default_growth_rate(sector)
        
        # Get generic trends
        trends = self._get_generic_trends(sector)
        
        # Generate trend analysis
        trend_analysis = {
            "sector": sector,
            "growth_rate": growth_rate,
            "trends": trends[:5],  # Take top 5 trends
            "technology_shifts": [],
            "regulations": [],
            "opportunity_areas": self._get_sector_opportunities(sector),
            "threat_areas": self._get_sector_threats(sector)
        }
        
        return self._normalize_sector_trends(trend_analysis, sector)

    def competitive_moat_analysis(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a company's competitive moat and defensibility.
        
        Args:
            company_data: Company data including metrics, technology, and business model
            
        Returns:
            Dictionary containing competitive moat analysis
        """
        logger.info(f"Analyzing competitive moat for {company_data.get('name', 'company')}")
        
        # Check cache first
        cache_key = f"moat_{company_data.get('name', 'company')}_{company_data.get('sector', '')}"
        cache_key = self._sanitize_cache_key(cache_key)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached moat analysis for {company_data.get('name', 'company')}")
            return cached_data
        
        # Get company sector
        sector = company_data.get("sector", "").lower()
        
        # Initialize moat analysis structure
        moat_analysis = {
            "overall_score": 0,
            "factors": {},
            "strongest": "",
            "weakest": "",
            "recommendations": []
        }
        
        # Calculate moat scores for different factors
        network_effects_score = self._calculate_network_effects_moat(company_data)
        switching_cost_score = self._calculate_switching_cost_moat(company_data)
        technology_score = self._calculate_technology_moat(company_data)
        brand_score = self._calculate_brand_moat(company_data)
        scale_score = self._calculate_scale_moat(company_data)
        
        # Store factors with scores
        factors = {
            "network_effects": network_effects_score,
            "switching_costs": switching_cost_score,
            "technology": technology_score,
            "brand": brand_score,
            "scale": scale_score
        }
        
        # Apply sector-specific adjustments
        factors = self._apply_sector_moat_adjustments(factors, sector)
        
        # Calculate overall score (weighted average of factors)
        overall_score = sum(factors.values()) / len(factors)
        
        # Find strongest and weakest factors
        strongest_factor = max(factors.items(), key=lambda x: x[1])[0]
        weakest_factor = min(factors.items(), key=lambda x: x[1])[0]
        
        # Generate recommendations
        recommendations = self._generate_moat_recommendations(factors, sector)
        
        # Compile final analysis
        moat_analysis = {
            "overall_score": overall_score,
            "factors": factors,
            "strongest": strongest_factor,
            "weakest": weakest_factor,
            "recommendations": recommendations
        }
        
        # Save to cache
        self._save_to_cache(cache_key, moat_analysis)
        
        return moat_analysis
    
    def _calculate_network_effects_moat(self, company_data: Dict[str, Any]) -> float:
        """Calculate network effects moat score."""
        # Default score
        score = 50.0
        
        try:
            # Extract relevant metrics
            user_count = company_data.get("current_users", 0)
            virality = company_data.get("viral_coefficient", 0)
            multi_sided = "marketplace" in company_data.get("business_model", "").lower()
            user_growth_rate = company_data.get("user_growth_rate", 0)
            
            # Base score on user count (logarithmic scale)
            if user_count > 0:
                user_factor = min(30, 5 * math.log10(user_count))
                score += user_factor
            
            # Adjust for virality
            if virality > 0:
                virality_factor = min(25, virality * 50)
                score += virality_factor
            
            # Adjust for multi-sided platform
            if multi_sided:
                score += 15
            
            # Adjust for growth rate
            if user_growth_rate > 0:
                growth_factor = min(15, user_growth_rate * 0.5)
                score += growth_factor
            
            # Cap at 100
            score = min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating network effects moat: {str(e)}")
        
        return score
    
    def _calculate_switching_cost_moat(self, company_data: Dict[str, Any]) -> float:
        """Calculate switching cost moat score."""
        # Default score
        score = 50.0
        
        try:
            # Extract relevant metrics
            business_model = company_data.get("business_model", "").lower()
            integrations = company_data.get("api_integrations_count", 0)
            data_exports = company_data.get("data_portability", True)
            contract_length = company_data.get("avg_contract_length_months", 0)
            retention_rate = company_data.get("retention_rate", 0)
            
            # Base score on business model
            if "subscription" in business_model:
                score += 10
            if "enterprise" in business_model:
                score += 15
            
            # Adjust for integrations
            if integrations > 0:
                integration_factor = min(25, 5 * integrations)
                score += integration_factor
            
            # Adjust for data portability (inverse - harder to export = higher switching cost)
            if not data_exports:
                score += 15
            
            # Adjust for contract length
            if contract_length > 0:
                contract_factor = min(20, contract_length / 2)
                score += contract_factor
            
            # Adjust for retention rate (as a proxy for switching cost)
            if retention_rate > 0:
                retention_factor = retention_rate * 20
                score += retention_factor
            
            # Cap at 100
            score = min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating switching cost moat: {str(e)}")
        
        return score
    
    def _calculate_technology_moat(self, company_data: Dict[str, Any]) -> float:
        """Calculate technology moat score."""
        # Default score
        score = 50.0
        
        try:
            # Extract relevant metrics
            patents = company_data.get("patent_count", 0)
            tech_team_size = company_data.get("development_team_size", 0)
            tech_team_ratio = company_data.get("tech_talent_ratio", 0)
            proprietary_tech = company_data.get("proprietary_technology_score", 0)
            
            # Base score on patents
            if patents > 0:
                patent_factor = min(30, patents * 3)
                score += patent_factor
            
            # Adjust for tech team size and ratio
            if tech_team_size > 0:
                team_factor = min(20, math.log2(tech_team_size) * 5)
                score += team_factor
            
            if tech_team_ratio > 0:
                ratio_factor = tech_team_ratio * 20
                score += ratio_factor
            
            # Adjust for proprietary technology
            if proprietary_tech > 0:
                tech_factor = proprietary_tech * 0.3
                score += tech_factor
            
            # Cap at 100
            score = min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating technology moat: {str(e)}")
        
        return score
    
    def _calculate_brand_moat(self, company_data: Dict[str, Any]) -> float:
        """Calculate brand moat score."""
        # Default score
        score = 50.0
        
        try:
            # Extract relevant metrics
            company_age = company_data.get("company_age_years", 0)
            sentiment = company_data.get("brand_sentiment", 0)
            social = company_data.get("social_media_following", 0)
            marketing_budget = company_data.get("marketing_budget", 0)
            b2c = "b2c" in company_data.get("business_model", "").lower()
            
            # Base score on company age
            if company_age > 0:
                age_factor = min(15, company_age)
                score += age_factor
            
            # Adjust for sentiment
            if sentiment > 0:
                sentiment_factor = sentiment * 0.2
                score += sentiment_factor
            
            # Adjust for social media
            if social > 0:
                social_factor = min(20, 3 * math.log10(social + 1))
                score += social_factor
            
            # Adjust for marketing budget
            if marketing_budget > 0:
                budget_factor = min(15, marketing_budget / 10000)
                score += budget_factor
            
            # Adjust for business model (B2C vs B2B)
            if b2c:
                score += 10
            
            # Cap at 100
            score = min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating brand moat: {str(e)}")
        
        return score
    
    def _calculate_scale_moat(self, company_data: Dict[str, Any]) -> float:
        """Calculate scale moat score."""
        # Default score
        score = 50.0
        
        try:
            # Extract relevant metrics
            revenue = company_data.get("monthly_revenue", 0) * 12  # Annual
            employees = company_data.get("employee_count", 0)
            market_share = company_data.get("market_share", 0)
            cost_structure = company_data.get("cost_structure", {})
            
            # Base score on revenue (logarithmic scale)
            if revenue > 0:
                revenue_factor = min(30, 3 * math.log10(revenue))
                score += revenue_factor
            
            # Adjust for team size
            if employees > 0:
                employee_factor = min(20, 5 * math.log10(employees + 1))
                score += employee_factor
            
            # Adjust for market share
            if market_share > 0:
                # Convert from percentage to decimal if needed
                if market_share > 1:
                    market_share = market_share / 100
                
                share_factor = market_share * 100
                score += share_factor
            
            # Adjust for cost structure
            if isinstance(cost_structure, dict) and "fixed_cost_ratio" in cost_structure:
                fixed_ratio = cost_structure["fixed_cost_ratio"]
                # Higher fixed costs = more scale advantage
                if fixed_ratio > 0.5:
                    scale_factor = (fixed_ratio - 0.5) * 40
                    score += scale_factor
            
            # Cap at 100
            score = min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating scale moat: {str(e)}")
        
        return score
    
    def _apply_sector_moat_adjustments(self, factors: Dict[str, float], sector: str) -> Dict[str, float]:
        """Apply sector-specific adjustments to moat factors."""
        adjusted_factors = factors.copy()
        
        # Software/SaaS
        if sector in ["saas", "software"]:
            adjusted_factors["switching_costs"] *= 1.2
            adjusted_factors["technology"] *= 1.1
            adjusted_factors["scale"] *= 0.9
        
        # Fintech
        elif sector == "fintech":
            adjusted_factors["scale"] *= 1.2
            adjusted_factors["technology"] *= 1.1
            adjusted_factors["brand"] *= 0.9
        
        # E-commerce
        elif sector in ["ecommerce", "retail"]:
            adjusted_factors["brand"] *= 1.2
            adjusted_factors["scale"] *= 1.1
            adjusted_factors["technology"] *= 0.9
        
        # Marketplace
        elif sector == "marketplace":
            adjusted_factors["network_effects"] *= 1.3
            adjusted_factors["scale"] *= 1.1
            adjusted_factors["technology"] *= 0.8
        
        # Social/Media
        elif sector in ["social", "media"]:
            adjusted_factors["network_effects"] *= 1.3
            adjusted_factors["brand"] *= 1.2
            adjusted_factors["switching_costs"] *= 0.8
        
        # Healthcare
        elif sector in ["healthcare", "biotech"]:
            adjusted_factors["technology"] *= 1.3
            adjusted_factors["scale"] *= 1.1
            adjusted_factors["network_effects"] *= 0.8
        
        # AI/ML
        elif sector in ["ai", "ml", "artificial intelligence"]:
            adjusted_factors["technology"] *= 1.3
            adjusted_factors["scale"] *= 1.1
            adjusted_factors["brand"] *= 0.9
        
        # Cap all scores at 100
        for factor in adjusted_factors:
            adjusted_factors[factor] = min(100, max(0, adjusted_factors[factor]))
        
        return adjusted_factors
    
    def _generate_moat_recommendations(self, factors: Dict[str, float], sector: str) -> List[Dict[str, Any]]:
        """Generate recommendations for improving competitive moat."""
        recommendations = []
        
        # Sort factors by score (ascending)
        sorted_factors = sorted(factors.items(), key=lambda x: x[1])
        
        # Focus on the 2 weakest factors
        for factor, score in sorted_factors[:2]:
            if factor == "network_effects" and score < 70:
                recommendations.append({
                    "factor": "network_effects",
                    "strength": "weak" if score < 50 else "moderate",
                    "recommendation": "Strengthen network effects by implementing viral features and incentives for user referrals."
                })
            
            elif factor == "switching_costs" and score < 70:
                recommendations.append({
                    "factor": "switching_costs",
                    "strength": "weak" if score < 50 else "moderate",
                    "recommendation": "Increase switching costs by expanding integrations with other tools and enhancing data value over time."
                })
            
            elif factor == "technology" and score < 70:
                recommendations.append({
                    "factor": "technology",
                    "strength": "weak" if score < 50 else "moderate",
                    "recommendation": "Strengthen technology moat by investing in proprietary technology and considering strategic patent filings."
                })
            
            elif factor == "brand" and score < 70:
                recommendations.append({
                    "factor": "brand",
                    "strength": "weak" if score < 50 else "moderate",
                    "recommendation": "Build brand strength through consistent messaging, thought leadership, and customer success stories."
                })
            
            elif factor == "scale" and score < 70:
                recommendations.append({
                    "factor": "scale",
                    "strength": "weak" if score < 50 else "moderate",
                    "recommendation": "Focus on scaling operations to achieve cost advantages and increase market presence."
                })
        
        # Add sector-specific recommendation
        if sector == "saas":
            recommendations.append({
                "factor": "ecosystem",
                "strength": "opportunity",
                "recommendation": "Develop a partner ecosystem and marketplace to create platform effects and increase stickiness."
            })
        elif sector == "fintech":
            recommendations.append({
                "factor": "regulation",
                "strength": "opportunity",
                "recommendation": "Use regulatory compliance as a competitive advantage by making it a core capability."
            })
        elif sector in ["ai", "ml"]:
            recommendations.append({
                "factor": "data_network",
                "strength": "opportunity",
                "recommendation": "Create data network effects by using customer data to improve the product for all users."
            })
        else:
            recommendations.append({
                "factor": "customer_success",
                "strength": "opportunity",
                "recommendation": "Invest in customer success and implementation services to improve retention and referrals."
            })
        
        return recommendations

# Add this function at the end of the file
def analyze_competitive_landscape(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze competitive landscape and market positioning
    
    Parameters:
    data (dict): Input data about the company and market
    
    Returns:
    dict: Comprehensive competitive analysis results
    """
    try:
        # Extract API keys if provided
        api_keys = data.get("api_keys", {})
        
        # Extract company and sector info
        company_name = data.get("company_name", "")
        sector = data.get("sector", "")
        keywords = data.get("keywords", [])
        
        # Create analyzer instance
        analyzer = CompetitiveIntelligence(api_keys=api_keys)
        
        # Get competitor analysis
        competitors = analyzer.get_competitors(
            company_name=company_name,
            sector=sector,
            keywords=keywords
        )
        
        # Convert competitor objects to dictionaries if needed
        if competitors and isinstance(competitors[0], Competitor):
            competitors = [c.to_dict() for c in competitors]
        
        # Get positioning analysis if company data provided
        positioning = {}
        if "company_data" in data:
            positioning = analyzer.competitive_positioning_analysis(
                company_data=data["company_data"],
                competitors=competitors
            )
        
        # Get market trends
        market_trends = analyzer.market_trends_analysis(sector=sector)
        
        # Get competitive moat analysis if company data provided
        moat_analysis = {}
        if "company_data" in data:
            moat_analysis = analyzer.competitive_moat_analysis(data["company_data"])
        
        # Combine all results
        result = {
            "company": company_name,
            "sector": sector,
            "competitors": competitors,
            "competitive_positioning": positioning,
            "market_trends": market_trends,
            "competitive_moat": moat_analysis,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        return result
    except Exception as e:
        # Return error with traceback for debugging
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }
