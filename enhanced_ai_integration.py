"""
Enhanced AI Integration for FlashDNA

This module provides robust integration with OpenAI and other AI services
to enhance the narrative capabilities of the FlashDNA system.

Key features:
1. Unified API client with proper error handling and retries
2. Secure credential management
3. Offline fallback mechanisms
4. Optimized prompt engineering for startup analysis
5. Caching to reduce API costs
"""

import os
import json
import time
import logging
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_ai")

# Type definitions
DocumentData = Dict[str, Any]
AIResponse = Dict[str, Any]
PromptData = Dict[str, Any]

class AICredentialManager:
    """
    Securely manages API credentials for AI services with fallback mechanisms.
    
    This class provides:
    1. Secure loading of credentials from environment variables
    2. Fallback to config files
    3. Credential rotation support
    4. Validation of credentials before use
    """
    
    def __init__(self):
        """Initialize the credential manager."""
        self.credentials = {}
        self.load_credentials()
    
    def load_credentials(self):
        """Load credentials from environment variables and config files."""
        # Try to load OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # If not in environment, try config file
        if not openai_key:
            try:
                if os.path.exists("config.py"):
                    from config import OPENAI_API_KEY as CONFIG_KEY
                    openai_key = CONFIG_KEY
            except (ImportError, AttributeError):
                logger.warning("OpenAI API key not found in config.py")
            
            # Try JSON config as another fallback
            if not openai_key and os.path.exists("config.json"):
                try:
                    with open("config.json", "r") as f:
                        config = json.load(f)
                        openai_key = config.get("OPENAI_API_KEY")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error loading config.json: {e}")
        
        if openai_key:
            self.credentials["openai"] = {"api_key": openai_key}
            logger.info("OpenAI API key loaded successfully")
        else:
            logger.warning("No OpenAI API key found. AI features will use fallback methods.")
        
        # Similarly load DeepSeek API key if it exists (for backward compatibility)
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            try:
                if os.path.exists("config.py"):
                    from config import DEEPSEEK_API_KEY as DS_CONFIG_KEY
                    deepseek_key = DS_CONFIG_KEY
            except (ImportError, AttributeError):
                pass
        
        if deepseek_key:
            self.credentials["deepseek"] = {"api_key": deepseek_key}
            logger.info("DeepSeek API key loaded successfully")
    
    def get_credential(self, service: str, credential_name: str = "api_key") -> Optional[str]:
        """
        Get a credential for the specified service.
        
        Args:
            service: The service name (e.g., "openai", "deepseek")
            credential_name: The name of the credential (default: "api_key")
            
        Returns:
            The credential value or None if not available
        """
        service_creds = self.credentials.get(service, {})
        return service_creds.get(credential_name)
    
    def has_credential(self, service: str) -> bool:
        """
        Check if credentials exist for the specified service.
        
        Args:
            service: The service name (e.g., "openai", "deepseek")
            
        Returns:
            True if credentials exist, False otherwise
        """
        return service in self.credentials and "api_key" in self.credentials[service]
    
    def validate_credential(self, service: str) -> bool:
        """
        Validate that a credential is working by making a test API call.
        
        Args:
            service: The service name (e.g., "openai", "deepseek")
            
        Returns:
            True if the credential is valid, False otherwise
        """
        if not self.has_credential(service):
            return False
        
        if service == "openai":
            return self._validate_openai()
        elif service == "deepseek":
            return self._validate_deepseek()
        
        return False
    
    def _validate_openai(self) -> bool:
        """Validate OpenAI API key with a minimal request."""
        try:
            api_key = self.get_credential("openai")
            if not api_key:
                return False
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Make a simple models list request (lightweight)
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI credential validation failed: {e}")
            return False
    
    def _validate_deepseek(self) -> bool:
        """Validate DeepSeek API key with a minimal request."""
        try:
            api_key = self.get_credential("deepseek")
            if not api_key:
                return False
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # This endpoint might need to be adjusted based on DeepSeek's API
            response = requests.get(
                "https://api.deepseek.com/v1/models",
                headers=headers,
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"DeepSeek credential validation failed: {e}")
            return False

class AIResponseCache:
    """
    Cache for AI responses to reduce API costs and improve performance.
    
    This class provides:
    1. Disk-based caching of AI responses
    2. TTL-based expiration
    3. Thread-safe operations
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_days: int = 7):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files (default: temp directory)
            ttl_days: Number of days before a cache entry expires
        """
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "flashdna_ai_cache")
        self.ttl = timedelta(days=ttl_days)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"AI response cache initialized at {self.cache_dir}")
    
    def _get_cache_key(self, prompt: str, service: str, model: str) -> str:
        """
        Generate a cache key from the prompt, service, and model.
        
        Args:
            prompt: The prompt string
            service: The AI service name
            model: The model name
            
        Returns:
            A hash string to use as the cache key
        """
        # Create a hash of the prompt + service + model
        hash_obj = hashlib.md5()
        hash_obj.update(f"{prompt}|{service}|{model}".encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: The cache key
            
        Returns:
            The file path for the cache entry
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, prompt: str, service: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if available and not expired.
        
        Args:
            prompt: The prompt string
            service: The AI service name
            model: The model name
            
        Returns:
            The cached response or None if not found or expired
        """
        try:
            cache_key = self._get_cache_key(prompt, service, model)
            cache_path = self._get_cache_path(cache_key)
            
            # Check if cache file exists
            if not os.path.exists(cache_path):
                return None
            
            # Check if cache entry is expired
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - cache_time > self.ttl:
                # Remove expired cache entry
                os.remove(cache_path)
                return None
            
            # Read and return the cached response
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    def set(self, prompt: str, service: str, model: str, response: Dict[str, Any]) -> bool:
        """
        Save a response to the cache.
        
        Args:
            prompt: The prompt string
            service: The AI service name
            model: The model name
            response: The response to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = self._get_cache_key(prompt, service, model)
            cache_path = self._get_cache_path(cache_key)
            
            # Save the response to the cache
            with open(cache_path, 'w') as f:
                json.dump(response, f)
            
            return True
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
            return False
    
    def clear(self, max_age_days: Optional[int] = None) -> int:
        """
        Clear expired cache entries or all entries.
        
        Args:
            max_age_days: Maximum age in days (if None, clears all)
            
        Returns:
            Number of entries cleared
        """
        try:
            count = 0
            now = datetime.now()
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                # If max_age_days is None, clear all entries
                if max_age_days is None:
                    os.remove(file_path)
                    count += 1
                    continue
                
                # Otherwise, check the file's age
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                max_age = timedelta(days=max_age_days)
                
                if now - file_time > max_age:
                    os.remove(file_path)
                    count += 1
            
            return count
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return 0

class AIClient:
    """
    Unified client for interacting with AI services with robust error handling.
    
    This class provides:
    1. Unified interface for multiple AI services
    2. Automatic retries and error handling
    3. Rate limiting support
    4. Response caching
    5. Fallback mechanisms for offline operation
    """
    
    def __init__(self, 
                 credential_manager: Optional[AICredentialManager] = None,
                 cache: Optional[AIResponseCache] = None,
                 max_retries: int = 3,
                 timeout: int = 60,
                 use_caching: bool = True):
        """
        Initialize the AI client.
        
        Args:
            credential_manager: Credential manager for API keys
            cache: Response cache
            max_retries: Maximum number of retries for API calls
            timeout: Timeout in seconds for API calls
            use_caching: Whether to use caching
        """
        self.credential_manager = credential_manager or AICredentialManager()
        self.cache = cache or AIResponseCache()
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_caching = use_caching
        
        # Create a session with retry functionality
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Determine which services are available
        self.available_services = self._get_available_services()
        self.preferred_service = self._determine_preferred_service()
        
        logger.info(f"AI client initialized with preferred service: {self.preferred_service}")
    
    def _get_available_services(self) -> List[str]:
        """
        Get a list of available AI services based on credentials.
        
        Returns:
            List of available service names
        """
        available = []
        
        if self.credential_manager.has_credential("openai"):
            available.append("openai")
        
        if self.credential_manager.has_credential("deepseek"):
            available.append("deepseek")
        
        return available
    
    def _determine_preferred_service(self) -> str:
        """
        Determine the preferred AI service based on availability and reliability.
        
        Returns:
            The preferred service name
        """
        # OpenAI is preferred if available
        if "openai" in self.available_services:
            return "openai"
        
        # Otherwise, use DeepSeek if available
        if "deepseek" in self.available_services:
            return "deepseek"
        
        # If no services are available, use offline fallback
        return "offline"
    
    def get_completion(self, 
                      prompt: str, 
                      service: Optional[str] = None,
                      model: Optional[str] = None,
                      temperature: float = 0.3,
                      max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Get a completion from an AI service.
        
        Args:
            prompt: The prompt string
            service: The AI service to use (default: preferred service)
            model: The model to use (default: service-specific default)
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response with text and metadata
        """
        # Determine which service to use
        service = service or self.preferred_service
        
        # If the service is not available, fall back to next best
        if service not in self.available_services and service != "offline":
            if "openai" in self.available_services:
                service = "openai"
            elif "deepseek" in self.available_services:
                service = "deepseek"
            else:
                service = "offline"
        
        # Check cache first if caching is enabled
        if self.use_caching:
            cached_response = self.cache.get(prompt, service, model or "default")
            if cached_response:
                logger.info(f"Using cached response for service: {service}")
                return cached_response
        
        # Make the API call based on the service
        try:
            if service == "openai":
                response = self._call_openai(prompt, model, temperature, max_tokens)
            elif service == "deepseek":
                response = self._call_deepseek(prompt, model, temperature, max_tokens)
            else:
                # Offline fallback
                response = self._get_offline_completion(prompt)
            
            # Cache the response if successful and caching is enabled
            if self.use_caching and response and "error" not in response:
                self.cache.set(prompt, service, model or "default", response)
            
            return response
        except Exception as e:
            logger.error(f"Error getting completion from {service}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return offline fallback on error
            return self._get_offline_completion(prompt)
    
    def _call_openai(self, 
                    prompt: str, 
                    model: Optional[str] = None,
                    temperature: float = 0.3,
                    max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt string
            model: The model to use (default: "gpt-3.5-turbo")
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response with text and metadata
        """
        api_key = self.credential_manager.get_credential("openai")
        if not api_key:
            logger.warning("No OpenAI API key available")
            return self._get_offline_completion(prompt)
        
        model = model or "gpt-3.5-turbo"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content from the response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    
                    return {
                        "text": content,
                        "service": "openai",
                        "model": model,
                        "usage": response_data.get("usage", {}),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.warning("Unexpected OpenAI response format")
                    return {
                        "error": "Unexpected response format",
                        "service": "openai"
                    }
            else:
                logger.warning(f"OpenAI API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "service": "openai",
                    "details": response.text
                }
        except requests.exceptions.Timeout:
            logger.warning("OpenAI API request timed out")
            return {
                "error": "Request timed out",
                "service": "openai"
            }
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenAI API request error: {e}")
            return {
                "error": f"Request error: {str(e)}",
                "service": "openai"
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {
                "error": f"Error: {str(e)}",
                "service": "openai"
            }
    
    def _call_deepseek(self, 
                      prompt: str, 
                      model: Optional[str] = None,
                      temperature: float = 0.3,
                      max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Call the DeepSeek API.
        
        Args:
            prompt: The prompt string
            model: The model to use (default: "deepseek-chat")
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response with text and metadata
        """
        api_key = self.credential_manager.get_credential("deepseek")
        if not api_key:
            logger.warning("No DeepSeek API key available")
            return self._get_offline_completion(prompt)
        
        model = model or "deepseek-chat"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = self.session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content from the response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    
                    return {
                        "text": content,
                        "service": "deepseek",
                        "model": model,
                        "usage": response_data.get("usage", {}),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.warning("Unexpected DeepSeek response format")
                    return {
                        "error": "Unexpected response format",
                        "service": "deepseek"
                    }
            else:
                logger.warning(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "service": "deepseek",
                    "details": response.text
                }
        except requests.exceptions.Timeout:
            logger.warning("DeepSeek API request timed out")
            return {
                "error": "Request timed out",
                "service": "deepseek"
            }
        except requests.exceptions.RequestException as e:
            logger.warning(f"DeepSeek API request error: {e}")
            return {
                "error": f"Request error: {str(e)}",
                "service": "deepseek"
            }
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            return {
                "error": f"Error: {str(e)}",
                "service": "deepseek"
            }
    
    def _get_offline_completion(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a reasonable fallback response when no AI services are available.
        
        Args:
            prompt: The prompt string
            
        Returns:
            A basic fallback response
        """
        # Extract analysis type from the prompt to generate a relevant response
        analysis_type = "general analysis"
        
        if "executive summary" in prompt.lower():
            analysis_type = "executive summary"
            response_text = self._generate_offline_executive_summary(prompt)
        elif "competitive" in prompt.lower():
            analysis_type = "competitive analysis"
            response_text = self._generate_offline_competitive_analysis(prompt)
        elif "market trends" in prompt.lower():
            analysis_type = "market trends analysis"
            response_text = self._generate_offline_market_trends(prompt)
        elif "recommendation" in prompt.lower():
            analysis_type = "recommendations"
            response_text = self._generate_offline_recommendations(prompt)
        elif "intangible" in prompt.lower():
            analysis_type = "intangible analysis"
            response_text = self._generate_offline_intangible_analysis(prompt)
        else:
            response_text = self._generate_generic_offline_response(prompt)
        
        return {
            "text": response_text,
            "service": "offline",
            "model": "fallback",
            "timestamp": datetime.now().isoformat(),
            "analysis_type": analysis_type
        }
    
    def _extract_company_details(self, prompt: str) -> Dict[str, Any]:
        """
        Extract company details from the prompt for better offline responses.
        
        Args:
            prompt: The prompt string
            
        Returns:
            Dictionary with extracted company details
        """
        details = {
            "name": "the company",
            "sector": "technology",
            "stage": "early stage"
        }
        
        # Try to extract company name
        name_matches = [
            line for line in prompt.split('\n') 
            if any(term in line.lower() for term in ["company:", "startup:", "name:"])
        ]
        
        if name_matches:
            parts = name_matches[0].split(":")
            if len(parts) > 1:
                details["name"] = parts[1].strip()
        
        # Try to extract sector
        sector_matches = [
            line for line in prompt.split('\n') 
            if any(term in line.lower() for term in ["sector:", "industry:"])
        ]
        
        if sector_matches:
            parts = sector_matches[0].split(":")
            if len(parts) > 1:
                details["sector"] = parts[1].strip()
        
        # Try to extract stage
        stage_matches = [
            line for line in prompt.split('\n') 
            if "stage:" in line.lower()
        ]
        
        if stage_matches:
            parts = stage_matches[0].split(":")
            if len(parts) > 1:
                details["stage"] = parts[1].strip()
        
        return details
    
    def _generate_offline_executive_summary(self, prompt: str) -> str:
        """Generate an offline executive summary."""
        details = self._extract_company_details(prompt)
        
        return f"""
Executive Summary

{details['name']} is a {details['stage']} company in the {details['sector']} sector. Based on the available data, the company shows signs of both promising growth potential and areas requiring attention.

The CAMP analysis indicates relative strengths in market positioning and team execution, while capital efficiency metrics suggest opportunities for optimization. The current business model appears viable, with market conditions generally supportive of the company's value proposition.

Key risk factors include competitive pressure and execution challenges typical of companies at this stage. However, the team's domain expertise and the expanding market create favorable conditions for growth if properly executed.

Investors should monitor key performance indicators closely, particularly customer acquisition costs and retention metrics, which will be critical determinants of long-term success.
        """.strip()
    
    def _generate_offline_competitive_analysis(self, prompt: str) -> str:
        """Generate an offline competitive analysis."""
        details = self._extract_company_details(prompt)
        
        return f"""
Competitive Analysis

Market Position
{details['name']} currently occupies a challenger position in the {details['sector']} market. The company faces competition from established players with greater resources and market share, but has identified a specific niche where it can differentiate through innovation and specialized offerings.

Competitive Advantages
The company's primary advantages include its agile development approach, lower overhead structure compared to incumbents, and targeted focus on underserved segments of the market. These advantages provide a foundation for growth, though they must be leveraged strategically to overcome the resource advantages of larger competitors.

Threats and Challenges
The most significant competitive threats come from: 1) Incumbent market leaders who may decide to enter the company's niche, 2) Other well-funded startups targeting the same market segment, and 3) Potential future consolidation in the industry that could change market dynamics significantly.

Differentiation Strategy
To strengthen its competitive position, {details['name']} should focus on: 1) Deepening its technological advantages in core areas, 2) Building stronger network effects through platform enhancements, 3) Establishing strategic partnerships to extend reach, and 4) Continuing to optimize its pricing strategy to maximize value perception against competitors.
        """.strip()
    
    def _generate_offline_market_trends(self, prompt: str) -> str:
        """Generate offline market trends analysis."""
        details = self._extract_company_details(prompt)
        
        return f"""
Market Trends Analysis

The {details['sector']} sector is currently experiencing several significant trends that will likely impact {details['name']}'s growth trajectory:

1. Digital Transformation Acceleration: Organizations across industries are accelerating their digital transformation initiatives, creating expanded market opportunities for technology solutions that enable this shift. This trend is expected to continue for the next 3-5 years, providing a favorable growth environment.

2. Shift to Cloud-Native Solutions: The market is increasingly favoring cloud-native architectures over traditional on-premises deployments, with implications for both product development priorities and go-to-market strategies. Companies that embrace this shift will be better positioned for long-term success.

3. Data-Driven Decision Making: Organizations are increasingly prioritizing solutions that provide actionable insights from their data, creating opportunities for analytical tools and platforms that can deliver meaningful business intelligence.

4. Remote Work Infrastructure: The permanent shift toward hybrid work models has created sustained demand for tools that support distributed teams, secure access, and seamless collaboration regardless of location.

5. AI/ML Integration: The integration of artificial intelligence and machine learning capabilities into mainstream business applications represents both an opportunity and a competitive necessity for companies in this space.

To capitalize on these trends, {details['name']} should align its product roadmap and market positioning to clearly address how it enables customers to leverage these shifts for competitive advantage.
        """.strip()
    
    def _generate_offline_recommendations(self, prompt: str) -> str:
        """Generate offline strategic recommendations."""
        details = self._extract_company_details(prompt)
        
        return f"""
Strategic Recommendations

## Optimize Unit Economics
Priority: High

The current LTV:CAC ratio could be improved to enhance overall capital efficiency. Focus on reducing customer acquisition costs through channel optimization and improving conversion rates. Simultaneously, develop strategies to increase customer lifetime value through improved retention and expansion revenue initiatives. Target achieving an LTV:CAC ratio of at least 3:1 within the next two quarters.

## Strengthen Product Differentiation
Priority: High

In this competitive market, product differentiation needs to be more clearly defined and communicated. Conduct competitive analysis to identify unique value propositions that can be strengthened. Focus development resources on features that highlight key differentiators rather than matching competitors feature-for-feature. Develop clear messaging that articulates these differentiators for sales and marketing materials.

## Expand Strategic Partnerships
Priority: Medium

Develop a structured partnership program to accelerate market reach and reduce direct sales costs. Identify potential partners whose offerings complement {details['name']}'s solution and whose customer base aligns with the ideal customer profile. Create clear partnership tiers with appropriate incentives, enablement resources, and co-marketing opportunities.

## Strengthen Governance Structure
Priority: Medium

As the company grows, formalize the governance structure with clear roles, responsibilities, and decision-making processes. Establish a regular cadence of strategic reviews with defined metrics and accountability mechanisms. Consider adding independent board members with relevant industry experience to provide additional oversight and guidance.
        """.strip()
    
    def _generate_offline_intangible_analysis(self, prompt: str) -> str:
        """Generate offline intangible analysis."""
        return json.dumps({
            "intangible_score": 65.0,
            "confidence": "medium",
            "team_conviction": 70.0,
            "vision_clarity": 65.0,
            "narrative_strength": 60.0,
            "execution_signals": 65.0,
            "key_positives": [
                "Clear articulation of market opportunity",
                "Demonstrated domain expertise",
                "Logical problem-solution framework"
            ],
            "key_concerns": [
                "Limited discussion of competitive differentiation",
                "Potential gaps in go-to-market strategy",
                "Ambitious timeline relative to resources"
            ],
            "overall_assessment": "The pitch demonstrates solid understanding of the market with a reasonable approach to addressing customer needs, though some aspects of execution planning could be strengthened."
        }, indent=2)
    
    def _generate_generic_offline_response(self, prompt: str) -> str:
        """Generate a generic offline response."""
        details = self._extract_company_details(prompt)
        
        return f"""
Based on the available information about {details['name']}, a {details['stage']} company in the {details['sector']} sector, we can provide an initial assessment.

The company appears to have both strengths and challenges typical of organizations at this stage. To provide more specific insights, additional data would be beneficial, particularly around key performance metrics, market positioning, and competitive landscape.

For a more comprehensive analysis, we recommend focusing on quantitative metrics such as growth rates, unit economics, and capital efficiency, as well as qualitative factors like team composition, product differentiation, and market trends.
        """.strip()

class EnhancedAINarrativeEngine:
    """
    Advanced AI narrative generation for startup analysis.
    
    This class provides:
    1. Optimized prompts for startup analysis
    2. Parallel processing for faster report generation
    3. Graceful fallbacks for missing data
    4. Consistent formatting and structure
    """
    
    def __init__(self, 
                 ai_client: Optional[AIClient] = None,
                 use_threading: bool = True,
                 max_workers: int = 4):
        """
        Initialize the narrative engine.
        
        Args:
            ai_client: AI client to use
            use_threading: Whether to use threading for parallel processing
            max_workers: Maximum number of worker threads
        """
        self.ai_client = ai_client or AIClient()
        self.use_threading = use_threading
        self.max_workers = max_workers
    
    def generate_executive_summary(self, doc: DocumentData) -> str:
        """
        Generate an AI-powered executive summary of the startup analysis.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative executive summary
        """
        prompt = self._create_executive_summary_prompt(doc)
        response = self.ai_client.get_completion(prompt)
        return response.get("text", "Executive summary generation failed.")
    
    def _create_executive_summary_prompt(self, doc: DocumentData) -> str:
        """Create the prompt for generating an executive summary."""
        # Extract key metrics
        name = doc.get("name", doc.get("company_name", "the startup"))
        sector = doc.get("sector", "technology")
        stage = doc.get("stage", "early stage")
        camp_score = doc.get("camp_score", 0)
        capital_score = doc.get("capital_score", 0)
        market_score = doc.get("market_score", 0)
        advantage_score = doc.get("advantage_score", 0)
        people_score = doc.get("people_score", 0)
        success_prob = doc.get("success_prob", 0)
        runway = doc.get("runway_months", 0)
        monthly_revenue = doc.get("monthly_revenue", 0)
        
        # Format currency values
        if monthly_revenue >= 1_000_000:
            monthly_revenue_str = f"${monthly_revenue/1_000_000:.1f}M"
        elif monthly_revenue >= 1_000:
            monthly_revenue_str = f"${monthly_revenue/1_000:.1f}K"
        else:
            monthly_revenue_str = f"${monthly_revenue:.2f}"
        
        # Extract patterns for strengths/weaknesses
        strengths = []
        weaknesses = []
        for pattern in doc.get("patterns_matched", []):
            if isinstance(pattern, dict):
                if pattern.get("is_positive", False):
                    strengths.append(pattern.get("name", ""))
                else:
                    weaknesses.append(pattern.get("name", ""))
        
        strengths_text = ", ".join(strengths[:3]) if strengths else "No clear strengths identified"
        weaknesses_text = ", ".join(weaknesses[:3]) if weaknesses else "No clear weaknesses identified"
        
        return f"""
        You are an expert startup analyst writing an executive summary for investors.
        Generate a concise, insightful executive summary for {name}, a {stage} company in the {sector} sector.
        
        Key metrics:
        - CAMP Score: {camp_score}/100
        - Capital Score: {capital_score}/100
        - Market Score: {market_score}/100
        - Advantage Score: {advantage_score}/100
        - People Score: {people_score}/100
        - Success Probability: {success_prob}%
        - Runway: {runway} months
        - Monthly Revenue: {monthly_revenue_str}
        
        Strengths: {strengths_text}
        Weaknesses: {weaknesses_text}
        
        The executive summary should:
        1. Open with a concise overview of the startup and its potential
        2. Highlight the most compelling aspects of the investment opportunity
        3. Identify key risks and how they might be mitigated
        4. Conclude with an overall assessment of the startup's prospects
        
        Write in a clear, professional tone that would appeal to sophisticated investors.
        Do not mention AI or that this was AI-generated.
        Do not use phrases like "based on the data provided" or "according to the metrics".
        Write as if you are a human analyst who has deeply studied this startup.
        Be concise but insightful, around 3-4 paragraphs total.
        """
    
    def generate_competitive_analysis(self, doc: DocumentData) -> str:
        """
        Generate an AI-powered competitive analysis of the startup.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative competitive analysis
        """
        prompt = self._create_competitive_analysis_prompt(doc)
        response = self.ai_client.get_completion(prompt)
        return response.get("text", "Competitive analysis generation failed.")
    
    def _create_competitive_analysis_prompt(self, doc: DocumentData) -> str:
        """Create the prompt for generating a competitive analysis."""
        name = doc.get("name", doc.get("company_name", "the startup"))
        sector = doc.get("sector", "technology")
        stage = doc.get("stage", "early stage")
        
        # Extract competitors
        competitors = doc.get("competitors", [])
        competitor_names = [c.get("name", f"Competitor {i+1}") 
                           for i, c in enumerate(competitors) if isinstance(c, dict)]
        
        competitors_text = ", ".join(competitor_names) if competitor_names else "unknown competitors"
        
        moat_analysis = doc.get("moat_analysis", {})
        moat_score = moat_analysis.get("overall_score", 0) if isinstance(moat_analysis, dict) else 0
        
        competitive_positioning = doc.get("competitive_positioning", {})
        position = competitive_positioning.get("position", "challenger") if isinstance(competitive_positioning, dict) else "challenger"
        
        return f"""
        You are an expert competitive analyst for venture capital investments.
        Generate a detailed competitive analysis for {name}, a {stage} company in the {sector} sector.
        
        The startup competes with: {competitors_text}
        Current competitive position: {position}
        Moat strength score: {moat_score}/100
        
        Please analyze the competitive landscape and include these sections:
        
        ## Market Position
        Analyze the startup's current position in the market and how it compares to competitors.
        
        ## Competitive Advantages
        Identify the startup's key competitive advantages and how defensible they are.
        
        ## Threats and Challenges
        Analyze the main competitive threats facing the startup.
        
        ## Differentiation Strategy
        Recommend how the startup can further differentiate from competitors.
        
        Write in a clear, professional tone. Be specific and actionable with your analysis.
        Do not mention that this was AI-generated.
        Do not use generic statements like "based on the information provided."
        Write as if you are a human analyst who has deeply studied this startup and its industry.
        """
    
    def generate_strategic_recommendations(self, doc: DocumentData) -> str:
        """
        Generate strategic recommendations based on the startup analysis.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative of strategic recommendations
        """
        prompt = self._create_recommendations_prompt(doc)
        response = self.ai_client.get_completion(prompt)
        return response.get("text", "Strategic recommendations generation failed.")
    
    def _create_recommendations_prompt(self, doc: DocumentData) -> str:
        """Create the prompt for generating strategic recommendations."""
        name = doc.get("name", doc.get("company_name", "the startup"))
        camp_score = doc.get("camp_score", 0)
        capital_score = doc.get("capital_score", 0)
        market_score = doc.get("market_score", 0)
        advantage_score = doc.get("advantage_score", 0)
        people_score = doc.get("people_score", 0)
        
        # Find lowest scoring area for focused recommendations
        scores = {
            "Capital Efficiency": capital_score,
            "Market Dynamics": market_score,
            "Advantage Moat": advantage_score,
            "People & Performance": people_score
        }
        weakest_area = min(scores.items(), key=lambda x: x[1])
        
        return f"""
        You are an expert startup advisor generating strategic recommendations for {name}.
        
        CAMP Framework Scores:
        - Overall: {camp_score}/100
        - Capital Efficiency: {capital_score}/100
        - Market Dynamics: {market_score}/100
        - Advantage Moat: {advantage_score}/100
        - People & Performance: {people_score}/100
        
        The startup's weakest area is: {weakest_area[0]} ({weakest_area[1]}/100)
        
        Generate 3-5 specific, actionable strategic recommendations.
        Each recommendation should:
        
        1. Focus on addressing key challenges or capitalizing on opportunities
        2. Be specific to this startup's situation, not generic advice
        3. Include clear next steps or implementation guidance
        4. Have a priority level (High, Medium, or Low)
        
        Format each recommendation as:
        
        ## [Recommendation Title]
        Priority: [Priority Level]
        
        [Detailed description with specific actions and expected outcomes]
        
        Make at least 2 recommendations that address the weakest area.
        Do not mention that this was AI-generated.
        Do not use clichés or generic startup advice.
        Write as if you are a veteran startup advisor with deep domain expertise.
        """
    
    def generate_market_trends_analysis(self, doc: DocumentData) -> str:
        """
        Generate an analysis of relevant market trends for the startup.
        
        Args:
            doc: The document containing all startup analysis data
            
        Returns:
            A narrative analysis of market trends
        """
        prompt = self._create_market_trends_prompt(doc)
        response = self.ai_client.get_completion(prompt)
        return response.get("text", "Market trends analysis generation failed.")
    
    def _create_market_trends_prompt(self, doc: DocumentData) -> str:
        """Create the prompt for generating market trends analysis."""
        sector = doc.get("sector", "technology")
        market_size = doc.get("market_size", 0)
        market_growth_rate = doc.get("market_growth_rate", 0)
        
        market_trends = doc.get("market_trends", {})
        existing_trends = []
        if isinstance(market_trends, dict) and "trends" in market_trends:
            existing_trends = market_trends.get("trends", [])
        
        trends_text = ", ".join(existing_trends) if existing_trends else "No specific trends identified"
        
        return f"""
        You are an expert market analyst specializing in the {sector} sector.
        Generate an insightful analysis of current and emerging market trends relevant to a startup in this space.
        
        Market data:
        - Market size: ${market_size:,.0f}
        - Market growth rate: {market_growth_rate}%/year
        - Identified trends: {trends_text}
        
        Your analysis should:
        1. Identify 3-5 key market trends that are most relevant to this sector
        2. Explain how each trend creates opportunities or threats
        3. Provide specific insights on how a startup could capitalize on these trends
        4. Include a forward-looking perspective on how the market might evolve in the next 2-3 years
        
        Write in a clear, data-driven, and insightful tone that demonstrates deep market expertise.
        Do not mention that this was AI-generated.
        Do not use phrases like "based on the data provided" or "according to the information."
        Write as if you are a human analyst who has been tracking this market for years.
        """
    
    def analyze_intangible_factors(self, pitch_text: str) -> Dict[str, Any]:
        """
        Perform deep analysis of intangible factors from pitch deck text.
        This goes beyond basic sentiment to extract nuanced signals about team,
        vision, and execution potential.
        
        Args:
            pitch_text: The extracted text from the pitch deck
            
        Returns:
            Dict containing intangible factor analysis
        """
        if not pitch_text or len(pitch_text.strip()) < 100:
            return {
                "intangible_score": 50.0,
                "confidence": "low",
                "team_conviction": 50.0,
                "vision_clarity": 50.0,
                "narrative_strength": 50.0,
                "execution_signals": 50.0,
                "overall_assessment": "Insufficient pitch data for intangible analysis"
            }
        
        prompt = self._create_intangible_analysis_prompt(pitch_text)
        response = self.ai_client.get_completion(prompt)
        
        try:
            # Try to parse JSON from the response
            import re
            text = response.get("text", "{}")
            
            # Find JSON within the response
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If not properly formatted, try to find JSON block directly
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
                
            # If all else fails, create a default response
            return {
                "intangible_score": 60.0,  # Slightly optimistic default
                "confidence": "low",
                "team_conviction": 60.0,
                "vision_clarity": 60.0,
                "narrative_strength": 60.0,
                "execution_signals": 60.0,
                "key_positives": ["Unable to extract specific positives"],
                "key_concerns": ["Unable to extract specific concerns"],
                "overall_assessment": "The pitch contains some promising elements but a full assessment requires more detailed analysis."
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse intangible analysis JSON")
            return {
                "intangible_score": 50.0,
                "confidence": "low",
                "team_conviction": 50.0,
                "vision_clarity": 50.0,
                "narrative_strength": 50.0,
                "execution_signals": 50.0,
                "overall_assessment": "Error parsing intangible analysis"
            }
    
    def _create_intangible_analysis_prompt(self, pitch_text: str) -> str:
        """Create the prompt for analyzing intangible factors from pitch text."""
        # Truncate pitch text if too long
        max_length = 8000
        if len(pitch_text) > max_length:
            pitch_text = pitch_text[:max_length] + "... [truncated]"
            
        return f"""
        You are an expert venture capital analyst who specializes in evaluating the intangible factors
        of startups based on their pitch materials. Analyze the following pitch deck text to extract
        signals about the team, vision, and execution potential.
        
        PITCH DECK TEXT:
        {pitch_text}
        
        Perform a comprehensive analysis of the intangible factors in this pitch. 
        Evaluate the following aspects:
        
        1. Team conviction - How passionate and committed does the team appear?
        2. Vision clarity - How clear and compelling is the vision?
        3. Narrative strength - How well-constructed and persuasive is the narrative?
        4. Execution signals - Are there indications of strong execution capability?
        
        Return your analysis as a JSON object with the following structure:
        
        ```json
        {{
            "intangible_score": <0-100 score>,
            "confidence": "<low|medium|high>",
            "team_conviction": <0-100 score>,
            "vision_clarity": <0-100 score>,
            "narrative_strength": <0-100 score>,
            "execution_signals": <0-100 score>,
            "key_positives": ["<positive signal 1>", "<positive signal 2>", ...],
            "key_concerns": ["<concern 1>", "<concern 2>", ...],
            "overall_assessment": "<1-2 sentence assessment>"
        }}
        ```
        
        Base your scores on patterns you've seen in successful vs. unsuccessful startups.
        Be honest in your assessment - neither overly optimistic nor overly critical.
        """
    
    def generate_report_content(self, doc: DocumentData, sections: List[str] = None) -> Dict[str, str]:
        """
        Generate multiple report sections in parallel for faster processing.
        
        Args:
            doc: The document containing all startup analysis data
            sections: List of section names to generate
            
        Returns:
            Dict mapping section names to generated content
        """
        if sections is None:
            sections = ["executive_summary", "competitive_analysis", "recommendations", "market_trends"]
        
        content = {}
        
        if self.use_threading:
            # Generate content in parallel
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                if "executive_summary" in sections:
                    futures["executive_summary"] = executor.submit(self.generate_executive_summary, doc)
                    
                if "competitive_analysis" in sections:
                    futures["competitive_analysis"] = executor.submit(self.generate_competitive_analysis, doc)
                    
                if "recommendations" in sections:
                    futures["recommendations"] = executor.submit(self.generate_strategic_recommendations, doc)
                    
                if "market_trends" in sections:
                    futures["market_trends"] = executor.submit(self.generate_market_trends_analysis, doc)
                
                # Collect results
                for section, future in futures.items():
                    try:
                        content[section] = future.result()
                    except Exception as e:
                        logger.error(f"Error generating {section}: {str(e)}")
                        content[section] = f"Error generating {section}: {str(e)}"
        else:
            # Generate content sequentially
            for section in sections:
                try:
                    if section == "executive_summary":
                        content[section] = self.generate_executive_summary(doc)
                    elif section == "competitive_analysis":
                        content[section] = self.generate_competitive_analysis(doc)
                    elif section == "recommendations":
                        content[section] = self.generate_strategic_recommendations(doc)
                    elif section == "market_trends":
                        content[section] = self.generate_market_trends_analysis(doc)
                except Exception as e:
                    logger.error(f"Error generating {section}: {str(e)}")
                    content[section] = f"Error generating {section}: {str(e)}"
        
        return content
    
    def compute_intangible_score(self, doc: DocumentData) -> float:
        """
        Enhanced replacement for the existing intangible_api.compute_intangible_llm function.
        Analyzes pitch text to derive a deep intangible score.
        
        Args:
            doc: The document containing pitch deck text
            
        Returns:
            Intangible score (0-100)
        """
        pitch_text = doc.get("pitch_deck_text", "")
        if not pitch_text or len(pitch_text.strip()) < 100:
            return 50.0  # Default score for insufficient data
            
        analysis = self.analyze_intangible_factors(pitch_text)
        return analysis.get("intangible_score", 50.0)

# Compatibility function to replace existing intangible_api.compute_intangible_llm
def compute_intangible_llm(doc: DocumentData) -> float:
    """
    Enhanced replacement for the existing compute_intangible_llm function.
    
    Args:
        doc: The document containing pitch deck text
        
    Returns:
        Intangible score (0-100)
    """
    engine = EnhancedAINarrativeEngine()
    return engine.compute_intangible_score(doc)

# Singleton instance of the narrative engine
_narrative_engine = None

def get_narrative_engine() -> EnhancedAINarrativeEngine:
    """
    Get the singleton instance of the narrative engine.
    
    Returns:
        EnhancedAINarrativeEngine instance
    """
    global _narrative_engine
    if _narrative_engine is None:
        _narrative_engine = EnhancedAINarrativeEngine()
    return _narrative_engine

# For backward compatibility with code that imports from ai_narrative_engine
initialize_narrative_engine = get_narrative_engine

# Compatibility function to replace existing ai_narrative_engine.AINarrativeEngine
class AINarrativeEngine(EnhancedAINarrativeEngine):
    """Compatibility class for backward compatibility with code using the old AINarrativeEngine class"""
    pass
