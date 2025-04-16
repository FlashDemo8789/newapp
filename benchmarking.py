import pandas as pd
import numpy as np
import logging
import aiohttp
import asyncio
import time
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """
    Container for results of a startup benchmarking analysis.
    
    Attributes:
        company_metrics: The extracted metrics from the company being benchmarked
        industry_benchmarks: Reference benchmarks for the company's sector and stage
        percentiles: The percentile rankings of the company's metrics compared to benchmarks
        performance_summary: Textual summary of the company's overall performance
        radar_data: Structured data for radar chart visualization
        recommendations: List of actionable recommendations based on the analysis
        metric_comparisons: Detailed comparison data for each metric
        percentile: Overall percentile rank compared to peers
        peer_comparisons: Comparison data with specific peers
        data_source: Source of the benchmark data (e.g., "default", "api", "cached")
        data_timestamp: When the benchmark data was last updated
    """
    company_metrics: Dict[str, float]
    industry_benchmarks: Dict[str, Dict[str, float]]
    percentiles: Dict[str, int]
    performance_summary: str
    radar_data: Dict[str, Any]
    recommendations: List[str]
    metric_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    percentile: int = 50
    peer_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    data_source: str = "default"
    data_timestamp: Optional[datetime] = None


class BenchmarkCache:
    """
    Cache manager for benchmark data to reduce API calls and improve performance.
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours for cached data
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.memory_cache = {}
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
                logger.info(f"Created cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to create cache directory: {str(e)}")
    
    def get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Create a safe filename from the key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def get(self, key: str) -> Optional[Dict]:
        """
        Get cached data if available and not expired.
        
        Args:
            key: Cache key (typically sector/stage)
            
        Returns:
            Cached data or None if not found or expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            if cache_entry["timestamp"] + timedelta(hours=self.ttl_hours) > datetime.now():
                logger.debug(f"Cache hit (memory) for {key}")
                return cache_entry["data"]
        
        # Check file cache
        cache_path = self.get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Check if cache is expired
                cache_time = datetime.fromisoformat(cache_entry["timestamp"])
                if cache_time + timedelta(hours=self.ttl_hours) > datetime.now():
                    # Update memory cache
                    self.memory_cache[key] = {
                        "data": cache_entry["data"],
                        "timestamp": cache_time
                    }
                    logger.debug(f"Cache hit (file) for {key}")
                    return cache_entry["data"]
                else:
                    logger.debug(f"Cache expired for {key}")
            except Exception as e:
                logger.warning(f"Error reading cache for {key}: {str(e)}")
        
        logger.debug(f"Cache miss for {key}")
        return None
    
    def set(self, key: str, data: Dict) -> None:
        """
        Store data in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        now = datetime.now()
        
        # Update memory cache
        self.memory_cache[key] = {
            "data": data,
            "timestamp": now
        }
        
        # Update file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "data": data,
                    "timestamp": now.isoformat()
                }, f)
            logger.debug(f"Cached data for {key}")
        except Exception as e:
            logger.warning(f"Error writing cache for {key}: {str(e)}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            # Clear specific key
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            cache_path = self.get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.debug(f"Cleared cache for {key}")
                except Exception as e:
                    logger.warning(f"Error clearing cache for {key}: {str(e)}")
        else:
            # Clear all cache
            self.memory_cache = {}
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        logger.warning(f"Error removing cache file {filename}: {str(e)}")
            
            logger.info("Cleared all benchmark cache")


class BenchmarkAPIClient:
    """
    Client for fetching benchmark data from external APIs.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_url: Optional[str] = None,
                timeout: int = 10,
                retries: int = 3):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for authentication
            api_url: Base URL for the API
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        self.api_key = api_key or os.environ.get("BENCHMARK_API_KEY")
        self.api_url = api_url or os.environ.get("BENCHMARK_API_URL", "https://api.flashdna.com/v1")
        self.timeout = timeout
        self.retries = retries
        
        if not self.api_key:
            logger.warning("No API key provided for benchmark data. Using default benchmarks only.")
    
    async def fetch_benchmarks(self, sector: str, stage: str) -> Optional[Dict]:
        """
        Fetch benchmark data from the API.
        
        Args:
            sector: Business sector (e.g., 'saas', 'fintech')
            stage: Company stage (e.g., 'seed', 'series-a')
            
        Returns:
            Dictionary of benchmark data or None if request failed
        """
        if not self.api_key:
            return None
        
        url = f"{self.api_url}/benchmarks"
        params = {
            "sector": sector,
            "stage": stage
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Implement retry logic
        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully fetched benchmark data for {sector}/{stage}")
                            return data
                        elif response.status == 404:
                            logger.warning(f"No benchmark data available for {sector}/{stage}")
                            return None
                        else:
                            logger.warning(f"API error: {response.status} - {await response.text()}")
                            
                            if attempt < self.retries - 1:
                                backoff_time = 2 ** attempt  # Exponential backoff
                                logger.info(f"Retrying in {backoff_time} seconds...")
                                await asyncio.sleep(backoff_time)
                            else:
                                logger.error(f"Failed to fetch benchmark data after {self.retries} attempts")
                                return None
                                
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for {sector}/{stage}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Benchmark API request timed out repeatedly")
                    return None
            except Exception as e:
                logger.error(f"Error fetching benchmark data: {str(e)}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    async def fetch_peer_data(self, 
                             company_id: Optional[str], 
                             sector: str, 
                             stage: str, 
                             limit: int = 5) -> List[Dict]:
        """
        Fetch data on peer companies for comparison.
        
        Args:
            company_id: ID of the company to exclude from peers (optional)
            sector: Business sector
            stage: Company stage
            limit: Maximum number of peers to return
            
        Returns:
            List of peer company data dictionaries
        """
        if not self.api_key:
            return []
        
        url = f"{self.api_url}/peers"
        params = {
            "sector": sector,
            "stage": stage,
            "limit": limit
        }
        
        if company_id:
            params["exclude"] = company_id
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched {len(data)} peer companies")
                        return data
                    else:
                        logger.warning(f"Failed to fetch peer data: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching peer data: {str(e)}")
            return []
    
    async def submit_company_metrics(self, company_data: Dict[str, Any]) -> bool:
        """
        Submit company metrics to the API for aggregation and improvement of benchmarks.
        
        Args:
            company_data: Company metrics data
            
        Returns:
            True if submission was successful, False otherwise
        """
        if not self.api_key:
            return False
        
        url = f"{self.api_url}/metrics"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=company_data,
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    if response.status in (200, 201, 202):
                        logger.info("Successfully submitted company metrics")
                        return True
                    else:
                        logger.warning(f"Failed to submit metrics: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error submitting metrics: {str(e)}")
            return False


class BenchmarkEngine:
    """
    Engine for comparing startup metrics against industry benchmarks.
    
    This class provides functionality to benchmark a startup's key metrics
    against industry standards for similar companies at the same stage and sector.
    It can generate percentile rankings, performance summaries, visualization data,
    and actionable recommendations.
    """

    # Metrics where higher values are better
    HIGHER_BETTER_METRICS = [
        'user_growth_rate', 'revenue_growth_rate', 'ltv_cac_ratio',
        'gross_margin', 'runway_months', 'net_retention_rate', 'dau_mau_ratio'
    ]
    
    # Metrics where lower values are better
    LOWER_BETTER_METRICS = [
        'churn_rate', 'cac_payback_months', 'burn_multiple'
    ]
    
    # Key metrics to include in radar visualization
    RADAR_METRICS = [
        'user_growth_rate', 'revenue_growth_rate', 'churn_rate',
        'ltv_cac_ratio', 'burn_multiple', 'gross_margin'
    ]
    
    # Metrics to display in the UI (with human-readable names)
    DISPLAY_METRICS = {
        'user_growth_rate': 'User Growth Rate',
        'revenue_growth_rate': 'Revenue Growth Rate', 
        'churn_rate': 'Churn Rate',
        'ltv_cac_ratio': 'LTV:CAC Ratio',
        'burn_multiple': 'Burn Multiple',
        'gross_margin': 'Gross Margin',
        'net_retention_rate': 'Net Retention Rate',
        'runway_months': 'Runway (months)',
        'cac_payback_months': 'CAC Payback (months)'
    }

    def __init__(self, 
                benchmark_db: Optional[Dict] = None,
                api_key: Optional[str] = None,
                api_url: Optional[str] = None,
                use_api: bool = True,
                cache_ttl_hours: int = 24):
        """
        Initialize the BenchmarkEngine with benchmark data.
        
        Args:
            benchmark_db: Optional dictionary containing benchmark data by sector and stage.
                         If None, default benchmarks will be used.
            api_key: API key for fetching external benchmark data
            api_url: Base URL for the benchmark API
            use_api: Whether to use the API for benchmark data (if available)
            cache_ttl_hours: How long to cache benchmark data (in hours)
        """
        self._default_benchmarks = benchmark_db or self._load_default_benchmarks()
        self._cache = BenchmarkCache(ttl_hours=cache_ttl_hours)
        self._use_api = use_api
        
        if use_api:
            self._api_client = BenchmarkAPIClient(api_key=api_key, api_url=api_url)
        else:
            self._api_client = None
            
        logger.info(f"BenchmarkEngine initialized with data for {len(self._default_benchmarks)} sectors")
        logger.info(f"API integration: {'Enabled' if use_api else 'Disabled'}")

    async def benchmark_startup_async(self, 
                                    startup_data: Dict[str, Any], 
                                    sector: Optional[str] = None, 
                                    stage: Optional[str] = None,
                                    submit_metrics: bool = False) -> BenchmarkResult:
        """
        Asynchronously benchmark a startup against industry standards.
        
        Args:
            startup_data: Dictionary containing the startup's metrics and information
            sector: Optional sector override. If None, will use sector from startup_data
            stage: Optional stage override. If None, will use stage from startup_data
            submit_metrics: Whether to submit the metrics to the API (if enabled)
            
        Returns:
            BenchmarkResult containing the analysis results
        """
        # Extract or default sector and stage, normalizing to lowercase
        try:
            sector = (sector or startup_data.get('sector', 'saas')).lower()
            stage = (stage or startup_data.get('stage', 'seed')).lower()
            
            # Log the benchmark request
            logger.info(f"Benchmarking startup in {sector} sector at {stage} stage")
            
            # Extract and process the startup's metrics
            company_metrics = self._extract_metrics(startup_data)
            
            # Try to get benchmarks from different sources in order:
            # 1. API (if enabled)
            # 2. Cache
            # 3. Default benchmarks
            benchmarks = {}
            data_source = "default"
            data_timestamp = datetime.now()
            
            # Try API first (if enabled)
            if self._use_api and self._api_client:
                # Try to get from cache first
                cache_key = f"{sector}_{stage}"
                cached_data = self._cache.get(cache_key)
                
                if cached_data:
                    benchmarks = cached_data
                    data_source = "cached"
                    logger.info(f"Using cached benchmark data for {sector}/{stage}")
                else:
                    # Fetch from API
                    api_data = await self._api_client.fetch_benchmarks(sector, stage)
                    if api_data:
                        benchmarks = api_data
                        data_source = "api"
                        # Cache the API response
                        self._cache.set(cache_key, api_data)
                        logger.info(f"Using API benchmark data for {sector}/{stage}")
            
            # If no benchmarks yet, use defaults
            if not benchmarks:
                benchmarks = self._get_benchmarks_for_sector_stage(sector, stage)
                data_source = "default"
                logger.info(f"Using default benchmark data for {sector}/{stage}")
            
            # Get peer data if available
            peers = []
            if self._use_api and self._api_client:
                company_id = startup_data.get('id')
                peers = await self._api_client.fetch_peer_data(company_id, sector, stage)
            
            # Submit metrics if requested
            if submit_metrics and self._use_api and self._api_client:
                # Submit asynchronously without waiting for response
                asyncio.create_task(self._api_client.submit_company_metrics({
                    "metrics": company_metrics,
                    "sector": sector,
                    "stage": stage,
                    "timestamp": datetime.now().isoformat()
                }))
            
            # Calculate percentiles for each metric
            percentiles = self._calculate_percentiles(company_metrics, benchmarks)
            
            # Calculate overall percentile
            overall_percentile = int(sum(percentiles.values()) / max(1, len(percentiles)))
            
            # Generate textual analysis
            summary = self._generate_performance_summary(percentiles, sector, stage)
            
            # Prepare data for radar visualization
            radar_data = self._prepare_radar_data(company_metrics, benchmarks, percentiles)
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(percentiles, company_metrics, benchmarks)
            
            # Generate metric comparisons for detailed table view
            metric_comparisons = self._generate_metric_comparisons(company_metrics, benchmarks, percentiles)
            
            # Generate peer comparisons
            peer_comparisons = self._process_peer_data(peers, company_metrics) if peers else []
            if not peer_comparisons:
                peer_comparisons = self._generate_synthetic_peer_comparisons(startup_data, sector, stage)
            
            # Create and return the result object
            result = BenchmarkResult(
                company_metrics=company_metrics,
                industry_benchmarks=benchmarks,
                percentiles=percentiles,
                performance_summary=summary,
                radar_data=radar_data,
                recommendations=recommendations,
                metric_comparisons=metric_comparisons,
                percentile=overall_percentile,
                peer_comparisons=peer_comparisons,
                data_source=data_source,
                data_timestamp=data_timestamp
            )
            
            logger.info(f"Successfully completed benchmarking for {sector} startup at {stage} stage")
            logger.debug(f"Overall percentile: {overall_percentile}")
            return result
            
        except Exception as e:
            logger.error(f"Error benchmarking startup: {str(e)}", exc_info=True)
            # Return a minimal valid result with error information instead of raising
            return BenchmarkResult(
                company_metrics={},
                industry_benchmarks={},
                percentiles={'error': 50},
                performance_summary=f"Error benchmarking startup: {str(e)}",
                radar_data={'metrics': [], 'company_values': [], 'median_values': []},
                recommendations=["Fix benchmarking data issues and try again."],
                percentile=50,
                data_source="error"
            )
            
    def benchmark_startup(self, 
                         startup_data: Dict[str, Any], 
                         sector: Optional[str] = None, 
                         stage: Optional[str] = None,
                         submit_metrics: bool = False) -> BenchmarkResult:
        """
        Synchronous version of benchmark_startup_async.
        
        This method handles running the async event loop for environments
        that don't support async/await directly.
        
        Args:
            startup_data: Dictionary containing the startup's metrics and information
            sector: Optional sector override. If None, will use sector from startup_data
            stage: Optional stage override. If None, will use stage from startup_data
            submit_metrics: Whether to submit the metrics to the API (if enabled)
            
        Returns:
            BenchmarkResult containing the analysis results
        """
        if self._use_api:
            # Use asyncio to run the async version
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(
                self.benchmark_startup_async(
                    startup_data=startup_data,
                    sector=sector,
                    stage=stage,
                    submit_metrics=submit_metrics
                )
            )
            return result
        else:
            # If API is disabled, we can just use synchronous code
            try:
                sector = (sector or startup_data.get('sector', 'saas')).lower()
                stage = (stage or startup_data.get('stage', 'seed')).lower()
                
                company_metrics = self._extract_metrics(startup_data)
                benchmarks = self._get_benchmarks_for_sector_stage(sector, stage)
                percentiles = self._calculate_percentiles(company_metrics, benchmarks)
                overall_percentile = int(sum(percentiles.values()) / max(1, len(percentiles)))
                summary = self._generate_performance_summary(percentiles, sector, stage)
                radar_data = self._prepare_radar_data(company_metrics, benchmarks, percentiles)
                recommendations = self._generate_recommendations(percentiles, company_metrics, benchmarks)
                metric_comparisons = self._generate_metric_comparisons(company_metrics, benchmarks, percentiles)
                peer_comparisons = self._generate_synthetic_peer_comparisons(startup_data, sector, stage)
                
                return BenchmarkResult(
                    company_metrics=company_metrics,
                    industry_benchmarks=benchmarks,
                    percentiles=percentiles,
                    performance_summary=summary,
                    radar_data=radar_data,
                    recommendations=recommendations,
                    metric_comparisons=metric_comparisons,
                    percentile=overall_percentile,
                    peer_comparisons=peer_comparisons,
                    data_source="default"
                )
            except Exception as e:
                logger.error(f"Error in synchronous benchmarking: {str(e)}", exc_info=True)
                return BenchmarkResult(
                    company_metrics={},
                    industry_benchmarks={},
                    percentiles={'error': 50},
                    performance_summary=f"Error benchmarking startup: {str(e)}",
                    radar_data={'metrics': [], 'company_values': [], 'median_values': []},
                    recommendations=["Fix benchmarking data issues and try again."],
                    percentile=50,
                    data_source="error"
                )

    def _extract_metrics(self, sd: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and normalize relevant metrics from startup data.
        
        Args:
            sd: Dictionary containing startup data
            
        Returns:
            Dictionary of normalized metrics ready for benchmarking
        """
        metrics = {}
        
        # Helper function to safely extract numeric values
        def safe_extract(key: str, default: float = 0.0, normalize: bool = False) -> float:
            """Extract a metric with proper type conversion and normalization"""
            value = sd.get(key, default)
            
            # Handle None values
            if value is None:
                return default
                
            # Ensure numeric type
            try:
                value = float(value)
            except (TypeError, ValueError):
                logger.warning(f"Non-numeric value for {key}: {value}, using default {default}")
                return default
                
            # Normalize percentages if needed (convert >1 values to 0-1 range)
            if normalize and value > 1.0 and value <= 100.0:
                return value / 100.0
                
            return value
        
        # Extract growth metrics
        metrics['user_growth_rate'] = safe_extract('user_growth_rate')
        metrics['revenue_growth_rate'] = safe_extract('revenue_growth_rate')
        
        # Extract unit economics
        metrics['ltv_cac_ratio'] = safe_extract('ltv_cac_ratio')
        metrics['cac_payback_months'] = safe_extract('cac_payback_months')
        
        # Extract retention metrics
        metrics['churn_rate'] = safe_extract('churn_rate', 0.05)
        metrics['net_retention_rate'] = safe_extract('net_retention_rate', 1.0, normalize=True)
        
        # Extract financial metrics
        metrics['gross_margin'] = safe_extract('gross_margin_percent', 70, normalize=True)
        if metrics['gross_margin'] == 0 and 'gross_margin' in sd:
            metrics['gross_margin'] = safe_extract('gross_margin', 0.7, normalize=True)
        
        # Calculate burn multiple
        burn_rate = safe_extract('burn_rate', 30000)
        revenue = safe_extract('monthly_revenue', 50000)
        revenue_growth = safe_extract('revenue_growth_rate', 0.1)
        net_rev_increase = revenue * revenue_growth
        
        if net_rev_increase <= 0:
            metrics['burn_multiple'] = 9.99  # Cap at a reasonable maximum
        else:
            metrics['burn_multiple'] = burn_rate / net_rev_increase
            # Cap unreasonably high values
            metrics['burn_multiple'] = min(metrics['burn_multiple'], 10.0)
        
        # Other metrics
        metrics['runway_months'] = safe_extract('runway_months', 12)
        metrics['dau_mau_ratio'] = safe_extract('dau_mau_ratio')
        
        logger.debug(f"Extracted metrics: {metrics}")
        return metrics

    def _get_benchmarks_for_sector_stage(self, sector: str, stage: str) -> Dict[str, Dict[str, float]]:
        """
        Get benchmark data for a specific sector and stage, with fallbacks.
        
        Args:
            sector: The startup's sector (e.g., 'saas', 'fintech')
            stage: The startup's stage (e.g., 'seed', 'series-a')
            
        Returns:
            Dictionary of benchmark data, or empty dict if no matching benchmarks found
        """
        # Try exact sector and stage match
        if sector in self._default_benchmarks and stage in self._default_benchmarks[sector]:
            logger.debug(f"Found exact benchmark match for {sector}/{stage}")
            return self._default_benchmarks[sector][stage]
        
        # Try alternative stage representations (e.g., 'series-a' vs 'series_a')
        normalized_stage = stage.replace('-', '_')
        if sector in self._default_benchmarks and normalized_stage in self._default_benchmarks[sector]:
            logger.debug(f"Found benchmark match with normalized stage for {sector}/{normalized_stage}")
            return self._default_benchmarks[sector][normalized_stage]
        
        # Try fallback to 'saas' sector
        if sector != 'saas' and 'saas' in self._default_benchmarks:
            if stage in self._default_benchmarks['saas']:
                logger.debug(f"Using fallback 'saas' benchmarks for {sector}/{stage}")
                return self._default_benchmarks['saas'][stage]
            elif normalized_stage in self._default_benchmarks['saas']:
                logger.debug(f"Using fallback 'saas' benchmarks with normalized stage for {sector}/{normalized_stage}")
                return self._default_benchmarks['saas'][normalized_stage]
                
        # Try the closest available stage
        if sector in self._default_benchmarks:
            available_stages = list(self._default_benchmarks[sector].keys())
            if available_stages:
                closest_stage = self._find_closest_stage(stage, available_stages)
                logger.debug(f"Using closest stage '{closest_stage}' benchmarks for {sector}/{stage}")
                return self._default_benchmarks[sector][closest_stage]
        
        # If sector is not available but 'saas' is, use 'saas' seed as default fallback
        if sector not in self._default_benchmarks and 'saas' in self._default_benchmarks and 'seed' in self._default_benchmarks['saas']:
            logger.warning(f"No benchmarks for {sector}/{stage}. Using 'saas/seed' as fallback.")
            return self._default_benchmarks['saas']['seed']
            
        # No benchmarks found, return empty dict as last resort
        logger.warning(f"No benchmarks found for {sector}/{stage}")
        return {}

    def _find_closest_stage(self, target_stage: str, available_stages: List[str]) -> str:
        """
        Find the closest available stage to the target.
        
        Args:
            target_stage: The desired stage to match
            available_stages: List of available stages
            
        Returns:
            The closest matching stage from available_stages
        """
        # Stage progression order (earlier to later)
        stage_order = [
            'pre-seed', 'preseed', 'pre_seed',
            'seed',
            'series-a', 'series_a', 'seriesa', 'series a', 
            'series-b', 'series_b', 'seriesb', 'series b',
            'series-c', 'series_c', 'seriesc', 'series c',
            'growth'
        ]
        
        # If exact match exists, return it
        if target_stage in available_stages:
            return target_stage
            
        # Find the position of the target stage in our ordering
        target_position = -1
        for i, stage in enumerate(stage_order):
            if stage.lower() == target_stage.lower():
                target_position = i
                break
                
        if target_position == -1:
            # If target stage isn't in our ordering, default to first available
            return available_stages[0]
            
        # Find the closest stage by position in the stage_order
        closest_stage = available_stages[0]
        closest_distance = float('inf')
        
        for avail_stage in available_stages:
            for i, ordered_stage in enumerate(stage_order):
                if ordered_stage.lower() == avail_stage.lower():
                    distance = abs(i - target_position)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_stage = avail_stage
                        
        return closest_stage

    def _calculate_percentiles(self, 
                              company_metrics: Dict[str, float],
                              benchmarks: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """
        Calculate percentile rankings for company metrics against benchmarks.
        
        Args:
            company_metrics: The company's metrics
            benchmarks: Industry benchmark data
            
        Returns:
            Dictionary mapping metric names to percentile rankings (0-100)
        """
        percentiles = {}
        
        for metric, company_value in company_metrics.items():
            # Skip if metric not in benchmarks
            if metric not in benchmarks:
                percentiles[metric] = 50  # Default to median
                continue
                
            benchmark = benchmarks[metric]
            
            # Ensure benchmark has required percentile points
            required_keys = ['p10', 'p25', 'p50', 'p75', 'p90']
            if not all(key in benchmark for key in required_keys):
                logger.warning(f"Incomplete benchmark data for {metric}, missing keys")
                percentiles[metric] = 50  # Default to median
                continue
                
            # Determine percentile based on metric type
            if metric in self.HIGHER_BETTER_METRICS:
                if company_value >= benchmark['p90']:
                    percentiles[metric] = 90
                elif company_value >= benchmark['p75']:
                    percentiles[metric] = 75
                elif company_value >= benchmark['p50']:
                    percentiles[metric] = 50
                elif company_value >= benchmark['p25']:
                    percentiles[metric] = 25
                else:
                    percentiles[metric] = 10
            elif metric in self.LOWER_BETTER_METRICS:
                if company_value <= benchmark['p10']:
                    percentiles[metric] = 90
                elif company_value <= benchmark['p25']:
                    percentiles[metric] = 75
                elif company_value <= benchmark['p50']:
                    percentiles[metric] = 50
                elif company_value <= benchmark['p75']:
                    percentiles[metric] = 25
                else:
                    percentiles[metric] = 10
            else:
                # For metrics not in either list, closer to p50 is better
                distances = [
                    (abs(company_value - benchmark['p10']), 10),
                    (abs(company_value - benchmark['p25']), 25),
                    (abs(company_value - benchmark['p50']), 50),
                    (abs(company_value - benchmark['p75']), 75),
                    (abs(company_value - benchmark['p90']), 90)
                ]
                percentiles[metric] = min(distances, key=lambda x: x[0])[1]
        
        # Log all percentiles for debugging
        for metric, percentile in percentiles.items():
            logger.debug(f"Percentile for {metric}: {percentile}")
        
        return percentiles

    def _generate_performance_summary(self, 
                                     percentiles: Dict[str, int],
                                     sector: str, 
                                     stage: str) -> str:
        """
        Generate a textual summary of the company's performance.
        
        Args:
            percentiles: Dictionary of percentile rankings for each metric
            sector: The company's sector
            stage: The company's stage
            
        Returns:
            String containing performance summary
        """
        if not percentiles:
            return f"Insufficient data to benchmark this {stage} {sector} startup."
            
        # Calculate average percentile across all metrics
        avg_percentile = sum(percentiles.values()) / max(1, len(percentiles))
        
        # Find strongest and weakest metrics
        sorted_metrics = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_metrics[0] if sorted_metrics else ("N/A", 50)
        weakest = sorted_metrics[-1] if sorted_metrics else ("N/A", 50)
        
        # Determine overall performance level
        if avg_percentile >= 75:
            performance = "outstanding"
        elif avg_percentile >= 60:
            performance = "strong"
        elif avg_percentile >= 45:
            performance = "average"
        elif avg_percentile >= 25:
            performance = "below average"
        else:
            performance = "poor"
        
        # Create metric name mappings for more readable output
        metric_names = {
            'user_growth_rate': 'User Growth',
            'revenue_growth_rate': 'Revenue Growth',
            'ltv_cac_ratio': 'LTV:CAC Ratio',
            'cac_payback_months': 'CAC Payback Period',
            'churn_rate': 'Churn Rate',
            'net_retention_rate': 'Net Retention',
            'gross_margin': 'Gross Margin',
            'burn_multiple': 'Burn Multiple',
            'runway_months': 'Runway',
            'dau_mau_ratio': 'Daily Active Usage'
        }
        
        # Use readable names for metrics if available
        strongest_name = metric_names.get(strongest[0], strongest[0])
        weakest_name = metric_names.get(weakest[0], weakest[0])
        
        summary = (
            f"This {stage} {sector} startup shows {performance} performance compared to industry benchmarks. "
            f"Strongest metric: {strongest_name} (better than {strongest[1]}% of peers). "
            f"Weakest metric: {weakest_name} (only better than {weakest[1]}% of peers)."
        )
        
        return summary

    def _prepare_radar_data(self, 
                           company_metrics: Dict[str, float],
                           benchmarks: Dict[str, Dict[str, float]],
                           percentiles: Dict[str, int]) -> Dict[str, Any]:
        """
        Prepare data for radar chart visualization.
        
        Args:
            company_metrics: The company's metrics
            benchmarks: Industry benchmark data
            percentiles: Percentile rankings for each metric
            
        Returns:
            Dictionary containing radar chart data
        """
        # Select metrics to show in the radar chart
        chosen_metrics = [m for m in self.RADAR_METRICS if m in company_metrics]
        
        # Initialize radar data structure
        data = {
            "metrics": [],
            "company_values": [],
            "median_values": [],
            "percentiles": []
        }
        
        # Create more readable names for display
        metric_labels = {
            'user_growth_rate': 'User Growth',
            'revenue_growth_rate': 'Revenue Growth',
            'churn_rate': 'Churn Rate',
            'ltv_cac_ratio': 'LTV:CAC',
            'burn_multiple': 'Burn Multiple',
            'gross_margin': 'Gross Margin',
            'runway_months': 'Runway',
            'net_retention_rate': 'Net Retention',
            'dau_mau_ratio': 'DAU/MAU'
        }
        
        # Process each metric
        for metric in chosen_metrics:
            # Add readable metric name
            data["metrics"].append(metric_labels.get(metric, metric))
            data["percentiles"].append(percentiles.get(metric, 50))
            
            # Get benchmark data for this metric
            benchmark = benchmarks.get(metric, {})
            company_value = company_metrics[metric]
            
            if benchmark and all(k in benchmark for k in ['p10', 'p50', 'p90']):
                # Calculate range for normalization
                value_range = benchmark['p90'] - benchmark['p10']
                if value_range <= 0:
                    value_range = 1  # Avoid division by zero
                
                # Normalize values based on whether higher or lower is better
                if metric in self.HIGHER_BETTER_METRICS:
                    norm_company = max(0, min(1, (company_value - benchmark['p10']) / value_range))
                    norm_median = max(0, min(1, (benchmark['p50'] - benchmark['p10']) / value_range))
                else:
                    norm_company = max(0, min(1, 1 - ((company_value - benchmark['p10']) / value_range)))
                    norm_median = max(0, min(1, 1 - ((benchmark['p50'] - benchmark['p10']) / value_range)))
            else:
                # Default to middle values if benchmark data is missing
                norm_company = 0.5
                norm_median = 0.5
            
            data["company_values"].append(norm_company)
            data["median_values"].append(norm_median)
        
        return data

    def _generate_recommendations(self, 
                                 percentiles: Dict[str, int],
                                 company_metrics: Dict[str, float],
                                 benchmarks: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Generate actionable recommendations based on benchmark comparison.
        
        Args:
            percentiles: Percentile rankings for each metric
            company_metrics: The company's metrics
            benchmarks: Industry benchmark data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Identify metrics with low percentiles (below 25th percentile)
        low_metrics = [m for m, p in percentiles.items() if p <= 25]
        
        # Recommendation mapping for each metric
        rec_map = {
            'user_growth_rate': "Improve user acquisition channels and marketing spend efficiency.",
            'revenue_growth_rate': "Explore upsell opportunities or refine pricing strategy to accelerate revenue growth.",
            'churn_rate': "Enhance customer retention through improved onboarding and product enhancements.",
            'ltv_cac_ratio': "Focus on unit economics by reducing CAC or increasing LTV through pricing or retention improvements.",
            'cac_payback_months': "Shorten payback period through conversion funnel or pricing optimization.",
            'burn_multiple': "Reduce burn rate or accelerate revenue growth to improve capital efficiency.",
            'gross_margin': "Optimize cost structure to improve margin through reduced COGS or increased pricing.",
            'runway_months': "Extend runway by reducing discretionary spending or raising additional capital.",
            'dau_mau_ratio': "Increase daily usage with improved product engagement features and notifications.",
            'net_retention_rate': "Improve expansion revenue through upsell and cross-sell strategies."
        }
        
        # Add recommendations for low-performing metrics
        for metric in low_metrics:
            if metric in rec_map:
                recommendations.append(rec_map[metric])
            else:
                recommendations.append(f"Consider improvements to {metric}.")
        
        # Cap recommendations at 5
        if len(recommendations) > 5:
            recommendations = recommendations[:5]
            
        # Add default recommendation if none found
        if not recommendations:
            # Get metrics at different performance levels
            good_metrics = [m for m, p in percentiles.items() if p >= 75]
            
            if good_metrics:
                metric = good_metrics[0]
                readable_metric = metric.replace('_', ' ').title()
                recommendations.append(f"Maintain strong performance in {readable_metric} while focusing on scaling operations.")
            else:
                recommendations.append("Focus on improving overall operational metrics as you scale.")
        
        return recommendations
        
    def _generate_metric_comparisons(self,
                                    company_metrics: Dict[str, float],
                                    benchmarks: Dict[str, Dict[str, float]],
                                    percentiles: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed metric comparisons for UI display.
        
        Args:
            company_metrics: The company's metrics
            benchmarks: Industry benchmark data
            percentiles: Percentile rankings for each metric
            
        Returns:
            Dictionary of metric comparison details
        """
        metric_comparisons = {}
        
        # Format values according to metric type
        for metric, value in company_metrics.items():
            if metric not in self.DISPLAY_METRICS:
                continue
                
            benchmark = benchmarks.get(metric, {})
            percentile = percentiles.get(metric, 50)
            
            # Format the value based on metric type
            if metric in ['user_growth_rate', 'revenue_growth_rate', 'churn_rate', 'gross_margin', 'net_retention_rate']:
                formatted_value = f"{value * 100:.1f}%"
                benchmark_value = f"{benchmark.get('p50', 0) * 100:.1f}%" if 'p50' in benchmark else "N/A"
            elif metric in ['ltv_cac_ratio', 'burn_multiple']:
                formatted_value = f"{value:.1f}x"
                benchmark_value = f"{benchmark.get('p50', 0):.1f}x" if 'p50' in benchmark else "N/A"
            elif metric in ['runway_months', 'cac_payback_months']:
                formatted_value = f"{value:.1f}"
                benchmark_value = f"{benchmark.get('p50', 0):.1f}" if 'p50' in benchmark else "N/A"
            else:
                formatted_value = f"{value:.2f}"
                benchmark_value = f"{benchmark.get('p50', 0):.2f}" if 'p50' in benchmark else "N/A"
                
            # Create comparison data
            metric_comparisons[metric] = {
                "display_name": self.DISPLAY_METRICS[metric],
                "company_value": value,
                "formatted_value": formatted_value,
                "benchmark_value": benchmark.get('p50', 0),
                "formatted_benchmark": benchmark_value,
                "percentile": percentile,
                "is_good": percentile >= 50
            }
            
        return metric_comparisons
        
    def _process_peer_data(self,
                          peers: List[Dict],
                          company_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Process peer company data from API into a comparable format.
        
        Args:
            peers: List of peer company data from API
            company_metrics: The company's metrics for comparison
            
        Returns:
            List of processed peer comparison data
        """
        if not peers:
            return []
            
        processed_peers = []
        
        for peer in peers:
            peer_data = {
                "name": peer.get("name", "Unknown Peer"),
                "metrics": {}
            }
            
            # Extract metrics that match our company metrics
            peer_metrics = peer.get("metrics", {})
            for metric in company_metrics.keys():
                if metric in peer_metrics:
                    peer_data["metrics"][metric] = peer_metrics[metric]
            
            processed_peers.append(peer_data)
            
        return processed_peers
        
    def _generate_synthetic_peer_comparisons(self,
                                           startup_data: Dict[str, Any],
                                           sector: str,
                                           stage: str) -> List[Dict[str, Any]]:
        """
        Generate synthetic peer company data based on benchmarks.
        
        Args:
            startup_data: Data for the company being benchmarked
            sector: The company's sector
            stage: The company's stage
            
        Returns:
            List of peer comparison data dictionaries
        """
        peers = []
        
        # Get relevant benchmarks
        benchmarks = self._get_benchmarks_for_sector_stage(sector, stage)
        if not benchmarks:
            return []
            
        # Create three synthetic peers at different performance levels
        peer_levels = [
            {"name": f"Top {sector.capitalize()} Performer", "percentile": 90},
            {"name": f"Average {sector.capitalize()} Company", "percentile": 50},
            {"name": f"Below Average {sector.capitalize()} Company", "percentile": 25}
        ]
        
        # Key metrics to compare
        key_metrics = [
            'user_growth_rate', 'revenue_growth_rate', 'churn_rate',
            'ltv_cac_ratio', 'gross_margin', 'net_retention_rate'
        ]
        
        for peer in peer_levels:
            peer_data = {"name": peer["name"], "metrics": {}}
            
            for metric in key_metrics:
                if metric in benchmarks:
                    b = benchmarks[metric]
                    # Get the appropriate percentile value
                    if peer["percentile"] == 90:
                        peer_data["metrics"][metric] = b.get('p90', 0)
                    elif peer["percentile"] == 50:
                        peer_data["metrics"][metric] = b.get('p50', 0)
                    elif peer["percentile"] == 25:
                        peer_data["metrics"][metric] = b.get('p25', 0)
            
            peers.append(peer_data)
            
        return peers

    def _load_default_benchmarks(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Load default benchmark data for common sectors and stages.
        
        Returns:
            Nested dictionary of benchmark data organized by sector, stage, metric, and percentile
        """
        return {
            'saas': {
                'seed': {
                    'user_growth_rate':     {'p10':0.03,'p25':0.05,'p50':0.08,'p75':0.15,'p90':0.25},
                    'revenue_growth_rate':  {'p10':0.05,'p25':0.08,'p50':0.15,'p75':0.25,'p90':0.40},
                    'ltv_cac_ratio':        {'p10':1.0,'p25':1.5,'p50':2.5,'p75':3.5,'p90':5.0},
                    'cac_payback_months':   {'p10':36 ,'p25':24 ,'p50':18 ,'p75':12 ,'p90':6 },
                    'churn_rate':           {'p10':0.15,'p25':0.10,'p50':0.07,'p75':0.05,'p90':0.03},
                    'net_retention_rate':   {'p10':0.70,'p25':0.85,'p50':0.95,'p75':1.05,'p90':1.20},
                    'gross_margin':         {'p10':0.55,'p25':0.65,'p50':0.72,'p75':0.80,'p90':0.85},
                    'burn_multiple':        {'p10':5.0,'p25':3.0,'p50':2.0,'p75':1.5,'p90':1.0},
                    'runway_months':        {'p10':6  ,'p25':9  ,'p50':12 ,'p75':18 ,'p90':24},
                    'dau_mau_ratio':        {'p10':0.05,'p25':0.10,'p50':0.15,'p75':0.25,'p90':0.40}
                },
                'series-a': {
                    'user_growth_rate':     {'p10':0.05,'p25':0.08,'p50':0.12,'p75':0.20,'p90':0.30},
                    'revenue_growth_rate':  {'p10':0.08,'p25':0.12,'p50':0.18,'p75':0.30,'p90':0.45},
                    'ltv_cac_ratio':        {'p10':1.5,'p25':2.0,'p50':3.0,'p75':4.0,'p90':6.0},
                    'cac_payback_months':   {'p10':24 ,'p25':18 ,'p50':12 ,'p75':9 ,'p90':6 },
                    'churn_rate':           {'p10':0.10,'p25':0.08,'p50':0.05,'p75':0.03,'p90':0.02},
                    'net_retention_rate':   {'p10':0.80,'p25':0.90,'p50':1.05,'p75':1.15,'p90':1.30},
                    'gross_margin':         {'p10':0.60,'p25':0.70,'p50':0.75,'p75':0.82,'p90':0.88},
                    'burn_multiple':        {'p10':4.0,'p25':2.5,'p50':1.5,'p75':1.0,'p90':0.7},
                    'runway_months':        {'p10':12 ,'p25':18 ,'p50':24 ,'p75':30 ,'p90':36},
                    'dau_mau_ratio':        {'p10':0.10,'p25':0.15,'p50':0.22,'p75':0.32,'p90':0.50}
                }
            },
            'fintech': {
                'seed': {
                    'user_growth_rate':     {'p10':0.03,'p25':0.05,'p50':0.07,'p75':0.12,'p90':0.20},
                    'revenue_growth_rate':  {'p10':0.04,'p25':0.07,'p50':0.12,'p75':0.20,'p90':0.35},
                    'ltv_cac_ratio':        {'p10':0.8,'p25':1.2,'p50':2.0,'p75':3.0,'p90':4.5},
                    'cac_payback_months':   {'p10':42 ,'p25':30 ,'p50':24 ,'p75':18 ,'p90':12},
                    'churn_rate':           {'p10':0.12,'p25':0.08,'p50':0.05,'p75':0.03,'p90':0.02},
                    'net_retention_rate':   {'p10':0.75,'p25':0.85,'p50':0.95,'p75':1.05,'p90':1.15},
                    'gross_margin':         {'p10':0.40,'p25':0.50,'p50':0.65,'p75':0.75,'p90':0.80},
                    'burn_multiple':        {'p10':6.0,'p25':4.0,'p50':3.0,'p75':2.0,'p90':1.2},
                    'runway_months':        {'p10':6  ,'p25':9  ,'p50':12 ,'p75':18 ,'p90':24},
                    'dau_mau_ratio':        {'p10':0.04,'p25':0.08,'p50':0.12,'p75':0.20,'p90':0.30}
                },
                'series-a': {
                    'user_growth_rate':     {'p10':0.04,'p25':0.06,'p50':0.10,'p75':0.15,'p90':0.25},
                    'revenue_growth_rate':  {'p10':0.06,'p25':0.10,'p50':0.15,'p75':0.25,'p90':0.40},
                    'ltv_cac_ratio':        {'p10':1.2,'p25':1.8,'p50':2.5,'p75':3.5,'p90':5.0},
                    'cac_payback_months':   {'p10':36 ,'p25':24 ,'p50':18 ,'p75':12 ,'p90':9 },
                    'churn_rate':           {'p10':0.08,'p25':0.06,'p50':0.04,'p75':0.02,'p90':0.01},
                    'net_retention_rate':   {'p10':0.85,'p25':0.95,'p50':1.05,'p75':1.15,'p90':1.25},
                    'gross_margin':         {'p10':0.45,'p25':0.55,'p50':0.68,'p75':0.78,'p90':0.85},
                    'burn_multiple':        {'p10':5.0,'p25':3.0,'p50':2.0,'p75':1.5,'p90':1.0},
                    'runway_months':        {'p10':12 ,'p25':18 ,'p50':24 ,'p75':30 ,'p90':36},
                    'dau_mau_ratio':        {'p10':0.08,'p25':0.12,'p50':0.18,'p75':0.25,'p90':0.40}
                }
            },
            'ecommerce': {
                'seed': {
                    'user_growth_rate':     {'p10':0.05,'p25':0.08,'p50':0.15,'p75':0.25,'p90':0.40},
                    'revenue_growth_rate':  {'p10':0.06,'p25':0.10,'p50':0.18,'p75':0.30,'p90':0.50},
                    'ltv_cac_ratio':        {'p10':0.8,'p25':1.2,'p50':1.8,'p75':2.5,'p90':3.5},
                    'cac_payback_months':   {'p10':12 ,'p25':9 ,'p50':6 ,'p75':3 ,'p90':1 },
                    'churn_rate':           {'p10':0.40,'p25':0.30,'p50':0.20,'p75':0.15,'p90':0.10},
                    'net_retention_rate':   {'p10':0.40,'p25':0.55,'p50':0.70,'p75':0.85,'p90':1.0},
                    'gross_margin':         {'p10':0.20,'p25':0.30,'p50':0.40,'p75':0.50,'p90':0.60},
                    'burn_multiple':        {'p10':6.0,'p25':4.0,'p50':3.0,'p75':2.0,'p90':1.0},
                    'runway_months':        {'p10':6 ,'p25':9 ,'p50':12 ,'p75':18 ,'p90':24},
                    'dau_mau_ratio':        {'p10':0.10,'p25':0.15,'p50':0.25,'p75':0.40,'p90':0.60}
                }
            },
            'ai': {
                'seed': {
                    'user_growth_rate':     {'p10':0.05,'p25':0.10,'p50':0.20,'p75':0.35,'p90':0.50},
                    'revenue_growth_rate':  {'p10':0.08,'p25':0.15,'p50':0.25,'p75':0.40,'p90':0.60},
                    'ltv_cac_ratio':        {'p10':0.7,'p25':1.0,'p50':1.5,'p75':2.5,'p90':4.0},
                    'cac_payback_months':   {'p10':36 ,'p25':24 ,'p50':18 ,'p75':12 ,'p90':6 },
                    'churn_rate':           {'p10':0.15,'p25':0.10,'p50':0.08,'p75':0.05,'p90':0.03},
                    'net_retention_rate':   {'p10':0.75,'p25':0.90,'p50':1.10,'p75':1.30,'p90':1.50},
                    'gross_margin':         {'p10':0.50,'p25':0.60,'p50':0.70,'p75':0.80,'p90':0.90},
                    'burn_multiple':        {'p10':8.0,'p25':5.0,'p50':3.0,'p75':2.0,'p90':1.0},
                    'runway_months':        {'p10':6 ,'p25':9 ,'p50':12 ,'p75':18 ,'p90':24},
                    'dau_mau_ratio':        {'p10':0.05,'p25':0.10,'p50':0.20,'p75':0.30,'p90':0.45}
                }
            }
        }

    def get_available_sectors(self) -> List[str]:
        """
        Get the list of sectors that have benchmark data available.
        
        Returns:
            List of sector names
        """
        return list(self._default_benchmarks.keys())
        
    def get_available_stages(self, sector: str = None) -> List[str]:
        """
        Get the list of stages that have benchmark data available for a sector.
        
        Args:
            sector: Optional sector to filter stages by. If None, returns all stages.
            
        Returns:
            List of stage names
        """
        if sector is not None and sector in self._default_benchmarks:
            return list(self._default_benchmarks[sector].keys())
            
        # Collect all unique stages across all sectors
        all_stages = set()
        for sector_data in self._default_benchmarks.values():
            all_stages.update(sector_data.keys())
            
        return list(all_stages)
        
    def clear_cache(self, sector: str = None, stage: str = None):
        """
        Clear benchmark data cache.
        
        Args:
            sector: Optional sector to clear cache for
            stage: Optional stage to clear cache for (requires sector)
        """
        if sector and stage:
            key = f"{sector}_{stage}"
            self._cache.clear(key)
        elif sector:
            # Clear all stages for this sector
            for stage in self.get_available_stages(sector):
                key = f"{sector}_{stage}"
                self._cache.clear(key)
        else:
            # Clear all cache
            self._cache.clear()
            
    def update_api_config(self, api_key: str = None, api_url: str = None, use_api: bool = None):
        """
        Update API client configuration.
        
        Args:
            api_key: New API key
            api_url: New API URL
            use_api: Whether to use the API
        """
        if api_key is not None or api_url is not None:
            # Recreate API client with new config
            current_key = self._api_client.api_key if self._api_client else None
            current_url = self._api_client.api_url if self._api_client else None
            
            self._api_client = BenchmarkAPIClient(
                api_key=api_key or current_key,
                api_url=api_url or current_url
            )
            
        if use_api is not None:
            self._use_api = use_api
            
        logger.info(f"Updated API configuration. API enabled: {self._use_api}")


# Example integration with render_benchmarks_tab function
def render_benchmarks_tab(doc: dict):
    """Render benchmark analysis."""
    import streamlit as st
    
    st.header("Benchmark Analysis")
    
    benchmarks = doc.get("benchmarks", {})
    
    if not benchmarks:
        st.warning("Benchmark analysis not available.")
        return
    
    # Display data source if available
    data_source = benchmarks.get("data_source", "default")
    if data_source == "api":
        st.info("Using real-time industry benchmark data from API.")
    elif data_source == "cached":
        timestamp = benchmarks.get("data_timestamp")
        time_str = timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else "unknown time"
        st.info(f"Using cached benchmark data from {time_str}.")
    
    # Overall benchmarking
    st.subheader("Sector Benchmarking")
    
    sector = doc.get("sector", "")
    stage = doc.get("stage", "")
    
    st.markdown(f"Benchmarking against **{sector.title()}** companies at **{stage.title()}** stage")
    
    # Percentile ranking
    percentile = benchmarks.get("percentile", 0)
    st.metric("Industry Percentile", f"{percentile:.0f}th")
    
    # Metric comparisons
    metric_comparisons = benchmarks.get("metric_comparisons", {})
    
    if metric_comparisons:
        # Create DataFrame
        data = []
        for metric, comparison in metric_comparisons.items():
            data.append({
                "Metric": comparison.get("display_name", metric),
                "Your Value": comparison.get("formatted_value", "N/A"),
                "Benchmark": comparison.get("formatted_benchmark", "N/A"),
                "Percentile": comparison.get("percentile", 0)
            })
        
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Create comparison chart using the pre-formatted values
        import plotly.graph_objects as go
        
        # Extract data for plotting
        metrics = [row["Metric"] for row in data]
        company_values = [comparison.get("company_value", 0) for comparison in metric_comparisons.values()]
        benchmark_values = [comparison.get("benchmark_value", 0) for comparison in metric_comparisons.values()]
        
        fig = go.Figure()
        
        # Add company values
        fig.add_trace(go.Bar(
            x=metrics,
            y=company_values,
            name='Your Company',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Add benchmark values
        fig.add_trace(go.Bar(
            x=metrics,
            y=benchmark_values,
            name='Industry Benchmark',
            marker_color='rgba(255, 127, 14, 0.8)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Metric Comparison to Industry Benchmarks',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        st.markdown("### Detailed Benchmark Comparison")
        st.dataframe(df)
    
    # Peer comparisons
    peer_comparisons = benchmarks.get("peer_comparisons", [])
    
    if peer_comparisons and all(isinstance(peer, dict) for peer in peer_comparisons):
        st.subheader("Peer Comparisons")
        
        # Create DataFrame
        companies = ["Your Company"] + [peer.get("name", f"Peer {i+1}") for i, peer in enumerate(peer_comparisons)]
        
        # Get metrics to compare
        all_metrics = set()
        for peer in peer_comparisons:
            all_metrics.update(peer.get("metrics", {}).keys())
        
        # Create comparison data
        comparison_data = {metric: [] for metric in all_metrics}
        
        # Add company data
        for metric in all_metrics:
            comparison_data[metric].append(doc.get(metric, 0))
        
        # Add peer data
        for peer in peer_comparisons:
            peer_metrics = peer.get("metrics", {})
            for metric in all_metrics:
                comparison_data[metric].append(peer_metrics.get(metric, 0))
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(comparison_data, index=companies)
        
        # Transpose for better display
        df_display = df.transpose()
        
        # Show table
        st.dataframe(df_display)
        
        # Create radar chart for key metrics
        key_metrics = ["revenue_growth_rate", "user_growth_rate", "ltv_cac_ratio", "gross_margin_percent"]
        key_metrics = [m for m in key_metrics if m in all_metrics]
        
        if key_metrics:
            # Create radar chart data
            radar_data = []
            
            for i, company in enumerate(companies):
                company_data = {"Company": company}
                for metric in key_metrics:
                    if i < len(comparison_data[metric]):
                        company_data[metric] = comparison_data[metric][i]
                radar_data.append(company_data)
            
            # Create radar chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for data in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=[data.get(metric, 0) for metric in key_metrics] + [data.get(key_metrics[0], 0)],  # Close the loop
                    theta=key_metrics + [key_metrics[0]],  # Close the loop
                    fill='toself',
                    name=data['Company']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                title="Key Metrics Comparison",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
