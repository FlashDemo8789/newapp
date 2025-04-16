"""
Service module for Competitive Intelligence Analysis
Integrates with the CompetitiveIntelligence from the competitive_intelligence.py module
"""
import sys
import os
import logging
import json
import traceback
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Add the project root to the Python path to import the module
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
try:
    from competitive_intelligence import CompetitiveIntelligence  # Fixed import name
except ImportError:
    # Create a fallback implementation if the module is not available
    class CompetitiveIntelligence:
        def analyze_competitive_landscape(self, **kwargs):
            return {"error": "CompetitiveIntelligence module not available"}

logger = logging.getLogger(__name__)

class CompetitiveIntelligenceService:
    """Service wrapper for CompetitiveIntelligence"""
    
    def __init__(self):
        """Initialize the competitive intelligence service"""
        try:
            self.analyzer = CompetitiveIntelligence()
            logger.info("CompetitiveIntelligenceService initialized")
        except Exception as e:
            logger.error(f"Error initializing CompetitiveIntelligenceService: {str(e)}")
            self.analyzer = None
    
    async def analyze_competitive_landscape(
        self, 
        startup_data: Dict[str, Any],
        competitor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitive landscape and positioning
        
        Args:
            startup_data: Core startup metrics data
            competitor_data: Optional data about competitors
            
        Returns:
            Dict containing competitive analysis results
        """
        try:
            # Log the incoming data for debugging
            logger.info(f"Analyzing competitive landscape for: {startup_data.get('name', 'Unknown Company')}")
            logger.debug(f"Startup data keys: {list(startup_data.keys())}")
            
            # Check if startup_data has the expected structure
            if not self._validate_input(startup_data):
                logger.warning("Input validation failed, using fallback analysis")
                return self._generate_fallback_competitive_analysis(startup_data)
            
            # Transform the data for the analyzer
            transformed_data = self._transform_data_for_analyzer(startup_data, competitor_data)
            
            # Perform the competitive intelligence analysis
            if self.analyzer is None:
                logger.error("Analyzer is not initialized")
                return self._generate_fallback_competitive_analysis(startup_data)
                
            competitive_results = self.analyzer.analyze_competitive_landscape(**transformed_data)
            
            # Convert to serializable dict and return
            if hasattr(competitive_results, "to_dict"):
                return competitive_results.to_dict()
            
            # Ensure the result is JSON serializable
            try:
                # Test JSON serialization
                result_str = json.dumps(competitive_results)
                return competitive_results
            except (TypeError, OverflowError) as json_err:
                logger.error(f"Result is not JSON serializable: {str(json_err)}")
                return self._generate_fallback_competitive_analysis(startup_data)
            
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a basic error response with some generated data
            return self._generate_fallback_competitive_analysis(startup_data)
    
    def _validate_input(self, startup_data: Dict[str, Any]) -> bool:
        """
        Validate that the input data has the required fields
        
        Args:
            startup_data: The startup data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for minimal required fields
        if not isinstance(startup_data, dict):
            logger.warning(f"startup_data is not a dict, type: {type(startup_data)}")
            return False
            
        # Accept various input formats
        # Format 1: company_info nested structure
        if "company_info" in startup_data:
            return True
            
        # Format 2: direct company fields
        if "name" in startup_data and "industry" in startup_data:
            return True
            
        # Format 3: CAMP framework fields
        has_market = "market_metrics" in startup_data or "market" in startup_data
        has_advantage = "advantage_metrics" in startup_data or "advantage" in startup_data
        if has_market or has_advantage:
            return True
            
        logger.warning(f"Missing required fields in startup_data: {list(startup_data.keys())}")
        return False
    
    def _transform_data_for_analyzer(
        self, 
        startup_data: Union[Dict[str, Any], str],
        competitor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Transform startup analysis data format to the format 
        expected by the CompetitiveIntelligence
        
        Args:
            startup_data: Startup data in CAMP framework format or JSON string
            competitor_data: Optional data about competitors
            
        Returns:
            Dict formatted for the competitive intelligence
        """
        # Convert string to dict if needed
        if isinstance(startup_data, str):
            try:
                startup_data = json.loads(startup_data)
            except json.JSONDecodeError:
                logger.error("Failed to parse startup_data as JSON")
                startup_data = {}
        
        # Handle different input formats
        company_info = {}
        if "company_info" in startup_data:
            company_info = startup_data["company_info"]
        else:
            company_info = {
                "name": startup_data.get("name", ""),
                "industry": startup_data.get("industry", ""),
                "stage": startup_data.get("stage", ""),
                "business_model": startup_data.get("business_model", "")
            }
        
        # Extract metrics with fallbacks for different field names
        market_metrics = {}
        for field in ["market_metrics", "market", "market_analysis"]:
            if field in startup_data and startup_data[field]:
                market_metrics = startup_data[field]
                break
        
        advantage_metrics = {}
        for field in ["advantage_metrics", "advantage", "product_metrics"]:
            if field in startup_data and startup_data[field]:
                advantage_metrics = startup_data[field]
                break
        
        # Handle both dictionary and nested object structures
        market_share = self._extract_metric(market_metrics, ["market_share", "share"])
        growth_rate = self._extract_metric(market_metrics, ["revenue_growth_rate", "growth_rate", "growth"])
        innovation_score = self._extract_metric(advantage_metrics, ["tech_innovation_score", "innovation_score", "innovation"])
        product_quality = self._extract_metric(advantage_metrics, ["product_quality", "quality"])
        feature_completeness = self._extract_metric(advantage_metrics, ["feature_completeness", "features"])
        user_satisfaction = self._extract_metric(advantage_metrics, ["user_satisfaction", "satisfaction"])
        nps_score = self._extract_metric(market_metrics, ["nps_score", "nps"])
        
        # Prepare data structure for the analyzer
        analyzer_data = {
            "company_data": {
                "info": company_info,
                "market_position": {
                    "market_share": market_share or 0.01,
                    "growth_rate": growth_rate or 0.3
                },
                "product_strengths": {
                    "innovation_score": innovation_score or 0.6,
                    "product_quality": product_quality or 0.7,
                    "feature_set": feature_completeness or 0.5
                },
                "user_metrics": {
                    "user_satisfaction": user_satisfaction or 0.7,
                    "nps": nps_score or 30
                }
            }
        }
        
        # Add competitor data if provided
        if competitor_data and "competitors" in competitor_data:
            analyzer_data["competitors"] = competitor_data["competitors"]
        else:
            # Generate synthetic competitors if not provided
            synthetic_competitors = self._generate_synthetic_competitors(startup_data)
            analyzer_data["competitors"] = synthetic_competitors.get("competitors", [])
        
        # Add industry data
        analyzer_data["industry_data"] = {
            "total_market_size": self._extract_metric(market_metrics, ["tam", "total_addressable_market", "market_size"]) or 1000000000,
            "industry_growth_rate": self._extract_metric(market_metrics, ["market_growth_rate", "industry_growth"]) or 0.15
        }
        
        return analyzer_data
    
    def _extract_metric(self, metrics_dict: Dict[str, Any], field_names: List[str]) -> Any:
        """
        Extract a metric from various possible field names
        
        Args:
            metrics_dict: The metrics dictionary
            field_names: List of possible field names
            
        Returns:
            The extracted value or None
        """
        if not metrics_dict or not isinstance(metrics_dict, dict):
            return None
            
        for field in field_names:
            if field in metrics_dict:
                return metrics_dict[field]
        return None
    
    def _generate_synthetic_competitors(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic competitor data based on startup's industry
        
        Args:
            startup_data: Startup data in CAMP framework format
            
        Returns:
            Dict with synthetic competitor data
        """
        # Extract industry information
        industry = ""
        if "company_info" in startup_data and "industry" in startup_data["company_info"]:
            industry = startup_data["company_info"]["industry"]
        elif "industry" in startup_data:
            industry = startup_data["industry"]
        
        # Extract metrics 
        market_metrics = startup_data.get("market_metrics", {}) or {}
        advantage_metrics = startup_data.get("advantage_metrics", {}) or {}
        
        market_share = market_metrics.get("market_share", 0.05)
        growth_rate = market_metrics.get("revenue_growth_rate", 0.3)
        innovation_score = advantage_metrics.get("tech_innovation_score", 0.6)
        
        # Generate competitors based on industry
        competitors = []
        competitor_names = []
        
        # Use industry-specific competitors if possible
        if industry.lower() in ["ai", "artificial intelligence", "machine learning"]:
            competitor_names = ["OpenAI", "Anthropic", "Cohere", "AI21 Labs", "Stability AI"]
        elif industry.lower() in ["software", "saas", "cloud"]:
            competitor_names = ["Microsoft", "Salesforce", "Oracle", "SAP", "Atlassian"]
        elif industry.lower() in ["fintech", "financial technology"]:
            competitor_names = ["Stripe", "Plaid", "Square", "Klarna", "Chime"]
        elif industry.lower() in ["ecommerce", "e-commerce"]:
            competitor_names = ["Shopify", "Amazon", "BigCommerce", "WooCommerce", "Magento"]
        elif industry.lower() in ["health", "healthcare", "healthtech"]:
            competitor_names = ["Epic Systems", "Cerner", "Teladoc", "Oscar Health", "Babylon Health"]
        else:
            competitor_names = [
                f"Competitor {i+1}" for i in range(4)
            ]
        
        # Generate competitor data
        for i, name in enumerate(competitor_names[:4]):
            if i == 0:  # Market leader
                competitor_market_share = max(0.2, market_share * 5)
                competitor_growth = max(0.05, growth_rate * 0.5)
                competitor_innovation = max(0.5, innovation_score * 0.8)
                strengths = ["Market dominance", "Brand recognition"]
                weaknesses = ["Less agile", "Legacy systems"]
            elif i == 1:  # Established player
                competitor_market_share = max(0.1, market_share * 3)
                competitor_growth = max(0.1, growth_rate * 0.7)
                competitor_innovation = max(0.6, innovation_score * 0.9)
                strengths = ["Established customer base", "Strong partnerships"]
                weaknesses = ["Average innovation", "Higher costs"]
            elif i == 2:  # Fast-growing challenger
                competitor_market_share = max(0.05, market_share * 2)
                competitor_growth = max(0.25, growth_rate * 1.2)
                competitor_innovation = max(0.7, innovation_score * 1.1)
                strengths = ["Fast growth", "Innovative approach"]
                weaknesses = ["Limited resources", "Narrow product range"]
            else:  # New entrant
                competitor_market_share = max(0.01, market_share * 0.5)
                competitor_growth = max(0.4, growth_rate * 1.5)
                competitor_innovation = max(0.8, innovation_score * 1.2)
                strengths = ["Cutting-edge technology", "Niche focus"]
                weaknesses = ["Limited market presence", "Early stage product"]
            
            competitors.append({
                "name": name,
                "description": f"A {industry} company focusing on {strengths[0].lower()}",
                "market_share": competitor_market_share,
                "growth_rate": competitor_growth,
                "innovation_score": competitor_innovation,
                "strengths": strengths,
                "weaknesses": weaknesses
            })
        
        return {
            "competitors": competitors,
            "industry_data": {
                "total_market_size": market_metrics.get("tam", 1000000000),
                "industry_growth_rate": market_metrics.get("market_growth_rate", 0.15)
            }
        }
    
    def _generate_fallback_competitive_analysis(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback competitive analysis when the actual analysis fails
        
        Args:
            startup_data: Basic startup data
            
        Returns:
            Dict with synthetic competitive analysis results
        """
        # Extract company name if available
        company_name = "Your Company"
        if "name" in startup_data:
            company_name = startup_data["name"]
        elif "company_info" in startup_data and "name" in startup_data["company_info"]:
            company_name = startup_data["company_info"]["name"]
        
        # Generate synthetic competitor data
        competitor_data = self._generate_synthetic_competitors(startup_data)
        competitors = competitor_data.get("competitors", [])
        
        # Get industry if available
        industry = ""
        if "industry" in startup_data:
            industry = startup_data["industry"]
        elif "company_info" in startup_data and "industry" in startup_data["company_info"]:
            industry = startup_data["company_info"]["industry"]
        
        # Create your_advantages based on industry
        your_advantages = []
        if industry.lower() in ["ai", "artificial intelligence", "machine learning"]:
            your_advantages = [
                "Specialized AI models with higher accuracy",
                "More efficient resource utilization",
                "Better integration with existing systems"
            ]
        elif industry.lower() in ["software", "saas", "cloud"]:
            your_advantages = [
                "Modern cloud-native architecture",
                "Superior user experience",
                "More flexible pricing model"
            ]
        elif industry.lower() in ["fintech", "financial technology"]:
            your_advantages = [
                "Lower transaction costs",
                "Faster processing times",
                "Enhanced security features"
            ]
        else:
            your_advantages = [
                "Innovative product approach",
                "Superior user experience",
                "More flexible solution"
            ]
        
        # Create market threats based on industry
        market_threats = []
        if industry.lower() in ["ai", "artificial intelligence", "machine learning"]:
            market_threats = [
                "Rapidly evolving regulatory environment",
                "Increasing concerns about AI ethics",
                "New specialized AI startups entering the market"
            ]
        elif industry.lower() in ["software", "saas", "cloud"]:
            market_threats = [
                "Consolidation of larger players",
                "Open-source alternatives gaining popularity",
                "Changing security and compliance requirements"
            ]
        elif industry.lower() in ["fintech", "financial technology"]:
            market_threats = [
                "Increasing regulatory scrutiny",
                "Big tech companies entering financial services",
                "Traditional financial institutions modernizing"
            ]
        else:
            market_threats = [
                "Larger competitors with more resources",
                "Potential new entrants",
                "Changing market dynamics"
            ]
        
        # Calculate basic positioning
        positioning = {
            "market_position": {
                "market_share_rank": 3,
                "growth_rate_rank": 2,
                "innovation_rank": 1
            },
            "swot_analysis": {
                "strengths": [
                    "Innovative product approach",
                    "Strong growth trajectory",
                    "Talented technical team"
                ],
                "weaknesses": [
                    "Limited market share",
                    "Early-stage product maturity",
                    "Resource constraints"
                ],
                "opportunities": [
                    "Expanding market size",
                    "Competitor weaknesses in innovation",
                    "New market segments"
                ],
                "threats": market_threats
            },
            "competitive_moat": {
                "score": 0.65,
                "factors": {
                    "technology": 0.8,
                    "network_effects": 0.5,
                    "switching_costs": 0.6,
                    "brand": 0.4
                }
            }
        }
        
        # Generate industry-specific recommendations
        recommendations = []
        if industry.lower() in ["ai", "artificial intelligence", "machine learning"]:
            recommendations = [
                "Focus on demonstrating ROI and business value of your AI solutions",
                "Create detailed case studies showing performance advantages over competitors",
                "Develop partnerships with data providers to strengthen your offerings"
            ]
        elif industry.lower() in ["software", "saas", "cloud"]:
            recommendations = [
                "Emphasize integration capabilities with popular enterprise systems",
                "Develop a clear product roadmap highlighting innovation areas",
                "Create a tiered pricing strategy to capture different market segments"
            ]
        else:
            recommendations = [
                "Focus on product differentiation through innovation",
                "Target underserved market segments to establish stronger position",
                "Develop strategic partnerships to expand reach"
            ]
        
        return {
            "competitors": competitors,
            "your_advantages": your_advantages,
            "market_threats": market_threats,
            "positioning": positioning,
            "recommendations": recommendations,
            "is_synthetic": True
        }

# Singleton instance
competitive_intelligence_service = CompetitiveIntelligenceService()
