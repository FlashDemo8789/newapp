from typing import Dict, Any, TypedDict, List, Optional, Union, Tuple, Callable
import logging
import json
import datetime
import uuid
import functools
from enum import Enum
import math
import os
from dataclasses import dataclass, field, asdict

# Configure logging with rotating file handler for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("venture_analytics.team_moat")

class InvestmentStage(Enum):
    """Investment stages used for benchmarking appropriate to company maturity"""
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b" 
    SERIES_C = "series_c"
    GROWTH = "growth"
    PRE_IPO = "pre_ipo"

class FounderBackground(Enum):
    """Types of founder backgrounds for experience categorization"""
    TECHNICAL = "technical"
    BUSINESS = "business"
    DOMAIN_EXPERT = "domain_expert"
    SERIAL_ENTREPRENEUR = "serial_entrepreneur"
    ACADEMIC = "academic"

class RiskFactors(TypedDict):
    """Type definition for risk factor breakdown"""
    founder_experience: float
    team_completeness: float
    domain_expertise: float
    team_size: float
    execution_history: float
    runway_risk: float
    key_person_dependency: float

class ExecutionRiskResult(TypedDict):
    """Type definition for execution risk analysis output"""
    execution_risk_score: float
    risk_factors: RiskFactors
    confidence_interval: Tuple[float, float]
    investor_recommendations: List[str]

class AnalysisResult(TypedDict):
    """Type definition for comprehensive analysis output"""
    team_depth_score: float
    competitive_moat_score: float
    execution_risk: float
    risk_breakdown: RiskFactors
    combined_score: float
    confidence_interval: Tuple[float, float]
    recommendations: List[str]
    industry_percentile: Optional[int]
    historical_trend: Optional[List[Dict[str, Any]]]
    analysis_id: str
    analysis_timestamp: str
    version: str

@dataclass
class BenchmarkData:
    """Industry benchmark data by sector and stage"""
    sector: str
    stage: InvestmentStage
    avg_team_score: float
    avg_moat_score: float
    avg_risk_score: float
    percentile_thresholds: Dict[str, Dict[int, float]] = field(default_factory=dict)

# Cache decorator for expensive operations
def memoize(seconds: int = 3600):
    """Cache decorator with timeout for expensive operations"""
    def decorator_memoize(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper_memoize(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = datetime.datetime.now()
            
            if key in cache:
                result, timestamp = cache[key]
                if (current_time - timestamp).total_seconds() < seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
            
        return wrapper_memoize
    return decorator_memoize

class TeamMoatAnalyzer:
    """
    Enterprise-grade analyzer for startup team strength, competitive moat, and execution risk
    based on various factors including team composition, experience, and capabilities.
    
    Features:
    - Comprehensive analysis with confidence intervals
    - Industry benchmarking
    - Historical trend analysis
    - Production-ready with caching and error handling
    - Detailed recommendations for investors
    """
    
    VERSION = "2.3.0"
    
    # Team depth score weights
    FOUNDER_EXIT_WEIGHT = 13
    FOUNDER_EXIT_MAX = 40
    DOMAIN_EXP_WEIGHT = 3
    DOMAIN_EXP_MAX = 30
    CTO_WEIGHT = 10
    CMO_WEIGHT = 7
    CFO_WEIGHT = 7
    TECH_RATIO_WEIGHT = 20
    TECH_RATIO_MAX = 10
    MGMT_SATISFACTION_FACTOR = 0.1
    MGMT_SATISFACTION_MAX = 10
    DIVERSITY_FACTOR = 0.1
    DIVERSITY_MAX = 10
    
    # Moat score weights
    PATENT_WEIGHT = 8
    PATENT_MAX = 25
    BRAND_FACTOR = 0.2
    BRAND_MAX = 20
    NETWORK_FACTOR = 50
    NETWORK_MAX = 25
    TECH_INNOVATION_FACTOR = 0.2
    TECH_INNOVATION_MAX = 20
    DATA_MOAT_FACTOR = 0.15
    DATA_MOAT_MAX = 15
    PARTNER_WEIGHT = 2
    PARTNER_MAX = 10
    BIZ_MODEL_FACTOR = 0.15
    BIZ_MODEL_MAX = 15
    LICENSE_WEIGHT = 5
    LICENSE_MAX = 10
    
    # Risk assessment factors
    BASE_RISK = 0.5
    EXIT_RISK_FACTOR = 0.1
    EXIT_RISK_MAX = 0.3
    ROLE_RISK_FACTOR = 0.05
    ROLE_RISK_MAX = 0.15
    DOMAIN_RISK_FACTOR = 0.02
    DOMAIN_RISK_MAX = 0.2
    RUNWAY_RISK_MAX = 0.2
    KEY_PERSON_RISK_MAX = 0.15
    EXECUTION_HISTORY_MAX = 0.15
    SMALL_TEAM_RISK = 0.15
    MEDIUM_TEAM_RISK = 0.05
    
    # Industry benchmarks - simplified version (would typically come from a database)
    BENCHMARKS = {
        "software": {
            InvestmentStage.SEED: BenchmarkData(
                sector="software",
                stage=InvestmentStage.SEED,
                avg_team_score=65.3,
                avg_moat_score=48.7,
                avg_risk_score=0.62,
                percentile_thresholds={
                    "team_depth_score": {25: 45.0, 50: 65.3, 75: 80.0, 90: 90.0},
                    "competitive_moat_score": {25: 30.0, 50: 48.7, 75: 65.0, 90: 80.0},
                    "combined_score": {25: 20.0, 50: 35.5, 75: 55.0, 90: 70.0},
                }
            ),
            InvestmentStage.SERIES_A: BenchmarkData(
                sector="software",
                stage=InvestmentStage.SERIES_A,
                avg_team_score=72.1,
                avg_moat_score=58.3,
                avg_risk_score=0.48,
                percentile_thresholds={
                    "team_depth_score": {25: 55.0, 50: 72.1, 75: 85.0, 90: 92.0},
                    "competitive_moat_score": {25: 40.0, 50: 58.3, 75: 75.0, 90: 85.0},
                    "combined_score": {25: 30.0, 50: 47.8, 75: 65.0, 90: 78.0},
                }
            ),
        },
        "biotech": {
            InvestmentStage.SEED: BenchmarkData(
                sector="biotech",
                stage=InvestmentStage.SEED,
                avg_team_score=70.5,
                avg_moat_score=55.2,
                avg_risk_score=0.68,
                percentile_thresholds={
                    "team_depth_score": {25: 55.0, 50: 70.5, 75: 82.0, 90: 91.0},
                    "competitive_moat_score": {25: 40.0, 50: 55.2, 75: 70.0, 90: 82.0},
                    "combined_score": {25: 25.0, 50: 40.2, 75: 58.0, 90: 72.0},
                }
            ),
        },
    }
    
    def __init__(self, 
                 cache_enabled: bool = True, 
                 cache_timeout: int = 3600,
                 data_store_path: Optional[str] = None):
        """
        Initialize the analyzer with configuration options.
        
        Args:
            cache_enabled: Whether to enable caching for expensive operations
            cache_timeout: Timeout in seconds for cached results
            data_store_path: Path to store historical analysis data
        """
        self.cache_enabled = cache_enabled
        self.cache_timeout = cache_timeout
        self.data_store_path = data_store_path
    
    @staticmethod
    def validate_input(doc: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input document has required fields.
        
        Args:
            doc: Input document dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Required fields with default values if missing
        required_fields = {
            "name": "Unnamed Startup",
            "stage": "seed",
            "sector": "software",
            "founder_exits": 0,
            "founder_domain_exp_yrs": 0,
            "employee_count": 1,
            "tech_talent_ratio": 0.5
        }
        
        # Check for missing fields and apply defaults
        for field, default_value in required_fields.items():
            if field not in doc or doc[field] is None:
                # Apply default value instead of rejecting
                doc[field] = default_value
                logger.info(f"Missing required field: {field}, using default value: {default_value}")
        
        # Additional validation rules
        if "tech_talent_ratio" in doc and (doc["tech_talent_ratio"] < 0 or doc["tech_talent_ratio"] > 1):
            return False, f"tech_talent_ratio must be between 0 and 1, got {doc['tech_talent_ratio']}"
        
        if "founder_domain_exp_yrs" in doc and doc["founder_domain_exp_yrs"] < 0:
            doc["founder_domain_exp_yrs"] = 0
            logger.warning(f"founder_domain_exp_yrs was negative, set to 0")
        
        if "founder_exits" in doc and doc["founder_exits"] < 0:
            doc["founder_exits"] = 0
            logger.warning(f"founder_exits was negative, set to 0")
        
        return True, None
    
    def compute_team_depth_score(self, doc: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate team depth score based on founder experience, domain expertise,
        key roles, technical talent, management satisfaction, and diversity.
        
        Args:
            doc: Dictionary containing team metrics and attributes
            
        Returns:
            Tuple of (team_depth_score, component_breakdown)
        """
        try:
            # Validate input
            is_valid, error = self.validate_input(doc)
            if not is_valid:
                logger.error(f"Input validation error: {error}")
                return 0.0, {}
            
            score = 0.0
            components = {}
            
            # Founder exits (weighted up to max)
            founder_exits = doc.get("founder_exits", 0)
            exit_score = min(self.FOUNDER_EXIT_MAX, founder_exits * self.FOUNDER_EXIT_WEIGHT)
            score += exit_score
            components["founder_exits"] = exit_score
            
            # Domain expertise (weighted up to max)
            domain_exp = doc.get("founder_domain_exp_yrs", 0)
            domain_score = min(self.DOMAIN_EXP_MAX, domain_exp * self.DOMAIN_EXP_WEIGHT)
            score += domain_score
            components["domain_expertise"] = domain_score
            
            # Key roles present
            key_roles_score = 0
            if doc.get("has_cto", False):
                key_roles_score += self.CTO_WEIGHT
            if doc.get("has_cmo", False):
                key_roles_score += self.CMO_WEIGHT
            if doc.get("has_cfo", False):
                key_roles_score += self.CFO_WEIGHT
            score += key_roles_score
            components["key_roles"] = key_roles_score
                
            # Technical talent ratio (weighted up to max)
            tech_ratio = doc.get("tech_talent_ratio", 0.0)
            tech_score = min(self.TECH_RATIO_MAX, tech_ratio * self.TECH_RATIO_WEIGHT)
            score += tech_score
            components["technical_talent"] = tech_score
            
            # Management satisfaction (scaled up to max)
            mgmt_score = doc.get("management_satisfaction_score", 0.0)
            mgmt_satisfaction = min(self.MGMT_SATISFACTION_MAX, 
                                    mgmt_score * self.MGMT_SATISFACTION_FACTOR)
            score += mgmt_satisfaction
            components["management_satisfaction"] = mgmt_satisfaction
            
            # Team diversity (scaled up to max)
            diversity_score = doc.get("founder_diversity_score", 0.0)
            diversity = min(self.DIVERSITY_MAX, diversity_score * self.DIVERSITY_FACTOR)
            score += diversity
            components["team_diversity"] = diversity
            
            # Advanced factors (if available)
            if "team_cohesion_score" in doc:
                cohesion_score = min(15, doc.get("team_cohesion_score", 0) * 0.15)
                score += cohesion_score
                components["team_cohesion"] = cohesion_score
            
            if "founder_complementary_skills" in doc:
                comp_skills = min(10, doc.get("founder_complementary_skills", 0) * 0.1)
                score += comp_skills
                components["complementary_skills"] = comp_skills
            
            # Cap at 100
            final_score = min(score, 100)
            
            # Calculate confidence factor based on data completeness
            confidence = min(1.0, len(components) / 8)
            components["confidence_factor"] = confidence
            
            return final_score, components
            
        except Exception as e:
            logger.error(f"Error computing team depth score: {e}", exc_info=True)
            return 0.0, {"error": str(e)}
    
    @memoize(3600)
    def compute_moat_score(self, doc: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate competitive moat score based on IP, brand strength, network effects,
        technical innovation, data advantages, partnerships, business model strength,
        and licensing.
        
        Args:
            doc: Dictionary containing moat-related metrics
            
        Returns:
            Tuple of (moat_score, component_breakdown)
        """
        try:
            score = 0.0
            components = {}
            
            # Patent count (weighted up to max)
            patents = doc.get("patent_count", 0)
            patent_score = min(self.PATENT_MAX, patents * self.PATENT_WEIGHT)
            score += patent_score
            components["patents"] = patent_score
            
            # Brand/category leadership (scaled up to max)
            brand = doc.get("category_leadership_score", 0)
            brand_score = min(self.BRAND_MAX, brand * self.BRAND_FACTOR)
            score += brand_score
            components["brand_leadership"] = brand_score
            
            # Network effects/viral coefficient (scaled up to max)
            network = doc.get("viral_coefficient", 0) * self.NETWORK_FACTOR
            network_score = min(self.NETWORK_MAX, network)
            score += network_score
            components["network_effects"] = network_score
            
            # Technical innovation (scaled up to max)
            tech_innovation = doc.get("technical_innovation_score", 0)
            tech_score = min(self.TECH_INNOVATION_MAX, 
                             tech_innovation * self.TECH_INNOVATION_FACTOR)
            score += tech_score
            components["technical_innovation"] = tech_score
            
            # Data advantages (scaled up to max)
            data_moat = doc.get("data_moat_strength", 0)
            data_score = min(self.DATA_MOAT_MAX, data_moat * self.DATA_MOAT_FACTOR)
            score += data_score
            components["data_advantages"] = data_score
            
            # Channel partners (weighted up to max)
            partners = doc.get("channel_partner_count", 0)
            partner_score = min(self.PARTNER_MAX, partners * self.PARTNER_WEIGHT)
            score += partner_score
            components["channel_partners"] = partner_score
            
            # Business model strength (scaled up to max)
            biz_model = doc.get("business_model_strength", 0)
            biz_score = min(self.BIZ_MODEL_MAX, biz_model * self.BIZ_MODEL_FACTOR)
            score += biz_score
            components["business_model"] = biz_score
            
            # Licenses (weighted up to max)
            licenses = doc.get("licenses_count", 0)
            license_score = min(self.LICENSE_MAX, licenses * self.LICENSE_WEIGHT)
            score += license_score
            components["licenses"] = license_score
            
            # Advanced factors (if available)
            if "switching_cost_rating" in doc:
                switching_cost = min(15, doc.get("switching_cost_rating", 0) * 0.15)
                score += switching_cost
                components["switching_costs"] = switching_cost
                
            if "regulatory_moat_strength" in doc:
                reg_moat = min(15, doc.get("regulatory_moat_strength", 0) * 0.15)
                score += reg_moat
                components["regulatory_advantage"] = reg_moat
            
            # Cap at 100
            final_score = min(100, score)
            
            # Calculate confidence factor based on data completeness
            confidence = min(1.0, len(components) / 10)
            components["confidence_factor"] = confidence
            
            return final_score, components
            
        except Exception as e:
            logger.error(f"Error computing moat score: {e}", exc_info=True)
            return 0.0, {"error": str(e)}
    
    def evaluate_team_execution_risk(self, doc: Dict[str, Any]) -> ExecutionRiskResult:
        """
        Evaluate execution risk based on founder experience, team completeness,
        domain expertise, and team size.
        
        Args:
            doc: Dictionary containing team metrics and attributes
            
        Returns:
            ExecutionRiskResult: Execution risk score (0-1) and breakdown of risk factors
        """
        try:
            # Start with base risk
            base_risk = self.BASE_RISK
            risk_components = {}
            
            # Factor 1: Founder exits/experience reduces risk
            founder_exits = doc.get("founder_exits", 0)
            experience_factor = min(self.EXIT_RISK_MAX, founder_exits * self.EXIT_RISK_FACTOR)
            
            # Factor 2: Team completeness reduces risk
            team_completeness = 0
            if doc.get("has_cto", False): 
                team_completeness += 1
            if doc.get("has_cmo", False): 
                team_completeness += 1
            if doc.get("has_cfo", False): 
                team_completeness += 1
            completeness_factor = min(self.ROLE_RISK_MAX, team_completeness * self.ROLE_RISK_FACTOR)
            
            # Factor 3: Domain expertise reduces risk
            domain_exp = doc.get("founder_domain_exp_yrs", 0)
            domain_factor = min(self.DOMAIN_RISK_MAX, domain_exp * self.DOMAIN_RISK_FACTOR)
            
            # Factor 4: Small team increases risk
            team_size = doc.get("employee_count", 1)
            if team_size < 3:
                size_factor = self.SMALL_TEAM_RISK
            elif team_size < 10:
                size_factor = self.MEDIUM_TEAM_RISK
            else:
                size_factor = 0
            
            # Factor 5: Runway risk (if available)
            runway_months = doc.get("runway_months", 12)
            if runway_months < 6:
                runway_factor = self.RUNWAY_RISK_MAX
            elif runway_months < 12:
                runway_factor = self.RUNWAY_RISK_MAX * 0.7
            elif runway_months < 18:
                runway_factor = self.RUNWAY_RISK_MAX * 0.3
            else:
                runway_factor = 0
                
            # Factor 6: Key person dependency
            key_person_dependency = doc.get("key_person_dependency", 0.5)
            key_person_factor = self.KEY_PERSON_RISK_MAX * key_person_dependency
            
            # Factor 7: Execution history
            execution_history = doc.get("execution_history_score", 0.5)
            execution_factor = self.EXECUTION_HISTORY_MAX * (1 - execution_history)
            
            # Calculate final risk score
            risk_score = (base_risk 
                        - experience_factor 
                        - completeness_factor 
                        - domain_factor 
                        + size_factor
                        + runway_factor
                        + key_person_factor
                        - (execution_factor if "execution_history_score" in doc else 0))
            
            # Bound between 0.1 and 0.9
            risk_score = max(0.1, min(0.9, risk_score))
            
            # Calculate confidence interval (±10% by default, narrower with more data)
            data_completeness = len([k for k in doc.keys() if k in [
                "founder_exits", "has_cto", "has_cmo", "has_cfo", 
                "founder_domain_exp_yrs", "employee_count",
                "runway_months", "key_person_dependency", "execution_history_score"
            ]]) / 9.0
            
            margin = 0.1 * (1 - data_completeness)
            confidence_interval = (max(0.1, risk_score - margin), min(0.9, risk_score + margin))
            
            # Calculate normalized risk factors (higher is riskier)
            risk_factors = {
                "founder_experience": 1 - experience_factor / self.EXIT_RISK_MAX,
                "team_completeness": 1 - completeness_factor / self.ROLE_RISK_MAX,
                "domain_expertise": 1 - domain_factor / self.DOMAIN_RISK_MAX,
                "team_size": size_factor / self.SMALL_TEAM_RISK if size_factor > 0 else 0,
                "runway_risk": runway_factor / self.RUNWAY_RISK_MAX if "runway_months" in doc else 0.5,
                "key_person_dependency": key_person_factor / self.KEY_PERSON_RISK_MAX if "key_person_dependency" in doc else 0.5,
                "execution_history": execution_factor / self.EXECUTION_HISTORY_MAX if "execution_history_score" in doc else 0.5
            }
            
            # Generate investor-focused recommendations based on risk factors
            recommendations = []
            
            if risk_factors["founder_experience"] > 0.7:
                recommendations.append("High risk: Limited founder exit experience. Consider adding experienced advisors or executives.")
            
            if risk_factors["team_completeness"] > 0.6:
                recommendations.append("Moderate risk: Incomplete executive team. Prioritize filling key C-suite roles.")
                
            if risk_factors["domain_expertise"] > 0.7:
                recommendations.append("High risk: Limited domain expertise. Consider adding domain experts to the team or advisory board.")
                
            if risk_factors["team_size"] > 0.8:
                recommendations.append("High risk: Team is too small for execution. Accelerate hiring in key roles.")
                
            if risk_factors.get("runway_risk", 0) > 0.7:
                recommendations.append("Critical risk: Limited runway. Prioritize fundraising or revenue acceleration.")
                
            if risk_factors.get("key_person_dependency", 0) > 0.7:
                recommendations.append("High risk: Strong dependency on key individuals. Develop succession planning and knowledge sharing.")
            
            return {
                "execution_risk_score": risk_score,
                "risk_factors": risk_factors,
                "confidence_interval": confidence_interval,
                "investor_recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error evaluating execution risk: {e}", exc_info=True)
            return {
                "execution_risk_score": 0.5,
                "risk_factors": {
                    "founder_experience": 0.5,
                    "team_completeness": 0.5,
                    "domain_expertise": 0.5,
                    "team_size": 0.5,
                    "runway_risk": 0.5,
                    "key_person_dependency": 0.5,
                    "execution_history": 0.5
                },
                "confidence_interval": (0.4, 0.6),
                "investor_recommendations": ["Error in risk analysis. Insufficient data for recommendations."]
            }

    def calculate_industry_percentile(self, 
                                      score: float, 
                                      metric: str, 
                                      sector: str, 
                                      stage: InvestmentStage) -> Optional[int]:
        """
        Calculate the percentile of a score against industry benchmarks
        
        Args:
            score: The score to evaluate
            metric: Which metric to compare (team_depth_score, competitive_moat_score, etc)
            sector: Industry sector (software, biotech, etc)
            stage: Investment stage
            
        Returns:
            Optional[int]: Percentile position (1-100) or None if benchmarks unavailable
        """
        try:
            if sector not in self.BENCHMARKS:
                return None
                
            if stage not in self.BENCHMARKS[sector]:
                return None
                
            benchmarks = self.BENCHMARKS[sector][stage]
            if metric not in benchmarks.percentile_thresholds:
                return None
                
            thresholds = benchmarks.percentile_thresholds[metric]
            
            # Find percentile position
            for percentile in sorted(thresholds.keys(), reverse=True):
                if score >= thresholds[percentile]:
                    return percentile
                    
            # Below lowest threshold
            lowest_percentile = min(thresholds.keys())
            return lowest_percentile // 2  # Rough estimate below the lowest threshold
            
        except Exception as e:
            logger.error(f"Error calculating percentile: {e}")
            return None
    
    def load_historical_data(self, company_id: str) -> List[Dict[str, Any]]:
        """
        Load historical analysis data for trend analysis
        
        Args:
            company_id: Unique identifier for the company
            
        Returns:
            List of historical analysis results
        """
        if not self.data_store_path:
            return []
            
        try:
            file_path = os.path.join(self.data_store_path, f"{company_id}_history.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return []
    
    def save_analysis_result(self, company_id: str, result: Dict[str, Any]) -> bool:
        """
        Save analysis result for historical tracking
        
        Args:
            company_id: Unique identifier for the company
            result: Analysis result to save
            
        Returns:
            bool: Success status
        """
        if not self.data_store_path:
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(self.data_store_path, exist_ok=True)
            
            # Load existing history
            file_path = os.path.join(self.data_store_path, f"{company_id}_history.json")
            history = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    history = json.load(f)
            
            # Add new result with timestamp
            timestamp = datetime.datetime.now().isoformat()
            history_entry = {
                "timestamp": timestamp,
                "team_score": result["team_depth_score"],
                "moat_score": result["competitive_moat_score"],
                "risk_score": result["execution_risk"],
                "combined_score": result["combined_score"]
            }
            history.append(history_entry)
            
            # Save updated history
            with open(file_path, 'w') as f:
                json.dump(history, f)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            return False
    
    def get_comprehensive_analysis(self, 
                                  company_data: Dict[str, Any],
                                  company_id: Optional[str] = None,
                                  sector: str = "software",
                                  stage: InvestmentStage = InvestmentStage.SEED) -> AnalysisResult:
        """
        Provides a comprehensive analysis of team strength, moat, and risk
        in a single function call with industry benchmarking.
        
        Args:
            company_data: Dictionary containing all team and moat metrics
            company_id: Optional unique identifier for the company (for historical tracking)
            sector: Industry sector for benchmarking
            stage: Investment stage for benchmarking
            
        Returns:
            AnalysisResult: Combined analysis results with scores and recommendations
        """
        # Generate analysis ID if not provided
        if not company_id:
            company_id = str(uuid.uuid4())
            
        # Calculate team score with component breakdown
        team_score, team_components = self.compute_team_depth_score(company_data)
        
        # Calculate moat score with component breakdown
        moat_score, moat_components = self.compute_moat_score(company_data)
        
        # Calculate risk analysis
        risk_analysis = self.evaluate_team_execution_risk(company_data)
        
        # Calculate combined score
        risk_adjustment = 1 - risk_analysis["execution_risk_score"]
        combined_score = (team_score + moat_score) / 2 * risk_adjustment
        
        # Calculate confidence interval for combined score
        team_confidence = team_components.get("confidence_factor", 0.8)
        moat_confidence = moat_components.get("confidence_factor", 0.8)
        risk_interval_width = risk_analysis["confidence_interval"][1] - risk_analysis["confidence_interval"][0]
        
        # Overall confidence is influenced by all three components
        avg_confidence = (team_confidence + moat_confidence + (1 - risk_interval_width)) / 3
        margin = 10 * (1 - avg_confidence)
        confidence_interval = (max(0, combined_score - margin), min(100, combined_score + margin))
        
        # Get industry percentile if benchmarks available
        industry_percentile = self.calculate_industry_percentile(
            combined_score, "combined_score", sector, stage
        )
        
        # Load historical trend data if company_id provided
        historical_trend = None
        if company_id:
            historical_data = self.load_historical_data(company_id)
            if historical_data:
                historical_trend = historical_data
        
        # Generate recommendations based on scores
        recommendations = []
        
        # Team recommendations
        if team_score < 50:
            recommendations.append("Team strengthening priority: Consider adding experienced executives with domain expertise")
        elif team_score < 70:
            recommendations.append("Team improvement area: Focus on filling key leadership roles and improving team diversity")
        
        # Moat recommendations
        if moat_score < 40:
            recommendations.append("Critical moat weakness: Develop stronger competitive barriers through patents, network effects, or data advantages")
        elif moat_score < 60:
            recommendations.append("Moat enhancement opportunity: Focus on strengthening your top two competitive advantages to create defensibility")
        
        # Risk-specific recommendations
        recommendations.extend(risk_analysis["investor_recommendations"])
        
        # Sector-specific recommendations
        if sector == "software" and moat_components.get("data_advantages", 0) < 5:
            recommendations.append("Software-specific recommendation: Develop a data acquisition and leverage strategy to create sustainable competitive advantage")
        
        elif sector == "biotech" and team_components.get("domain_expertise", 0) < 20:
            recommendations.append("Biotech-specific recommendation: Add scientific advisory board members with deep domain expertise")
        
        # Create result object
        result = {
            "team_depth_score": team_score,
            "team_components": team_components,
            "competitive_moat_score": moat_score,
            "moat_components": moat_components,
            "execution_risk": risk_analysis["execution_risk_score"],
            "risk_breakdown": risk_analysis["risk_factors"],
            "combined_score": combined_score,
            "confidence_interval": confidence_interval,
            "recommendations": recommendations,
            "industry_percentile": industry_percentile,
            "historical_trend": historical_trend,
            "analysis_id": str(uuid.uuid4()),
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "version": self.VERSION
        }
        
        # Save result for historical tracking if company_id provided
        if company_id:
            self.save_analysis_result(company_id, result)
        
        return result

# Create a global analyzer instance for the wrapper functions
_analyzer = TeamMoatAnalyzer(cache_enabled=True, data_store_path="./analytics_data")

# Create wrapper functions that the main application expects to import
def compute_team_depth_score(doc: Dict[str, Any]) -> float:
    """
    Wrapper function to compute team depth score using the TeamMoatAnalyzer.
    This maintains compatibility with the existing import structure.
    
    Args:
        doc: Dictionary containing team metrics and attributes
        
    Returns:
        float: Team depth score (0-100)
    """
    score, _ = _analyzer.compute_team_depth_score(doc)
    return score

def compute_moat_score(doc: Dict[str, Any]) -> float:
    """
    Wrapper function to compute competitive moat score using the TeamMoatAnalyzer.
    This maintains compatibility with the existing import structure.
    
    Args:
        doc: Dictionary containing moat metrics
        
    Returns:
        float: Moat score (0-100)
    """
    score, _ = _analyzer.compute_moat_score(doc)
    return score

def evaluate_team_execution_risk(doc: Dict[str, Any]) -> ExecutionRiskResult:
    """
    Wrapper function to evaluate team execution risk using the TeamMoatAnalyzer.
    This maintains compatibility with the existing import structure.
    
    Args:
        doc: Dictionary containing team metrics and attributes
        
    Returns:
        ExecutionRiskResult: Object containing risk score and breakdown
    """
    return _analyzer.evaluate_team_execution_risk(doc)

# Add this function at the end of the file
def analyze_team(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze team moat data - interface function for module_exports
    
    Parameters:
    data (dict): Input data about the company team
    
    Returns:
    dict: Formatted team analysis results
    """
    try:
        analyzer = TeamMoatAnalyzer()
        
        # Extract sector and stage if available
        sector = data.get("sector", "software")
        stage_str = data.get("investment_stage", "seed")
        
        # Convert stage string to enum
        try:
            stage = InvestmentStage(stage_str.lower())
        except (ValueError, AttributeError):
            stage = InvestmentStage.SEED
        
        # Get company ID if available
        company_id = data.get("company_id")
        
        # Run comprehensive analysis
        result = analyzer.get_comprehensive_analysis(
            company_data=data,
            company_id=company_id,
            sector=sector,
            stage=stage
        )
        
        # Convert to dictionary if it's not already
        if not isinstance(result, dict):
            result = {k: v for k, v in result.items()}
        
        return result
    except Exception as e:
        # Return error with traceback for debugging
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }

# For testing
if __name__ == "__main__":
    # Sample company data
    sample_company = {
        "founder_exits": 2,
        "founder_domain_exp_yrs": 8,
        "has_cto": True,
        "has_cmo": True,
        "has_cfo": False,
        "tech_talent_ratio": 0.4,
        "management_satisfaction_score": 85,
        "founder_diversity_score": 70,
        "patent_count": 2,
        "category_leadership_score": 65,
        "viral_coefficient": 0.3,
        "technical_innovation_score": 75,
        "data_moat_strength": 60,
        "channel_partner_count": 4,
        "business_model_strength": 70,
        "licenses_count": 1,
        "employee_count": 15,
        "runway_months": 14,
        "key_person_dependency": 0.4,
        "execution_history_score": 0.7
    }
    
    # Test the wrapper functions
    team_score = compute_team_depth_score(sample_company)
    moat_score = compute_moat_score(sample_company)
    risk_result = evaluate_team_execution_risk(sample_company)
    
    print(f"Team Score: {team_score}")
    print(f"Moat Score: {moat_score}")
    print(f"Execution Risk: {risk_result['execution_risk_score']}")
