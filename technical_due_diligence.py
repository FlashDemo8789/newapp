import os
import json
import logging
import re
import time
import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tech_due_diligence")

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

class TechMaturityLevel(str, Enum):
    """Industry-standard technology maturity classification"""
    EMERGING = "Emerging"      # New technology, rapidly evolving
    GROWING = "Growing"        # Gaining adoption, still evolving
    MATURE = "Mature"          # Widely adopted, stable ecosystem
    DECLINING = "Declining"    # Decreasing relevance, maintenance mode
    LEGACY = "Legacy"          # Outdated, minimal maintenance


class RiskLevel(str, Enum):
    """Standardized risk classification for technical assessment"""
    CRITICAL = "Critical"      # Immediate attention required
    HIGH = "High"              # Significant concern, action needed soon
    MEDIUM = "Medium"          # Moderate concern, plan to address
    LOW = "Low"                # Minor concern, monitor
    NEGLIGIBLE = "Negligible"  # No significant concern


class TechCategory(str, Enum):
    """Technical component categorization"""
    LANGUAGE = "Language"
    FRAMEWORK = "Framework"
    DATABASE = "Database"
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    INFRASTRUCTURE = "Infrastructure"
    CLOUD_SERVICE = "Cloud Service"
    DEVOPS = "DevOps" 
    SECURITY = "Security"
    ANALYTICS = "Analytics"
    MESSAGING = "Messaging"
    CACHE = "Cache"
    MOBILE = "Mobile"
    API = "API"
    TESTING = "Testing"
    ML_AI = "Machine Learning/AI"
    OTHER = "Other"


class ArchitecturePattern(str, Enum):
    """Enterprise architecture patterns"""
    MONOLITH = "Monolithic"
    MICROSERVICES = "Microservices"
    SERVERLESS = "Serverless"
    EVENT_DRIVEN = "Event-Driven"
    LAYERED = "Layered"
    HEXAGONAL = "Hexagonal/Ports & Adapters"
    CQRS = "CQRS"
    DDD = "Domain-Driven Design"
    SOA = "Service-Oriented Architecture"
    SPACE_BASED = "Space-Based Architecture"
    MESH = "Service Mesh"
    HYBRID = "Hybrid"
    UNDEFINED = "Undefined"


@dataclass
class TechStackItem:
    """Comprehensive technology stack component assessment"""
    name: str
    category: TechCategory
    version: Optional[str] = None
    
    # Maturity metrics
    maturity_level: TechMaturityLevel = TechMaturityLevel.GROWING
    maturity_score: float = 0.5
    market_adoption: float = 0.5
    
    # Performance characteristics
    scalability: float = 0.5  # 0 to 1 scale
    reliability: float = 0.5  # 0 to 1 scale
    performance: float = 0.5  # 0 to 1 scale
    
    # Risk factors
    security_score: float = 0.5
    expertise_required: float = 0.5
    vendor_lock_in: float = 0.5
    community_support: float = 0.5
    
    # Operational factors
    cost_efficiency: float = 0.5
    operational_complexity: float = 0.5
    
    # Meta data
    last_major_release: Optional[str] = None
    release_frequency: Optional[str] = None
    license_type: Optional[str] = None
    
    # Specific vulnerabilities and risks
    known_vulnerabilities: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate scores and metrics"""
        score_fields = [
            'maturity_score', 'market_adoption', 'scalability', 'reliability',
            'performance', 'security_score', 'expertise_required', 'vendor_lock_in',
            'community_support', 'cost_efficiency', 'operational_complexity'
        ]
        for field_name in score_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be a number between 0 and 1")
        
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        
        if not isinstance(self.category, TechCategory):
            raise ValueError("category must be a valid TechCategory")
        
        if not isinstance(self.maturity_level, TechMaturityLevel):
            raise ValueError("maturity_level must be a valid TechMaturityLevel")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper enum handling"""
        result = asdict(self)
        # Handle enum conversion
        if isinstance(self.category, Enum):
            result['category'] = self.category.value
        if isinstance(self.maturity_level, Enum):
            result['maturity_level'] = self.maturity_level.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TechStackItem':
        """Create from dictionary with enum handling"""
        # Convert string values to enums
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = TechCategory(data['category'])
        if 'maturity_level' in data and isinstance(data['maturity_level'], str):
            data['maturity_level'] = TechMaturityLevel(data['maturity_level'])
        return cls(**data)


@dataclass
class SecurityAssessment:
    """Comprehensive security assessment based on industry standards"""
    # Overall scores
    overall_score: float
    
    # Compliance assessments
    compliance_frameworks: Dict[str, float] = field(default_factory=dict)
    
    # OWASP Top 10 assessment
    owasp_risk_profile: Dict[str, float] = field(default_factory=dict)
    
    # Component analysis
    dependency_vulnerabilities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    critical_vulnerabilities_count: int = 0
    high_vulnerabilities_count: int = 0
    medium_vulnerabilities_count: int = 0
    low_vulnerabilities_count: int = 0
    
    # Security practices assessment
    security_practices: Dict[str, float] = field(default_factory=dict)
    
    # Authentication and authorization
    auth_assessment: Dict[str, float] = field(default_factory=dict)
    
    # Data protection
    data_protection_assessment: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    critical_recommendations: List[str] = field(default_factory=list)
    high_priority_recommendations: List[str] = field(default_factory=list)
    medium_priority_recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate security assessment data"""
        # Validate overall score
        if not isinstance(self.overall_score, (int, float)) or not 0 <= self.overall_score <= 1:
            raise ValueError("overall_score must be a number between 0 and 1")
        
        # Validate vulnerability counts
        for field_name in ['critical_vulnerabilities_count', 'high_vulnerabilities_count', 
                          'medium_vulnerabilities_count', 'low_vulnerabilities_count']:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        
        # Validate assessment dictionaries
        score_dicts = [
            ('compliance_frameworks', self.compliance_frameworks),
            ('owasp_risk_profile', self.owasp_risk_profile),
            ('security_practices', self.security_practices),
            ('auth_assessment', self.auth_assessment),
            ('data_protection_assessment', self.data_protection_assessment)
        ]
        
        for name, score_dict in score_dicts:
            if not isinstance(score_dict, dict):
                raise ValueError(f"{name} must be a dictionary")
            for key, value in score_dict.items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"All scores in {name} must be numbers between 0 and 1")
        
        # Validate recommendation lists
        for field_name in ['critical_recommendations', 'high_priority_recommendations', 'medium_priority_recommendations']:
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"All items in {field_name} must be strings")
        
        # Validate dependency vulnerabilities
        if not isinstance(self.dependency_vulnerabilities, dict):
            raise ValueError("dependency_vulnerabilities must be a dictionary")
        for key, value in self.dependency_vulnerabilities.items():
            if not isinstance(value, list):
                raise ValueError(f"Value for {key} in dependency_vulnerabilities must be a list")
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"All items in dependency_vulnerabilities[{key}] must be dictionaries")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper enum handling"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], TechStackItem):
                result[key] = [item.to_dict() for item in value]
            elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


@dataclass
class ScalabilityAssessment:
    """Detailed scalability assessment with capacity planning"""
    # Required fields (no defaults)
    overall_score: float
    infrastructure_scalability: float
    database_scalability: float
    compute_scalability: float
    network_scalability: float
    architecture_scalability: float
    
    # Optional fields (with defaults)
    bottlenecks: List[str] = field(default_factory=list)
    throughput_capability: Optional[str] = None
    latency_profile: Optional[str] = None
    load_test_results: Dict[str, Any] = field(default_factory=dict)
    horizontal_scaling: bool = False
    vertical_scaling: bool = False
    data_partitioning: bool = False
    caching_strategy: bool = False
    max_theoretical_capacity: Optional[str] = None
    scaling_recommendations: List[str] = field(default_factory=list)
    scaling_costs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalDebtAssessment:
    """Comprehensive technical debt assessment"""
    overall_score: float
    
    # Quality metrics
    code_quality: Dict[str, float] = field(default_factory=dict)
    test_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Code metrics
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    duplication_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Repository metrics
    commit_frequency: Optional[float] = None
    lead_time: Optional[float] = None
    
    # Maintenance burden
    maintenance_burden: float = 0.5
    modernization_needs: List[str] = field(default_factory=list)
    
    # Refactoring needs
    critical_refactoring_areas: List[str] = field(default_factory=list)
    refactoring_cost_estimate: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DORAMetrics:
    """DevOps Research and Assessment metrics for delivery performance"""
    deployment_frequency: str  # e.g., "Multiple deploys per day", "Between once per week and once per month"
    lead_time_for_changes: str  # e.g., "Less than one hour", "Between one week and one month"
    time_to_restore_service: str  # e.g., "Less than one hour", "Less than one day"
    change_failure_rate: float  # Percentage of changes that lead to degraded service
    
    # Custom metrics beyond DORA's core four
    deployment_success_rate: float = 0.0
    automated_test_coverage: float = 0.0
    mean_time_between_failures: Optional[float] = None
    incident_resolution_time: Optional[float] = None
    
    # Classification
    performance_category: str = "Medium"  # Elite, High, Medium, Low
    
    def __post_init__(self):
        """Validate DORA metrics"""
        valid_categories = {"Elite", "High", "Medium", "Low"}
        if self.performance_category not in valid_categories:
            raise ValueError(f"performance_category must be one of: {', '.join(valid_categories)}")
        
        if not isinstance(self.change_failure_rate, (int, float)) or not 0 <= self.change_failure_rate <= 100:
            raise ValueError("change_failure_rate must be a percentage between 0 and 100")
        
        if not isinstance(self.deployment_success_rate, (int, float)) or not 0 <= self.deployment_success_rate <= 100:
            raise ValueError("deployment_success_rate must be a percentage between 0 and 100")
        
        if not isinstance(self.automated_test_coverage, (int, float)) or not 0 <= self.automated_test_coverage <= 100:
            raise ValueError("automated_test_coverage must be a percentage between 0 and 100")
        
        if self.mean_time_between_failures is not None and (not isinstance(self.mean_time_between_failures, (int, float)) or self.mean_time_between_failures < 0):
            raise ValueError("mean_time_between_failures must be a non-negative number")
        
        if self.incident_resolution_time is not None and (not isinstance(self.incident_resolution_time, (int, float)) or self.incident_resolution_time < 0):
            raise ValueError("incident_resolution_time must be a non-negative number")


@dataclass
class CompetitivePositioning:
    """Technical competitive positioning assessment"""
    relative_tech_strength: float  # 0-1 scale
    technical_advantages: List[str] = field(default_factory=list)
    technical_disadvantages: List[str] = field(default_factory=list)
    differentiation_factors: List[str] = field(default_factory=list)
    competitive_threats: List[str] = field(default_factory=list)
    industry_benchmark_position: Optional[str] = None  # e.g., "Top 10%", "Average", "Below average"
    
    def __post_init__(self):
        """Validate competitive positioning metrics"""
        if not isinstance(self.relative_tech_strength, (int, float)):
            raise ValueError("relative_tech_strength must be a number between 0 and 1")
            
        if not 0 <= self.relative_tech_strength <= 1:
            raise ValueError("relative_tech_strength must be a number between 0 and 1")
        
        if self.industry_benchmark_position is not None:
            valid_positions = {"Top 10%", "Top 25%", "Average", "Below average", "Bottom 25%"}
            if self.industry_benchmark_position not in valid_positions:
                raise ValueError(f"industry_benchmark_position must be one of: {', '.join(valid_positions)}")
        
        for field_name in ['technical_advantages', 'technical_disadvantages', 'differentiation_factors', 'competitive_threats']:
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"All items in {field_name} must be strings")


@dataclass
class TalentAssessment:
    """Engineering talent and skill assessment"""
    # Required fields (no defaults)
    overall_score: float
    team_size: int
    senior_engineers_ratio: float
    knowledge_sharing_score: float
    documentation_quality: float
    onboarding_efficiency: float
    market_competitiveness: float
    
    # Optional fields (with defaults)
    skills_coverage: Dict[str, float] = field(default_factory=dict)
    key_person_dependencies: int = 0
    hiring_difficulty: Dict[str, float] = field(default_factory=dict)
    estimated_replacement_time: Dict[str, int] = field(default_factory=dict)
    talent_recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate talent assessment data"""
        # Validate score fields
        score_fields = [
            'overall_score', 'senior_engineers_ratio', 'knowledge_sharing_score',
            'documentation_quality', 'onboarding_efficiency', 'market_competitiveness'
        ]
        for field_name in score_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be a number between 0 and 1")
        
        # Validate team size
        if not isinstance(self.team_size, int) or self.team_size < 0:
            raise ValueError("team_size must be a non-negative integer")
        
        # Validate key person dependencies
        if not isinstance(self.key_person_dependencies, int) or self.key_person_dependencies < 0:
            raise ValueError("key_person_dependencies must be a non-negative integer")
        
        # Validate dictionary fields
        for field_name in ['skills_coverage', 'hiring_difficulty']:
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                raise ValueError(f"{field_name} must be a dictionary")
            for k, v in value.items():
                if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                    raise ValueError(f"All scores in {field_name} must be numbers between 0 and 1")
        
        # Validate estimated replacement time
        if not isinstance(self.estimated_replacement_time, dict):
            raise ValueError("estimated_replacement_time must be a dictionary")
        for k, v in self.estimated_replacement_time.items():
            if not isinstance(v, int) or v < 0:
                raise ValueError("All values in estimated_replacement_time must be non-negative integers")
        
        # Validate recommendations
        if not isinstance(self.talent_recommendations, list):
            raise ValueError("talent_recommendations must be a list")
        if not all(isinstance(item, str) for item in self.talent_recommendations):
            raise ValueError("All items in talent_recommendations must be strings")


@dataclass
class OperationalAssessment:
    """Production operations and reliability assessment"""
    # Required fields (no defaults)
    overall_score: float
    reliability_score: float
    uptime_percentage: float
    observability_score: float
    monitoring_coverage: float
    alerting_effectiveness: float
    incident_management_score: float
    sre_maturity: float
    
    # Optional fields (with defaults)
    mean_time_between_failures: Optional[float] = None
    mean_time_to_recovery: Optional[float] = None
    error_budget_policy: bool = False
    chaos_engineering: bool = False
    operational_recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate operational assessment data"""
        # Validate score fields
        score_fields = [
            'overall_score', 'reliability_score', 'observability_score',
            'monitoring_coverage', 'alerting_effectiveness',
            'incident_management_score', 'sre_maturity'
        ]
        for field_name in score_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be a number between 0 and 1")
        
        # Validate uptime percentage
        if not isinstance(self.uptime_percentage, (int, float)) or not 0 <= self.uptime_percentage <= 100:
            raise ValueError("uptime_percentage must be a number between 0 and 100")
        
        # Validate optional time metrics
        for field_name in ['mean_time_between_failures', 'mean_time_to_recovery']:
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative number")
        
        # Validate recommendations
        if not isinstance(self.operational_recommendations, list):
            raise ValueError("operational_recommendations must be a list")
        if not all(isinstance(item, str) for item in self.operational_recommendations):
            raise ValueError("All items in operational_recommendations must be strings")


@dataclass
class TechnicalRoadmap:
    """Strategic technical evolution planning"""
    horizon_1_initiatives: List[Dict[str, Any]]  # 0-6 months
    horizon_2_initiatives: List[Dict[str, Any]]  # 6-18 months
    horizon_3_initiatives: List[Dict[str, Any]]  # 18+ months
    
    # Cost and resource estimates
    implementation_costs: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Risk assessment
    implementation_risks: Dict[str, RiskLevel] = field(default_factory=dict)
    migration_complexity: Dict[str, float] = field(default_factory=dict)
    
    # Business alignment
    business_impact_assessment: Dict[str, Any] = field(default_factory=dict)
    strategic_alignment_score: float = 0.0
    
    def __post_init__(self):
        """Validate technical roadmap data"""
        # Validate initiatives
        for field_name in ['horizon_1_initiatives', 'horizon_2_initiatives', 'horizon_3_initiatives']:
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"All items in {field_name} must be dictionaries")
        
        # Validate implementation risks
        for key, value in self.implementation_risks.items():
            if not isinstance(value, RiskLevel):
                raise ValueError(f"Implementation risk '{key}' must have a valid RiskLevel value")
        
        # Validate migration complexity scores
        for key, value in self.migration_complexity.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Migration complexity score for '{key}' must be a number between 0 and 1")
        
        # Validate strategic alignment score
        if not isinstance(self.strategic_alignment_score, (int, float)) or not 0 <= self.strategic_alignment_score <= 1:
            raise ValueError("strategic_alignment_score must be a number between 0 and 1")


@dataclass
class TechnicalAssessment:
    """Comprehensive technical assessment for institutional investors"""
    # Required fields (no defaults)
    overall_score: float
    confidence_score: float
    architecture_pattern: ArchitecturePattern
    architecture_score: float
    architecture_maturity: float
    tech_stack: List[TechStackItem]
    scalability: ScalabilityAssessment
    security: SecurityAssessment
    technical_debt: TechnicalDebtAssessment
    operational: OperationalAssessment
    talent: TalentAssessment
    
    # Optional fields (with defaults)
    assessment_date: str = field(default_factory=lambda: datetime.now().isoformat())
    dora_metrics: Optional[DORAMetrics] = None
    competitive_positioning: Optional[CompetitivePositioning] = None
    technical_roadmap: Optional[TechnicalRoadmap] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    critical_recommendations: List[str] = field(default_factory=list)
    high_priority_recommendations: List[str] = field(default_factory=list)
    medium_priority_recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    investment_considerations: List[Dict[str, Any]] = field(default_factory=list)
    assessment_methodology: str = "EnterpriseGrade TDD v3.0"
    assessor_information: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate assessment scores"""
        for field_name, field_value in self.__dict__.items():
            if field_name.endswith('_score') and isinstance(field_value, (int, float)):
                if not 0 <= field_value <= 1:
                    raise ValueError(f"{field_name} must be between 0 and 1")
            elif field_name == 'uptime_percentage' and isinstance(field_value, (int, float)):
                if not 0 <= field_value <= 100:
                    raise ValueError("uptime_percentage must be between 0 and 100")
            elif field_name == 'senior_engineers_ratio' and isinstance(field_value, (int, float)):
                if not 0 <= field_value <= 1:
                    raise ValueError("senior_engineers_ratio must be between 0 and 1")
            elif field_name == 'team_size' and isinstance(field_value, int):
                if field_value < 0:
                    raise ValueError("team_size cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper enum handling"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], TechStackItem):
                result[key] = [item.to_dict() for item in value]
            elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# -----------------------------------------------------------------------------
# Knowledge Base
# -----------------------------------------------------------------------------

class TechnologyKnowledgeBase:
    """
    Comprehensive technology knowledge base with industry-standard assessments
    
    Features:
    - Technology stack assessment based on industry data
    - Maturity models for technologies
    - Risk profiles for technology choices
    - Integration with external data sources (simulated)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the knowledge base
        
        Args:
            data_path: Path to load technology data from
        """
        self.logger = logging.getLogger("tech_due_diligence.knowledge_base")
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), "data")
        
        # Initialize knowledge base
        self.tech_stack_db = self._load_tech_stack_database()
        self.vulnerability_db = self._load_vulnerability_database()
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.tech_trends = self._load_technology_trends()
        
        self.logger.info(f"Initialized technology knowledge base with "
                        f"{sum(len(v) for v in self.tech_stack_db.values())} technology entries")
    
    def _load_tech_stack_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load the technology stack database with comprehensive assessments"""
        # In a production system, this would load from a database or API
        # Here we simulate with an enhanced in-memory database
        
        # Base categories from existing code
        tech_db = {
            str(TechCategory.LANGUAGE.value): [],
            str(TechCategory.FRAMEWORK.value): [],
            str(TechCategory.DATABASE.value): [], 
            str(TechCategory.FRONTEND.value): [],
            str(TechCategory.BACKEND.value): [],
            str(TechCategory.INFRASTRUCTURE.value): [],
            str(TechCategory.CLOUD_SERVICE.value): [],
            str(TechCategory.DEVOPS.value): [],
            str(TechCategory.SECURITY.value): [],
            str(TechCategory.MESSAGING.value): [],
            str(TechCategory.CACHE.value): [],
            str(TechCategory.MOBILE.value): []
        }
        
        # Enhanced language entries with comprehensive metrics
        tech_db[str(TechCategory.LANGUAGE.value)] = [
            {
                "name": "Java",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.85,
                "scalability": 0.75,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.3,
                "community_support": 0.9,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.7
            },
            {
                "name": "Python",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.9,
                "scalability": 0.65,
                "reliability": 0.8,
                "security_score": 0.75,
                "expertise_required": 0.5,
                "vendor_lock_in": 0.2,
                "community_support": 0.95,
                "release_frequency": "Yearly major releases",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.5
            },
            {
                "name": "JavaScript",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.95,
                "scalability": 0.7,
                "reliability": 0.75,
                "security_score": 0.65,
                "expertise_required": 0.5,
                "vendor_lock_in": 0.2,
                "community_support": 0.95,
                "release_frequency": "Annual ECMAScript updates",
                "cost_efficiency": 0.9,
                "operational_complexity": 0.6
            },
            {
                "name": "TypeScript",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.8,
                "scalability": 0.75,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.4,
                "community_support": 0.85,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.6
            },
            {
                "name": "Go",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.75,
                "market_adoption": 0.7,
                "scalability": 0.9,
                "reliability": 0.85,
                "security_score": 0.85,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.3,
                "community_support": 0.8,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.5
            },
            {
                "name": "Rust",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.65,
                "market_adoption": 0.5,
                "scalability": 0.85,
                "reliability": 0.9,
                "security_score": 0.95,
                "expertise_required": 0.85,
                "vendor_lock_in": 0.2,
                "community_support": 0.85,
                "release_frequency": "Every 6 weeks",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.7
            },
            {
                "name": "C#",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.75,
                "scalability": 0.8,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.6,
                "community_support": 0.8,
                "release_frequency": "Yearly",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.6
            },
            {
                "name": "PHP",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.7,
                "scalability": 0.6,
                "reliability": 0.7,
                "security_score": 0.6,
                "expertise_required": 0.5,
                "vendor_lock_in": 0.2,
                "community_support": 0.7,
                "release_frequency": "Yearly",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.5
            },
            {
                "name": "Ruby",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.75,
                "market_adoption": 0.6,
                "scalability": 0.6,
                "reliability": 0.75,
                "security_score": 0.7,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.2,
                "community_support": 0.7,
                "release_frequency": "Yearly",
                "cost_efficiency": 0.75,
                "operational_complexity": 0.6
            },
            {
                "name": "Swift",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.7,
                "market_adoption": 0.6,
                "scalability": 0.7,
                "reliability": 0.8,
                "security_score": 0.8,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.6,
                "community_support": 0.75,
                "release_frequency": "Yearly",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.65
            }
        ]
        
        # Enhanced database entries
        tech_db[str(TechCategory.DATABASE.value)] = [
            {
                "name": "PostgreSQL",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.85,
                "scalability": 0.8,
                "reliability": 0.9,
                "security_score": 0.85,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.2,
                "community_support": 0.9,
                "release_frequency": "Yearly",
                "cost_efficiency": 0.9,
                "operational_complexity": 0.7
            },
            {
                "name": "MySQL",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.9,
                "scalability": 0.75,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.3,
                "community_support": 0.85,
                "release_frequency": "Every 2-3 years",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.65
            },
            {
                "name": "MongoDB",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.8,
                "scalability": 0.85,
                "reliability": 0.8,
                "security_score": 0.75,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.4,
                "community_support": 0.8,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.75,
                "operational_complexity": 0.7
            },
            {
                "name": "Redis",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.8,
                "scalability": 0.85,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.2,
                "community_support": 0.85,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.6
            },
            {
                "name": "DynamoDB",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.75,
                "market_adoption": 0.65,
                "scalability": 0.95,
                "reliability": 0.9,
                "security_score": 0.85,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.9,
                "community_support": 0.7,
                "release_frequency": "Continuous updates by AWS",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.6
            },
            {
                "name": "Cassandra",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.6,
                "scalability": 0.95,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.8,
                "vendor_lock_in": 0.3,
                "community_support": 0.75,
                "release_frequency": "Every 6-12 months",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.85
            },
            {
                "name": "Elasticsearch",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.75,
                "scalability": 0.85,
                "reliability": 0.8,
                "security_score": 0.75,
                "expertise_required": 0.75,
                "vendor_lock_in": 0.5,
                "community_support": 0.8,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.8
            }
        ]
        
        # Enhanced frontend entries
        tech_db[str(TechCategory.FRONTEND.value)] = [
            {
                "name": "React",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.9,
                "scalability": 0.85,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.4,
                "community_support": 0.95,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.7
            },
            {
                "name": "Angular",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.8,
                "scalability": 0.8,
                "reliability": 0.85,
                "security_score": 0.85,
                "expertise_required": 0.75,
                "vendor_lock_in": 0.5,
                "community_support": 0.85,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.75,
                "operational_complexity": 0.8
            },
            {
                "name": "Vue",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.8,
                "market_adoption": 0.75,
                "scalability": 0.8,
                "reliability": 0.8,
                "security_score": 0.8,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.3,
                "community_support": 0.85,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.6
            },
            {
                "name": "Svelte",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.7,
                "market_adoption": 0.5,
                "scalability": 0.8,
                "reliability": 0.8,
                "security_score": 0.75,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.2,
                "community_support": 0.75,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.9,
                "operational_complexity": 0.5
            }
        ]
        
        # Enhanced backend entries
        tech_db[str(TechCategory.BACKEND.value)] = [
            {
                "name": "Node.js",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.85,
                "scalability": 0.8,
                "reliability": 0.8,
                "security_score": 0.75,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.2,
                "community_support": 0.9,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.6
            },
            {
                "name": "Django",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.7,
                "scalability": 0.75,
                "reliability": 0.85,
                "security_score": 0.85,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.3,
                "community_support": 0.8,
                "release_frequency": "Every year",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.6
            },
            {
                "name": "Spring Boot",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.8,
                "scalability": 0.85,
                "reliability": 0.9,
                "security_score": 0.85,
                "expertise_required": 0.8,
                "vendor_lock_in": 0.4,
                "community_support": 0.85,
                "release_frequency": "Every 6 months",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.75
            },
            {
                "name": "Flask",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.75,
                "scalability": 0.7,
                "reliability": 0.75,
                "security_score": 0.7,
                "expertise_required": 0.5,
                "vendor_lock_in": 0.2,
                "community_support": 0.8,
                "release_frequency": "Variable",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.5
            },
            {
                "name": "FastAPI",
                "maturity_level": TechMaturityLevel.GROWING.value,
                "maturity_score": 0.7,
                "market_adoption": 0.6,
                "scalability": 0.8,
                "reliability": 0.8,
                "security_score": 0.8,
                "expertise_required": 0.6,
                "vendor_lock_in": 0.2,
                "community_support": 0.8,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.85,
                "operational_complexity": 0.5
            }
        ]
        
        # Enhanced infrastructure entries
        tech_db[str(TechCategory.INFRASTRUCTURE.value)] = [
            {
                "name": "Kubernetes",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.8,
                "scalability": 0.95,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.85,
                "vendor_lock_in": 0.3,
                "community_support": 0.9,
                "release_frequency": "Every 3 months",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.9
            },
            {
                "name": "Docker",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.9,
                "scalability": 0.85,
                "reliability": 0.85,
                "security_score": 0.75,
                "expertise_required": 0.7,
                "vendor_lock_in": 0.3,
                "community_support": 0.9,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.7
            },
            {
                "name": "AWS",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.95,
                "market_adoption": 0.95,
                "scalability": 0.95,
                "reliability": 0.9,
                "security_score": 0.85,
                "expertise_required": 0.8,
                "vendor_lock_in": 0.9,
                "community_support": 0.9,
                "release_frequency": "Continuous",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.8
            },
            {
                "name": "Azure",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.9,
                "market_adoption": 0.85,
                "scalability": 0.9,
                "reliability": 0.85,
                "security_score": 0.85,
                "expertise_required": 0.75,
                "vendor_lock_in": 0.9,
                "community_support": 0.85,
                "release_frequency": "Continuous",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.8
            },
            {
                "name": "Google Cloud Platform",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.85,
                "market_adoption": 0.75,
                "scalability": 0.95,
                "reliability": 0.9,
                "security_score": 0.85,
                "expertise_required": 0.75,
                "vendor_lock_in": 0.9,
                "community_support": 0.8,
                "release_frequency": "Continuous",
                "cost_efficiency": 0.7,
                "operational_complexity": 0.75
            },
            {
                "name": "Terraform",
                "maturity_level": TechMaturityLevel.MATURE.value,
                "maturity_score": 0.8,
                "market_adoption": 0.75,
                "scalability": 0.85,
                "reliability": 0.85,
                "security_score": 0.8,
                "expertise_required": 0.75,
                "vendor_lock_in": 0.4,
                "community_support": 0.85,
                "release_frequency": "Every few months",
                "cost_efficiency": 0.8,
                "operational_complexity": 0.7
            }
        ]
        
        # Add more categories and technologies as needed
        
        return tech_db
    
    def _load_vulnerability_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load known vulnerabilities database"""
        # In a production system, this would integrate with CVE databases
        # Here we simulate with a representative sample
        vuln_db = {
            "Java": [
                {"id": "CVE-2023-25193", "severity": "High", "description": "Remote code execution vulnerability in Java JDK"}
            ],
            "Log4j": [
                {"id": "CVE-2021-44228", "severity": "Critical", "description": "Log4j JNDI features vulnerable to RCE"}
            ],
            "Spring": [
                {"id": "CVE-2022-22965", "severity": "Critical", "description": "Spring Framework RCE vulnerability"}
            ],
            "React": [
                {"id": "CVE-2018-6341", "severity": "Medium", "description": "XSS vulnerability in React DOM"}
            ],
            "MongoDB": [
                {"id": "CVE-2019-2386", "severity": "Medium", "description": "MongoDB vulnerability allowing privilege escalation"}
            ],
            "Docker": [
                {"id": "CVE-2021-41091", "severity": "High", "description": "Incorrect permission handling in Docker"}
            ],
            "Kubernetes": [
                {"id": "CVE-2020-8554", "severity": "Medium", "description": "Man-in-the-middle vulnerability in Kubernetes"}
            ]
        }
        return vuln_db
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load industry benchmarks for comparative assessment"""
        # In a production system, this would be loaded from a database with real benchmarks
        # Here we simulate with reasonable estimates
        benchmarks = {
            "fintech": {
                "tech_stack_maturity": 0.85,
                "security_requirements": 0.9,
                "scalability_requirements": 0.85,
                "reliability_requirements": 0.95,
                "common_technologies": ["Java", "Spring", "React", "PostgreSQL", "Kubernetes", "AWS"],
                "architecture_patterns": ["Microservices", "Event-Driven"],
                "regulatory_requirements": ["PCI-DSS", "SOC2", "GDPR"],
                "testing_requirements": 0.9
            },
            "ecommerce": {
                "tech_stack_maturity": 0.8,
                "security_requirements": 0.85,
                "scalability_requirements": 0.9,
                "reliability_requirements": 0.9,
                "common_technologies": ["Java", "Python", "React", "MySQL", "Redis", "AWS"],
                "architecture_patterns": ["Microservices", "CQRS"],
                "regulatory_requirements": ["PCI-DSS", "GDPR"],
                "testing_requirements": 0.85
            },
            "healthcare": {
                "tech_stack_maturity": 0.75,
                "security_requirements": 0.95,
                "scalability_requirements": 0.8,
                "reliability_requirements": 0.95,
                "common_technologies": ["Java", ".NET", "Angular", "Oracle", "Azure"],
                "architecture_patterns": ["SOA", "Layered"],
                "regulatory_requirements": ["HIPAA", "GDPR", "SOC2"],
                "testing_requirements": 0.95
            },
            "saas": {
                "tech_stack_maturity": 0.85,
                "security_requirements": 0.85,
                "scalability_requirements": 0.85,
                "reliability_requirements": 0.9,
                "common_technologies": ["Python", "JavaScript", "TypeScript", "React", "MongoDB", "AWS", "Kubernetes"],
                "architecture_patterns": ["Microservices", "Serverless"],
                "regulatory_requirements": ["SOC2", "GDPR"],
                "testing_requirements": 0.85
            },
            "ai": {
                "tech_stack_maturity": 0.8,
                "security_requirements": 0.8,
                "scalability_requirements": 0.85,
                "reliability_requirements": 0.85,
                "common_technologies": ["Python", "TensorFlow", "PyTorch", "FastAPI", "Docker", "Kubernetes", "AWS"],
                "architecture_patterns": ["Microservices", "Event-Driven"],
                "regulatory_requirements": ["GDPR", "Ethical AI Guidelines"],
                "testing_requirements": 0.8
            }
        }
        return benchmarks
    
    def _load_technology_trends(self) -> Dict[str, Dict[str, Any]]:
        """Load current technology trends for future-proofing assessment"""
        # In a production system, this would be updated regularly from market research
        trends = {
            "rising_technologies": [
                {"name": "WebAssembly", "adoption_rate": 0.4, "maturity": 0.6, "relevance_by_sector": {"fintech": 0.6, "ecommerce": 0.5, "saas": 0.7}},
                {"name": "GraphQL", "adoption_rate": 0.6, "maturity": 0.7, "relevance_by_sector": {"fintech": 0.7, "ecommerce": 0.8, "saas": 0.8}},
                {"name": "Rust", "adoption_rate": 0.5, "maturity": 0.7, "relevance_by_sector": {"fintech": 0.7, "ecommerce": 0.5, "saas": 0.6}},
                {"name": "Deno", "adoption_rate": 0.3, "maturity": 0.5, "relevance_by_sector": {"fintech": 0.4, "ecommerce": 0.5, "saas": 0.6}},
                {"name": "Svelte", "adoption_rate": 0.4, "maturity": 0.6, "relevance_by_sector": {"fintech": 0.5, "ecommerce": 0.6, "saas": 0.7}}
            ],
            "declining_technologies": [
                {"name": "jQuery", "adoption_rate": 0.5, "maturity": 0.9, "relevance_by_sector": {"fintech": 0.3, "ecommerce": 0.4, "saas": 0.2}},
                {"name": "PHP", "adoption_rate": 0.5, "maturity": 0.8, "relevance_by_sector": {"fintech": 0.3, "ecommerce": 0.5, "saas": 0.3}},
                {"name": "Angular.js", "adoption_rate": 0.2, "maturity": 0.8, "relevance_by_sector": {"fintech": 0.2, "ecommerce": 0.2, "saas": 0.1}},
                {"name": "SOAP", "adoption_rate": 0.3, "maturity": 0.9, "relevance_by_sector": {"fintech": 0.4, "ecommerce": 0.2, "saas": 0.2}}
            ],
            "emerging_architectures": [
                {"name": "Micro-Frontends", "adoption_rate": 0.5, "maturity": 0.6, "relevance_by_sector": {"fintech": 0.7, "ecommerce": 0.8, "saas": 0.7}},
                {"name": "Serverless", "adoption_rate": 0.7, "maturity": 0.7, "relevance_by_sector": {"fintech": 0.7, "ecommerce": 0.7, "saas": 0.8}},
                {"name": "Edge Computing", "adoption_rate": 0.4, "maturity": 0.5, "relevance_by_sector": {"fintech": 0.6, "ecommerce": 0.7, "saas": 0.6}}
            ]
        }
        return trends
    
    @lru_cache(maxsize=512)
    def get_tech_profile(self, tech_name: str) -> Dict[str, Any]:
        """
        Get the comprehensive profile for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Dictionary with technology profile information
        """
        # Normalize the name for lookup
        tech_name_norm = tech_name.lower()
        
        # Search in all categories
        for category, technologies in self.tech_stack_db.items():
            for tech in technologies:
                if tech["name"].lower() == tech_name_norm:
                    # Create a copy of the tech data
                    tech_profile = tech.copy()
                    tech_profile["category"] = category
                    
                    # Add vulnerability information if available
                    if tech["name"] in self.vulnerability_db:
                        tech_profile["vulnerabilities"] = self.vulnerability_db[tech["name"]]
                    
                    return tech_profile
        
        # If not found, provide a default profile with limited information
        return {
            "name": tech_name,
            "category": "Unknown",
            "maturity_level": TechMaturityLevel.GROWING.value,
            "maturity_score": 0.5,
            "market_adoption": 0.5,
            "scalability": 0.5,
            "security_score": 0.5,
            "expertise_required": 0.5,
            "community_support": 0.5
        }
    
    def categorize_tech(self, tech_name: str) -> TechCategory:
        """
        Categorize a technology based on knowledge base
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            TechCategory representing the type of technology
        """
        name_lower = tech_name.lower()
        
        # Check direct matches in the database
        for category_str, technologies in self.tech_stack_db.items():
            for tech in technologies:
                if tech["name"].lower() == name_lower:
                    return TechCategory(category_str)
        
        # Pattern-based categorization
        if any(db.lower() in name_lower for db in ['sql', 'db', 'database', 'mongo', 'postgres', 'mysql', 'oracle', 'redis']):
            return TechCategory.DATABASE
        elif any(lang.lower() in name_lower for lang in ['java', 'python', 'ruby', 'php', 'go', 'rust', 'c#', '.net', 'typescript', 'javascript', 'ts']):
            return TechCategory.LANGUAGE
        elif any(fr.lower() in name_lower for fr in ['react', 'vue', 'angular', 'svelte', 'ember']):
            return TechCategory.FRONTEND
        elif any(bk.lower() in name_lower for bk in ['node', 'express', 'django', 'flask', 'rails', 'spring']):
            return TechCategory.BACKEND
        elif any(infra.lower() in name_lower for infra in ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker']):
            return TechCategory.INFRASTRUCTURE
        elif any(devops.lower() in name_lower for devops in ['jenkins', 'gitlab', 'github', 'circleci', 'terraform', 'ansible']):
            return TechCategory.DEVOPS
        elif any(msg.lower() in name_lower for msg in ['kafka', 'rabbitmq', 'activemq', 'sqs', 'pubsub']):
            return TechCategory.MESSAGING
        elif any(mobile.lower() in name_lower for mobile in ['ios', 'android', 'flutter', 'react native', 'cordova']):
            return TechCategory.MOBILE
        
        # Default category
        return TechCategory.OTHER
    
    def get_tech_maturity(self, tech_name: str) -> Tuple[TechMaturityLevel, float]:
        """
        Get the maturity level and score for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Tuple of (TechMaturityLevel, maturity_score)
        """
        tech_profile = self.get_tech_profile(tech_name)
        
        maturity_level = tech_profile.get("maturity_level", TechMaturityLevel.GROWING.value)
        if isinstance(maturity_level, str):
            maturity_level = TechMaturityLevel(maturity_level)
        
        maturity_score = tech_profile.get("maturity_score", 0.5)
        
        return maturity_level, maturity_score
    
    def get_tech_scalability(self, tech_name: str) -> float:
        """
        Get the scalability score for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Scalability score (0-1 scale)
        """
        tech_profile = self.get_tech_profile(tech_name)
        return tech_profile.get("scalability", 0.5)
    
    def get_tech_market_adoption(self, tech_name: str) -> float:
        """
        Get the market adoption score for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Market adoption score (0-1 scale)
        """
        tech_profile = self.get_tech_profile(tech_name)
        return tech_profile.get("market_adoption", 0.5)
    
    def get_tech_expertise_required(self, tech_name: str) -> float:
        """
        Get the expertise level required for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Expertise required score (0-1 scale)
        """
        tech_profile = self.get_tech_profile(tech_name)
        return tech_profile.get("expertise_required", 0.5)
    
    def get_tech_security_profile(self, tech_name: str) -> Dict[str, Any]:
        """
        Get the security profile for a technology
        
        Args:
            tech_name: Name of the technology
            
        Returns:
            Security profile with vulnerabilities and security score
        """
        tech_profile = self.get_tech_profile(tech_name)
        
        security_profile = {
            "security_score": tech_profile.get("security_score", 0.5),
            "vulnerabilities": tech_profile.get("vulnerabilities", [])
        }
        
        return security_profile
    
    def get_benchmark_data(self, sector: str) -> Dict[str, Any]:
        """
        Get benchmark data for a specific sector
        
        Args:
            sector: Industry sector
            
        Returns:
            Benchmark data for the sector
        """
        # Normalize sector name
        sector_norm = sector.lower()
        
        # Find closest match
        if sector_norm in self.industry_benchmarks:
            return self.industry_benchmarks[sector_norm]
        
        # Try to find a partial match
        for benchmark_sector in self.industry_benchmarks:
            if benchmark_sector in sector_norm or sector_norm in benchmark_sector:
                return self.industry_benchmarks[benchmark_sector]
        
        # Default to SaaS benchmarks if no match
        return self.industry_benchmarks.get("saas", {})
    
    def get_tech_trend_analysis(self, tech_stack: List[str], sector: str) -> Dict[str, Any]:
        """
        Analyze a technology stack against current trends
        
        Args:
            tech_stack: List of technology names
            sector: Industry sector
            
        Returns:
            Analysis of the tech stack against current trends
        """
        sector_norm = sector.lower()
        tech_stack_lower = [t.lower() for t in tech_stack]
        
        # Initialize results
        analysis = {
            "rising_tech_alignment": 0.0,
            "legacy_tech_risk": 0.0,
            "architecture_alignment": 0.0,
            "rising_technologies_present": [],
            "declining_technologies_present": [],
            "recommended_technologies": [],
            "tech_refresh_candidates": []
        }
        
        # Check for rising technologies
        for tech in self.tech_trends["rising_technologies"]:
            if any(tech["name"].lower() in t for t in tech_stack_lower):
                analysis["rising_technologies_present"].append(tech["name"])
                if sector_norm in tech.get("relevance_by_sector", {}):
                    analysis["rising_tech_alignment"] += tech["relevance_by_sector"][sector_norm]
        
        # Normalize rising tech alignment
        if analysis["rising_technologies_present"]:
            analysis["rising_tech_alignment"] /= len(analysis["rising_technologies_present"])
        
        # Check for declining technologies
        for tech in self.tech_trends["declining_technologies"]:
            if any(tech["name"].lower() in t for t in tech_stack_lower):
                analysis["declining_technologies_present"].append(tech["name"])
                if sector_norm in tech.get("relevance_by_sector", {}):
                    analysis["legacy_tech_risk"] += 1 - tech["relevance_by_sector"][sector_norm]
        
        # Normalize legacy tech risk
        if analysis["declining_technologies_present"]:
            analysis["legacy_tech_risk"] /= len(analysis["declining_technologies_present"])
        
        # Recommend rising technologies not present
        sector_relevant_rising = [
            tech for tech in self.tech_trends["rising_technologies"] 
            if sector_norm in tech.get("relevance_by_sector", {}) and 
            tech["relevance_by_sector"][sector_norm] > 0.6 and
            not any(tech["name"].lower() in t for t in tech_stack_lower)
        ]
        
        # Sort by relevance and adoption rate
        sector_relevant_rising.sort(
            key=lambda x: (x["relevance_by_sector"][sector_norm], x["adoption_rate"]), 
            reverse=True
        )
        
        analysis["recommended_technologies"] = [tech["name"] for tech in sector_relevant_rising[:3]]
        
        # Identify technologies to consider replacing
        if analysis["declining_technologies_present"]:
            for declining_tech in analysis["declining_technologies_present"]:
                # Find a suitable rising technology in the same category
                declining_category = self.categorize_tech(declining_tech)
                
                replacement_candidates = [
                    tech for tech in sector_relevant_rising 
                    if self.categorize_tech(tech["name"]) == declining_category
                ]
                
                if replacement_candidates:
                    # Sort by relevance and adoption
                    replacement_candidates.sort(
                        key=lambda x: (x["relevance_by_sector"][sector_norm], x["adoption_rate"]), 
                        reverse=True
                    )
                    
                    analysis["tech_refresh_candidates"].append({
                        "current": declining_tech,
                        "recommended": replacement_candidates[0]["name"]
                    })
        
        return analysis


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------

def generate_assessment_report(assessment: TechnicalAssessment, format: str = "markdown") -> str:
    """
    Generate a comprehensive technical assessment report in markdown format
    
    Args:
        assessment: TechnicalAssessment object
        format: Output format (currently only supports 'markdown')
        
    Returns:
        Report as a string in the specified format
    """
    if format != "markdown":
        raise ValueError(f"Unsupported report format: {format}")
    
    report = []
    
    # Title and summary
    report.append(f"# Technical Due Diligence Assessment Report")
    report.append(f"## Executive Summary")
    report.append(f"**Overall Technical Score**: {assessment.overall_score:.2f}/1.0")
    report.append(f"**Assessment Date**: {assessment.assessment_date}")
    report.append(f"**Architecture Pattern**: {assessment.architecture_pattern.value}")
    report.append(f"**Confidence Level**: {assessment.confidence_score:.2f}/1.0")
    report.append("")
    
    # Top strengths and weaknesses
    report.append("### Key Strengths")
    for strength in assessment.strengths[:5]:
        report.append(f"- {strength}")
    report.append("")
    
    report.append("### Key Weaknesses")
    for weakness in assessment.weaknesses[:5]:
        report.append(f"- {weakness}")
    report.append("")
    
    # Risk assessment
    if assessment.risk_assessment and "overall_risk" in assessment.risk_assessment:
        overall_risk = assessment.risk_assessment["overall_risk"]
        report.append("### Risk Assessment")
        report.append(f"**Overall Risk Level**: {overall_risk.get('level').value}")
        report.append(f"**Risk Summary**: {overall_risk.get('summary')}")
        report.append("")
        
        report.append("**Top Risks:**")
        for risk in overall_risk.get("top_risks", []):
            factors = ", ".join(risk.get("factors", []))
            report.append(f"- **{risk.get('area')}** ({risk.get('level').value}): {factors}")
        report.append("")
    
    # Critical recommendations
    report.append("### Critical Recommendations")
    for rec in assessment.critical_recommendations:
        report.append(f"- {rec}")
    report.append("")
    
    # Investment considerations
    if assessment.investment_considerations:
        report.append("### Investment Considerations")
        for consideration in assessment.investment_considerations:
            report.append(f"- **{consideration.get('area')}**: {consideration.get('consideration')}")
            report.append(f"  - Impact: {consideration.get('impact')}")
            report.append(f"  - Timeframe: {consideration.get('timeframe')}")
            report.append(f"  - Estimated Investment: ${consideration.get('investment_required'):,.2f}")
        report.append("")
    
    # Detailed assessments
    report.append("## Detailed Assessment")
    
    # Architecture
    report.append("### Architecture Assessment")
    report.append(f"**Score**: {assessment.architecture_score:.2f}/1.0")
    report.append(f"**Pattern**: {assessment.architecture_pattern.value}")
    report.append(f"**Maturity**: {assessment.architecture_maturity:.2f}/1.0")
    report.append("")
    
    # Technology stack
    report.append("### Technology Stack Assessment")
    report.append("| Technology | Category | Maturity | Market Adoption | Scalability | Security |")
    report.append("|------------|----------|----------|-----------------|------------|----------|")
    for item in assessment.tech_stack:
        report.append(f"| {item.name} | {item.category.value} | {item.maturity_score:.2f} | {item.market_adoption:.2f} | {item.scalability:.2f} | {item.security_score:.2f} |")
    report.append("")
    
    # Scalability
    report.append("### Scalability Assessment")
    report.append(f"**Overall Score**: {assessment.scalability.overall_score:.2f}/1.0")
    report.append(f"**Infrastructure Scalability**: {assessment.scalability.infrastructure_scalability:.2f}/1.0")
    report.append(f"**Database Scalability**: {assessment.scalability.database_scalability:.2f}/1.0")
    
    if assessment.scalability.bottlenecks:
        report.append("**Bottlenecks:**")
        for bottleneck in assessment.scalability.bottlenecks:
            report.append(f"- {bottleneck}")
    report.append("")
    
    # Security
    report.append("### Security Assessment")
    report.append(f"**Overall Score**: {assessment.security.overall_score:.2f}/1.0")
    report.append(f"**Critical Vulnerabilities**: {assessment.security.critical_vulnerabilities_count}")
    report.append(f"**High Vulnerabilities**: {assessment.security.high_vulnerabilities_count}")
    
    if assessment.security.critical_recommendations:
        report.append("**Critical Security Recommendations:**")
        for rec in assessment.security.critical_recommendations:
            report.append(f"- {rec}")
    report.append("")
    
    # Technical debt
    report.append("### Technical Debt Assessment")
    report.append(f"**Overall Score**: {assessment.technical_debt.overall_score:.2f}/1.0")
    
    if assessment.technical_debt.critical_refactoring_areas:
        report.append("**Critical Refactoring Areas:**")
        for area in assessment.technical_debt.critical_refactoring_areas:
            report.append(f"- {area}")
    report.append("")
    
    # Operational
    report.append("### Operational Assessment")
    report.append(f"**Overall Score**: {assessment.operational.overall_score:.2f}/1.0")
    report.append(f"**Reliability Score**: {assessment.operational.reliability_score:.2f}/1.0")
    report.append(f"**Uptime Percentage**: {assessment.operational.uptime_percentage:.2f}%")
    report.append("")
    
    # Talent
    report.append("### Talent Assessment")
    report.append(f"**Overall Score**: {assessment.talent.overall_score:.2f}/1.0")
    report.append(f"**Team Size**: {assessment.talent.team_size}")
    report.append(f"**Senior Engineers Ratio**: {assessment.talent.senior_engineers_ratio:.2f}")
    report.append(f"**Key Person Dependencies**: {assessment.talent.key_person_dependencies}")
    report.append("")
    
    # Technical roadmap
    if assessment.technical_roadmap:
        report.append("## Technical Roadmap")
        
        report.append("### Short-term (0-6 months)")
        for initiative in assessment.technical_roadmap.horizon_1_initiatives:
            report.append(f"- **{initiative.get('name')}** - Priority: {initiative.get('priority')}, Effort: {initiative.get('estimated_effort')}")
        report.append("")
        
        report.append("### Medium-term (6-18 months)")
        for initiative in assessment.technical_roadmap.horizon_2_initiatives:
            report.append(f"- **{initiative.get('name')}** - Priority: {initiative.get('priority')}, Effort: {initiative.get('estimated_effort')}")
        report.append("")
        
        report.append("### Long-term (18+ months)")
        for initiative in assessment.technical_roadmap.horizon_3_initiatives:
            report.append(f"- **{initiative.get('name')}** - Priority: {initiative.get('priority')}, Effort: {initiative.get('estimated_effort')}")
        report.append("")
    
    # DORA metrics if available
    if assessment.dora_metrics:
        report.append("## DORA Metrics")
        report.append(f"**Performance Category**: {assessment.dora_metrics.performance_category}")
        report.append(f"**Deployment Frequency**: {assessment.dora_metrics.deployment_frequency}")
        report.append(f"**Lead Time for Changes**: {assessment.dora_metrics.lead_time_for_changes}")
        report.append(f"**Time to Restore Service**: {assessment.dora_metrics.time_to_restore_service}")
        report.append(f"**Change Failure Rate**: {assessment.dora_metrics.change_failure_rate*100:.1f}%")
        report.append("")
    
    # Footer
    report.append("---")
    report.append(f"*This report was generated using {assessment.assessment_methodology} on {assessment.assessment_date}.*")
    
    return "\n".join(report)


# -----------------------------------------------------------------------------
# Analyzer Implementation
# -----------------------------------------------------------------------------

class EnterpriseGradeTechnicalDueDiligence:
    """
    Enterprise-grade technical due diligence system for high-stakes investment decisions
    
    Features:
    - Comprehensive technical assessment based on industry standards
    - Risk evaluation using advanced risk models
    - Integration with external data sources
    - Scenario planning and technical roadmapping
    - Investment-grade reporting
    - Competitive benchmarking and analysis
    - Talent and organizational assessment
    - Security and compliance evaluation
    
    This system is designed for institutional investors, private equity firms,
    and corporate development teams conducting technical due diligence on
    technology companies for acquisition, investment, or partnership purposes.
    """
    
    # Version information
    VERSION = "3.0.0"
    BUILD_DATE = "2025-03-30"
    
    def __init__(self, enable_external_integrations: bool = False, data_path: Optional[str] = None):
        """
        Initialize the technical due diligence system
        
        Args:
            enable_external_integrations: Whether to enable integration with external APIs
        """
        self.logger = logging.getLogger("tech_due_diligence")
        
        # Initialize components
        self.knowledge_base = TechnologyKnowledgeBase(data_path)
        self.enable_external_integrations = enable_external_integrations
        
        # Performance monitoring
        self.assessment_timings = {}
        self.assessments_count = 0
        self.last_execution_time = 0
        
        # Cache for performance optimization
        self._cache = {}
        
        self.logger.info(f"Initialized EnterpriseGradeTechnicalDueDiligence system v{self.VERSION}")
    
    def assess_technical_architecture(self, tech_data: Dict[str, Any], generate_report: bool = False) -> Union[TechnicalAssessment, Tuple[TechnicalAssessment, str]]:
        """
        Perform comprehensive technical assessment
        
        Args:
            tech_data: Dictionary containing technical data
            generate_report: Whether to generate a markdown report
            
        Returns:
            TechnicalAssessment object with detailed assessment results,
            or a tuple of (TechnicalAssessment, report_str) if generate_report is True
        """
        assessment_start_time = time.time()
        
        # Extract and analyze the tech stack
        stack = self._extract_tech_stack(tech_data)
        
        # Determine architecture pattern
        arch_pattern = self._determine_architecture_pattern(tech_data)
        
        # Perform detailed assessments
        arch_score, arch_maturity = self._assess_architecture(tech_data, stack, arch_pattern)
        scalability = self._assess_scalability(tech_data, stack, arch_pattern)
        security = self._assess_security(tech_data, stack)
        tech_debt = self._assess_technical_debt(tech_data, stack)
        operational = self._assess_operational_capabilities(tech_data, stack)
        talent = self._assess_talent_requirements(tech_data, stack)
        
        # Calculate DORA metrics if data is available
        dora_metrics = self._calculate_dora_metrics(tech_data) if self._has_dora_data(tech_data) else None
        
        # Perform competitive analysis
        competitive = self._assess_competitive_positioning(tech_data, stack)
        
        # Generate technical roadmap
        roadmap = self._generate_technical_roadmap(tech_data, stack, tech_debt, scalability, security)
        
        # Identify SWOT
        strengths = self._identify_strengths(tech_data, stack, arch_score, scalability.overall_score, tech_debt.overall_score)
        weaknesses = self._identify_weaknesses(tech_data, stack, arch_score, scalability.overall_score, tech_debt.overall_score)
        opportunities = self._identify_opportunities(tech_data, stack, scalability, security)
        threats = self._identify_threats(tech_data, stack, security, tech_debt, operational)
        
        # Generate recommendations
        critical_recs = self._generate_critical_recommendations(tech_data, stack, weaknesses, security, scalability)
        high_recs = self._generate_high_recommendations(tech_data, stack, weaknesses, tech_debt)
        medium_recs = self._generate_medium_recommendations(tech_data, stack, weaknesses, operational)
        
        # Calculate comprehensive risk assessment
        risk_assessment = self._assess_risks(tech_data, stack, security, scalability, tech_debt, operational)
        
        # Calculate investment considerations
        investment_considerations = self._generate_investment_considerations(
            tech_data, stack, arch_score, scalability, security, tech_debt, operational, risk_assessment
        )
        
        # Calculate overall score with weighted components
        weights = {
            "architecture": 0.2,
            "scalability": 0.2,
            "security": 0.2,
            "tech_debt": 0.15,
            "operational": 0.15,
            "talent": 0.1
        }
        
        overall_score = (
            arch_score * weights["architecture"] +
            scalability.overall_score * weights["scalability"] +
            security.overall_score * weights["security"] +
            tech_debt.overall_score * weights["tech_debt"] +
            operational.overall_score * weights["operational"] +
            talent.overall_score * weights["talent"]
        )
        
        # Calculate confidence score based on data completeness
        confidence_score = self._calculate_confidence_score(tech_data)
        
        # Create the assessment object
        assessment = TechnicalAssessment(
            overall_score=overall_score,
            confidence_score=confidence_score,
            architecture_pattern=arch_pattern,
            architecture_score=arch_score,
            architecture_maturity=arch_maturity,
            tech_stack=stack,
            scalability=scalability,
            security=security,
            technical_debt=tech_debt,
            operational=operational,
            talent=talent,
            dora_metrics=dora_metrics,
            competitive_positioning=competitive,
            technical_roadmap=roadmap,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            critical_recommendations=critical_recs,
            high_priority_recommendations=high_recs,
            medium_priority_recommendations=medium_recs,
            risk_assessment=risk_assessment,
            investment_considerations=investment_considerations
        )
        
        # Record assessment time
        assessment_time = time.time() - assessment_start_time
        self.assessment_timings[tech_data.get("name", "unknown")] = assessment_time
        
        self.logger.info(f"Completed technical assessment in {assessment_time:.2f} seconds "
                         f"with overall score {overall_score:.2f}")
        
        # Update metrics
        self.last_execution_time = assessment_time
        self.assessments_count += 1
        
        # Generate report if requested
        if generate_report:
            report = generate_assessment_report(assessment)
            return assessment, report
        
        return assessment
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the due diligence system
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "version": self.VERSION,
            "build_date": self.BUILD_DATE,
            "assessments_completed": self.assessments_count,
            "average_assessment_time": sum(self.assessment_timings.values()) / max(1, len(self.assessment_timings)),
            "last_assessment_time": self.last_execution_time,
            "assessment_timings": self.assessment_timings
        }
        
        return metrics
    
    def generate_report(self, assessment: TechnicalAssessment, format: str = "markdown") -> str:
        """
        Generate a comprehensive report from an assessment
        
        Args:
            assessment: TechnicalAssessment object
            format: Output format (currently only supports 'markdown')
            
        Returns:
            Report as a string in the specified format
        """
        return generate_assessment_report(assessment, format)
    
    def _extract_tech_stack(self, td: Dict[str, Any]) -> List[TechStackItem]:
        """
        Extract and analyze the technology stack with enhanced pattern recognition
        
        Args:
            td: Technical data dictionary
            
        Returns:
            List of TechStackItem objects with comprehensive metadata
        """
        stack = []
        
        # Case 1: Explicit tech stack provided
        if 'tech_stack' in td and isinstance(td['tech_stack'], list):
            for it in td['tech_stack']:
                if isinstance(it, dict) and 'name' in it:
                    # Get technology name
                    name = it['name']
                    
                    # Get or determine category
                    category_str = it.get('category')
                    if category_str and isinstance(category_str, str):
                        try:
                            category = TechCategory(category_str)
                        except ValueError:
                            category = self.knowledge_base.categorize_tech(name)
                    else:
                        category = self.knowledge_base.categorize_tech(name)
                    
                    # Get technology profile from knowledge base
                    tech_profile = self.knowledge_base.get_tech_profile(name)
                    
                    # Determine maturity level
                    maturity_level_str = it.get('maturity_level', tech_profile.get('maturity_level'))
                    if maturity_level_str and isinstance(maturity_level_str, str):
                        try:
                            maturity_level = TechMaturityLevel(maturity_level_str)
                        except ValueError:
                            maturity_level = TechMaturityLevel.GROWING
                    else:
                        maturity_level, _ = self.knowledge_base.get_tech_maturity(name)
                    
                    # Create tech stack item with all available information
                    stack_item = TechStackItem(
                        name=name,
                        category=category,
                        version=it.get('version'),
                        maturity_level=maturity_level,
                        maturity_score=it.get('maturity', tech_profile.get('maturity_score', 0.5)),
                        market_adoption=it.get('market_adoption', tech_profile.get('market_adoption', 0.5)),
                        scalability=it.get('scalability', tech_profile.get('scalability', 0.5)),
                        reliability=it.get('reliability', tech_profile.get('reliability', 0.5)),
                        performance=it.get('performance', tech_profile.get('performance', 0.5)),
                        security_score=it.get('security_score', tech_profile.get('security_score', 0.5)),
                        expertise_required=it.get('expertise_required', tech_profile.get('expertise_required', 0.5)),
                        vendor_lock_in=it.get('vendor_lock_in', tech_profile.get('vendor_lock_in', 0.5)),
                        community_support=it.get('community_support', tech_profile.get('community_support', 0.5)),
                        cost_efficiency=it.get('cost_efficiency', tech_profile.get('cost_efficiency', 0.5)),
                        operational_complexity=it.get('operational_complexity', tech_profile.get('operational_complexity', 0.5)),
                        last_major_release=it.get('last_major_release'),
                        release_frequency=it.get('release_frequency', tech_profile.get('release_frequency')),
                        license_type=it.get('license_type'),
                        known_vulnerabilities=it.get('known_vulnerabilities', tech_profile.get('vulnerabilities', [])),
                        known_limitations=it.get('known_limitations', [])
                    )
                    
                    stack.append(stack_item)
        
        # Case 2: Tech description provided as text
        elif 'tech_description' in td and isinstance(td['tech_description'], str):
            # Enhanced pattern matching for technology extraction
            desc = td['tech_description']
            
            # Define patterns to match technologies
            patterns = [
                r'using ([A-Za-z0-9#+.\-_]+)',
                r'built with ([A-Za-z0-9#+.\-_]+)',
                r'([A-Za-z0-9#+.\-_]+) for (?:backend|frontend|database)',
                r'([A-Za-z0-9#+.\-_]+) (?:stack|framework|language|database|infrastructure)',
                r'(?:powered by|runs on) ([A-Za-z0-9#+.\-_]+)',
                r'(?:deployed on|hosted on) ([A-Za-z0-9#+.\-_]+)',
                r'(?:leverages|utilizes) ([A-Za-z0-9#+.\-_]+)'
            ]
            
            found_technologies = set()
            
            # Apply each pattern and collect matches
            for pattern in patterns:
                matches = re.finditer(pattern, desc, re.IGNORECASE)
                for match in matches:
                    tech_name = match.group(1).strip()
                    if tech_name and len(tech_name) > 1:
                        found_technologies.add(tech_name)
            
            # Create tech stack items from extracted technologies
            for tech_name in found_technologies:
                # Get technology details from knowledge base
                category = self.knowledge_base.categorize_tech(tech_name)
                maturity_level, maturity_score = self.knowledge_base.get_tech_maturity(tech_name)
                tech_profile = self.knowledge_base.get_tech_profile(tech_name)
                
                # Create stack item
                stack_item = TechStackItem(
                    name=tech_name,
                    category=category,
                    maturity_level=maturity_level,
                    maturity_score=maturity_score,
                    market_adoption=tech_profile.get('market_adoption', 0.5),
                    scalability=tech_profile.get('scalability', 0.5),
                    reliability=tech_profile.get('reliability', 0.5),
                    security_score=tech_profile.get('security_score', 0.5),
                    expertise_required=tech_profile.get('expertise_required', 0.5),
                    vendor_lock_in=tech_profile.get('vendor_lock_in', 0.5),
                    community_support=tech_profile.get('community_support', 0.5),
                    cost_efficiency=tech_profile.get('cost_efficiency', 0.5),
                    operational_complexity=tech_profile.get('operational_complexity', 0.5)
                )
                
                stack.append(stack_item)
        
        # Case 3: Generate default stack if nothing is provided
        if not stack:
            stack = self._generate_default_stack(td)
        
        # Ensure we have at least one technology per category
        self._ensure_stack_completeness(stack, td)
        
        return stack
    
    def _ensure_stack_completeness(self, stack: List[TechStackItem], td: Dict[str, Any]) -> None:
        """
        Ensure the technology stack has at least one technology per essential category
        
        Args:
            stack: Current technology stack
            td: Technical data dictionary
        """
        # Essential categories that should be present in a complete stack
        essential_categories = [
            TechCategory.LANGUAGE,
            TechCategory.DATABASE,
            TechCategory.FRONTEND,
            TechCategory.BACKEND,
            TechCategory.INFRASTRUCTURE
        ]
        
        # Check which categories are missing
        current_categories = set(item.category for item in stack)
        missing_categories = [cat for cat in essential_categories if cat not in current_categories]
        
        # Add default technologies for missing categories
        for category in missing_categories:
            default_tech = self._get_default_tech_for_category(category, td)
            stack.append(default_tech)
    
    def _get_default_tech_for_category(self, category: TechCategory, td: Dict[str, Any]) -> TechStackItem:
        """
        Get a default technology for a specific category based on context
        
        Args:
            category: Technology category
            td: Technical data dictionary
            
        Returns:
            Default TechStackItem for the category
        """
        sector = td.get('sector', '').lower()
        
        # Get appropriate defaults based on sector and category
        if category == TechCategory.LANGUAGE:
            if sector in ['fintech', 'banking', 'enterprise']:
                return TechStackItem(name="Java", category=category, maturity_level=TechMaturityLevel.MATURE)
            elif sector in ['ai', 'ml', 'data']:
                return TechStackItem(name="Python", category=category, maturity_level=TechMaturityLevel.MATURE)
            else:
                return TechStackItem(name="JavaScript", category=category, maturity_level=TechMaturityLevel.MATURE)
        
        elif category == TechCategory.DATABASE:
            if sector in ['fintech', 'banking', 'enterprise']:
                return TechStackItem(name="PostgreSQL", category=category, maturity_level=TechMaturityLevel.MATURE)
            elif sector in ['ecommerce', 'retail']:
                return TechStackItem(name="MySQL", category=category, maturity_level=TechMaturityLevel.MATURE)
            else:
                return TechStackItem(name="MongoDB", category=category, maturity_level=TechMaturityLevel.MATURE)
        
        elif category == TechCategory.FRONTEND:
            return TechStackItem(name="React", category=category, maturity_level=TechMaturityLevel.MATURE)
        
        elif category == TechCategory.BACKEND:
            if sector in ['fintech', 'banking', 'enterprise']:
                return TechStackItem(name="Spring Boot", category=category, maturity_level=TechMaturityLevel.MATURE)
            elif sector in ['ai', 'ml', 'data']:
                return TechStackItem(name="FastAPI", category=category, maturity_level=TechMaturityLevel.GROWING)
            else:
                return TechStackItem(name="Node.js", category=category, maturity_level=TechMaturityLevel.MATURE)
        
        elif category == TechCategory.INFRASTRUCTURE:
            return TechStackItem(name="AWS", category=category, maturity_level=TechMaturityLevel.MATURE)
        
        # Default for any other category
        return TechStackItem(name=f"Unknown {category.value}", category=category, maturity_level=TechMaturityLevel.GROWING)
    
    def _generate_default_stack(self, tech_data: Dict[str, Any]) -> List[TechStackItem]:
        """
        Generate a default technology stack based on company sector and context
        
        Args:
            tech_data: Technical data dictionary
            
        Returns:
            Default technology stack
        """
        sector = tech_data.get('sector', '').lower()
        stack = []
        
        if sector in ['fintech', 'banking']:
            stack.append(TechStackItem(
                name="Java", 
                category=TechCategory.LANGUAGE, 
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.7
            ))
            stack.append(TechStackItem(
                name="Spring Boot", 
                category=TechCategory.BACKEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.8,
                expertise_required=0.7
            ))
            stack.append(TechStackItem(
                name="PostgreSQL", 
                category=TechCategory.DATABASE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="React", 
                category=TechCategory.FRONTEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="AWS", 
                category=TechCategory.INFRASTRUCTURE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.9,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.7
            ))
            
        elif sector in ['ecommerce', 'retail']:
            stack.append(TechStackItem(
                name="PHP", 
                category=TechCategory.LANGUAGE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.7,
                scalability=0.6,
                reliability=0.7,
                security_score=0.6,
                expertise_required=0.5
            ))
            stack.append(TechStackItem(
                name="MySQL", 
                category=TechCategory.DATABASE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="React", 
                category=TechCategory.FRONTEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="Redis", 
                category=TechCategory.CACHE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="AWS", 
                category=TechCategory.INFRASTRUCTURE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.9,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.7
            ))
            
        elif sector in ['saas', 'enterprise']:
            stack.append(TechStackItem(
                name="Python", 
                category=TechCategory.LANGUAGE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.5
            ))
            stack.append(TechStackItem(
                name="Django", 
                category=TechCategory.BACKEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.7,
                reliability=0.8,
                security_score=0.8,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="PostgreSQL", 
                category=TechCategory.DATABASE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="React", 
                category=TechCategory.FRONTEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="Docker", 
                category=TechCategory.INFRASTRUCTURE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.7
            ))
            
        elif sector in ['ai', 'ml', 'data']:
            stack.append(TechStackItem(
                name="Python", 
                category=TechCategory.LANGUAGE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.5
            ))
            stack.append(TechStackItem(
                name="FastAPI", 
                category=TechCategory.BACKEND,
                maturity_level=TechMaturityLevel.GROWING,
                maturity_score=0.7,
                market_adoption=0.6,
                scalability=0.8,
                reliability=0.8,
                security_score=0.8,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="PostgreSQL", 
                category=TechCategory.DATABASE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="React", 
                category=TechCategory.FRONTEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="Docker", 
                category=TechCategory.INFRASTRUCTURE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.7
            ))
            stack.append(TechStackItem(
                name="TensorFlow", 
                category=TechCategory.ML_AI,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.8
            ))
            
        else:
            # Default stack for other sectors
            stack.append(TechStackItem(
                name="JavaScript", 
                category=TechCategory.LANGUAGE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.5
            ))
            stack.append(TechStackItem(
                name="Node.js", 
                category=TechCategory.BACKEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.7,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="MongoDB", 
                category=TechCategory.DATABASE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.8,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="React", 
                category=TechCategory.FRONTEND,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.8,
                market_adoption=0.9,
                scalability=0.8,
                reliability=0.8,
                security_score=0.7,
                expertise_required=0.6
            ))
            stack.append(TechStackItem(
                name="AWS", 
                category=TechCategory.INFRASTRUCTURE,
                maturity_level=TechMaturityLevel.MATURE,
                maturity_score=0.9,
                market_adoption=0.9,
                scalability=0.9,
                reliability=0.9,
                security_score=0.8,
                expertise_required=0.7
            ))
        
        return stack
    
    def _determine_architecture_pattern(self, td: Dict[str, Any]) -> ArchitecturePattern:
        """
        Determine the architecture pattern based on available information
        
        Args:
            td: Technical data dictionary
            
        Returns:
            ArchitecturePattern representing the system architecture
        """
        # Check if architecture type is explicitly provided
        arch_type = td.get("architecture_type", "").lower()
        
        # Match to known patterns
        if 'microservice' in arch_type or 'micro-service' in arch_type:
            return ArchitecturePattern.MICROSERVICES
            
        elif 'monolith' in arch_type:
            return ArchitecturePattern.MONOLITH
            
        elif 'serverless' in arch_type:
            return ArchitecturePattern.SERVERLESS
            
        elif 'event' in arch_type and ('driven' in arch_type or 'sourcing' in arch_type):
            return ArchitecturePattern.EVENT_DRIVEN
            
        elif 'service oriented' in arch_type or 'soa' in arch_type:
            return ArchitecturePattern.SOA
            
        elif 'layered' in arch_type:
            return ArchitecturePattern.LAYERED
            
        elif 'hexagonal' in arch_type or 'ports and adapters' in arch_type:
            return ArchitecturePattern.HEXAGONAL
            
        elif 'cqrs' in arch_type:
            return ArchitecturePattern.CQRS
            
        elif 'ddd' in arch_type or 'domain driven' in arch_type:
            return ArchitecturePattern.DDD
            
        elif 'mesh' in arch_type:
            return ArchitecturePattern.MESH
            
        elif 'space' in arch_type:
            return ArchitecturePattern.SPACE_BASED
        
        # If not explicitly stated, try to infer from technology stack
        tech_description = td.get("tech_description", "").lower()
        
        if any(t in tech_description for t in ['kubernetes', 'microservice', 'service mesh', 'istio']):
            return ArchitecturePattern.MICROSERVICES
            
        elif any(t in tech_description for t in ['lambda', 'aws lambda', 'serverless', 'azure functions']):
            return ArchitecturePattern.SERVERLESS
            
        elif any(t in tech_description for t in ['kafka', 'event', 'pubsub', 'message queue', 'rabbitmq']):
            return ArchitecturePattern.EVENT_DRIVEN
            
        elif any(t in tech_description for t in ['monolith', 'single application']):
            return ArchitecturePattern.MONOLITH
        
        # Default to hybrid if unclear
        return ArchitecturePattern.HYBRID
    
    def _assess_architecture(
        self, 
        td: Dict[str, Any], 
        stack: List[TechStackItem], 
        arch_pattern: ArchitecturePattern
    ) -> Tuple[float, float]:
        """
        Assess the architecture quality and maturity
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            arch_pattern: Architecture pattern
            
        Returns:
            Tuple of (architecture_score, architecture_maturity)
        """
        # Start with baseline scores
        score = 0.7  # Default to slightly above average
        maturity = 0.6
        
        # Adjust based on architecture pattern
        if arch_pattern == ArchitecturePattern.MICROSERVICES:
            score += 0.1
            # Microservices are good for scalability but add complexity
            if td.get("engineering_team_size", 0) < 10:
                score -= 0.05  # Small team may struggle with microservice complexity
        
        elif arch_pattern == ArchitecturePattern.MONOLITH:
            score -= 0.05  # Monoliths are generally less flexible
            # But monoliths can be good for small teams and early stage
            if td.get("engineering_team_size", 0) < 5 or td.get("stage", "") in ["pre-seed", "seed"]:
                score += 0.05
        
        elif arch_pattern == ArchitecturePattern.SERVERLESS:
            score += 0.1  # Serverless architectures often scale well and reduce ops burden
            # But may have cold-start and other limitations
            if td.get("current_users", 0) > 100000:
                score -= 0.05  # Very high scale may hit serverless limitations
        
        elif arch_pattern == ArchitecturePattern.EVENT_DRIVEN:
            score += 0.05  # Good for decoupling
            # Check if appropriate messaging tech is present
            has_messaging = any(item.category == TechCategory.MESSAGING for item in stack)
            if not has_messaging:
                score -= 0.1  # Event-driven without messaging tech is a concern
        
        # Adjust based on tech stack maturity
        if stack:
            avg_maturity = sum(item.maturity_score for item in stack) / len(stack)
            avg_adoption = sum(item.market_adoption for item in stack) / len(stack)
            
            # Influence score and maturity by tech stack characteristics
            score += (avg_maturity - 0.5) * 0.2
            score += (avg_adoption - 0.5) * 0.1
            maturity = avg_maturity * 0.6 + avg_adoption * 0.4
        
        # Adjust based on reported issues and documentation
        if td.get("reported_issues", 0) > 10:
            score -= 0.1
        elif td.get("reported_issues", 0) > 5:
            score -= 0.05
        
        if td.get("has_architecture_docs", False):
            score += 0.05
            maturity += 0.1
        
        # Adjust based on team experience with the architecture pattern
        team_exp = td.get("team_architecture_experience", 0)
        if team_exp > 3:
            score += 0.05
            maturity += 0.1
        elif team_exp < 1:
            score -= 0.05
            maturity -= 0.1
        
        # Ensure scores are within bounds
        score = max(0.1, min(1.0, score))
        maturity = max(0.1, min(1.0, maturity))
        
        return score, maturity
    
    def _assess_scalability(
        self, 
        td: Dict[str, Any], 
        stack: List[TechStackItem],
        arch_pattern: ArchitecturePattern
    ) -> ScalabilityAssessment:
        """
        Assess the scalability characteristics of the system
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            arch_pattern: Architecture pattern
            
        Returns:
            ScalabilityAssessment with detailed scalability metrics
        """
        # Start with baseline scores based on architecture pattern
        overall_score = 0.6  # Default to slightly above average
        
        # Architecture-specific scalability scores
        arch_scalability = 0.0
        if arch_pattern == ArchitecturePattern.MICROSERVICES:
            arch_scalability = 0.85
        elif arch_pattern == ArchitecturePattern.SERVERLESS:
            arch_scalability = 0.9
        elif arch_pattern == ArchitecturePattern.EVENT_DRIVEN:
            arch_scalability = 0.8
        elif arch_pattern == ArchitecturePattern.MONOLITH:
            arch_scalability = 0.5
        elif arch_pattern == ArchitecturePattern.SPACE_BASED:
            arch_scalability = 0.9
        elif arch_pattern == ArchitecturePattern.MESH:
            arch_scalability = 0.85
        else:
            arch_scalability = 0.7  # Hybrid or other architectures
        
        # Current scale indicators
        current_users = td.get("current_users", 0)
        
        # Scale-based adjustments
        if current_users > 1_000_000:
            overall_score += 0.15  # System already handles large scale
            arch_scalability += 0.05
        elif current_users > 100_000:
            overall_score += 0.1
        elif current_users > 10_000:
            overall_score += 0.05
        
        # Technology stack scalability analysis
        infra_items = [item for item in stack if item.category == TechCategory.INFRASTRUCTURE]
        db_items = [item for item in stack if item.category == TechCategory.DATABASE]
        
        # Analyze infrastructure scalability
        infra_scalability = 0.6  # Default
        if infra_items:
            infra_scalability = sum(item.scalability for item in infra_items) / len(infra_items)
            
            # Check for specific scalable infrastructure technologies
            for item in infra_items:
                if 'kubernetes' in item.name.lower():
                    infra_scalability = max(infra_scalability, 0.9)
                elif 'aws' in item.name.lower() and not td.get("aws_limited_to_ec2_only", False):
                    infra_scalability = max(infra_scalability, 0.85)
                elif 'azure' in item.name.lower() and not td.get("azure_limited_to_vms_only", False):
                    infra_scalability = max(infra_scalability, 0.85)
                elif 'gcp' in item.name.lower():
                    infra_scalability = max(infra_scalability, 0.85)
                elif 'docker' in item.name.lower() and not any('kubernetes' in i.name.lower() for i in infra_items):
                    infra_scalability = max(infra_scalability, 0.7)  # Docker without orchestration
        
        # Database scalability analysis
        db_scalability = 0.5  # Default
        if db_items:
            db_scalability = sum(item.scalability for item in db_items) / len(db_items)
            
            # Check for specific database technologies
            for item in db_items:
                name = item.name.lower()
                if any(db in name for db in ['dynamodb', 'cosmosdb', 'cassandra']):
                    db_scalability = max(db_scalability, 0.9)
                elif any(db in name for db in ['postgres', 'mysql']) and td.get("db_has_sharding", False):
                    db_scalability = max(db_scalability, 0.8)
                elif any(db in name for db in ['mongodb']) and td.get("db_has_sharding", False):
                    db_scalability = max(db_scalability, 0.85)
                elif any(db in name for db in ['sqlite', 'access']):
                    db_scalability = min(db_scalability, 0.3)
        
        # Compute/Application tier scalability
        compute_scalability = 0.6  # Default
        
        # Check for horizontal scaling indicators
        horizontal_scaling = td.get("implements_horizontal_scaling", False) or arch_pattern in [
            ArchitecturePattern.MICROSERVICES, 
            ArchitecturePattern.SERVERLESS,
            ArchitecturePattern.SPACE_BASED
        ]
        
        if horizontal_scaling:
            compute_scalability = 0.8
        
        # Check for vertical scaling indicators
        vertical_scaling = td.get("implements_vertical_scaling", False)
        if vertical_scaling:
            compute_scalability = max(compute_scalability, 0.7)
        
        # Network scalability (CDN, load balancing, etc.)
        network_scalability = 0.6  # Default
        
        # Check for CDN usage
        if td.get("uses_cdn", False):
            network_scalability += 0.1
        
        # Check for load balancing
        if td.get("uses_load_balancing", False):
            network_scalability += 0.1
        
        # Caching strategies
        caching_strategy = False
        cache_items = [item for item in stack if item.category == TechCategory.CACHE]
        if cache_items or td.get("implements_caching", False):
            network_scalability += 0.1
            compute_scalability += 0.05
            db_scalability += 0.05
            caching_strategy = True
        
        # Data partitioning
        data_partitioning = False
        if td.get("implements_data_partitioning", False) or td.get("db_has_sharding", False):
            db_scalability += 0.1
            data_partitioning = True
        
        # Identify bottlenecks
        bottlenecks = []
        
        if db_scalability < 0.6:
            bottlenecks.append("Database scaling limitations")
        
        if infra_scalability < 0.6:
            bottlenecks.append("Infrastructure scaling constraints")
        
        if not horizontal_scaling and not vertical_scaling:
            bottlenecks.append("Lack of defined scaling strategy")
        
        if arch_pattern == ArchitecturePattern.MONOLITH and current_users > 50000:
            bottlenecks.append("Monolithic architecture at scale")
        
        if not td.get("uses_load_balancing", False) and current_users > 10000:
            bottlenecks.append("Missing load balancing")
            
        if not cache_items and not td.get("implements_caching", False) and current_users > 10000:
            bottlenecks.append("No caching strategy identified")
        
        # Performance metrics if available
        throughput_capability = None
        latency_profile = None
        
        if td.get("performance_metrics", {}):
            perf = td.get("performance_metrics", {})
            if "throughput" in perf:
                throughput_capability = f"{perf['throughput']} {perf.get('throughput_unit', 'req/s')}"
            if "latency" in perf:
                latency_profile = f"p50: {perf.get('latency_p50', 'N/A')}ms, p95: {perf.get('latency_p95', 'N/A')}ms, p99: {perf.get('latency_p99', 'N/A')}ms"
        
        # Load test results
        load_test_results = td.get("load_test_results", {})
        
        # Calculate overall score
        overall_score = (
            arch_scalability * 0.25 +
            infra_scalability * 0.25 + 
            db_scalability * 0.2 + 
            compute_scalability * 0.15 +
            network_scalability * 0.15
        )
        
        # Capacity planning and recommendations
        max_theoretical_capacity = None
        if current_users > 0 and td.get("peak_concurrent_users", 0) > 0:
            max_users = current_users * 5  # Conservative 5x estimate
            max_theoretical_capacity = f"~{max_users:,} users"
        
        # Generate scaling recommendations
        scaling_recommendations = []
        
        if arch_pattern == ArchitecturePattern.MONOLITH and current_users > 50000:
            scaling_recommendations.append("Consider migrating from monolith to microservices architecture")
        
        if not horizontal_scaling and current_users > 10000:
            scaling_recommendations.append("Implement horizontal scaling for application tier")
        
        if db_scalability < 0.7 and current_users > 50000:
            scaling_recommendations.append("Evaluate database scaling solutions (sharding, read replicas, etc.)")
        
        if not caching_strategy and current_users > 10000:
            scaling_recommendations.append("Implement caching strategy to reduce database load")
        
        if not td.get("uses_cdn", False) and current_users > 10000:
            scaling_recommendations.append("Implement CDN for static assets")
            
        if not data_partitioning and current_users > 100000:
            scaling_recommendations.append("Implement data partitioning strategy")
        
        # Scaling costs estimation
        scaling_costs = {}
        if current_users > 0:
            # Create simplified cost model
            scaling_costs = {
                "current_estimated_monthly": td.get("infrastructure_costs_monthly", 0),
                "10x_growth_estimated": td.get("infrastructure_costs_monthly", 0) * 4,  # Economies of scale
                "100x_growth_estimated": td.get("infrastructure_costs_monthly", 0) * 20  # Economies of scale
            }
        
        # Create the scalability assessment
        assessment = ScalabilityAssessment(
            overall_score=max(0.1, min(1.0, overall_score)),
            infrastructure_scalability=infra_scalability,
            database_scalability=db_scalability,
            compute_scalability=compute_scalability,
            network_scalability=network_scalability,
            architecture_scalability=arch_scalability,
            bottlenecks=bottlenecks,
            throughput_capability=throughput_capability,
            latency_profile=latency_profile,
            load_test_results=load_test_results,
            horizontal_scaling=horizontal_scaling,
            vertical_scaling=vertical_scaling,
            data_partitioning=data_partitioning,
            caching_strategy=caching_strategy,
            max_theoretical_capacity=max_theoretical_capacity,
            scaling_recommendations=scaling_recommendations,
            scaling_costs=scaling_costs
        )
        
        return assessment
    
    def _assess_security(
        self, 
        td: Dict[str, Any], 
        stack: List[TechStackItem]
    ) -> SecurityAssessment:
        """
        Perform a comprehensive security assessment
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            
        Returns:
            SecurityAssessment with detailed security metrics
        """
        # Calculate baseline security score
        base_score = 0.65  # Start with a slightly above average score
        
        # Compliance frameworks assessment
        compliance_frameworks = {}
        
        # Check for common compliance requirements
        if td.get("compliance_requirements", []):
            for compliance in td.get("compliance_requirements", []):
                if isinstance(compliance, dict):
                    name = compliance.get("name", "")
                    score = compliance.get("score", 0.5)
                    compliance_frameworks[name] = score
                elif isinstance(compliance, str):
                    compliance_frameworks[compliance] = 0.7  # Default score if only name provided
        
        # Infer compliance needs based on sector
        sector = td.get("sector", "").lower()
        if sector in ["fintech", "banking", "finance"]:
            if "PCI-DSS" not in compliance_frameworks:
                compliance_frameworks["PCI-DSS"] = 0.5  # Assumed score
            if "SOC2" not in compliance_frameworks:
                compliance_frameworks["SOC2"] = 0.5
        elif sector in ["healthcare", "health", "medical"]:
            if "HIPAA" not in compliance_frameworks:
                compliance_frameworks["HIPAA"] = 0.5
        
        # OWASP Top 10 risk assessment
        owasp_risk_profile = {
            "A01:2021-Broken Access Control": 0.6,
            "A02:2021-Cryptographic Failures": 0.7,
            "A03:2021-Injection": 0.7,
            "A04:2021-Insecure Design": 0.6,
            "A05:2021-Security Misconfiguration": 0.6,
            "A06:2021-Vulnerable Components": 0.5,
            "A07:2021-Auth Failures": 0.6,
            "A08:2021-Software and Data Integrity": 0.7,
            "A09:2021-Logging Failures": 0.6,
            "A10:2021-SSRF": 0.7
        }
        
        # Adjust OWASP scores based on available information
        if td.get("security_practices", {}):
            sec_practices = td.get("security_practices", {})
            
            if sec_practices.get("code_scanning", False):
                owasp_risk_profile["A03:2021-Injection"] += 0.2
                owasp_risk_profile["A06:2021-Vulnerable Components"] += 0.2
            
            if sec_practices.get("dependency_scanning", False):
                owasp_risk_profile["A06:2021-Vulnerable Components"] += 0.3
            
            if sec_practices.get("secrets_scanning", False):
                owasp_risk_profile["A02:2021-Cryptographic Failures"] += 0.1
                owasp_risk_profile["A05:2021-Security Misconfiguration"] += 0.1
            
            if sec_practices.get("pen_testing", False):
                for key in owasp_risk_profile:
                    owasp_risk_profile[key] += 0.1
        
        # Ensure OWASP scores are within bounds
        for key in owasp_risk_profile:
            owasp_risk_profile[key] = max(0.1, min(1.0, owasp_risk_profile[key]))
        
        # Dependency vulnerabilities analysis
        dependency_vulnerabilities = {}
        critical_vulns = 0
        high_vulns = 0
        medium_vulns = 0
        low_vulns = 0
        
        # Check known vulnerabilities in the stack
        for item in stack:
            if item.known_vulnerabilities:
                dependency_vulnerabilities[item.name] = item.known_vulnerabilities
                
                # Count vulnerabilities by severity
                for vuln in item.known_vulnerabilities:
                    severity = vuln.get("severity", "").lower()
                    if severity == "critical":
                        critical_vulns += 1
                        base_score -= 0.05
                    elif severity == "high":
                        high_vulns += 1
                        base_score -= 0.03
                    elif severity == "medium":
                        medium_vulns += 1
                        base_score -= 0.01
                    elif severity == "low":
                        low_vulns += 1
        
        # Security practices assessment
        security_practices = {
            "secure_sdlc": 0.5,
            "security_testing": 0.5,
            "dependency_management": 0.5,
            "incident_response": 0.5,
            "devsecops_integration": 0.5
        }
        
        # Update security practices based on available data
        if td.get("security_practices", {}):
            sec = td.get("security_practices", {})
            
            if sec.get("secure_sdlc", False):
                security_practices["secure_sdlc"] = 0.8
                base_score += 0.05
            
            if sec.get("security_testing", False):
                security_practices["security_testing"] = 0.8
                base_score += 0.05
            
            if sec.get("dependency_scanning", False):
                security_practices["dependency_management"] = 0.8
                base_score += 0.05
            
            if sec.get("incident_response_plan", False):
                security_practices["incident_response"] = 0.8
                base_score += 0.03
            
            if sec.get("devsecops", False):
                security_practices["devsecops_integration"] = 0.8
                base_score += 0.05
        
        # Authentication and authorization assessment
        auth_assessment = {
            "multi_factor_auth": 0.5,
            "role_based_access": 0.5,
            "session_management": 0.5,
            "password_policies": 0.5
        }
        
        # Update auth assessment based on available data
        if td.get("auth_details", {}):
            auth = td.get("auth_details", {})
            
            if auth.get("mfa", False):
                auth_assessment["multi_factor_auth"] = 0.9
                base_score += 0.05
            
            if auth.get("rbac", False):
                auth_assessment["role_based_access"] = 0.8
                base_score += 0.03
            
            if auth.get("secure_session", False):
                auth_assessment["session_management"] = 0.8
                base_score += 0.02
            
            if auth.get("password_policy", False):
                auth_assessment["password_policies"] = 0.8
                base_score += 0.02
        
        # Data protection assessment
        data_protection_assessment = {
            "encryption_at_rest": 0.5,
            "encryption_in_transit": 0.5,
            "data_classification": 0.5,
            "data_retention": 0.5
        }
        
        # Update data protection assessment based on available data
        if td.get("data_protection", {}):
            data_prot = td.get("data_protection", {})
            
            if data_prot.get("encryption_at_rest", False):
                data_protection_assessment["encryption_at_rest"] = 0.9
                base_score += 0.05
            
            if data_prot.get("encryption_in_transit", False):
                data_protection_assessment["encryption_in_transit"] = 0.9
                base_score += 0.05
            
            if data_prot.get("data_classification", False):
                data_protection_assessment["data_classification"] = 0.8
                base_score += 0.03
            
            if data_prot.get("data_retention_policy", False):
                data_protection_assessment["data_retention"] = 0.8
                base_score += 0.02
        
        # Generate security recommendations
        critical_recommendations = []
        high_priority_recommendations = []
        medium_priority_recommendations = []
        
        # Critical recommendations based on vulnerabilities
        if critical_vulns > 0:
            critical_recommendations.append(f"Address {critical_vulns} critical vulnerabilities in dependencies")
        
        if high_vulns > 0:
            critical_recommendations.append(f"Address {high_vulns} high severity vulnerabilities in dependencies")
        
        # Check for essential security practices
        if not td.get("security_practices", {}).get("dependency_scanning", False):
            high_priority_recommendations.append("Implement automated dependency scanning")
        
        if not td.get("security_practices", {}).get("code_scanning", False):
            high_priority_recommendations.append("Implement SAST (Static Application Security Testing)")
        
        if not td.get("auth_details", {}).get("mfa", False) and sector in ["fintech", "banking", "healthcare"]:
            high_priority_recommendations.append("Implement multi-factor authentication")
        
        if not td.get("data_protection", {}).get("encryption_at_rest", False) and sector in ["fintech", "banking", "healthcare"]:
            high_priority_recommendations.append("Implement encryption for data at rest")
        
        # Medium priority recommendations
        if not td.get("security_practices", {}).get("pen_testing", False):
            medium_priority_recommendations.append("Conduct regular penetration testing")
        
        if not td.get("security_practices", {}).get("security_training", False):
            medium_priority_recommendations.append("Implement security training for engineering team")
        
        # Overall security score calculation
        # Calculate average OWASP score
        avg_owasp = sum(owasp_risk_profile.values()) / len(owasp_risk_profile)
        
        # Calculate average security practices score
        avg_practices = sum(security_practices.values()) / len(security_practices)
        
        # Calculate average auth score
        avg_auth = sum(auth_assessment.values()) / len(auth_assessment)
        
        # Calculate average data protection score
        avg_data = sum(data_protection_assessment.values()) / len(data_protection_assessment)
        
        # Weighted overall score
        overall_score = (
            base_score * 0.3 +
            avg_owasp * 0.2 +
            avg_practices * 0.2 +
            avg_auth * 0.15 +
            avg_data * 0.15
        )
        
        # Ensure overall score is within bounds
        overall_score = max(0.1, min(1.0, overall_score))
        
        # Create security assessment
        assessment = SecurityAssessment(
            overall_score=overall_score,
            compliance_frameworks=compliance_frameworks,
            owasp_risk_profile=owasp_risk_profile,
            dependency_vulnerabilities=dependency_vulnerabilities,
            critical_vulnerabilities_count=critical_vulns,
            high_vulnerabilities_count=high_vulns,
            medium_vulnerabilities_count=medium_vulns,
            low_vulnerabilities_count=low_vulns,
            security_practices=security_practices,
            auth_assessment=auth_assessment,
            data_protection_assessment=data_protection_assessment,
            critical_recommendations=critical_recommendations,
            high_priority_recommendations=high_priority_recommendations,
            medium_priority_recommendations=medium_priority_recommendations
        )
        
        return assessment
    
    def _assess_technical_debt(
        self, 
        td: Dict[str, Any],
        stack: List[TechStackItem]
    ) -> TechnicalDebtAssessment:
        """
        Assess technical debt and code quality
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            
        Returns:
            TechnicalDebtAssessment with detailed debt metrics
        """
        # Start with baseline score
        base_score = 0.5  # Average technical debt
        
        # Code quality metrics
        code_quality = {
            "maintainability": 0.5,
            "readability": 0.5,
            "modularity": 0.5,
            "documentation": 0.5
        }
        
        # Test coverage metrics
        test_coverage = {
            "unit_test_coverage": td.get("test_coverage", 0) / 100,
            "integration_test_coverage": td.get("integration_test_coverage", 0) / 100,
            "e2e_test_coverage": td.get("e2e_test_coverage", 0) / 100,
            "manual_test_coverage": td.get("manual_test_coverage", 0) / 100
        }
        
        # Code complexity metrics
        complexity_metrics = {
            "cyclomatic_complexity": 0.5,
            "method_length": 0.5,
            "class_complexity": 0.5,
            "dependency_graph": 0.5
        }
        
        # Duplication metrics
        duplication_metrics = {
            "code_duplication": 0.5,
            "copy_paste_score": 0.5
        }
        
        # Update metrics based on available data
        if td.get("code_quality_metrics", {}):
            cq = td.get("code_quality_metrics", {})
            
            # Update code quality metrics
            if "maintainability_index" in cq:
                mi = cq["maintainability_index"] / 100 if cq["maintainability_index"] <= 100 else 1.0
                code_quality["maintainability"] = mi
            
            if "readability_score" in cq:
                code_quality["readability"] = cq["readability_score"] / 100
            
            if "modularity_score" in cq:
                code_quality["modularity"] = cq["modularity_score"] / 100
            
            if "documentation_ratio" in cq:
                code_quality["documentation"] = cq["documentation_ratio"] / 100
            
            # Update complexity metrics
            if "cyclomatic_complexity" in cq:
                # Convert cyclomatic complexity to a score (lower is better)
                cc = cq["cyclomatic_complexity"]
                complexity_metrics["cyclomatic_complexity"] = max(0, min(1, 1 - (cc - 5) / 15))
            
            if "method_length" in cq:
                # Convert method length to a score (lower is better)
                ml = cq["method_length"]
                complexity_metrics["method_length"] = max(0, min(1, 1 - (ml - 15) / 85))
            
            if "class_complexity" in cq:
                cc = cq["class_complexity"]
                complexity_metrics["class_complexity"] = max(0, min(1, 1 - (cc - 10) / 90))
            
            if "dependency_graph_complexity" in cq:
                complexity_metrics["dependency_graph"] = cq["dependency_graph_complexity"] / 100
            
            # Update duplication metrics
            if "code_duplication_percentage" in cq:
                dup = cq["code_duplication_percentage"] / 100
                duplication_metrics["code_duplication"] = max(0, min(1, 1 - dup))
            
            if "copy_paste_detection_score" in cq:
                duplication_metrics["copy_paste_score"] = cq["copy_paste_detection_score"] / 100
        
        # Repository metrics
        commit_frequency = td.get("commit_frequency_per_week", None)
        lead_time = td.get("lead_time_for_changes_days", None)
        
        # Calculate maintenance burden
        maintenance_burden = 0.5  # Default average burden
        
        # Update maintenance burden based on metrics
        avg_code_quality = sum(code_quality.values()) / len(code_quality)
        avg_complexity = sum(complexity_metrics.values()) / len(complexity_metrics)
        avg_duplication = sum(duplication_metrics.values()) / len(duplication_metrics)
        
        # Maintenance burden is inverse of quality metrics
        maintenance_burden = 1 - (
            avg_code_quality * 0.3 +
            avg_complexity * 0.4 +
            avg_duplication * 0.3
        )
        
        # Adjust for test coverage
        test_coverage_avg = sum(test_coverage.values()) / len(test_coverage)
        if test_coverage_avg > 0.7:
            maintenance_burden -= 0.1
        elif test_coverage_avg < 0.3:
            maintenance_burden += 0.1
        
        # Modernization needs based on technology stack
        modernization_needs = []
        
        # Identify outdated or risky technologies
        for item in stack:
            if item.maturity_level == TechMaturityLevel.DECLINING or item.maturity_level == TechMaturityLevel.LEGACY:
                modernization_needs.append(f"Replace {item.name} with modern alternative")
            elif item.maturity_score < 0.5 and item.market_adoption < 0.4:
                modernization_needs.append(f"Evaluate continued use of {item.name}")
        
        # Check for monolithic architecture modernization
        if td.get("architecture_type", "").lower() == "monolith" and td.get("current_users", 0) > 50000:
            modernization_needs.append("Consider breaking monolith into microservices")
        
        # Identify critical refactoring areas
        critical_refactoring_areas = []
        
        if td.get("test_coverage", 0) < 30:
            critical_refactoring_areas.append("Improve test coverage (currently below 30%)")
        
        if duplication_metrics["code_duplication"] < 0.7:  # More than 30% duplication
            critical_refactoring_areas.append("Address high code duplication")
        
        if complexity_metrics["cyclomatic_complexity"] < 0.5:  # High complexity
            critical_refactoring_areas.append("Reduce code complexity in core modules")
        
        if td.get("open_bugs", 0) > 50:
            critical_refactoring_areas.append(f"Address backlog of {td.get('open_bugs', 0)} open bugs")
        
        # Refactoring cost estimate
        refactoring_cost_estimate = {}
        
        if critical_refactoring_areas or modernization_needs:
            dev_team_size = td.get("engineering_team_size", 5)
            monthly_dev_cost = dev_team_size * 15000  # Estimated monthly dev cost
            
            # Estimate refactoring effort
            if len(critical_refactoring_areas) + len(modernization_needs) > 5:
                effort_months = 6
            elif len(critical_refactoring_areas) + len(modernization_needs) > 2:
                effort_months = 3
            else:
                effort_months = 1.5
            
            refactoring_cost_estimate = {
                "estimated_effort_months": effort_months,
                "estimated_cost": monthly_dev_cost * effort_months,
                "opportunity_cost": monthly_dev_cost * effort_months * 0.5  # Lost feature development
            }
        
        # Calculate overall technical debt score
        # Average of code quality, test coverage, complexity, and duplication
        code_quality_score = sum(code_quality.values()) / len(code_quality)
        test_coverage_score = sum(test_coverage.values()) / len(test_coverage)
        complexity_score = sum(complexity_metrics.values()) / len(complexity_metrics)
        duplication_score = sum(duplication_metrics.values()) / len(duplication_metrics)
        
        # Adjust score based on specific indicators
        if td.get("has_code_reviews", False):
            base_score += 0.1
        
        if td.get("has_documentation", False):
            base_score += 0.05
        
        if td.get("open_bugs", 0) > 100:
            base_score -= 0.2
        elif td.get("open_bugs", 0) > 50:
            base_score -= 0.1
        
        if td.get("acknowledged_tech_debt", 0) > 0:
            # Having acknowledged tech debt is good, but higher values mean more debt
            base_score += 0.05
            base_score -= min(0.2, td.get("acknowledged_tech_debt", 0) / 10)
        
        if td.get("regular_refactoring", False):
            base_score += 0.1
        
        # Weighted overall score
        overall_score = (
            base_score * 0.2 +
            code_quality_score * 0.2 +
            test_coverage_score * 0.2 +
            complexity_score * 0.2 +
            duplication_score * 0.2
        )
        
        # Ensure overall score is within bounds
        overall_score = max(0.1, min(1.0, overall_score))
        
        # Create technical debt assessment
        assessment = TechnicalDebtAssessment(
            overall_score=overall_score,
            code_quality=code_quality,
            test_coverage=test_coverage,
            complexity_metrics=complexity_metrics,
            duplication_metrics=duplication_metrics,
            commit_frequency=commit_frequency,
            lead_time=lead_time,
            maintenance_burden=maintenance_burden,
            modernization_needs=modernization_needs,
            critical_refactoring_areas=critical_refactoring_areas,
            refactoring_cost_estimate=refactoring_cost_estimate
        )
        
        return assessment
    
    def _assess_operational_capabilities(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem]
    ) -> OperationalAssessment:
        """
        Assess operational capabilities and reliability
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            
        Returns:
            OperationalAssessment with detailed operations metrics
        """
        # Start with baseline score
        base_score = 0.6  # Slightly above average operations
        
        # Reliability metrics
        reliability_score = 0.7  # Default good reliability
        uptime_percentage = td.get("uptime_percentage", 99.5)
        mean_time_between_failures = td.get("mtbf_hours", None)
        mean_time_to_recovery = td.get("mttr_minutes", None)
        
        # Adjust reliability score based on uptime
        if uptime_percentage >= 99.99:
            reliability_score = 0.95  # Excellent
        elif uptime_percentage >= 99.9:
            reliability_score = 0.85  # Very good
        elif uptime_percentage >= 99.5:
            reliability_score = 0.75  # Good
        elif uptime_percentage >= 99.0:
            reliability_score = 0.65  # Adequate
        elif uptime_percentage >= 98.0:
            reliability_score = 0.5   # Concerning
        else:
            reliability_score = 0.4   # Poor
        
        # Adjust for MTTR if available
        if mean_time_to_recovery is not None:
            if mean_time_to_recovery < 15:
                reliability_score += 0.05  # Very fast recovery
            elif mean_time_to_recovery > 120:
                reliability_score -= 0.05  # Slow recovery
        
        # Observability assessment
        observability_score = 0.5  # Default average observability
        
        # Adjust based on monitoring tools and practices
        if td.get("monitoring_tools", []):
            monitoring_tools = td.get("monitoring_tools", [])
            if len(monitoring_tools) >= 3:
                observability_score += 0.1
            elif len(monitoring_tools) > 0:
                observability_score += 0.05
        
        if td.get("has_apm", False):  # Application Performance Monitoring
            observability_score += 0.1
        
        if td.get("has_distributed_tracing", False):
            observability_score += 0.1
        
        if td.get("has_centralized_logging", False):
            observability_score += 0.1
        
        # Monitoring coverage
        monitoring_coverage = 0.6  # Default above-average coverage
        
        if td.get("monitoring_coverage_percentage", 0) > 0:
            monitoring_coverage = td.get("monitoring_coverage_percentage", 0) / 100
        else:
            # Estimate monitoring coverage from tools and practices
            if td.get("has_infrastructure_monitoring", False):
                monitoring_coverage += 0.1
            
            if td.get("has_application_monitoring", False):
                monitoring_coverage += 0.1
            
            if td.get("has_database_monitoring", False):
                monitoring_coverage += 0.1
            
            if td.get("has_endpoint_monitoring", False):
                monitoring_coverage += 0.1
            
            # Ensure within bounds
            monitoring_coverage = max(0.1, min(1.0, monitoring_coverage))
        
        # Alerting effectiveness
        alerting_effectiveness = 0.5  # Default average alerting
        
        if td.get("has_alerting", False):
            alerting_effectiveness += 0.1
        
        if td.get("has_paging", False):
            alerting_effectiveness += 0.1
        
        if td.get("has_alert_prioritization", False):
            alerting_effectiveness += 0.1
        
        if td.get("low_alert_noise", False):  # Low alert fatigue
            alerting_effectiveness += 0.1
        
        # Incident management
        incident_management_score = 0.5  # Default average incident management
        
        if td.get("has_incident_management_process", False):
            incident_management_score += 0.1
        
        if td.get("has_postmortems", False):
            incident_management_score += 0.1
        
        if td.get("has_incident_database", False):
            incident_management_score += 0.1
        
        if td.get("has_on_call_rotation", False):
            incident_management_score += 0.1
        
        # SRE practices
        sre_maturity = 0.3  # Default low SRE maturity
        
        if td.get("has_sre_team", False):
            sre_maturity += 0.2
        
        if td.get("has_slos", False):
            sre_maturity += 0.2
        
        error_budget_policy = td.get("has_error_budget_policy", False)
        if error_budget_policy:
            sre_maturity += 0.2
        
        chaos_engineering = td.get("practices_chaos_engineering", False)
        if chaos_engineering:
            sre_maturity += 0.1
        
        # Generate operational recommendations
        operational_recommendations = []
        
        if not td.get("has_centralized_logging", False):
            operational_recommendations.append("Implement centralized logging")
        
        if not td.get("has_distributed_tracing", False):
            operational_recommendations.append("Implement distributed tracing for better observability")
        
        if not td.get("has_apm", False):
            operational_recommendations.append("Implement application performance monitoring")
        
        if not td.get("has_incident_management_process", False):
            operational_recommendations.append("Establish formal incident management process")
        
        if not td.get("has_postmortems", False):
            operational_recommendations.append("Conduct postmortems for all significant incidents")
        
        if not td.get("has_slos", False):
            operational_recommendations.append("Define service level objectives (SLOs)")
        
        if uptime_percentage < 99.5:
            operational_recommendations.append("Improve system reliability to achieve at least 99.9% uptime")
        
        if mean_time_to_recovery and mean_time_to_recovery > 60:
            operational_recommendations.append("Reduce mean time to recovery (MTTR)")
        
        # Calculate overall score
        overall_score = (
            reliability_score * 0.35 +
            observability_score * 0.2 +
            monitoring_coverage * 0.1 +
            alerting_effectiveness * 0.1 +
            incident_management_score * 0.15 +
            sre_maturity * 0.1
        )
        
        # Ensure overall score is within bounds
        overall_score = max(0.1, min(1.0, overall_score))
        
        # Create operational assessment
        assessment = OperationalAssessment(
            overall_score=overall_score,
            reliability_score=reliability_score,
            uptime_percentage=uptime_percentage,
            observability_score=observability_score,
            monitoring_coverage=monitoring_coverage,
            alerting_effectiveness=alerting_effectiveness,
            incident_management_score=incident_management_score,
            sre_maturity=sre_maturity,
            mean_time_between_failures=mean_time_between_failures,
            mean_time_to_recovery=mean_time_to_recovery,
            error_budget_policy=error_budget_policy,
            chaos_engineering=chaos_engineering,
            operational_recommendations=operational_recommendations
        )
        
        return assessment
    
    def _assess_talent_requirements(
        self, 
        td: Dict[str, Any],
        stack: List[TechStackItem]
    ) -> TalentAssessment:
        """
        Assess engineering talent requirements and risks
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            
        Returns:
            TalentAssessment with detailed talent metrics
        """
        # Get team size
        team_size = td.get("engineering_team_size", 0)
        
        # Default values
        senior_engineers_ratio = 0.3  # Default 30% senior engineers
        
        # Update from data if available
        if td.get("senior_engineers", 0) > 0 and team_size > 0:
            senior_engineers_ratio = td.get("senior_engineers", 0) / team_size
        
        # Skills coverage analysis
        skills_coverage = {}
        
        # Analyze required skills based on technology stack
        required_skills = set()
        for item in stack:
            cat = item.category.value if isinstance(item.category, Enum) else str(item.category)
            
            if cat not in skills_coverage:
                skills_coverage[cat] = 0.5  # Default coverage
            
            required_skills.add(item.name)
        
        # Update skills coverage if available
        if td.get("skills_coverage", {}):
            for skill, coverage in td.get("skills_coverage", {}).items():
                if isinstance(coverage, (int, float)):
                    skills_coverage[skill] = coverage / 100 if coverage > 1 else coverage
        
        # Key person dependencies
        key_person_dependencies = td.get("key_person_dependencies", 0)
        
        # Knowledge sharing metrics
        knowledge_sharing_score = 0.5  # Default average knowledge sharing
        documentation_quality = 0.5  # Default average documentation
        onboarding_efficiency = 0.5  # Default average onboarding
        
        # Update from data if available
        if td.get("knowledge_sharing_score", 0) > 0:
            knowledge_sharing_score = td.get("knowledge_sharing_score", 0) / 100
        else:
            # Estimate from other factors
            if td.get("has_code_reviews", False):
                knowledge_sharing_score += 0.1
            
            if td.get("has_documentation", False):
                knowledge_sharing_score += 0.1
                documentation_quality = 0.7
            
            if td.get("pair_programming", False):
                knowledge_sharing_score += 0.1
            
            if td.get("knowledge_sharing_sessions", False):
                knowledge_sharing_score += 0.1
        
        if td.get("documentation_quality", 0) > 0:
            documentation_quality = td.get("documentation_quality", 0) / 100
        
        if td.get("onboarding_time_days", 0) > 0:
            # Convert onboarding time to efficiency score (lower time = higher efficiency)
            onboarding_days = td.get("onboarding_time_days", 0)
            if onboarding_days < 15:
                onboarding_efficiency = 0.9
            elif onboarding_days < 30:
                onboarding_efficiency = 0.7
            elif onboarding_days < 60:
                onboarding_efficiency = 0.5
            else:
                onboarding_efficiency = 0.3
        
        # Talent market factors
        hiring_difficulty = {}
        market_competitiveness = 0.7  # Default competitive market
        
        # Assess hiring difficulty for each technology
        for item in stack:
            tech_name = item.name
            expertise_required = item.expertise_required
            market_adoption = item.market_adoption
            
            # Calculate hiring difficulty (high expertise + low adoption = harder to hire)
            difficulty = expertise_required * (1 - market_adoption)
            hiring_difficulty[tech_name] = difficulty
        
        # Estimate replacement time
        estimated_replacement_time = {}
        
        for tech_name, difficulty in hiring_difficulty.items():
            # Estimate replacement time in weeks based on difficulty
            if difficulty > 0.7:
                estimated_replacement_time[tech_name] = 10  # 10 weeks
            elif difficulty > 0.4:
                estimated_replacement_time[tech_name] = 6  # 6 weeks
            else:
                estimated_replacement_time[tech_name] = 4  # 4 weeks
        
        # Generate talent recommendations
        talent_recommendations = []
        
        if key_person_dependencies > (team_size * 0.3) and team_size > 3:
            talent_recommendations.append("Reduce key person dependencies through knowledge sharing")
        
        if documentation_quality < 0.5:
            talent_recommendations.append("Improve technical documentation for knowledge transfer")
        
        if onboarding_efficiency < 0.5:
            talent_recommendations.append("Streamline onboarding process to reduce time to productivity")
        
        # Identify technologies with high hiring difficulty
        difficult_techs = [tech for tech, diff in hiring_difficulty.items() if diff > 0.7]
        if difficult_techs:
            talent_recommendations.append(f"Consider talent market constraints for: {', '.join(difficult_techs[:3])}")
        
        if senior_engineers_ratio < 0.2 and team_size > 5:
            talent_recommendations.append("Increase senior engineering representation for technical leadership")
        
        # Calculate overall talent score
        overall_score = (
            knowledge_sharing_score * 0.25 +
            documentation_quality * 0.2 +
            onboarding_efficiency * 0.15 +
            (1 - (sum(hiring_difficulty.values()) / len(hiring_difficulty) if hiring_difficulty else 0.5)) * 0.2 +
            (senior_engineers_ratio if senior_engineers_ratio <= 1 else 1) * 0.2
        )
        
        # Adjust for key person dependencies
        if team_size > 0 and key_person_dependencies > 0:
            kpd_ratio = key_person_dependencies / team_size
            overall_score -= kpd_ratio * 0.2
        
        # Ensure overall score is within bounds
        overall_score = max(0.1, min(1.0, overall_score))
        
        # Create talent assessment
        assessment = TalentAssessment(
            overall_score=overall_score,
            team_size=team_size,
            senior_engineers_ratio=senior_engineers_ratio,
            knowledge_sharing_score=knowledge_sharing_score,
            documentation_quality=documentation_quality,
            onboarding_efficiency=onboarding_efficiency,
            market_competitiveness=market_competitiveness,
            # Optional fields
            skills_coverage=skills_coverage,
            key_person_dependencies=key_person_dependencies,
            hiring_difficulty=hiring_difficulty,
            estimated_replacement_time=estimated_replacement_time,
            talent_recommendations=talent_recommendations
        )
        
        return assessment
        
    def _has_dora_data(self, td: Dict[str, Any]) -> bool:
        """
        Check if sufficient DORA metrics data is available
        
        Args:
            td: Technical data dictionary
            
        Returns:
            True if sufficient DORA data is available, False otherwise
        """
        # Check for essential DORA metric fields
        essential_fields = [
            "deployment_frequency", 
            "lead_time_for_changes", 
            "mean_time_to_recovery", 
            "change_failure_rate"
        ]
        
        # Check if at least 2 DORA metrics are available
        available_count = sum(1 for field in essential_fields if field in td)
        return available_count >= 2
    
    def _calculate_dora_metrics(self, td: Dict[str, Any]) -> DORAMetrics:
        """
        Calculate DORA metrics for engineering delivery performance
        
        Args:
            td: Technical data dictionary
            
        Returns:
            DORAMetrics object with DORA metrics assessment
        """
        # Extract DORA metrics from data or use defaults with reasonable estimates
        
        # Deployment Frequency
        deploy_freq = td.get("deployment_frequency", "").lower()
        if not deploy_freq:
            # Estimate from deployments per week or month
            deploys_per_week = td.get("deployments_per_week", 0)
            if deploys_per_week >= 5:
                deploy_freq = "Multiple deploys per day"
            elif deploys_per_week >= 1:
                deploy_freq = "Between once per day and once per week"
            elif deploys_per_week >= 0.25:
                deploy_freq = "Between once per week and once per month"
            else:
                deploy_freq = "Less than once per month"
        
        # Lead Time for Changes
        lead_time = td.get("lead_time_for_changes", "").lower()
        if not lead_time:
            # Estimate from lead time in days
            lt_days = td.get("lead_time_for_changes_days", 0)
            if lt_days <= 0.5:
                lead_time = "Less than one day"
            elif lt_days <= 7:
                lead_time = "Between one day and one week"
            elif lt_days <= 30:
                lead_time = "Between one week and one month"
            else:
                lead_time = "More than one month"
        
        # Time to Restore Service
        restore_time = td.get("time_to_restore_service", "").lower()
        if not restore_time:
            # Estimate from MTTR in minutes
            mttr_minutes = td.get("mttr_minutes", 0)
            if mttr_minutes <= 60:
                restore_time = "Less than one hour"
            elif mttr_minutes <= 1440:  # 24 hours
                restore_time = "Less than one day"
            elif mttr_minutes <= 10080:  # One week
                restore_time = "Less than one week"
            else:
                restore_time = "More than one week"
        
        # Change Failure Rate
        change_failure_rate = td.get("change_failure_rate", 0)
        if change_failure_rate > 1:
            # Convert from percentage to decimal
            change_failure_rate = change_failure_rate / 100
        
        # Additional metrics
        deployment_success_rate = 1 - change_failure_rate
        automated_test_coverage = td.get("test_coverage", 0) / 100 if td.get("test_coverage", 0) > 1 else td.get("test_coverage", 0)
        mean_time_between_failures = td.get("mtbf_hours", None)
        incident_resolution_time = td.get("mttr_minutes", None) / 60 if td.get("mttr_minutes", None) is not None else None
        
        # Determine performance category
        performance_category = "Medium"  # Default
        
        # Criteria for Elite performance
        if (deploy_freq == "Multiple deploys per day" and
            lead_time == "Less than one day" and
            restore_time == "Less than one hour" and
            change_failure_rate <= 0.15):
            performance_category = "Elite"
        
        # Criteria for High performance
        elif (deploy_freq in ["Multiple deploys per day", "Between once per day and once per week"] and
            lead_time in ["Less than one day", "Between one day and one week"] and
            restore_time in ["Less than one hour", "Less than one day"] and
            change_failure_rate <= 0.3):
            performance_category = "High"
        
        # Criteria for Low performance
        elif (deploy_freq == "Less than once per month" or
            lead_time == "More than one month" or
            restore_time == "More than one week" or
            change_failure_rate >= 0.6):
            performance_category = "Low"
        
        # Create DORA metrics
        dora_metrics = DORAMetrics(
            deployment_frequency=deploy_freq,
            lead_time_for_changes=lead_time,
            time_to_restore_service=restore_time,
            change_failure_rate=change_failure_rate,
            deployment_success_rate=deployment_success_rate,
            automated_test_coverage=automated_test_coverage,
            mean_time_between_failures=mean_time_between_failures,
            incident_resolution_time=incident_resolution_time,
            performance_category=performance_category
        )
        
        return dora_metrics
    
    def _assess_competitive_positioning(
        self, 
        td: Dict[str, Any],
        stack: List[TechStackItem]
    ) -> CompetitivePositioning:
        """
        Assess technical competitive positioning
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            
        Returns:
            CompetitivePositioning assessment
        """
        # Get sector for benchmark comparison
        sector = td.get("sector", "").lower()
        
        # Get sector benchmarks
        sector_benchmark = self.knowledge_base.get_benchmark_data(sector)
        
        # Calculate relative technical strength
        relative_strength = 0.5  # Default average strength
        
        # Compare tech stack to sector benchmarks
        if sector_benchmark.get("common_technologies"):
            benchmark_techs = set(tech.lower() for tech in sector_benchmark.get("common_technologies", []))
            stack_techs = set(item.name.lower() for item in stack)
            
            # Calculate overlap with benchmark technologies
            overlap = len(benchmark_techs.intersection(stack_techs))
            if len(benchmark_techs) > 0:
                alignment = overlap / len(benchmark_techs)
                # Higher alignment is generally good, but not always
                relative_strength += (alignment - 0.5) * 0.2
        
        # Compare architecture pattern to sector benchmarks
        if sector_benchmark.get("architecture_patterns"):
            benchmark_patterns = set(pattern.lower() for pattern in sector_benchmark.get("architecture_patterns", []))
            arch_pattern = td.get("architecture_type", "").lower()
            
            if any(pattern in arch_pattern for pattern in benchmark_patterns):
                relative_strength += 0.05
        
        # Adjust for technology maturity and innovation
        avg_maturity = sum(item.maturity_score for item in stack) / len(stack) if stack else 0.5
        
        # Balance between maturity and innovation
        if avg_maturity > 0.8:
            # Very mature stack - good for stability, potentially less innovative
            relative_strength += 0.05
        elif avg_maturity < 0.4:
            # Very new stack - potentially innovative but risky
            relative_strength -= 0.05
        
        # Adjust for scalability if available
        if td.get("scalability_score", 0) > 0:
            scalability = td.get("scalability_score", 0)
            benchmark_scalability = sector_benchmark.get("scalability_requirements", 0.7)
            
            relative_strength += (scalability - benchmark_scalability) * 0.2
        
        # Identify technical advantages
        technical_advantages = []
        
        # Check for scalability advantage
        if td.get("scalability_score", 0) > sector_benchmark.get("scalability_requirements", 0.7):
            technical_advantages.append("Superior scalability architecture")
        
        # Check for security advantage
        if td.get("security_score", 0) > sector_benchmark.get("security_requirements", 0.7):
            technical_advantages.append("Enhanced security posture")
        
        # Check for modern stack advantage
        if avg_maturity > 0.7 and td.get("technical_innovation_score", 0) > 0.7:
            technical_advantages.append("Modern, innovative technology stack")
        
        # Check for API-driven advantage
        if td.get("api_integrations_count", 0) > 5:
            technical_advantages.append("Extensive API ecosystem")
        
        # Check for data advantage
        if td.get("data_volume_tb", 0) > 100:
            technical_advantages.append("Large proprietary dataset")
        
        # Identify technical disadvantages
        technical_disadvantages = []
        
        # Check for scalability disadvantage
        if td.get("scalability_score", 0) < sector_benchmark.get("scalability_requirements", 0.7):
            technical_disadvantages.append("Below-market scalability capabilities")
        
        # Check for security disadvantage
        if td.get("security_score", 0) < sector_benchmark.get("security_requirements", 0.7):
            technical_disadvantages.append("Security gaps compared to industry standards")
        
        # Check for outdated stack disadvantage
        if avg_maturity < 0.5:
            technical_disadvantages.append("Outdated or immature technology stack")
        
        # Check for tech debt disadvantage
        if td.get("tech_debt_score", 0) < 0.5:
            technical_disadvantages.append("Significant technical debt")
        
        # Check for reliability disadvantage
        if td.get("uptime_percentage", 0) < 99.5:
            technical_disadvantages.append("Below-market reliability metrics")
        
        # Identify differentiation factors
        differentiation_factors = []
        
        # Check for unique technologies
        unique_techs = set(item.name.lower() for item in stack) - set(tech.lower() for tech in sector_benchmark.get("common_technologies", []))
        if unique_techs:
            top_unique = list(unique_techs)[:2]
            if top_unique:
                differentiation_factors.append(f"Unique technology choices: {', '.join(top_unique)}")
        
        # Check for unique architecture
        if (td.get("architecture_type", "").lower() not in 
            [p.lower() for p in sector_benchmark.get("architecture_patterns", [])]):
            differentiation_factors.append(f"Differentiated {td.get('architecture_type', '')} architecture")
        
        # Check for ML/AI differentiation
        if any(item.category == TechCategory.ML_AI for item in stack):
            differentiation_factors.append("ML/AI capabilities integration")
        
        # Check for performance differentiation
        if td.get("performance_metrics", {}).get("throughput", 0) > 0:
            differentiation_factors.append("High-performance technical infrastructure")
        
        # Identify competitive threats
        competitive_threats = []
        
        # Check for talent competition
        difficult_techs = [item.name for item in stack if item.expertise_required > 0.7 and item.market_adoption < 0.6]
        if difficult_techs:
            competitive_threats.append(f"Talent competition for specialized skills: {', '.join(difficult_techs[:2])}")
        
        # Check for obsolescence risk
        declining_techs = [item.name for item in stack if item.maturity_level in [TechMaturityLevel.DECLINING, TechMaturityLevel.LEGACY]]
        if declining_techs:
            competitive_threats.append(f"Technology obsolescence risk: {', '.join(declining_techs[:2])}")
        
        # Check for scaling competition
        if sector_benchmark.get("scalability_requirements", 0.7) > 0.8 and td.get("scalability_score", 0) < 0.8:
            competitive_threats.append("Competitors with superior scalability capabilities")
        
        # Check for innovation competition
        if td.get("technical_innovation_score", 0) < 0.6:
            competitive_threats.append("Faster-innovating competitors")
        
        # Industry benchmark position estimation
        industry_benchmark_position = None
        
        if relative_strength > 0.8:
            industry_benchmark_position = "Top 10%"
        elif relative_strength > 0.7:
            industry_benchmark_position = "Top 25%"
        elif relative_strength > 0.5:
            industry_benchmark_position = "Above Average"
        elif relative_strength > 0.3:
            industry_benchmark_position = "Average"
        else:
            industry_benchmark_position = "Below Average"
        
        # Create competitive positioning assessment
        assessment = CompetitivePositioning(
            relative_tech_strength=relative_strength,
            technical_advantages=technical_advantages,
            technical_disadvantages=technical_disadvantages,
            differentiation_factors=differentiation_factors,
            competitive_threats=competitive_threats,
            industry_benchmark_position=industry_benchmark_position
        )
        
        return assessment
    
    def _generate_technical_roadmap(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        tech_debt: TechnicalDebtAssessment,
        scalability: ScalabilityAssessment,
        security: SecurityAssessment
    ) -> TechnicalRoadmap:
        """
        Generate strategic technical roadmap
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            tech_debt: Technical debt assessment
            scalability: Scalability assessment
            security: SecurityAssessment
            
        Returns:
            TechnicalRoadmap with prioritized initiatives
        """
        # Horizon 1: Critical initiatives (0-6 months)
        horizon_1 = []
        
        # Add critical security initiatives
        for rec in security.critical_recommendations:
            horizon_1.append({
                "name": rec,
                "type": "Security",
                "priority": "Critical",
                "estimated_effort": "Medium",
                "business_impact": "High"
            })
        
        # Add critical scalability initiatives if high growth
        if td.get("user_growth_rate", 0) > 0.2 or td.get("revenue_growth_rate", 0) > 0.2:
            for bottle in scalability.bottlenecks[:2]:
                horizon_1.append({
                    "name": f"Address scalability bottleneck: {bottle}",
                    "type": "Scalability",
                    "priority": "Critical",
                    "estimated_effort": "Medium",
                    "business_impact": "High"
                })
        
        # Add critical tech debt if very high
        if tech_debt.overall_score < 0.4:
            for area in tech_debt.critical_refactoring_areas[:2]:
                horizon_1.append({
                    "name": f"Critical refactoring: {area}",
                    "type": "Technical Debt",
                    "priority": "High",
                    "estimated_effort": "Medium",
                    "business_impact": "Medium"
                })
        
        # Horizon 2: Strategic initiatives (6-18 months)
        horizon_2 = []
        
        # Add scalability initiatives
        for rec in scalability.scaling_recommendations[:2]:
            horizon_2.append({
                "name": rec,
                "type": "Scalability",
                "priority": "High",
                "estimated_effort": "High",
                "business_impact": "High"
            })
        
        # Add security initiatives
        for rec in security.high_priority_recommendations[:2]:
            horizon_2.append({
                "name": rec,
                "type": "Security",
                "priority": "High",
                "estimated_effort": "Medium",
                "business_impact": "Medium"
            })
        
        # Add tech debt initiatives
        if tech_debt.modernization_needs:
            for mod in tech_debt.modernization_needs[:2]:
                horizon_2.append({
                    "name": mod,
                    "type": "Modernization",
                    "priority": "Medium",
                    "estimated_effort": "High",
                    "business_impact": "Medium"
                })
        
        # Horizon 3: Transformative initiatives (18+ months)
        horizon_3 = []
        
        # Add architecture evolution
        arch_type = td.get("architecture_type", "").lower()
        if "monolith" in arch_type and td.get("current_users", 0) > 10000:
            horizon_3.append({
                "name": "Microservices transformation",
                "type": "Architecture",
                "priority": "Medium",
                "estimated_effort": "Very High",
                "business_impact": "High"
            })
        
        # Add data platform initiatives
        if td.get("data_volume_tb", 0) > 10:
            horizon_3.append({
                "name": "Next-generation data platform",
                "type": "Data",
                "priority": "Medium",
                "estimated_effort": "High",
                "business_impact": "High"
            })
        
        # Add AI/ML initiatives if relevant
        if "ai" in td.get("sector", "").lower() or td.get("ai_ml_relevance", 0) > 0.5:
            horizon_3.append({
                "name": "AI/ML capabilities integration",
                "type": "Innovation",
                "priority": "Medium",
                "estimated_effort": "High",
                "business_impact": "High"
            })
        
        # Implementation costs estimation
        implementation_costs = {
            "horizon_1": sum(len(horizon_1) * 2) * 50000,  # $50k per engineer-month
            "horizon_2": sum(len(horizon_2) * 3) * 50000,
            "horizon_3": sum(len(horizon_3) * 6) * 50000
        }
        
        # Resource requirements
        team_size = td.get("engineering_team_size", 5)
        resource_requirements = {
            "horizon_1": {
                "engineering_headcount": max(1, int(team_size * 0.2)),
                "duration_months": 6
            },
            "horizon_2": {
                "engineering_headcount": max(2, int(team_size * 0.3)),
                "duration_months": 12
            },
            "horizon_3": {
                "engineering_headcount": max(3, int(team_size * 0.4)),
                "duration_months": 18
            }
        }
        
        # Implementation risks
        implementation_risks = {}
        
        for i, initiative in enumerate(horizon_1 + horizon_2 + horizon_3):
            name = initiative["name"]
            initiative_type = initiative["type"]
            
            # Assign risk level based on type and effort
            if initiative["estimated_effort"] == "Very High":
                risk_level = RiskLevel.HIGH
            elif initiative["estimated_effort"] == "High":
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Adjust based on type
            if initiative_type == "Architecture":
                risk_level = RiskLevel.HIGH
            elif initiative_type == "Security" and initiative["priority"] == "Critical":
                risk_level = RiskLevel.CRITICAL
            
            implementation_risks[name] = risk_level
        
        # Migration complexity
        migration_complexity = {}
        
        for initiative in horizon_1 + horizon_2 + horizon_3:
            name = initiative["name"]
            
            # Default medium complexity
            complexity = 0.5
            
            # Adjust based on type and effort
            if "architecture" in name.lower() or "transformation" in name.lower():
                complexity = 0.8
            elif "refactoring" in name.lower():
                complexity = 0.7
            elif "security" in name.lower():
                complexity = 0.6
            
            migration_complexity[name] = complexity
        
        # Business impact assessment
        business_impact_assessment = {}
        
        for initiative in horizon_1 + horizon_2 + horizon_3:
            name = initiative["name"]
            impact = initiative["business_impact"]
            
            if impact == "High":
                business_impact_assessment[name] = {
                    "revenue_impact": "Significant",
                    "cost_reduction": "Moderate",
                    "risk_reduction": "Significant"
                }
            elif impact == "Medium":
                business_impact_assessment[name] = {
                    "revenue_impact": "Moderate",
                    "cost_reduction": "Moderate",
                    "risk_reduction": "Moderate"
                }
            else:
                business_impact_assessment[name] = {
                    "revenue_impact": "Limited",
                    "cost_reduction": "Limited",
                    "risk_reduction": "Moderate"
                }
        
        # Calculate strategic alignment score
        strategic_alignment_score = 0.7  # Default good alignment
        
        # Adjust based on alignment with business goals
        if td.get("business_goals", {}):
            goals = td.get("business_goals", {})
            alignment_scores = []
            
            # For each initiative, check alignment with business goals
            for initiative in horizon_1 + horizon_2 + horizon_3:
                init_type = initiative["type"].lower()
                
                if "growth" in goals and init_type in ["scalability", "architecture"]:
                    alignment_scores.append(0.9)
                elif "efficiency" in goals and init_type in ["technical debt", "modernization"]:
                    alignment_scores.append(0.9)
                elif "security" in goals and init_type == "security":
                    alignment_scores.append(0.9)
                else:
                    alignment_scores.append(0.5)
            
            # Average alignment score
            if alignment_scores:
                strategic_alignment_score = sum(alignment_scores) / len(alignment_scores)
        
        # Create technical roadmap
        roadmap = TechnicalRoadmap(
            horizon_1_initiatives=horizon_1,
            horizon_2_initiatives=horizon_2,
            horizon_3_initiatives=horizon_3,
            implementation_costs=implementation_costs,
            resource_requirements=resource_requirements,
            implementation_risks=implementation_risks,
            migration_complexity=migration_complexity,
            business_impact_assessment=business_impact_assessment,
            strategic_alignment_score=strategic_alignment_score
        )
        
        return roadmap
    
    def _identify_strengths(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        arch_score: float,
        scal_score: float,
        debt_score: float
    ) -> List[str]:
        """
        Identify technical strengths
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            arch_score: Architecture score
            scal_score: Scalability score
            debt_score: Technical debt score
            
        Returns:
            List of identified strengths
        """
        strengths = []
        
        # Architecture strengths
        if arch_score > 0.7:
            arch_type = td.get("architecture_type", "").lower()
            if "microservice" in arch_type:
                strengths.append("Well-designed microservice architecture enables independent scaling and development")
            elif "serverless" in arch_type:
                strengths.append("Serverless architecture provides cost efficiency and automatic scaling")
            elif "event" in arch_type:
                strengths.append("Event-driven architecture enables loose coupling and extensibility")
            else:
                strengths.append("Well-structured architecture supports business requirements and future growth")
        
        # Technology stack strengths
        modern_tech = [it for it in stack if it.maturity_score > 0.7 and it.market_adoption > 0.7]
        if len(modern_tech) >= 2:
            tech_names = ", ".join(it.name for it in modern_tech[:2])
            strengths.append(f"Modern technology stack ({tech_names}) with strong community support and maturity")
        
        # Scalability strengths
        if scal_score > 0.7:
            strengths.append("Strong scaling capabilities to support growth and traffic spikes")
        
        # Technical debt strengths
        if debt_score > 0.7:
            if td.get("test_coverage", 0) > 70:
                strengths.append(f"High test coverage ({td.get('test_coverage', 0)}%) reduces regression risk and supports refactoring")
            if td.get("has_code_reviews", False):
                strengths.append("Consistent code review process ensures code quality and knowledge sharing")
            if td.get("regular_refactoring", False):
                strengths.append("Regular refactoring practice keeps technical debt in check")
        
        # Security strengths
        if td.get("security_score", 0) > 0.7:
            strengths.append("Robust security practices protect customer data and business operations")
        
        # Operational strengths
        if td.get("uptime_percentage", 0) > 99.9:
            strengths.append(f"Excellent reliability with {td.get('uptime_percentage', 0)}% uptime")
        
        # Engineering strengths
        if td.get("engineering_team_size", 0) > 10 and td.get("senior_engineers_ratio", 0) > 0.3:
            strengths.append("Experienced engineering team with strong domain expertise")
        
        # Add default strength if none identified
        if not strengths:
            strengths.append("Technology stack is adequate for current business requirements")
        
        return strengths
    
    def _identify_weaknesses(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        arch_score: float,
        scal_score: float,
        debt_score: float
    ) -> List[str]:
        """
        Identify technical weaknesses
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            arch_score: Architecture score
            scal_score: Scalability score
            debt_score: Technical debt score
            
        Returns:
            List of identified weaknesses
        """
        weaknesses = []
        
        # Architecture weaknesses
        if arch_score < 0.5:
            arch_type = td.get("architecture_type", "").lower()
            if "monolith" in arch_type and td.get("current_users", 0) > 50000:
                weaknesses.append("Monolithic architecture limits scalability and development velocity at current scale")
            else:
                weaknesses.append("Suboptimal architecture creates maintenance challenges and limits growth potential")
        
        # Technology stack weaknesses
        niche_tech = [it for it in stack if it.market_adoption < 0.4]
        if len(niche_tech) >= 2:
            tech_names = ", ".join(it.name for it in niche_tech[:2])
            weaknesses.append(f"Reliance on niche technologies ({tech_names}) creates hiring and support challenges")
        
        high_expert = [it for it in stack if it.expertise_required > 0.8]
        if len(high_expert) >= 2:
            tech_names = ", ".join(it.name for it in high_expert[:2])
            weaknesses.append(f"Complex technology stack ({tech_names}) requires specialized expertise")
        
        # Scalability weaknesses
        if scal_score < 0.5:
            weaknesses.append("Significant architecture and infrastructure changes required to support scale")
        
        # Technical debt weaknesses
        if debt_score < 0.5:
            if td.get("test_coverage", 0) < 30:
                weaknesses.append(f"Low test coverage ({td.get('test_coverage', 0)}%) increases risk of regressions and slows development")
            if td.get("open_bugs", 0) > 50:
                weaknesses.append(f"Large backlog of {td.get('open_bugs', 0)} open bugs indicates quality issues")
            if not td.get("has_documentation", False):
                weaknesses.append("Lack of technical documentation creates knowledge silos and slows onboarding")
        
        # Security weaknesses
        if td.get("security_score", 0) < 0.5:
            weaknesses.append("Security gaps pose risk to customer data and business operations")
        
        # Operational weaknesses
        if td.get("uptime_percentage", 0) < 99.5:
            weaknesses.append(f"Below-industry-standard reliability ({td.get('uptime_percentage', 0)}% uptime)")
        
        # Talent weaknesses
        if td.get("key_person_dependencies", 0) > 2:
            weaknesses.append(f"Key person dependencies ({td.get('key_person_dependencies', 0)}) create continuity risk")
        
        # Add default weakness if none identified
        if not weaknesses:
            weaknesses.append("Technical stack may require optimization as business scales")
        
        return weaknesses
    
    def _identify_opportunities(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        scalability: ScalabilityAssessment,
        security: SecurityAssessment
    ) -> List[str]:
        """
        Identify technical opportunities
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            scalability: Scalability assessment
            security: Security assessment
            
        Returns:
            List of identified opportunities
        """
        opportunities = []
        
        # Analyze technology trends
        sector = td.get("sector", "").lower()
        tech_names = [item.name for item in stack]
        trend_analysis = self.knowledge_base.get_tech_trend_analysis(tech_names, sector)
        
        # Add technology trend opportunities
        if trend_analysis["recommended_technologies"]:
            techs = ", ".join(trend_analysis["recommended_technologies"][:2])
            opportunities.append(f"Adopt emerging technologies ({techs}) for competitive advantage")
        
        # Architecture evolution opportunities
        arch_type = td.get("architecture_type", "").lower()
        if "monolith" in arch_type and td.get("current_users", 0) > 10000:
            opportunities.append("Gradual migration to microservices architecture to improve scalability and team velocity")
        
        # Scalability opportunities
        if scalability.horizontal_scaling is False and td.get("user_growth_rate", 0) > 0.1:
            opportunities.append("Implement horizontal scaling to support projected user growth")
        
        if scalability.caching_strategy is False:
            opportunities.append("Implement comprehensive caching strategy to improve performance and reduce costs")
        
        # Cloud optimization opportunities
        if any("aws" in item.name.lower() for item in stack) and not td.get("aws_optimization", False):
            opportunities.append("Optimize AWS infrastructure for significant cost reduction")
        
        # Security opportunities
        if security.overall_score < 0.7:
            opportunities.append("Enhance security posture to meet industry best practices and compliance requirements")
        
        # Automation opportunities
        if not td.get("ci_cd", False):
            opportunities.append("Implement CI/CD pipeline for faster, more reliable deployments")
        
        # Data opportunities
        if td.get("data_volume_tb", 0) > 10 and not any(item.category == TechCategory.ANALYTICS for item in stack):
            opportunities.append("Build data analytics capabilities to extract business insights from existing data")
        
        # ML/AI opportunities
        if (td.get("data_volume_tb", 0) > 50 and 
            not any(item.category == TechCategory.ML_AI for item in stack) and
            sector in ["ecommerce", "fintech", "saas"]):
            opportunities.append("Leverage machine learning for personalization and business intelligence")
        
        # Mobile opportunities
        if not any(item.category == TechCategory.MOBILE for item in stack) and sector in ["ecommerce", "saas"]:
            opportunities.append("Develop mobile application to expand user engagement channels")
        
        # Add default opportunity if none identified
        if not opportunities:
            opportunities.append("Continue technology consolidation and optimization for operational efficiency")
        
        return opportunities
    
    def _identify_threats(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        security: SecurityAssessment,
        tech_debt: TechnicalDebtAssessment,
        operational: OperationalAssessment
    ) -> List[str]:
        """
        Identify technical threats
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            security: Security assessment
            tech_debt: Technical debt assessment
            operational: Operational assessment
            
        Returns:
            List of identified threats
        """
        threats = []
        
        # Security threats
        if security.critical_vulnerabilities_count > 0:
            threats.append(f"{security.critical_vulnerabilities_count} critical security vulnerabilities pose significant business risk")
        elif security.high_vulnerabilities_count > 0:
            threats.append(f"{security.high_vulnerabilities_count} high-severity security vulnerabilities require remediation")
        
        # Scalability threats
        if operational.uptime_percentage < 99.5 and td.get("user_growth_rate", 0) > 0.2:
            threats.append("Current reliability issues will be amplified by projected growth")
        
        # Technical debt threats
        if tech_debt.overall_score < 0.4:
            threats.append("Accumulating technical debt will increasingly impact development velocity")
        
        # Technology obsolescence threats
        legacy_tech = [item.name for item in stack if item.maturity_level in [TechMaturityLevel.DECLINING, TechMaturityLevel.LEGACY]]
        if legacy_tech:
            tech_names = ", ".join(legacy_tech[:2])
            threats.append(f"Dependency on declining technologies ({tech_names}) creates growing maintenance burden")
        
        # Talent threats
        difficult_tech = [item.name for item in stack if item.expertise_required > 0.7 and item.market_adoption < 0.5]
        if difficult_tech:
            tech_names = ", ".join(difficult_tech[:2])
            threats.append(f"Hiring challenges for specialized skills ({tech_names}) could impact growth execution")
        
        # Cloud spend threats
        if "aws" in " ".join(item.name.lower() for item in stack) and td.get("cloud_cost_growth_rate", 0) > 0.3:
            threats.append("Rapidly growing cloud costs could impact unit economics and profitability")
        
        # Compliance threats
        sector = td.get("sector", "").lower()
        if sector in ["fintech", "healthcare"] and security.overall_score < 0.6:
            threats.append("Regulatory compliance gaps could lead to penalties and business disruption")
        
        # Competitive threats
        if tech_debt.overall_score < 0.5 and td.get("revenue_growth_rate", 0) > 0.2:
            threats.append("Technical constraints could limit ability to respond to competitive pressures")
        
        # Operational threats
        if td.get("key_person_dependencies", 0) > 3 and td.get("engineering_team_size", 0) < 10:
            threats.append("Key person dependencies create significant business continuity risk")
        
        # Add default threat if none identified
        if not threats:
            threats.append("Growing technical complexity may require additional investment to maintain velocity")
        
        return threats
    
    def _generate_critical_recommendations(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        weaknesses: List[str],
        security: SecurityAssessment,
        scalability: ScalabilityAssessment
    ) -> List[str]:
        """
        Generate critical technical recommendations
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            weaknesses: Identified weaknesses
            security: Security assessment
            scalability: Scalability assessment
            
        Returns:
            List of critical recommendations
        """
        recommendations = []
        
        # Include critical security recommendations
        recommendations.extend(security.critical_recommendations)
        
        # Address critical scalability issues
        if (td.get("user_growth_rate", 0) > 0.3 and 
            scalability.overall_score < 0.6 and 
            td.get("current_users", 0) > 10000):
            recommendations.append("Urgent scalability improvements required to support growth trajectory")
        
        # Address monolith issues at scale
        if (td.get("architecture_type", "").lower() == "monolith" and 
            td.get("current_users", 0) > 100000):
            recommendations.append("Begin phased migration from monolith to microservices to address scaling limitations")
        
        # Address critical talent risks
        if td.get("key_person_dependencies", 0) > 3 and td.get("engineering_team_size", 0) < 10:
            recommendations.append("Implement knowledge sharing program to mitigate key person dependencies")
        
        # Address severe technical debt
        if td.get("test_coverage", 0) < 20 and td.get("open_bugs", 0) > 100:
            recommendations.append("Establish quality improvement program to address critical technical debt")
        
        # Limit to top 3 critical recommendations
        return recommendations[:3]
    
    def _generate_high_recommendations(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        weaknesses: List[str],
        tech_debt: TechnicalDebtAssessment
    ) -> List[str]:
        """
        Generate high-priority technical recommendations
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            weaknesses: Identified weaknesses
            tech_debt: Technical debt assessment
            
        Returns:
            List of high-priority recommendations
        """
        recommendations = []
        
        # Address technical debt
        if tech_debt.overall_score < 0.5:
            for area in tech_debt.critical_refactoring_areas[:2]:
                recommendations.append(f"Address technical debt: {area}")
        
        # Address test coverage
        if td.get("test_coverage", 0) < 40:
            recommendations.append("Implement test coverage improvement plan with clear targets")
        
        # Address documentation
        if not td.get("has_documentation", False):
            recommendations.append("Create technical documentation for critical components and onboarding")
        
        # Address code reviews
        if not td.get("has_code_reviews", False):
            recommendations.append("Implement mandatory code review process to improve quality and knowledge sharing")
        
        # Address CI/CD
        if not td.get("ci_cd", False):
            recommendations.append("Implement CI/CD pipeline to improve deployment reliability and frequency")
        
        # Limit to top 5 high-priority recommendations
        return recommendations[:5]
    
    def _generate_medium_recommendations(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        weaknesses: List[str],
        operational: OperationalAssessment
    ) -> List[str]:
        """
        Generate medium-priority technical recommendations
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            weaknesses: Identified weaknesses
            operational: Operational assessment
            
        Returns:
            List of medium-priority recommendations
        """
        recommendations = []
        
        # Include operational recommendations
        recommendations.extend(operational.operational_recommendations[:3])
        
        # Address tech stack modernization
        legacy_tech = [item for item in stack if item.maturity_level in [TechMaturityLevel.DECLINING, TechMaturityLevel.LEGACY]]
        if legacy_tech:
            tech_name = legacy_tech[0].name
            recommendations.append(f"Create migration plan for {tech_name} to reduce technical risk")
        
        # Address monitoring
        if not td.get("has_monitoring", False):
            recommendations.append("Implement comprehensive monitoring for system health and performance")
        
        # Address performance
        if not td.get("has_performance_testing", False):
            recommendations.append("Establish regular performance testing to identify bottlenecks proactively")
        
        # Limit to top 5 medium-priority recommendations
        return recommendations[:5]
    
    def _assess_risks(
        self,
        td: Dict[str, Any],
        stack: List[TechStackItem],
        security: SecurityAssessment,
        scalability: ScalabilityAssessment,
        tech_debt: TechnicalDebtAssessment,
        operational: OperationalAssessment
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive risk assessment
        
        Args:
            td: Technical data dictionary
            stack: Technology stack
            security: Security assessment
            scalability: Scalability assessment
            tech_debt: Technical debt assessment
            operational: Operational assessment
            
        Returns:
            Dictionary mapping risk areas to risk assessments
        """
        risks = {}
        
        # Security risk
        security_risk = {
            "level": RiskLevel.MEDIUM,  # Default medium risk
            "factors": [],
            "impact": "High",
            "mitigation": []
        }
        
        # Adjust security risk level
        if security.critical_vulnerabilities_count > 0:
            security_risk["level"] = RiskLevel.CRITICAL
            security_risk["factors"].append(f"{security.critical_vulnerabilities_count} critical vulnerabilities")
            security_risk["mitigation"].append("Immediately address critical vulnerabilities")
        elif security.high_vulnerabilities_count > 0:
            security_risk["level"] = RiskLevel.HIGH
            security_risk["factors"].append(f"{security.high_vulnerabilities_count} high-severity vulnerabilities")
            security_risk["mitigation"].append("Address high-severity vulnerabilities in next sprint")
        elif security.overall_score < 0.5:
            security_risk["level"] = RiskLevel.HIGH
            security_risk["factors"].append("Overall security posture below industry standards")
            security_risk["mitigation"].append("Implement security improvement program")
        elif security.overall_score > 0.7:
            security_risk["level"] = RiskLevel.LOW
        
        # Add compliance factors
        sector = td.get("sector", "").lower()
        if sector in ["fintech", "healthcare"] and security.overall_score < 0.6:
            security_risk["factors"].append("Potential compliance gaps in regulated industry")
            security_risk["mitigation"].append("Conduct compliance gap assessment and remediation")
        
        risks["security_risk"] = security_risk
        
        # Scalability risk
        scalability_risk = {
            "level": RiskLevel.MEDIUM,  # Default medium risk
            "factors": [],
            "impact": "High",
            "mitigation": []
        }
        
        # Adjust scalability risk level
        if scalability.overall_score < 0.5 and td.get("user_growth_rate", 0) > 0.2:
            scalability_risk["level"] = RiskLevel.HIGH
            scalability_risk["factors"].append("Significant scalability gaps with high growth rate")
            scalability_risk["mitigation"].append("Implement scalability improvement program")
        elif scalability.overall_score < 0.5:
            scalability_risk["level"] = RiskLevel.MEDIUM
            scalability_risk["factors"].append("Scalability limitations could impact future growth")
            scalability_risk["mitigation"].append("Address key scalability bottlenecks")
        elif scalability.overall_score > 0.7:
            scalability_risk["level"] = RiskLevel.LOW
        
        # Add bottleneck factors
        for bottleneck in scalability.bottlenecks[:2]:
            scalability_risk["factors"].append(bottleneck)
        
        # Add scaling strategy factors
        if not scalability.horizontal_scaling and not scalability.vertical_scaling:
            scalability_risk["factors"].append("No defined scaling strategy")
            scalability_risk["mitigation"].append("Develop and implement scaling strategy")
        
        risks["scalability_risk"] = scalability_risk
        
        # Technical debt risk
        tech_debt_risk = {
            "level": RiskLevel.MEDIUM,  # Default medium risk
            "factors": [],
            "impact": "Medium",
            "mitigation": []
        }
        
        # Adjust technical debt risk level
        if tech_debt.overall_score < 0.4:
            tech_debt_risk["level"] = RiskLevel.HIGH
            tech_debt_risk["factors"].append("Significant technical debt impacting development velocity")
            tech_debt_risk["mitigation"].append("Implement technical debt reduction program")
        elif tech_debt.overall_score < 0.6:
            tech_debt_risk["level"] = RiskLevel.MEDIUM
            tech_debt_risk["factors"].append("Moderate technical debt requiring attention")
            tech_debt_risk["mitigation"].append("Address critical technical debt areas")
        elif tech_debt.overall_score > 0.7:
            tech_debt_risk["level"] = RiskLevel.LOW
            tech_debt_risk["factors"].append("Low technical debt")
            tech_debt_risk["mitigation"].append("Continue monitoring technical debt")
        
        risks["tech_debt_risk"] = tech_debt_risk
        
        # Operational risk
        operational_risk = {
            "level": RiskLevel.MEDIUM,  # Default medium risk
            "factors": [],
            "impact": "High",
            "mitigation": []
        }
        
        # Adjust operational risk level
        if operational.overall_score < 0.4:
            operational_risk["level"] = RiskLevel.HIGH
            operational_risk["factors"].append("Significant operational gaps")
            operational_risk["mitigation"].append("Implement comprehensive operational improvements")
        elif operational.overall_score < 0.6:
            operational_risk["level"] = RiskLevel.MEDIUM
            operational_risk["factors"].append("Moderate operational challenges")
            operational_risk["mitigation"].append("Address key operational bottlenecks")
        elif operational.overall_score > 0.7:
            operational_risk["level"] = RiskLevel.LOW
            operational_risk["factors"].append("Strong operational practices")
            operational_risk["mitigation"].append("Continue monitoring operational metrics")
        
        # Add reliability factors
        if operational.uptime_percentage < 99.9:
            operational_risk["factors"].append(f"Uptime below 99.9% ({operational.uptime_percentage}%)")
            operational_risk["mitigation"].append("Improve system reliability and redundancy")
        
        # Add observability factors
        if operational.observability_score < 0.6:
            operational_risk["factors"].append("Limited observability")
            operational_risk["mitigation"].append("Enhance monitoring and observability")
        
        risks["operational_risk"] = operational_risk
        
        return risks
    
    def _calculate_risk_level(self, score: float, thresholds: Dict[str, float] = None) -> RiskLevel:
        """
        Calculate risk level based on score and thresholds
        
        Args:
            score: Score value between 0 and 1
            thresholds: Dictionary with threshold values for different risk levels
                       Default: {'critical': 0.3, 'high': 0.5, 'medium': 0.7}
        
        Returns:
            Appropriate RiskLevel enum value
        """
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("Score must be a number between 0 and 1")
            
        thresholds = thresholds or {
            'critical': 0.3,
            'high': 0.5,
            'medium': 0.7
        }
        
        if score < thresholds['critical']:
            return RiskLevel.CRITICAL
        elif score < thresholds['high']:
            return RiskLevel.HIGH
        elif score < thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _create_risk_assessment(self, score: float, impact: str = "Medium",
                               thresholds: Dict[str, float] = None,
                               factors: List[str] = None,
                               mitigations: List[str] = None) -> Dict[str, Any]:
        """
        Create a standardized risk assessment
        
        Args:
            score: Score value between 0 and 1
            impact: Impact level (High, Medium, Low)
            thresholds: Optional custom thresholds for risk levels
            factors: Optional list of risk factors
            mitigations: Optional list of mitigation strategies
        
        Returns:
            Dictionary containing the risk assessment
        """
        if not isinstance(impact, str) or impact not in ["High", "Medium", "Low"]:
            raise ValueError("Impact must be one of: High, Medium, Low")
            
        assessment = {
            "level": self._calculate_risk_level(score, thresholds),
            "factors": factors or [],
            "impact": impact,
            "mitigation": mitigations or []
        }
        
        return assessment