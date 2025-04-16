import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pmf_analyzer")

class PMFStage(Enum):
    """Enumeration of Product-Market Fit stages"""
    PRE_PMF = "pre-PMF"
    EARLY_PMF = "early-PMF"
    PMF = "PMF"
    SCALING = "scaling"
    
    def __str__(self) -> str:
        return self.value
        
    @classmethod
    def from_string(cls, stage_str: str) -> 'PMFStage':
        """Convert string to PMFStage enum"""
        for stage in cls:
            if stage.value == stage_str:
                return stage
        raise ValueError(f"Invalid PMF stage: {stage_str}")

@dataclass
class PMFScoreDimension:
    """Detailed dimension of the PMF score with contributing factors"""
    name: str
    score: float
    weight: float
    contributing_metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    
    @property
    def weighted_score(self) -> float:
        """Calculate the weighted contribution to overall PMF score"""
        return self.score * self.weight

@dataclass
class PMFMetrics:
    """Comprehensive Product-Market Fit analysis results"""
    # Overall scores
    pmf_score: float
    stage: PMFStage
    
    # Dimension scores
    dimensions: Dict[str, PMFScoreDimension]
    
    # Key metrics 
    retention_rate: float
    churn_rate: float
    user_growth_rate: float
    nps_score: float
    dau_mau_ratio: float
    
    # Additional analysis
    factors: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    industry_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization"""
        result = {
            "pmf_score": self.pmf_score,
            "stage": str(self.stage),
            "retention_rate": self.retention_rate,
            "churn_rate": self.churn_rate,
            "user_growth_rate": self.user_growth_rate,
            "nps_score": self.nps_score,
            "dau_mau_ratio": self.dau_mau_ratio,
            "factors": self.factors,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "industry_benchmarks": self.industry_benchmarks,
            "analyzed_at": self.analyzed_at.isoformat(),
            "version": self.version
        }
        
        # Add dimension scores
        result["dimensions"] = {}
        for name, dimension in self.dimensions.items():
            result["dimensions"][name] = {
                "score": dimension.score,
                "weight": dimension.weight,
                "contributing_metrics": dimension.contributing_metrics,
                "insights": dimension.insights
            }
        
        return result
    
    def generate_report(self) -> str:
        """Generate a formatted text report of the PMF analysis"""
        report = []
        report.append("=" * 60)
        report.append(f"PRODUCT-MARKET FIT ANALYSIS - SCORE: {self.pmf_score:.1f}/100")
        report.append("=" * 60)
        report.append(f"CURRENT STAGE: {self.stage}")
        report.append("-" * 60)
        
        # Key Metrics
        report.append("KEY METRICS:")
        report.append(f"- Retention Rate: {self.retention_rate:.1f}% (Benchmark: {self.industry_benchmarks.get('retention_rate', 0):.1f}%)")
        report.append(f"- Churn Rate: {self.churn_rate:.1f}% (Benchmark: {self.industry_benchmarks.get('churn_rate', 0):.1f}%)")
        report.append(f"- User Growth Rate: {self.user_growth_rate:.1f}% (Benchmark: {self.industry_benchmarks.get('user_growth_rate', 0):.1f}%)")
        report.append(f"- NPS Score: {self.nps_score:.1f} (Benchmark: {self.industry_benchmarks.get('nps_score', 0):.1f})")
        report.append(f"- DAU/MAU Ratio: {self.dau_mau_ratio:.2f} (Benchmark: {self.industry_benchmarks.get('dau_mau_ratio', 0):.2f})")
        report.append("-" * 60)
        
        # Dimension Scores
        report.append("DIMENSION SCORES:")
        for name, dimension in self.dimensions.items():
            report.append(f"- {name}: {dimension.score:.1f}/100 (Weight: {dimension.weight:.2f})")
            for insight in dimension.insights:
                report.append(f"  - {insight}")
        report.append("-" * 60)
        
        # Strengths
        report.append("STRENGTHS:")
        for strength in self.strengths:
            report.append(f"- {strength}")
        report.append("-" * 60)
        
        # Weaknesses
        report.append("WEAKNESSES:")
        for weakness in self.weaknesses:
            report.append(f"- {weakness}")
        report.append("-" * 60)
        
        # Recommendations
        report.append("RECOMMENDED ACTIONS:")
        for i, recommendation in enumerate(self.recommendations, 1):
            report.append(f"{i}. {recommendation}")
        report.append("-" * 60)
        
        # Analysis timestamp
        report.append(f"Analysis performed: {self.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis version: {self.version}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_radar_chart(self) -> plt.Figure:
        """Generate a radar chart visualization of PMF dimensions"""
        try:
            # Extract dimension names and scores
            dimensions = list(self.dimensions.keys())
            scores = [dim.score for dim in self.dimensions.values()]
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Compute angle for each dimension
            angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
            
            # Make the plot circular by appending the first value to the end
            scores.append(scores[0])
            angles.append(angles[0])
            dimensions.append(dimensions[0])
            
            # Plot data
            ax.plot(angles, scores, linewidth=2, linestyle='solid')
            ax.fill(angles, scores, alpha=0.25)
            
            # Set labels and styling
            ax.set_thetagrids(np.degrees(angles[:-1]), dimensions[:-1])
            ax.set_ylim(0, 100)
            ax.set_title(f"Product-Market Fit Analysis - Score: {self.pmf_score:.1f}/100", size=15, pad=20)
            ax.grid(True)
            
            # Add benchmark comparison if available
            if self.industry_benchmarks:
                benchmark_scores = []
                for dim in dimensions[:-1]:  # Exclude the duplicated last element
                    if dim.lower() == "retention":
                        benchmark_scores.append(self.industry_benchmarks.get('retention_rate', 0))
                    elif dim.lower() == "growth":
                        benchmark_scores.append(self.industry_benchmarks.get('user_growth_rate', 0))
                    elif dim.lower() == "engagement":
                        benchmark_scores.append(self.industry_benchmarks.get('dau_mau_ratio', 0) * 100)
                    elif dim.lower() == "nps":
                        # Scale NPS from -100...100 to 0...100
                        nps_benchmark = self.industry_benchmarks.get('nps_score', 0)
                        benchmark_scores.append((nps_benchmark + 100) / 2)
                    else:
                        benchmark_scores.append(50)  # Default benchmark
                
                # Append first element to close the loop
                benchmark_scores.append(benchmark_scores[0])
                
                ax.plot(angles, benchmark_scores, linewidth=1.5, linestyle='dashed', color='gray', alpha=0.7)
                ax.fill(angles, benchmark_scores, alpha=0.1, color='gray')
                plt.legend(['Your Product', 'Industry Benchmark'], loc='upper right')
            
            return fig
        except Exception as e:
            logger.error(f"Error generating radar chart: {str(e)}")
            # Return an empty figure in case of error
            return plt.figure()

class ProductMarketFitAnalyzer:
    """
    Advanced Product-Market Fit analysis engine.
    
    This class provides comprehensive analysis of a product's PMF status by examining
    multiple dimensions including retention, engagement, growth, user satisfaction,
    and qualitative feedback. The analysis produces actionable insights and
    recommendations tailored to the product's current PMF stage.
    """

    # Default dimension weights
    DEFAULT_WEIGHTS = {
        "Retention": 0.35,
        "Engagement": 0.25,
        "Growth": 0.20,
        "NPS": 0.10,
        "Qualitative": 0.10
    }

    # Industry benchmark data by sector
    INDUSTRY_BENCHMARKS = {
        "saas": {
            "retention_rate": 85,
            "churn_rate": 5,
            "user_growth_rate": 15,
            "nps_score": 40,
            "dau_mau_ratio": 0.35,
            "activation_rate": 60,
            "feature_adoption_rate": 0.65,
            "session_frequency": 12,
            "referral_rate": 0.10,
            "ltv_cac_ratio": 3.0
        },
        "fintech": {
            "retention_rate": 82,
            "churn_rate": 4.5,
            "user_growth_rate": 12,
            "nps_score": 35,
            "dau_mau_ratio": 0.45,
            "activation_rate": 55,
            "feature_adoption_rate": 0.55,
            "session_frequency": 20,
            "referral_rate": 0.08,
            "ltv_cac_ratio": 3.5
        },
        "ecommerce": {
            "retention_rate": 70,
            "churn_rate": 8.5,
            "user_growth_rate": 20,
            "nps_score": 30,
            "dau_mau_ratio": 0.15,
            "activation_rate": 45,
            "feature_adoption_rate": 0.40,
            "session_frequency": 7,
            "referral_rate": 0.12,
            "ltv_cac_ratio": 2.5
        },
        "marketplace": {
            "retention_rate": 75,
            "churn_rate": 7.0,
            "user_growth_rate": 25,
            "nps_score": 32,
            "dau_mau_ratio": 0.20,
            "activation_rate": 50,
            "feature_adoption_rate": 0.45,
            "session_frequency": 9,
            "referral_rate": 0.15,
            "ltv_cac_ratio": 2.8
        },
        "mobile": {
            "retention_rate": 60,
            "churn_rate": 12.0,
            "user_growth_rate": 30,
            "nps_score": 28,
            "dau_mau_ratio": 0.25,
            "activation_rate": 40,
            "feature_adoption_rate": 0.35,
            "session_frequency": 15,
            "referral_rate": 0.06,
            "ltv_cac_ratio": 2.0
        },
        "default": {
            "retention_rate": 75,
            "churn_rate": 7.5,
            "user_growth_rate": 20,
            "nps_score": 35,
            "dau_mau_ratio": 0.25,
            "activation_rate": 50,
            "feature_adoption_rate": 0.50,
            "session_frequency": 10,
            "referral_rate": 0.10,
            "ltv_cac_ratio": 3.0
        }
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the PMF Analyzer.
        
        Args:
            custom_weights: Optional custom dimension weights
                            (must sum to 1.0 and include all dimensions)
        """
        self.weights = self.DEFAULT_WEIGHTS.copy()
        
        if custom_weights:
            self._validate_weights(custom_weights)
            self.weights.update(custom_weights)
        
        logger.info("ProductMarketFitAnalyzer initialized with weights: %s", self.weights)
    
    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Validate custom weights"""
        # Check if all dimensions are present
        for dim in self.DEFAULT_WEIGHTS.keys():
            if dim not in weights:
                raise ValueError(f"Missing weight for dimension: {dim}")
        
        # Check if weights sum to 1.0 (with small tolerance for floating point)
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def analyze_pmf(self, company_data: Dict[str, Any], 
                   user_data: Optional[Dict[str, Any]] = None,
                   feedback_data: Optional[Dict[str, Any]] = None) -> PMFMetrics:
        """
        Analyze product-market fit using multiple data sources.
        
        Args:
            company_data: Primary company metrics data
            user_data: Optional detailed user behavior data
            feedback_data: Optional user feedback and qualitative data
            
        Returns:
            PMFMetrics object containing comprehensive PMF analysis
        """
        logger.info("Starting PMF analysis for %s", company_data.get('name', 'company'))
        
        # Initialize all needed variables
        company_metrics = self._extract_metrics(company_data)
        
        # Get industry benchmarks based on sector
        sector = company_data.get('sector', 'default').lower()
        benchmarks = self.INDUSTRY_BENCHMARKS.get(sector, self.INDUSTRY_BENCHMARKS['default'])
        
        # Calculate dimension scores
        dimensions = {}
        
        # Retention dimension
        retention_dim = self._analyze_retention_dimension(company_metrics, user_data, benchmarks)
        dimensions["Retention"] = retention_dim
        
        # Engagement dimension
        engagement_dim = self._analyze_engagement_dimension(company_metrics, user_data, benchmarks)
        dimensions["Engagement"] = engagement_dim
        
        # Growth dimension
        growth_dim = self._analyze_growth_dimension(company_metrics, benchmarks)
        dimensions["Growth"] = growth_dim
        
        # NPS dimension
        nps_dim = self._analyze_nps_dimension(company_metrics, feedback_data, benchmarks)
        dimensions["NPS"] = nps_dim
        
        # Qualitative dimension
        qualitative_dim = self._analyze_qualitative_dimension(company_metrics, feedback_data, benchmarks)
        dimensions["Qualitative"] = qualitative_dim
        
        # Calculate overall PMF score
        pmf_score = sum(dim.weighted_score for dim in dimensions.values())
        
        # Determine PMF stage
        pmf_stage = self._determine_pmf_stage(pmf_score, company_metrics)
        
        # Extract key metrics for the report
        retention_rate = company_metrics.get('retention', 0)
        if retention_rate <= 1:
            retention_rate *= 100
            
        churn_rate = company_metrics.get('churn_rate', 0)
        if churn_rate <= 1:
            churn_rate *= 100
        elif churn_rate == 0 and retention_rate > 0:
            churn_rate = 100 - retention_rate
            
        user_growth_rate = company_metrics.get('user_growth', 0)
        if user_growth_rate <= 1:
            user_growth_rate *= 100
            
        nps_score = company_metrics.get('nps', 0)
        dau_mau_ratio = company_metrics.get('dau_mau', 0)
        
        # Generate strengths, weaknesses, and recommendations
        strengths = self._identify_strengths(pmf_score, dimensions, company_metrics)
        weaknesses = self._identify_weaknesses(pmf_score, dimensions, company_metrics)
        recommendations = self._generate_recommendations(pmf_stage, weaknesses, company_metrics, sector)
        
        # Calculate additional factors
        factors = self._calculate_additional_factors(company_metrics, user_data, feedback_data)
        
        # Create PMFMetrics object
        pmf_metrics = PMFMetrics(
            pmf_score=pmf_score,
            stage=pmf_stage,
            dimensions=dimensions,
            retention_rate=retention_rate,
            churn_rate=churn_rate,
            user_growth_rate=user_growth_rate,
            nps_score=nps_score,
            dau_mau_ratio=dau_mau_ratio,
            factors=factors,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            industry_benchmarks=benchmarks
        )
        
        logger.info("PMF analysis complete. Score: %.1f, Stage: %s", pmf_score, pmf_stage)
        return pmf_metrics
    
    def _extract_metrics(self, company_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize metrics from company data"""
        metrics = {}
        
        # Handle retention/churn (ensure they're complementary)
        retention = company_data.get('retention_rate', None)
        churn = company_data.get('churn_rate', None)
        
        if retention is not None:
            metrics['retention'] = retention
            if retention <= 1:
                metrics['churn_rate'] = 1 - retention
            else:
                metrics['churn_rate'] = 100 - retention
        elif churn is not None:
            metrics['churn_rate'] = churn
            if churn <= 1:
                metrics['retention'] = 1 - churn
            else:
                metrics['retention'] = 100 - churn
        else:
            # Default values if neither is provided
            metrics['retention'] = 0.75  # 75%
            metrics['churn_rate'] = 0.25  # 25%
        
        # User growth metrics
        metrics['user_growth'] = company_data.get('user_growth_rate', 0)
        metrics['active_users'] = company_data.get('monthly_active_users', 0)
        metrics['referral_rate'] = company_data.get('referral_rate', 0)
        
        # Engagement metrics
        metrics['dau_mau'] = company_data.get('dau_mau_ratio', 0)
        metrics['session_frequency'] = company_data.get('session_frequency', 0)
        metrics['feature_adoption'] = company_data.get('feature_adoption_rate', 0)
        metrics['activation_rate'] = company_data.get('activation_rate', 0)
        
        # Satisfaction metrics
        metrics['nps'] = company_data.get('nps_score', 0)
        metrics['positive_feedback_rate'] = company_data.get('positive_feedback_rate', 0)
        
        # Business metrics
        metrics['lifetime_value_ltv'] = company_data.get('lifetime_value_ltv', 0)
        metrics['avg_revenue_per_user'] = company_data.get('avg_revenue_per_user', 0)
        metrics['paid_conversion'] = company_data.get('paid_conversion_rate', 0)
        metrics['customer_acquisition_cost'] = company_data.get('customer_acquisition_cost', 0)
        metrics['ltv_cac_ratio'] = company_data.get('ltv_cac_ratio', 0)
        
        # Support metrics
        metrics['support_ticket_volume'] = company_data.get('support_ticket_volume', 0)
        metrics['feature_request_count'] = company_data.get('feature_request_count', 0)
        metrics['support_ticket_sla_percent'] = company_data.get('support_ticket_sla_percent', 0)
        
        return metrics
    
    def _analyze_retention_dimension(self, metrics: Dict[str, float], 
                                    user_data: Optional[Dict[str, Any]], 
                                    benchmarks: Dict[str, float]) -> PMFScoreDimension:
        """Analyze retention dimension with sophisticated scoring and insights"""
        contributing_metrics = {}
        insights = []
        
        # Normalize retention rate to percentage
        retention_rate = metrics.get('retention', 0)
        if retention_rate <= 1:
            retention_rate *= 100
        
        contributing_metrics['retention_rate'] = retention_rate
        
        # Benchmark comparison
        benchmark_retention = benchmarks.get('retention_rate', 75)
        retention_vs_benchmark = retention_rate - benchmark_retention
        
        # Calculate retention score with benchmark awareness
        if retention_rate >= 90:
            retention_score = 95 + min(5, (retention_rate - 90))  # Max score 100
            insights.append(f"Exceptional retention rate of {retention_rate:.1f}% (top tier)")
        elif retention_rate >= benchmark_retention + 5:
            retention_score = 80 + (retention_rate - benchmark_retention) / 5 * 10
            insights.append(f"Strong retention rate of {retention_rate:.1f}% (above industry benchmark by {retention_vs_benchmark:.1f}%)")
        elif retention_rate >= benchmark_retention - 5:
            retention_score = 70 + (retention_rate - (benchmark_retention - 5)) / 10 * 10
            insights.append(f"Good retention rate of {retention_rate:.1f}% (near industry benchmark)")
        elif retention_rate >= 50:
            retention_score = 40 + (retention_rate - 50) / 20 * 30
            insights.append(f"Below-average retention rate of {retention_rate:.1f}% (below industry benchmark by {-retention_vs_benchmark:.1f}%)")
        else:
            retention_score = max(10, retention_rate)
            insights.append(f"Critical retention issue with only {retention_rate:.1f}% retention (significantly below benchmark)")
        
        # Cohort retention pattern analysis if user_data provided
        if user_data and 'cohort_retention' in user_data:
            cohort_data = user_data['cohort_retention']
            # Analyze cohort retention curve
            if isinstance(cohort_data, dict) and 'months' in cohort_data and 'values' in cohort_data:
                months = cohort_data['months']
                values = cohort_data['values']
                
                if len(months) >= 3 and len(values) >= 3:
                    # Check for retention curve shape
                    initial_drop = values[0] - values[1]
                    second_drop = values[1] - values[2]
                    
                    if initial_drop > 25 and second_drop < 10:
                        insights.append("High initial churn followed by loyal user retention - indicates potential onboarding issues")
                    elif initial_drop < 15 and second_drop < 10:
                        insights.append("Shallow retention curve - product shows strong stickiness after initial adoption")
                        retention_score += 5  # Bonus for good retention curve
                    elif initial_drop > 40:
                        insights.append("Very steep initial drop-off - indicates potential product value discovery issues")
                        retention_score -= 5  # Penalty for poor initial retention
        
        # Additional retention analysis based on available metrics
        ltv = metrics.get('lifetime_value_ltv', 0)
        if ltv > 0:
            contributing_metrics['ltv'] = ltv
            if ltv > 3 * metrics.get('customer_acquisition_cost', 0) and metrics.get('customer_acquisition_cost', 0) > 0:
                insights.append(f"Strong LTV:CAC ratio indicates healthy unit economics supporting retention")
                retention_score += 3  # Small bonus for good LTV:CAC
        
        return PMFScoreDimension(
            name="Retention",
            score=min(100, max(0, retention_score)),  # Ensure score is between 0-100
            weight=self.weights["Retention"],
            contributing_metrics=contributing_metrics,
            insights=insights
        )
    
    def _analyze_engagement_dimension(self, metrics: Dict[str, float], 
                                     user_data: Optional[Dict[str, Any]], 
                                     benchmarks: Dict[str, float]) -> PMFScoreDimension:
        """Analyze user engagement dimension with sophisticated scoring"""
        contributing_metrics = {}
        insights = []
        sub_scores = []
        
        # DAU/MAU ratio (daily active users / monthly active users)
        dau_mau = metrics.get('dau_mau', 0)
        if dau_mau > 0:
            contributing_metrics['dau_mau_ratio'] = dau_mau
            benchmark_dau_mau = benchmarks.get('dau_mau_ratio', 0.25)
            
            # Score DAU/MAU relative to benchmark and theoretical maximum (1.0)
            if dau_mau >= 0.5:  # Exceptional engagement
                dau_mau_score = 90 + min(10, (dau_mau - 0.5) * 100)
                insights.append(f"Exceptional DAU/MAU ratio of {dau_mau:.2f} (users engage almost daily)")
            elif dau_mau >= benchmark_dau_mau:
                dau_mau_score = 70 + (dau_mau - benchmark_dau_mau) / (0.5 - benchmark_dau_mau) * 20
                insights.append(f"Strong DAU/MAU ratio of {dau_mau:.2f} (above industry benchmark)")
            elif dau_mau >= benchmark_dau_mau * 0.7:
                dau_mau_score = 50 + (dau_mau - benchmark_dau_mau * 0.7) / (benchmark_dau_mau * 0.3) * 20
                insights.append(f"Moderate DAU/MAU ratio of {dau_mau:.2f} (approaching industry benchmark)")
            else:
                dau_mau_score = max(10, dau_mau * 250)  # Linear scaling for low values
                insights.append(f"Below-average DAU/MAU ratio of {dau_mau:.2f} (below industry benchmark)")
                
            sub_scores.append((dau_mau_score, 0.4))  # DAU/MAU has high importance
        
        # Session frequency
        session_freq = metrics.get('session_frequency', 0)
        if session_freq > 0:
            contributing_metrics['session_frequency'] = session_freq
            benchmark_frequency = benchmarks.get('session_frequency', 10)
            
            # Score session frequency with logarithmic scaling to handle wide range
            if session_freq >= benchmark_frequency * 1.5:
                freq_score = 90 + min(10, (session_freq - benchmark_frequency * 1.5) / benchmark_frequency * 10)
                insights.append(f"Exceptional session frequency of {session_freq:.1f} per month")
            elif session_freq >= benchmark_frequency:
                freq_score = 70 + (session_freq - benchmark_frequency) / (benchmark_frequency * 0.5) * 20
                insights.append(f"Strong session frequency of {session_freq:.1f} per month")
            elif session_freq >= benchmark_frequency * 0.6:
                freq_score = 50 + (session_freq - benchmark_frequency * 0.6) / (benchmark_frequency * 0.4) * 20
                insights.append(f"Moderate session frequency of {session_freq:.1f} per month")
            else:
                freq_score = max(10, min(50, session_freq / benchmark_frequency * 0.6 * 80))
                insights.append(f"Below-average session frequency of {session_freq:.1f} per month")
                
            sub_scores.append((freq_score, 0.3))  # Session frequency has medium importance
        
        # Feature adoption rate
        feature_adoption = metrics.get('feature_adoption', 0)
        if feature_adoption > 0:
            # Normalize to percentage if needed
            if feature_adoption <= 1:
                feature_adoption_pct = feature_adoption * 100
            else:
                feature_adoption_pct = feature_adoption
                
            contributing_metrics['feature_adoption_rate'] = feature_adoption_pct
            benchmark_adoption = benchmarks.get('feature_adoption_rate', 0.5) * 100
            
            # Score feature adoption
            if feature_adoption_pct >= 80:
                adoption_score = 90 + min(10, (feature_adoption_pct - 80) / 2)
                insights.append(f"Exceptional feature adoption rate of {feature_adoption_pct:.1f}%")
            elif feature_adoption_pct >= benchmark_adoption:
                adoption_score = 70 + (feature_adoption_pct - benchmark_adoption) / (80 - benchmark_adoption) * 20
                insights.append(f"Strong feature adoption rate of {feature_adoption_pct:.1f}%")
            elif feature_adoption_pct >= 30:
                adoption_score = 40 + (feature_adoption_pct - 30) / (benchmark_adoption - 30) * 30
                insights.append(f"Moderate feature adoption rate of {feature_adoption_pct:.1f}%")
            else:
                adoption_score = max(10, feature_adoption_pct * 1.5)
                insights.append(f"Limited feature adoption rate of {feature_adoption_pct:.1f}% (users not exploring full product)")
                
            sub_scores.append((adoption_score, 0.3))  # Feature adoption has medium importance
        
        # Activation rate (if available)
        activation_rate = metrics.get('activation_rate', 0)
        if activation_rate > 0:
            # Normalize to percentage if needed
            if activation_rate <= 1:
                activation_pct = activation_rate * 100
            else:
                activation_pct = activation_rate
                
            contributing_metrics['activation_rate'] = activation_pct
            benchmark_activation = benchmarks.get('activation_rate', 50)
            
            if activation_pct >= benchmark_activation + 20:
                insights.append(f"Excellent activation rate of {activation_pct:.1f}% - users quickly find value")
                # We don't add this to sub_scores as it's a leading indicator for engagement
            elif activation_pct < benchmark_activation - 15:
                insights.append(f"Below-average activation rate of {activation_pct:.1f}% - users struggling to find initial value")
        
        # Calculate overall engagement score
        if sub_scores:
            total_weight = sum(weight for _, weight in sub_scores)
            engagement_score = sum(score * weight for score, weight in sub_scores) / total_weight
        else:
            engagement_score = 50  # Default score if no engagement metrics available
            insights.append("Limited engagement data available - score based on available signals")
        
        # Look for advanced engagement patterns in user_data
        if user_data and isinstance(user_data, dict):
            # Check for power user percentage
            if 'power_user_percentage' in user_data:
                power_user_pct = user_data['power_user_percentage']
                contributing_metrics['power_user_percentage'] = power_user_pct
                
                if power_user_pct > 20:
                    insights.append(f"Strong power user base ({power_user_pct:.1f}% of users) indicates product resonance")
                    engagement_score = min(100, engagement_score + 5)
                elif power_user_pct < 5:
                    insights.append(f"Limited power user base ({power_user_pct:.1f}% of users) suggests engagement challenges")
            
            # Check for feature engagement distribution
            if 'feature_engagement' in user_data and isinstance(user_data['feature_engagement'], dict):
                feature_engagement = user_data['feature_engagement']
                if feature_engagement:
                    feature_count = len(feature_engagement)
                    engaged_features = sum(1 for v in feature_engagement.values() if v > 0.3)
                    feature_ratio = engaged_features / feature_count if feature_count > 0 else 0
                    
                    if feature_ratio > 0.7:
                        insights.append(f"Users engage with {engaged_features}/{feature_count} features - broad product usage")
                    elif feature_ratio < 0.3:
                        insights.append(f"Users only engage with {engaged_features}/{feature_count} features - concentrated usage")
        
        return PMFScoreDimension(
            name="Engagement",
            score=min(100, max(0, engagement_score)),  # Ensure score is between 0-100
            weight=self.weights["Engagement"],
            contributing_metrics=contributing_metrics,
            insights=insights
        )
    
    def _analyze_growth_dimension(self, metrics: Dict[str, float], 
                                 benchmarks: Dict[str, float]) -> PMFScoreDimension:
        """Analyze growth dimension with sophisticated scoring"""
        contributing_metrics = {}
        insights = []
        
        # User growth rate
        growth_rate = metrics.get('user_growth', 0)
        # Normalize to percentage if needed
        if growth_rate <= 1:
            growth_pct = growth_rate * 100
        else:
            growth_pct = growth_rate
            
        contributing_metrics['user_growth_rate'] = growth_pct
        benchmark_growth = benchmarks.get('user_growth_rate', 15)
        growth_vs_benchmark = growth_pct - benchmark_growth
        
        # Calculate growth score with benchmark awareness
        if growth_pct >= benchmark_growth * 2:
            growth_score = 90 + min(10, (growth_pct - benchmark_growth * 2) / benchmark_growth * 5)
            insights.append(f"Exceptional growth rate of {growth_pct:.1f}% monthly (2x industry benchmark)")
        elif growth_pct >= benchmark_growth:
            growth_score = 70 + (growth_pct - benchmark_growth) / benchmark_growth * 20
            insights.append(f"Strong growth rate of {growth_pct:.1f}% monthly (above industry benchmark)")
        elif growth_pct >= benchmark_growth * 0.5:
            growth_score = 50 + (growth_pct - benchmark_growth * 0.5) / (benchmark_growth * 0.5) * 20
            insights.append(f"Moderate growth rate of {growth_pct:.1f}% monthly (below industry benchmark)")
        elif growth_pct > 0:
            growth_score = max(20, 20 + growth_pct / (benchmark_growth * 0.5) * 30)
            insights.append(f"Limited growth rate of {growth_pct:.1f}% monthly (significantly below benchmark)")
        else:
            growth_score = max(0, 20 + growth_pct)  # Handle negative growth with penalty
            insights.append(f"Negative growth rate of {growth_pct:.1f}% monthly - urgent attention needed")
        
        # Adjust score based on additional growth signals
        
        # Referral rate
        referral_rate = metrics.get('referral_rate', 0)
        if referral_rate > 0:
            # Normalize to percentage if needed
            if referral_rate <= 1:
                referral_pct = referral_rate * 100
            else:
                referral_pct = referral_rate
                
            contributing_metrics['referral_rate'] = referral_pct
            benchmark_referral = benchmarks.get('referral_rate', 0.1) * 100
            
            if referral_pct >= benchmark_referral * 1.5:
                growth_score += 10
                insights.append(f"Strong referral rate of {referral_pct:.1f}% indicates organic growth potential")
            elif referral_pct >= benchmark_referral:
                growth_score += 5
                insights.append(f"Healthy referral rate of {referral_pct:.1f}% supports sustainable growth")
            elif referral_pct < benchmark_referral * 0.5:
                growth_score -= 5
                insights.append(f"Below-average referral rate of {referral_pct:.1f}% limits organic growth")
        
        # Viral coefficient (k-factor) if available
        viral_coefficient = metrics.get('viral_coefficient', 0)
        if viral_coefficient > 0:
            contributing_metrics['viral_coefficient'] = viral_coefficient
            
            if viral_coefficient >= 1.0:
                growth_score += 15
                insights.append(f"Viral coefficient of {viral_coefficient:.2f} enables exponential growth")
            elif viral_coefficient >= 0.7:
                growth_score += 7
                insights.append(f"Viral coefficient of {viral_coefficient:.2f} significantly reduces CAC")
            elif viral_coefficient < 0.3 and viral_coefficient > 0:
                insights.append(f"Low viral coefficient of {viral_coefficient:.2f} indicates limited word-of-mouth")
        
        # Consider absolute user base size
        active_users = metrics.get('active_users', 0)
        if active_users > 0:
            contributing_metrics['monthly_active_users'] = active_users
            
            if active_users > 1000000:
                insights.append(f"Large user base of {active_users:,} MAUs - growth from large base is significant")
                # No score adjustment - large base makes growth harder but more impressive
            elif active_users < 1000:
                insights.append(f"Small user base of {active_users:,} MAUs - growth percentage less meaningful")
                # Small adjustment to prevent overvaluing high growth percentages on tiny base
                if growth_pct > benchmark_growth:
                    growth_score = max(50, growth_score - 5)
        
        # Paid acquisition efficiency 
        ltv_cac = metrics.get('ltv_cac_ratio', 0)
        if ltv_cac > 0:
            contributing_metrics['ltv_cac_ratio'] = ltv_cac
            benchmark_ltv_cac = benchmarks.get('ltv_cac_ratio', 3.0)
            
            if ltv_cac >= benchmark_ltv_cac * 1.5:
                growth_score += 8
                insights.append(f"Excellent LTV:CAC ratio of {ltv_cac:.1f}x enables efficient growth scaling")
            elif ltv_cac >= benchmark_ltv_cac:
                growth_score += 4
                insights.append(f"Healthy LTV:CAC ratio of {ltv_cac:.1f}x supports sustainable paid acquisition")
            elif ltv_cac < 1.5:
                growth_score -= 5
                insights.append(f"Poor LTV:CAC ratio of {ltv_cac:.1f}x limits paid acquisition efficiency")
        
        return PMFScoreDimension(
            name="Growth",
            score=min(100, max(0, growth_score)),  # Ensure score is between 0-100
            weight=self.weights["Growth"],
            contributing_metrics=contributing_metrics,
            insights=insights
        )
    
    def _analyze_nps_dimension(self, metrics: Dict[str, float], 
                              feedback_data: Optional[Dict[str, Any]],
                              benchmarks: Dict[str, float]) -> PMFScoreDimension:
        """Analyze NPS and customer satisfaction dimension"""
        contributing_metrics = {}
        insights = []
        
        # NPS score (-100 to 100 scale)
        nps = metrics.get('nps', 0)
        contributing_metrics['nps_score'] = nps
        benchmark_nps = benchmarks.get('nps_score', 30)
        
        # Calculate NPS score (convert from -100...100 scale to 0...100 score)
        # Uses benchmark awareness in scoring
        if nps >= 70:
            nps_score = 90 + min(10, (nps - 70) / 3)
            insights.append(f"World-class NPS of {nps} (exceptional user satisfaction)")
        elif nps >= 50:
            nps_score = 80 + (nps - 50) / 2
            insights.append(f"Excellent NPS of {nps} (strong user satisfaction)")
        elif nps >= benchmark_nps:
            nps_score = 65 + (nps - benchmark_nps) / (50 - benchmark_nps) * 15
            insights.append(f"Good NPS of {nps} (above industry benchmark)")
        elif nps >= 0:
            nps_score = 50 + nps / benchmark_nps * 15
            insights.append(f"Average NPS of {nps} (below industry benchmark)")
        elif nps >= -30:
            nps_score = 30 + (nps + 30) / 30 * 20
            insights.append(f"Below-average NPS of {nps} (significant user satisfaction issues)")
        else:
            nps_score = max(10, 30 - (-30 - nps) / 7)
            insights.append(f"Critical NPS of {nps} (serious user satisfaction problems)")
        
        # Enhance analysis with NPS breakdown if available
        promoters_pct = None
        detractors_pct = None
        
        if feedback_data and isinstance(feedback_data, dict) and 'nps_breakdown' in feedback_data:
            nps_breakdown = feedback_data['nps_breakdown']
            if isinstance(nps_breakdown, dict):
                promoters_pct = nps_breakdown.get('promoters_percentage', None)
                detractors_pct = nps_breakdown.get('detractors_percentage', None)
                passives_pct = nps_breakdown.get('passives_percentage', None)
                
                if promoters_pct is not None:
                    contributing_metrics['promoters_percentage'] = promoters_pct
                    
                    if promoters_pct >= 50:
                        insights.append(f"Strong promoter base ({promoters_pct:.1f}%) indicates enthusiastic users")
                    elif promoters_pct <= 20 and promoters_pct is not None:
                        insights.append(f"Limited promoter base ({promoters_pct:.1f}%) restricts organic growth")
                
                if detractors_pct is not None:
                    contributing_metrics['detractors_percentage'] = detractors_pct
                    
                    if detractors_pct >= 30:
                        insights.append(f"High detractor percentage ({detractors_pct:.1f}%) risks negative word-of-mouth")
                        nps_score -= 5  # Penalty for high detractor percentage
                    elif detractors_pct <= 10:
                        insights.append(f"Low detractor percentage ({detractors_pct:.1f}%) indicates minimal user dissatisfaction")
                        nps_score += 3  # Bonus for low detractor percentage
        
        # Look for top NPS drivers in feedback data
        if feedback_data and isinstance(feedback_data, dict) and 'nps_drivers' in feedback_data:
            nps_drivers = feedback_data['nps_drivers']
            if isinstance(nps_drivers, dict):
                positive_drivers = nps_drivers.get('positive', [])
                negative_drivers = nps_drivers.get('negative', [])
                
                if positive_drivers and len(positive_drivers) > 0:
                    top_positives = positive_drivers[:2]
                    insights.append(f"Top positive NPS drivers: {', '.join(top_positives)}")
                
                if negative_drivers and len(negative_drivers) > 0:
                    top_negatives = negative_drivers[:2]
                    insights.append(f"Top negative NPS drivers: {', '.join(top_negatives)}")
        
        return PMFScoreDimension(
            name="NPS",
            score=min(100, max(0, nps_score)),  # Ensure score is between 0-100
            weight=self.weights["NPS"],
            contributing_metrics=contributing_metrics,
            insights=insights
        )
    
    def _analyze_qualitative_dimension(self, metrics: Dict[str, float], 
                                      feedback_data: Optional[Dict[str, Any]],
                                      benchmarks: Dict[str, float]) -> PMFScoreDimension:
        """Analyze qualitative feedback and support metrics"""
        contributing_metrics = {}
        insights = []
        qualitative_score = 50  # Default starting score
        
        # Positive feedback rate
        positive_feedback = metrics.get('positive_feedback_rate', 0)
        if positive_feedback > 0:
            # Normalize to percentage if needed
            if positive_feedback <= 1:
                positive_pct = positive_feedback * 100
            else:
                positive_pct = positive_feedback
                
            contributing_metrics['positive_feedback_rate'] = positive_pct
            
            if positive_pct >= 80:
                qualitative_score = 90 + min(10, (positive_pct - 80) / 2)
                insights.append(f"Exceptional positive feedback rate of {positive_pct:.1f}%")
            elif positive_pct >= 65:
                qualitative_score = 70 + (positive_pct - 65) / 15 * 20
                insights.append(f"Strong positive feedback rate of {positive_pct:.1f}%")
            elif positive_pct >= 50:
                qualitative_score = 50 + (positive_pct - 50) / 15 * 20
                insights.append(f"Moderate positive feedback rate of {positive_pct:.1f}%")
            else:
                qualitative_score = max(20, positive_pct)
                insights.append(f"Limited positive feedback rate of {positive_pct:.1f}%")
        
        # Support ticket volume relative to user base
        support_volume = metrics.get('support_ticket_volume', 0)
        active_users = metrics.get('active_users', 0)
        
        if support_volume > 0 and active_users > 0:
            ticket_ratio = support_volume / active_users
            contributing_metrics['support_ticket_ratio'] = ticket_ratio
            
            if ticket_ratio > 0.1:
                qualitative_score -= 20
                insights.append(f"High support ticket volume ({support_volume:,} for {active_users:,} users) indicates product friction")
            elif ticket_ratio > 0.05:
                qualitative_score -= 10
                insights.append(f"Above-average support volume ({support_volume:,} for {active_users:,} users)")
            elif ticket_ratio < 0.01:
                qualitative_score += 10
                insights.append(f"Low support volume ({support_volume:,} for {active_users:,} users) indicates smooth user experience")
        
        # Support SLA percentage
        support_sla = metrics.get('support_ticket_sla_percent', 0)
        if support_sla > 0:
            # Normalize to percentage if needed
            if support_sla <= 1:
                sla_pct = support_sla * 100
            else:
                sla_pct = support_sla
                
            contributing_metrics['support_sla_percentage'] = sla_pct
            
            if sla_pct >= 95:
                qualitative_score += 8
                insights.append(f"Excellent support SLA adherence of {sla_pct:.1f}%")
            elif sla_pct < 80:
                qualitative_score -= 8
                insights.append(f"Poor support SLA adherence of {sla_pct:.1f}%")
        
        # Feature request analysis
        feature_requests = metrics.get('feature_request_count', 0)
        if feature_requests > 0 and active_users > 0:
            request_ratio = feature_requests / active_users
            contributing_metrics['feature_request_ratio'] = request_ratio
            
            # Moderate feature requests is good (engaged users providing feedback)
            # Too many can indicate missing critical functionality
            # Too few can indicate disengaged users
            if 0.01 <= request_ratio <= 0.05:
                qualitative_score += 5
                insights.append(f"Healthy feature request volume indicates engaged user base")
            elif request_ratio > 0.1:
                qualitative_score -= 5
                insights.append(f"High feature request volume may indicate significant product gaps")
        
        # Analyze feedback sentiment if available
        if feedback_data and isinstance(feedback_data, dict):
            if 'sentiment' in feedback_data:
                sentiment = feedback_data['sentiment']
                if isinstance(sentiment, dict):
                    sentiment_score = sentiment.get('overall_score', None)
                    if sentiment_score is not None:
                        # Typically 0-1 scale
                        contributing_metrics['sentiment_score'] = sentiment_score
                        
                        if sentiment_score >= 0.8:
                            qualitative_score += 10
                            insights.append(f"Extremely positive sentiment in user feedback ({sentiment_score:.2f}/1.0)")
                        elif sentiment_score >= 0.6:
                            qualitative_score += 5
                            insights.append(f"Positive sentiment in user feedback ({sentiment_score:.2f}/1.0)")
                        elif sentiment_score < 0.4:
                            qualitative_score -= 10
                            insights.append(f"Negative sentiment in user feedback ({sentiment_score:.2f}/1.0)")
            
            # Analyze recurring themes in feedback
            if 'themes' in feedback_data:
                themes = feedback_data['themes']
                if isinstance(themes, dict):
                    positive_themes = themes.get('positive', [])
                    negative_themes = themes.get('negative', [])
                    
                    if positive_themes and len(positive_themes) > 0:
                        insights.append(f"Top positive feedback themes: {', '.join(positive_themes[:2])}")
                    
                    if negative_themes and len(negative_themes) > 0:
                        insights.append(f"Top negative feedback themes: {', '.join(negative_themes[:2])}")
                        
                        # Adjust score based on severity of negative themes
                        if any(t in [t.lower() for t in negative_themes[:3]] for t in 
                              ['crash', 'bug', 'broken', "doesn't work", 'unusable']):
                            qualitative_score -= 10
                            insights.append("Critical usability issues identified in feedback")
        
        return PMFScoreDimension(
            name="Qualitative",
            score=min(100, max(0, qualitative_score)),  # Ensure score is between 0-100
            weight=self.weights["Qualitative"],
            contributing_metrics=contributing_metrics,
            insights=insights
        )
    
    def _determine_pmf_stage(self, pmf_score: float, metrics: Dict[str, float]) -> PMFStage:
        """Determine the PMF stage based on the score and key metrics"""
        # Basic determination based on score
        if pmf_score >= 80:
            stage = PMFStage.SCALING
        elif pmf_score >= 65:
            stage = PMFStage.PMF
        elif pmf_score >= 50:
            stage = PMFStage.EARLY_PMF
        else:
            stage = PMFStage.PRE_PMF
        
        # Apply adjustments based on critical metrics
        
        # Retention is critical for true PMF
        retention = metrics.get('retention', 0)
        if retention <= 1:
            retention *= 100
            
        # A product can't be in PMF with very low retention
        if retention < 40 and stage in [PMFStage.PMF, PMFStage.SCALING]:
            stage = PMFStage.EARLY_PMF
            
        # A product can't be scaling with negative growth
        growth = metrics.get('user_growth', 0)
        if growth <= 0 and stage == PMFStage.SCALING:
            stage = PMFStage.PMF
        
        # Very high growth can indicate PMF even with other limitations
        if growth > 50 and stage == PMFStage.EARLY_PMF:
            stage = PMFStage.PMF
        
        return stage
    
    def _identify_strengths(self, pmf_score: float, 
                           dimensions: Dict[str, PMFScoreDimension], 
                           metrics: Dict[str, float]) -> List[str]:
        """Identify product strengths based on metrics and dimension scores"""
        strengths = []
        
        # Extract high-performing dimensions
        high_dims = [dim for name, dim in dimensions.items() if dim.score >= 70]
        high_dims.sort(key=lambda x: x.score, reverse=True)
        
        # Add dimension-based strengths
        for dim in high_dims[:3]:  # Top 3 dimensions
            if dim.name == "Retention" and dim.score >= 70:
                retention = metrics.get('retention', 0)
                if retention <= 1:
                    retention *= 100
                strengths.append(f"Strong user retention ({retention:.1f}%) indicates clear product value and user satisfaction")
            
            elif dim.name == "Engagement" and dim.score >= 70:
                dau_mau = metrics.get('dau_mau', 0)
                if dau_mau >= 0.2:
                    strengths.append(f"High user engagement (DAU/MAU ratio of {dau_mau:.2f}) demonstrates product stickiness")
                
                session_freq = metrics.get('session_frequency', 0)
                if session_freq >= 10:
                    strengths.append(f"Frequent usage ({session_freq:.1f} sessions per month) indicates strong user habits")
            
            elif dim.name == "Growth" and dim.score >= 70:
                growth = metrics.get('user_growth', 0)
                if growth <= 1:
                    growth *= 100
                strengths.append(f"Strong growth rate ({growth:.1f}% monthly) indicates market validation and product-market fit")
                
                referral = metrics.get('referral_rate', 0)
                if referral > 0.1:
                    if referral <= 1:
                        referral *= 100
                    strengths.append(f"Healthy referral rate ({referral:.1f}%) creates organic acquisition channel")
            
            elif dim.name == "NPS" and dim.score >= 70:
                nps = metrics.get('nps', 0)
                strengths.append(f"Strong NPS score of {nps} demonstrates high user satisfaction and advocacy potential")
            
            elif dim.name == "Qualitative" and dim.score >= 70:
                pos_feedback = metrics.get('positive_feedback_rate', 0)
                if pos_feedback <= 1:
                    pos_feedback *= 100
                strengths.append(f"Positive user feedback ({pos_feedback:.1f}%) confirms product value delivery")
        
        # Add PMF stage-based strength
        if pmf_score >= 80:
            strengths.append("Strong product-market fit enables focusing on growth and scaling operations")
        elif pmf_score >= 65:
            strengths.append("Established product-market fit validates core value proposition and business model")
        
        # Add business model strengths
        ltv_cac = metrics.get('ltv_cac_ratio', 0)
        if ltv_cac >= 3:
            strengths.append(f"Excellent unit economics (LTV:CAC ratio of {ltv_cac:.1f}x) enables sustainable growth")
        
        # Ensure we have at least one strength
        if not strengths:
            # Find the highest scoring dimension
            best_dim = max(dimensions.values(), key=lambda x: x.score)
            strengths.append(f"Relative strength in {best_dim.name.lower()} dimension (score: {best_dim.score:.1f}/100)")
        
        return strengths[:5]  # Return top 5 strengths
    
    def _identify_weaknesses(self, pmf_score: float, 
                            dimensions: Dict[str, PMFScoreDimension], 
                            metrics: Dict[str, float]) -> List[str]:
        """Identify product weaknesses based on metrics and dimension scores"""
        weaknesses = []
        
        # Extract low-performing dimensions
        low_dims = [dim for name, dim in dimensions.items() if dim.score < 50]
        low_dims.sort(key=lambda x: x.score)
        
        # Add dimension-based weaknesses
        for dim in low_dims[:3]:  # Top 3 weaknesses
            if dim.name == "Retention" and dim.score < 50:
                retention = metrics.get('retention', 0)
                if retention <= 1:
                    retention *= 100
                weaknesses.append(f"Low retention rate ({retention:.1f}%) indicates potential product-market fit issues")
            
            elif dim.name == "Engagement" and dim.score < 50:
                dau_mau = metrics.get('dau_mau', 0)
                if dau_mau < 0.1:
                    weaknesses.append(f"Low daily engagement (DAU/MAU ratio of {dau_mau:.2f}) suggests limited product stickiness")
                
                feature_adoption = metrics.get('feature_adoption', 0)
                if feature_adoption <= 0.4:
                    if feature_adoption <= 1:
                        feature_adoption *= 100
                    weaknesses.append(f"Limited feature adoption ({feature_adoption:.1f}%) indicates users not experiencing full product value")
            
            elif dim.name == "Growth" and dim.score < 50:
                growth = metrics.get('user_growth', 0)
                if growth <= 1:
                    growth *= 100
                weaknesses.append(f"Insufficient growth rate ({growth:.1f}% monthly) suggests limited market traction")
            
            elif dim.name == "NPS" and dim.score < 50:
                nps = metrics.get('nps', 0)
                weaknesses.append(f"Low NPS score of {nps} indicates user satisfaction issues that could limit growth")
            
            elif dim.name == "Qualitative" and dim.score < 50:
                support_volume = metrics.get('support_ticket_volume', 0)
                active_users = metrics.get('active_users', 0)
                if support_volume > 0 and active_users > 0:
                    ticket_ratio = support_volume / active_users
                    if ticket_ratio > 0.05:
                        weaknesses.append(f"High support volume ({support_volume:,} tickets) indicates user experience friction")
        
        # Add PMF stage-based weakness
        if pmf_score < 50:
            weaknesses.append("Pre-product-market fit stage requires fundamental product-value refinement")
        elif pmf_score < 65:
            weaknesses.append("Early product-market fit signals require validation and reinforcement")
        
        # Add business model weaknesses
        ltv_cac = metrics.get('ltv_cac_ratio', 0)
        if ltv_cac > 0 and ltv_cac < 2:
            weaknesses.append(f"Suboptimal unit economics (LTV:CAC ratio of {ltv_cac:.1f}x) limits sustainable growth")
        
        # Add metric-specific weaknesses not covered by dimensions
        activation = metrics.get('activation_rate', 0)
        if activation > 0 and activation < 40:
            if activation <= 1:
                activation *= 100
            weaknesses.append(f"Low activation rate ({activation:.1f}%) indicates onboarding or initial value discovery issues")
        
        # Ensure we have at least one weakness if nothing else was identified
        if not weaknesses:
            # Find the lowest scoring dimension
            worst_dim = min(dimensions.values(), key=lambda x: x.score)
            weaknesses.append(f"Relative weakness in {worst_dim.name.lower()} dimension (score: {worst_dim.score:.1f}/100)")
        
        return weaknesses[:5]  # Return top 5 weaknesses
    
    def _generate_recommendations(self, pmf_stage: PMFStage, 
                                weaknesses: List[str], 
                                metrics: Dict[str, float],
                                sector: str) -> List[str]:
        """Generate actionable recommendations based on PMF stage and identified weaknesses"""
        recommendations = []
        
        # Stage-specific recommendations
        if pmf_stage == PMFStage.PRE_PMF:
            recommendations.extend([
                "Conduct customer discovery interviews with at least 20 users to identify core value drivers and pain points",
                "Implement cohort analysis to track retention improvements as you iterate on product",
                "Focus on a single, narrow use case to achieve strong PMF within a specific customer segment",
                "Instrument detailed analytics to identify user drop-off points in key user journeys",
                "Establish weekly customer feedback sessions to rapidly iterate on product improvements"
            ])
        
        elif pmf_stage == PMFStage.EARLY_PMF:
            recommendations.extend([
                "Optimize onboarding flow to improve activation metrics and first-week retention",
                "Implement NPS surveys with open-ended feedback to identify improvement opportunities",
                "Analyze most engaged user segments to define ideal customer profile more precisely",
                "Conduct churn interviews to identify and address top reasons for user abandonment",
                "Double down on features with highest engagement metrics while deprioritizing low-usage features"
            ])
        
        elif pmf_stage == PMFStage.PMF:
            recommendations.extend([
                "Develop a systematic growth model identifying key acquisition channels for scaling",
                "Implement customer success processes to maintain high retention while scaling user base",
                "Create a referral program to leverage existing customer satisfaction for organic growth",
                "Conduct competitive analysis to ensure differentiation remains strong as you scale",
                "Begin exploring adjacent market opportunities and expansion use cases"
            ])
        
        elif pmf_stage == PMFStage.SCALING:
            recommendations.extend([
                "Systematically test and optimize multiple acquisition channels to drive efficient growth",
                "Develop customer segmentation model to tailor messaging and features for different user types",
                "Implement advanced retention strategies focused on increasing lifetime value of existing users",
                "Build scalable customer feedback systems to maintain product quality during rapid growth",
                "Explore international expansion and localization opportunities"
            ])
        
        # Add recommendations based on specific weaknesses
        for weakness in weaknesses:
            w_lower = weakness.lower()
            
            if 'retention' in w_lower:
                recommendations.append("Implement detailed cohort analysis to identify specific drop-off points in the user lifecycle")
                recommendations.append("Launch re-engagement campaigns to reactivate dormant users with targeted value propositions")
            
            elif 'engagement' in w_lower:
                recommendations.append("Analyze feature usage patterns to identify underutilized high-value features and promote them")
                recommendations.append("Implement in-app guides and tooltips to help users discover key functionality")
                
            elif 'growth' in w_lower:
                recommendations.append("Test multiple acquisition channels with small budgets to identify most efficient growth paths")
                recommendations.append("Optimize conversion funnel to reduce drop-offs between acquisition and activation")
            
            elif 'nps' in w_lower:
                recommendations.append("Collect detailed feedback from detractors to identify and address top concerns")
                recommendations.append("Segment NPS by user cohort and acquisition channel to identify problem areas")
            
            elif 'support' in w_lower:
                recommendations.append("Analyze support ticket categories to identify and fix common product friction points")
                recommendations.append("Enhance self-service support with improved documentation and in-app guidance")
        
        # Add sector-specific recommendations
        if sector == "saas":
            recommendations.append("Develop integration strategy to embed product within customer workflows and increase switching costs")
        elif sector == "fintech":
            recommendations.append("Focus on trust and security enhancements to address common adoption barriers in financial services")
        elif sector == "ecommerce":
            recommendations.append("Optimize conversion funnel with particular focus on checkout abandonment and recovery")
        elif sector == "marketplace":
            recommendations.append("Analyze supply-demand balance metrics to identify and address marketplace liquidity challenges")
        elif sector == "mobile":
            recommendations.append("Optimize app store presence and implement app rating prompts to improve discovery")
        
        # Prioritize and return a reasonable number of recommendations
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:7]  # Return top 7 recommendations
    
    def _calculate_additional_factors(self, metrics: Dict[str, float], 
                                    user_data: Optional[Dict[str, Any]], 
                                    feedback_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate additional PMF factors beyond the core dimensions"""
        factors = {}
        
        # Sean Ellis test score (if available)
        if feedback_data and 'sean_ellis_test' in feedback_data:
            sean_ellis = feedback_data['sean_ellis_test']
            if isinstance(sean_ellis, dict) and 'very_disappointed_percentage' in sean_ellis:
                very_disappointed = sean_ellis['very_disappointed_percentage']
                factors['sean_ellis_score'] = very_disappointed
                
                # Sean Ellis PMF threshold is 40% "very disappointed" if product taken away
                if very_disappointed >= 40:
                    factors['sean_ellis_pmf'] = 1
                else:
                    factors['sean_ellis_pmf'] = 0
        
        # Activation metrics
        activation_rate = metrics.get('activation_rate', 0)
        if activation_rate > 0:
            factors['activation_score'] = min(100, activation_rate * 100 if activation_rate <= 1 else activation_rate)
        
        # Frequency metrics for recurring value
        session_freq = metrics.get('session_frequency', 0)
        if session_freq > 0:
            # Convert to weekly frequency for easier interpretation
            weekly_freq = session_freq / 4.3  # Average weeks per month
            factors['weekly_frequency'] = weekly_freq
            
            # Score based on ideal frequency for product type
            # (This is a simplification - would be customized per product)
            if weekly_freq >= 3:  # Daily/near-daily use
                factors['frequency_score'] = 90
            elif weekly_freq >= 1:  # Weekly use
                factors['frequency_score'] = 70
            elif weekly_freq >= 0.5:  # Bi-weekly use
                factors['frequency_score'] = 50
            else:  # Monthly or less
                factors['frequency_score'] = 30
        
        # Growth efficiency
        ltv_cac = metrics.get('ltv_cac_ratio', 0)
        if ltv_cac > 0:
            factors['growth_efficiency'] = min(100, ltv_cac * 25)  # Score of 75 for LTV:CAC of 3
        
        # Product stickiness
        dau_mau = metrics.get('dau_mau', 0)
        if dau_mau > 0:
            factors['stickiness_score'] = min(100, dau_mau * 200)  # Score of 100 for DAU/MAU of 0.5
        
        # Customer health score
        if 'nps' in metrics and 'retention' in metrics:
            nps = metrics['nps']
            retention = metrics['retention']
            if retention <= 1:
                retention *= 100
                
            # Simple health score calculation combining NPS and retention
            # Normalized to 0-100 scale
            nps_component = (nps + 100) / 2  # Convert -100...100 to 0...100
            health_score = (nps_component * 0.4) + (retention * 0.6)
            factors['customer_health_score'] = health_score
        
        return factors

    @staticmethod
    def visualize_pmf_journey(company_data: Dict[str, Any], 
                             historical_data: List[Dict[str, Any]]) -> plt.Figure:
        """
        Generate a visualization of PMF journey over time.
        
        Args:
            company_data: Current company metrics
            historical_data: List of historical PMF measurements
            
        Returns:
            Matplotlib figure with PMF journey visualization
        """
        try:
            # Extract historical PMF scores and dates
            dates = []
            pmf_scores = []
            retention_scores = []
            growth_scores = []
            
            for data_point in historical_data:
                if 'date' in data_point and 'pmf_score' in data_point:
                    dates.append(data_point['date'])
                    pmf_scores.append(data_point['pmf_score'])
                    
                    if 'dimensions' in data_point and isinstance(data_point['dimensions'], dict):
                        dimensions = data_point['dimensions']
                        if 'Retention' in dimensions and 'score' in dimensions['Retention']:
                            retention_scores.append(dimensions['Retention']['score'])
                        else:
                            retention_scores.append(None)
                            
                        if 'Growth' in dimensions and 'score' in dimensions['Growth']:
                            growth_scores.append(dimensions['Growth']['score'])
                        else:
                            growth_scores.append(None)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot overall PMF score
            axes[0].plot(dates, pmf_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='#1f77b4')
            axes[0].set_title('Product-Market Fit Journey', fontsize=16)
            axes[0].set_ylabel('PMF Score', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Add PMF threshold lines
            axes[0].axhline(y=65, color='green', linestyle='--', alpha=0.7, label='PMF Threshold')
            axes[0].axhline(y=80, color='purple', linestyle='--', alpha=0.7, label='Scaling Threshold')
            
            # Annotate current PMF stage
            if len(pmf_scores) > 0:
                current_score = pmf_scores[-1]
                if current_score >= 80:
                    stage_text = "SCALING"
                    color = 'purple'
                elif current_score >= 65:
                    stage_text = "PMF"
                    color = 'green'
                elif current_score >= 50:
                    stage_text = "EARLY PMF"
                    color = 'orange'
                else:
                    stage_text = "PRE-PMF"
                    color = 'red'
                
                axes[0].annotate(f'Current: {stage_text}',
                             xy=(dates[-1], current_score),
                             xytext=(10, -30),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color=color),
                             color=color,
                             fontsize=12,
                             fontweight='bold')
            
            axes[0].legend()
            
            # Plot key dimension scores
            if retention_scores and growth_scores:
                valid_indices = [i for i, (r, g) in enumerate(zip(retention_scores, growth_scores)) 
                               if r is not None and g is not None]
                
                valid_dates = [dates[i] for i in valid_indices]
                valid_retention = [retention_scores[i] for i in valid_indices]
                valid_growth = [growth_scores[i] for i in valid_indices]
                
                if valid_dates and valid_retention and valid_growth:
                    axes[1].plot(valid_dates, valid_retention, marker='s', linestyle='-', label='Retention', color='#ff7f0e')
                    axes[1].plot(valid_dates, valid_growth, marker='^', linestyle='-', label='Growth', color='#2ca02c')
                    axes[1].set_ylabel('Dimension Scores', fontsize=12)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend()
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add annotations for major events if provided
            if 'events' in company_data and isinstance(company_data['events'], list):
                events = company_data['events']
                for event in events:
                    if 'date' in event and 'description' in event:
                        event_date = event['date']
                        if event_date in dates:
                            idx = dates.index(event_date)
                            axes[0].annotate(event['description'],
                                         xy=(event_date, pmf_scores[idx]),
                                         xytext=(0, 20),
                                         textcoords='offset points',
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error generating PMF journey visualization: {str(e)}")
            # Return an empty figure in case of error
            return plt.figure()
