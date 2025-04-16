"""
Acquisition Fit Assessment module for FlashDNA Infinity.

This module provides comprehensive acquisition potential analysis for startups,
evaluating their attractiveness as acquisition targets, potential synergies,
valuation ranges, and integration risks.

Usage:
    from acquisition_fit import AcquisitionFitAnalyzer
    
    analyzer = AcquisitionFitAnalyzer()
    result = analyzer.analyze_acquisition_fit(startup_data)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

# Setup logging
logger = logging.getLogger("acquisition_fit")

@dataclass
class AcquisitionSynergy:
    """Represents potential acquisition synergies with different acquirer types."""
    acquirer_type: str
    strategic_fit_score: float
    revenue_synergy: float
    cost_synergy: float
    time_to_value: int  # months
    cultural_fit_score: float
    overall_score: float
    justification: str = ""
    potential_acquirers: List[str] = field(default_factory=list)

@dataclass
class IntegrationRisk:
    """Represents potential integration risks and complexities."""
    risk_type: str
    probability: float  # 0-1
    impact: float  # 0-1
    risk_score: float  # probability * impact
    description: str
    mitigation_strategy: str = ""

@dataclass
class AcquisitionValuation:
    """Represents acquisition valuation estimations."""
    method: str
    low_value: float
    base_value: float
    high_value: float
    assumptions: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""

@dataclass
class AcquisitionFitResult:
    """Results from acquisition fit analysis."""
    overall_acquisition_readiness: float  # 0-100 score
    readiness_by_dimension: Dict[str, float]
    primary_acquisition_appeal: str
    top_synergies: List[AcquisitionSynergy]
    valuations: List[AcquisitionValuation]
    integration_risks: List[IntegrationRisk]
    recommendations: List[str]
    key_metrics: Dict[str, Any]

class AcquisitionFitAnalyzer:
    """
    Analyzes startups for acquisition fit, valuation, and potential synergies.
    Implements HPC synergy BFS–free approach combined with multi-dimensional
    acquirer fit assessment.
    """

    def __init__(self, industry_comparables: Dict[str, Any] = None):
        """
        Initialize the acquisition fit analyzer.
        
        Args:
            industry_comparables: Optional dictionary with industry acquisition data
        """
        self.industry_comparables = industry_comparables or self._load_default_comparables()
        self.acquirer_profiles = self._initialize_acquirer_profiles()
        self.acquisition_readiness_dimensions = [
            "product_readiness",
            "technology_readiness",
            "market_position",
            "financial_performance",
            "team_completeness",
            "operational_scalability",
            "ip_strength",
            "customer_diversification",
            "documentation_completeness",
            "regulatory_readiness"
        ]

    def analyze_acquisition_fit(self, startup_data: Dict[str, Any]) -> AcquisitionFitResult:
        """
        Perform comprehensive acquisition fit analysis for a startup.
        
        Args:
            startup_data: Dictionary containing startup metrics and data
            
        Returns:
            AcquisitionFitResult with comprehensive acquisition assessment
        """
        try:
            # Assess acquisition readiness dimensions
            readiness_scores = self._assess_acquisition_readiness(startup_data)
            overall_readiness = self._calculate_overall_readiness(readiness_scores)
            
            # Identify primary acquisition appeal
            primary_appeal = self._identify_primary_appeal(startup_data, readiness_scores)
            
            # Calculate valuations using different methodologies
            valuations = self._calculate_acquisition_valuations(startup_data)
            
            # Analyze potential synergies with different acquirer types
            synergies = self._analyze_acquisition_synergies(startup_data)
            
            # Identify integration risks
            integration_risks = self._identify_integration_risks(startup_data, readiness_scores)
            
            # Generate key metrics
            key_metrics = self._extract_key_acquisition_metrics(startup_data)
            
            # Generate recommendations
            recommendations = self._generate_acquisition_recommendations(
                startup_data, 
                readiness_scores, 
                synergies, 
                integration_risks
            )
            
            return AcquisitionFitResult(
                overall_acquisition_readiness=overall_readiness,
                readiness_by_dimension=readiness_scores,
                primary_acquisition_appeal=primary_appeal,
                top_synergies=sorted(synergies, key=lambda x: x.overall_score, reverse=True),
                valuations=valuations,
                integration_risks=sorted(integration_risks, key=lambda x: x.risk_score, reverse=True),
                recommendations=recommendations,
                key_metrics=key_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in acquisition fit analysis: {str(e)}")
            # Return fallback result with error information
            return self._generate_fallback_result(startup_data, str(e))

    def _assess_acquisition_readiness(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess acquisition readiness across multiple dimensions.
        
        Args:
            data: Startup data dictionary
            
        Returns:
            Dictionary mapping each dimension to a readiness score (0-100)
        """
        scores = {}
        
        # Product readiness assessment
        product_maturity = data.get("product_maturity_score", 0)
        product_quality = data.get("product_quality_score", data.get("nps_score", 0) / 2 + 50 if data.get("nps_score") is not None else 50)
        feature_completeness = data.get("feature_completeness", 50)
        
        scores["product_readiness"] = min(100, (
            product_maturity * 0.4 +
            product_quality * 0.4 + 
            feature_completeness * 0.2
        ))
        
        # Technology readiness assessment
        scalability = data.get("scalability_score", 0)
        tech_debt = 100 - data.get("technical_debt_score", 50)  # Invert: lower tech debt = higher readiness
        documentation = data.get("has_documentation", False) * 80 + 20
        system_stability = data.get("system_stability_score", data.get("uptime_percent", 0))
        
        scores["technology_readiness"] = min(100, (
            scalability * 0.3 +
            tech_debt * 0.3 +
            documentation * 0.2 +
            system_stability * 0.2
        ))
        
        # Market position assessment
        market_share = data.get("market_share", 0) * 100  # Convert fraction to percentage
        growth_rate = data.get("user_growth_rate", 0) * 100  # Convert fraction to percentage
        category_leadership = data.get("category_leadership_score", 0)
        
        scores["market_position"] = min(100, (
            market_share * 3 +  # Amplify importance of market share
            growth_rate * 0.3 +
            category_leadership * 0.4
        ))
        
        # Financial performance assessment
        revenue = data.get("monthly_revenue", 0) * 12  # Annualized
        revenue_multiple = self._get_sector_revenue_multiple(data.get("sector", ""))
        gross_margin = data.get("gross_margin_percent", 0) if data.get("gross_margin_percent", 0) <= 100 else data.get("gross_margin_percent", 0) / 100
        
        # Higher score for higher revenue, up to $10M ARR
        revenue_score = min(100, revenue / 10000000 * 100)
        
        scores["financial_performance"] = min(100, (
            revenue_score * 0.4 +
            gross_margin * 100 * 0.4 +
            min(100, data.get("runway_months", 0) * 5) * 0.2  # 20+ months runway = 100
        ))
        
        # Team completeness assessment
        has_exec_team = sum([
            data.get("has_ceo", True),  # Assume true if not specified
            data.get("has_cto", False),
            data.get("has_cfo", False),
            data.get("has_cmo", False),
            data.get("has_coo", False),
        ]) / 5 * 100
        
        team_expertise = data.get("team_score", 0)
        documentation = data.get("has_team_documentation", False) * 100
        
        scores["team_completeness"] = min(100, (
            has_exec_team * 0.5 +
            team_expertise * 0.3 +
            documentation * 0.2
        ))
        
        # Operational scalability assessment
        process_documentation = data.get("has_process_documentation", False) * 100
        automation_level = data.get("automation_level", 50)
        operational_efficiency = data.get("operational_efficiency", 50)
        
        scores["operational_scalability"] = min(100, (
            process_documentation * 0.3 +
            automation_level * 0.4 +
            operational_efficiency * 0.3
        ))
        
        # IP strength assessment
        patent_count = min(100, data.get("patent_count", 0) * 20)  # 5+ patents = 100
        trademark_count = min(100, data.get("trademark_count", 0) * 25)  # 4+ trademarks = 100
        proprietary_tech = data.get("technical_innovation_score", 0)
        
        scores["ip_strength"] = min(100, (
            patent_count * 0.4 +
            trademark_count * 0.2 +
            proprietary_tech * 0.4
        ))
        
        # Customer diversification assessment
        customer_concentration = data.get("customer_concentration", 0.2)  # Default 20% concentration
        customer_count = data.get("customer_count", data.get("monthly_active_users", 0) / 100)
        
        # Lower concentration is better (inverse relationship)
        concentration_score = max(0, 100 - customer_concentration * 100)
        
        # More customers is better, up to 100 enterprise or 10,000 consumer customers
        sector = data.get("sector", "").lower()
        is_enterprise = sector in ["enterprise", "saas", "b2b"]
        customer_count_score = min(100, (customer_count / (100 if is_enterprise else 10000)) * 100)
        
        scores["customer_diversification"] = min(100, (
            concentration_score * 0.6 +
            customer_count_score * 0.4
        ))
        
        # Documentation completeness assessment
        technical_docs = data.get("has_technical_documentation", False) * 100
        business_docs = data.get("has_business_documentation", False) * 100
        legal_docs = data.get("has_legal_documentation", False) * 100
        
        scores["documentation_completeness"] = min(100, (
            technical_docs * 0.4 +
            business_docs * 0.3 +
            legal_docs * 0.3
        ))
        
        # Regulatory readiness assessment
        compliance_status = data.get("compliance_status", 50)
        regulatory_issues = data.get("has_regulatory_issues", False)
        
        # Severe penalty for regulatory issues
        if regulatory_issues:
            compliance_status *= 0.5
            
        scores["regulatory_readiness"] = compliance_status
        
        return scores

    def _calculate_overall_readiness(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate overall acquisition readiness score based on dimension scores.
        
        Args:
            dimension_scores: Dictionary mapping dimensions to scores
            
        Returns:
            Overall acquisition readiness score (0-100)
        """
        # Define dimension weights
        weights = {
            "product_readiness": 0.15,
            "technology_readiness": 0.15,
            "market_position": 0.15,
            "financial_performance": 0.15,
            "team_completeness": 0.10,
            "operational_scalability": 0.05,
            "ip_strength": 0.10,
            "customer_diversification": 0.05,
            "documentation_completeness": 0.05,
            "regulatory_readiness": 0.05
        }
        
        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0.1)  # Default weight for unknown dimensions
            weighted_sum += score * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 50  # Default moderate score

    def _identify_primary_appeal(self, data: Dict[str, Any], readiness_scores: Dict[str, float]) -> str:
        """
        Identify the primary acquisition appeal based on startup data and readiness scores.
        
        Args:
            data: Startup data dictionary
            readiness_scores: Dictionary of readiness scores by dimension
            
        Returns:
            String describing primary acquisition appeal
        """
        # Get top dimensions
        sorted_dims = sorted(readiness_scores.items(), key=lambda x: x[1], reverse=True)
        top_dim = sorted_dims[0][0] if sorted_dims else None
        
        # Get sector, stage, and other key data
        sector = data.get("sector", "").lower()
        stage = data.get("stage", "").lower()
        
        # Check for specific patterns that would indicate primary appeal
        if "ip_strength" == top_dim or data.get("patent_count", 0) > 3:
            return "Intellectual Property & Technology"
            
        if "market_position" == top_dim or data.get("market_share", 0) > 0.1:
            return "Market Position & Customer Base"
            
        if "team_completeness" == top_dim or data.get("team_score", 0) > 75:
            return "Talent Acquisition"
            
        if "product_readiness" == top_dim and data.get("product_maturity_score", 0) > 70:
            return "Product Capabilities & Features"
        
        if "technology_readiness" == top_dim and data.get("technical_innovation_score", 0) > 70:
            return "Technical Innovation & Capabilities"
        
        # Sector-specific appeals
        if sector in ["ai", "ml", "data"]:
            return "AI/ML/Data Capabilities & Talent"
            
        if sector in ["fintech", "finance"] and readiness_scores.get("regulatory_readiness", 0) > 70:
            return "Regulatory Positioning & Compliance"
        
        if sector in ["biotech", "medtech", "pharma"] and data.get("clinical_phase", 0) > 1:
            return "Clinical Progress & Regulatory Pathway"
            
        # Default appeals based on stage
        if stage in ["pre-seed", "seed"]:
            return "Early Technology & Talent"
            
        if stage in ["series-a", "series-b"]:
            return "Market Traction & Growth Potential"
            
        if stage in ["series-c", "growth"]:
            return "Market Position & Revenue Scale"
            
        # Fallback
        return "Balanced Acquisition Target"

    def _calculate_acquisition_valuations(self, data: Dict[str, Any]) -> List[AcquisitionValuation]:
        """
        Calculate acquisition valuations using multiple methodologies.
        
        Args:
            data: Startup data dictionary
            
        Returns:
            List of AcquisitionValuation objects with different valuation methods
        """
        valuations = []
        
        # Get key financial metrics
        monthly_revenue = data.get("monthly_revenue", 0)
        annual_revenue = monthly_revenue * 12
        growth_rate = data.get("revenue_growth_rate", data.get("user_growth_rate", 0.1))
        if growth_rate > 1:  # Convert percentage to decimal if needed
            growth_rate = growth_rate / 100
            
        gross_margin = data.get("gross_margin_percent", 70)
        if gross_margin > 1:  # Convert percentage to decimal if needed
            gross_margin = gross_margin / 100
            
        # Calculate forward revenue (12 months)
        forward_revenue = annual_revenue * (1 + growth_rate)
        
        # Get sector-specific revenue multiple
        sector = data.get("sector", "").lower()
        stage = data.get("stage", "").lower()
        
        revenue_multiple = self._get_sector_revenue_multiple(sector)
        
        # Adjust multiple based on growth rate
        if growth_rate > 1.0:
            revenue_multiple *= 1.5
        elif growth_rate > 0.5:
            revenue_multiple *= 1.3
        elif growth_rate > 0.3:
            revenue_multiple *= 1.2
        elif growth_rate < 0.1:
            revenue_multiple *= 0.8
            
        # Adjust multiple based on gross margin
        if gross_margin > 0.8:
            revenue_multiple *= 1.2
        elif gross_margin > 0.7:
            revenue_multiple *= 1.1
        elif gross_margin < 0.5:
            revenue_multiple *= 0.8
            
        # Adjust for stage
        if stage in ["pre-seed", "seed"]:
            revenue_multiple *= 0.9
        elif stage in ["series-c", "growth"]:
            revenue_multiple *= 1.1
            
        # 1. Revenue Multiple Method
        base_value = annual_revenue * revenue_multiple
        valuations.append(AcquisitionValuation(
            method="Revenue Multiple",
            low_value=base_value * 0.8,
            base_value=base_value,
            high_value=base_value * 1.2,
            assumptions={
                "annual_revenue": annual_revenue,
                "revenue_multiple": revenue_multiple,
                "growth_rate": growth_rate,
                "gross_margin": gross_margin
            },
            justification=f"Based on {revenue_multiple:.1f}x revenue multiple for {sector} sector, adjusted for growth and margins"
        ))
        
        # 2. Forward Revenue Multiple Method
        forward_base = forward_revenue * revenue_multiple
        valuations.append(AcquisitionValuation(
            method="Forward Revenue Multiple",
            low_value=forward_base * 0.8,
            base_value=forward_base,
            high_value=forward_base * 1.2,
            assumptions={
                "forward_revenue": forward_revenue,
                "revenue_multiple": revenue_multiple,
                "growth_rate": growth_rate
            },
            justification=f"Based on {revenue_multiple:.1f}x multiple of projected revenue ({growth_rate*100:.1f}% growth)"
        ))
        
        # 3. Discounted Cash Flow (simplified)
        try:
            # Project 5 years of cash flows
            cf_projections = []
            r = annual_revenue
            growth_decay = 0.9  # Growth rate decays by 10% per year
            g = growth_rate
            margin = gross_margin
            
            for year in range(1, 6):
                r = r * (1 + g)
                cf = r * margin * 0.7  # Assuming 70% of gross profit becomes FCF
                cf_projections.append(cf)
                g = g * growth_decay  # Decay growth rate
                
            # Terminal value (using perpetuity growth formula)
            terminal_growth = 0.03  # Long-term growth rate
            discount_rate = 0.20  # High discount rate for startups
            terminal_value = cf_projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            
            # Discount cash flows
            dcf_value = 0
            for i, cf in enumerate(cf_projections):
                dcf_value += cf / ((1 + discount_rate) ** (i+1))
                
            # Add discounted terminal value
            dcf_value += terminal_value / ((1 + discount_rate) ** 5)
            
            valuations.append(AcquisitionValuation(
                method="Discounted Cash Flow",
                low_value=dcf_value * 0.7,  # Higher uncertainty
                base_value=dcf_value,
                high_value=dcf_value * 1.3,
                assumptions={
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "initial_growth": growth_rate,
                    "growth_decay": growth_decay
                },
                justification=f"DCF with {discount_rate*100:.1f}% discount rate and {terminal_growth*100:.1f}% terminal growth"
            ))
        except Exception as e:
            logger.warning(f"Error calculating DCF valuation: {str(e)}")
        
        # 4. Comparable Transactions
        comp_transactions = self._get_comparable_transactions(sector, stage)
        if comp_transactions:
            # Calculate average and median transaction values
            values = [t.get("value", 0) for t in comp_transactions]
            avg_value = sum(values) / len(values) if values else 0
            
            # Use sales multiple from comparables if available
            multiples = [t.get("sales_multiple", 0) for t in comp_transactions if t.get("sales_multiple")]
            comp_multiple = sum(multiples) / len(multiples) if multiples else revenue_multiple
            
            comp_value = annual_revenue * comp_multiple
            
            valuations.append(AcquisitionValuation(
                method="Comparable Transactions",
                low_value=min(comp_value * 0.8, avg_value * 0.8),
                base_value=comp_value,
                high_value=max(comp_value * 1.2, avg_value * 1.2),
                assumptions={
                    "comparable_count": len(comp_transactions),
                    "comparable_multiple": comp_multiple,
                    "annual_revenue": annual_revenue
                },
                justification=f"Based on {len(comp_transactions)} comparable {sector} acquisitions with {comp_multiple:.1f}x avg multiple"
            ))
        
        # 5. Strategic Value Method (for high IP or market share)
        if data.get("patent_count", 0) > 2 or data.get("market_share", 0) > 0.05:
            strategic_premium = 1.5  # 50% premium for strategic value
            strategic_value = max(base_value, forward_base) * strategic_premium
            
            valuations.append(AcquisitionValuation(
                method="Strategic Value",
                low_value=strategic_value * 0.8,
                base_value=strategic_value,
                high_value=strategic_value * 1.2,
                assumptions={
                    "strategic_premium": strategic_premium,
                    "base_value": max(base_value, forward_base),
                    "patent_count": data.get("patent_count", 0),
                    "market_share": data.get("market_share", 0)
                },
                justification="Includes strategic premium for IP assets and market position"
            ))
        
        return valuations

    def _analyze_acquisition_synergies(self, data: Dict[str, Any]) -> List[AcquisitionSynergy]:
        """
        Analyze potential acquisition synergies with different acquirer types.
        
        Args:
            data: Startup data dictionary
            
        Returns:
            List of AcquisitionSynergy objects for different acquirer types
        """
        synergies = []
        
        sector = data.get("sector", "").lower()
        business_model = data.get("business_model", "").lower()
        
        # Determine relevant acquirer types based on sector and business model
        relevant_acquirers = self._get_relevant_acquirer_types(sector, business_model)
        
        for acquirer_type in relevant_acquirers:
            # Get acquirer profile
            profile = self.acquirer_profiles.get(acquirer_type, {})
            
            # Calculate strategic fit
            strategic_fit = self._calculate_strategic_fit(data, profile)
            
            # Calculate potential revenue synergies
            revenue_synergy = self._calculate_revenue_synergy(data, profile)
            
            # Calculate potential cost synergies
            cost_synergy = self._calculate_cost_synergy(data, profile)
            
            # Estimate time to value
            time_to_value = self._estimate_time_to_value(data, profile)
            
            # Assess cultural fit
            cultural_fit = self._assess_cultural_fit(data, profile)
            
            # Calculate overall synergy score
            overall_score = (
                strategic_fit * 0.3 +
                revenue_synergy * 0.3 +
                cost_synergy * 0.2 +
                (100 - min(time_to_value, 36) * 100 / 36) * 0.1 +  # Lower time is better
                cultural_fit * 0.1
            )
            
            # Generate justification
            justification = self._generate_synergy_justification(
                acquirer_type, strategic_fit, revenue_synergy, cost_synergy, time_to_value, cultural_fit
            )
            
            # Identify potential specific acquirers
            potential_acquirers = self._identify_potential_acquirers(acquirer_type, data)
            
            synergies.append(AcquisitionSynergy(
                acquirer_type=acquirer_type,
                strategic_fit_score=strategic_fit,
                revenue_synergy=revenue_synergy,
                cost_synergy=cost_synergy,
                time_to_value=time_to_value,
                cultural_fit_score=cultural_fit,
                overall_score=overall_score,
                justification=justification,
                potential_acquirers=potential_acquirers
            ))
        
        return synergies

    def _identify_integration_risks(self, data: Dict[str, Any], readiness_scores: Dict[str, float]) -> List[IntegrationRisk]:
        """
        Identify potential integration risks for acquisition.
        
        Args:
            data: Startup data dictionary
            readiness_scores: Dictionary of readiness scores by dimension
            
        Returns:
            List of IntegrationRisk objects
        """
        risks = []
        
        # Technical integration risk
        tech_readiness = readiness_scores.get("technology_readiness", 50)
        tech_debt = data.get("technical_debt_score", 50)
        
        tech_risk_prob = max(0.1, 1 - (tech_readiness / 100))
        tech_risk_impact = max(0.3, 1 - (tech_debt / 100))
        
        risks.append(IntegrationRisk(
            risk_type="Technical Integration",
            probability=tech_risk_prob,
            impact=tech_risk_impact,
            risk_score=tech_risk_prob * tech_risk_impact,
            description="Risk of technical integration challenges due to architecture compatibility, tech debt, or system complexity",
            mitigation_strategy="Conduct thorough technical due diligence; plan phased integration with clear milestones"
        ))
        
        # Cultural integration risk
        team_size = data.get("employee_count", 10)
        team_tenure = data.get("avg_employee_tenure", 1)
        
        # Larger teams with longer tenure = higher cultural risk
        cultural_risk_prob = min(0.9, (team_size / 100) * 0.5 + (team_tenure / 5) * 0.5)
        cultural_risk_impact = 0.7  # Cultural issues usually have high impact
        
        risks.append(IntegrationRisk(
            risk_type="Cultural Integration",
            probability=cultural_risk_prob,
            impact=cultural_risk_impact,
            risk_score=cultural_risk_prob * cultural_risk_impact,
            description="Risk of cultural clashes, retention issues, or productivity loss during integration",
            mitigation_strategy="Develop detailed cultural integration plan; identify and retain key employees; communicate clearly"
        ))
        
        # Customer transition risk
        customer_concentration = data.get("customer_concentration", 0.2)
        customer_count = max(1, data.get("customer_count", 10))
        
        # Higher concentration = higher risk
        customer_risk_prob = min(0.9, customer_concentration * 2)
        customer_risk_impact = min(0.9, 0.3 + customer_concentration * 0.6)
        
        risks.append(IntegrationRisk(
            risk_type="Customer Transition",
            probability=customer_risk_prob,
            impact=customer_risk_impact,
            risk_score=customer_risk_prob * customer_risk_impact,
            description="Risk of customer churn or dissatisfaction during ownership transition",
            mitigation_strategy="Develop customer communication plan; maintain service continuity; offer incentives for key accounts"
        ))
        
        # Operational integration risk
        ops_readiness = readiness_scores.get("operational_scalability", 50)
        process_documentation = data.get("has_process_documentation", False)
        
        ops_risk_prob = max(0.2, 1 - (ops_readiness / 100))
        ops_risk_impact = 0.6 if not process_documentation else 0.4
        
        risks.append(IntegrationRisk(
            risk_type="Operational Integration",
            probability=ops_risk_prob,
            impact=ops_risk_impact,
            risk_score=ops_risk_prob * ops_risk_impact,
            description="Risk of operational disruption, process incompatibility, or efficiency loss",
            mitigation_strategy="Map key processes; identify integration points; maintain parallel operations initially"
        ))
        
        # Regulatory/compliance risk
        sector = data.get("sector", "").lower()
        compliance_status = data.get("compliance_status", 50) / 100
        has_regulatory_issues = data.get("has_regulatory_issues", False)
        
        # Higher risk for regulated industries
        sector_risk_factor = 1.5 if sector in ["fintech", "healthcare", "biotech", "finance"] else 1.0
        
        reg_risk_prob = max(0.1, (1 - compliance_status) * sector_risk_factor)
        if has_regulatory_issues:
            reg_risk_prob = max(0.7, reg_risk_prob)
            
        reg_risk_impact = 0.8 if sector in ["fintech", "healthcare", "biotech", "finance"] else 0.5
        
        risks.append(IntegrationRisk(
            risk_type="Regulatory & Compliance",
            probability=reg_risk_prob,
            impact=reg_risk_impact,
            risk_score=reg_risk_prob * reg_risk_impact,
            description="Risk of regulatory issues, compliance gaps, or legal complications during integration",
            mitigation_strategy="Conduct regulatory due diligence; prepare compliance transition plan; engage regulators early"
        ))
        
        # IP transition risk
        patent_count = data.get("patent_count", 0)
        ip_readiness = readiness_scores.get("ip_strength", 50)
        
        ip_risk_prob = max(0.1, min(0.8, 0.8 - (ip_readiness / 100) * 0.5))
        ip_risk_impact = min(0.9, 0.3 + (patent_count / 10) * 0.6)  # More patents = higher impact
        
        risks.append(IntegrationRisk(
            risk_type="IP & Knowledge Transfer",
            probability=ip_risk_prob,
            impact=ip_risk_impact,
            risk_score=ip_risk_prob * ip_risk_impact,
            description="Risk of IP ownership issues, knowledge transfer gaps, or loss of key IP during transition",
            mitigation_strategy="Conduct IP audit; document key knowledge; retain key technical staff"
        ))
        
        return risks

    def _extract_key_acquisition_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics relevant for acquisition assessment.
        
        Args:
            data: Startup data dictionary
            
        Returns:
            Dictionary with key acquisition metrics
        """
        metrics = {}
        
        # Financial metrics
        metrics["monthly_revenue"] = data.get("monthly_revenue", 0)
        metrics["annual_revenue"] = metrics["monthly_revenue"] * 12
        metrics["burn_rate"] = data.get("burn_rate", 0)
        
        growth_rate = data.get("revenue_growth_rate", data.get("user_growth_rate", 0))
        if growth_rate > 1:  # Convert percentage to decimal if needed
            growth_rate = growth_rate / 100
        metrics["growth_rate"] = growth_rate
        
        # User/customer metrics
        metrics["active_users"] = data.get("monthly_active_users", 0)
        metrics["customer_count"] = data.get("customer_count", metrics["active_users"] / 100)
        metrics["acquisition_cost_per_customer"] = data.get("customer_acquisition_cost", 0)
        
        # Calculate user value
        arpu = data.get("avg_revenue_per_user", 0)
        if metrics["active_users"] > 0 and arpu == 0:
            arpu = metrics["monthly_revenue"] / metrics["active_users"]
        metrics["arpu"] = arpu
        
        # Team metrics
        metrics["employee_count"] = data.get("employee_count", 0)
        metrics["cost_per_employee"] = data.get("avg_employee_cost", 0)
        if metrics["employee_count"] > 0 and metrics["cost_per_employee"] == 0:
            metrics["cost_per_employee"] = metrics["burn_rate"] * 0.7 / metrics["employee_count"]
        
        # IP metrics
        metrics["patent_count"] = data.get("patent_count", 0)
        metrics["trademark_count"] = data.get("trademark_count", 0)
        
        # Market position
        metrics["market_share"] = data.get("market_share", 0)
        metrics["competitor_count"] = data.get("competitor_count", 5)
        
        # Calculated metrics
        gross_margin = data.get("gross_margin_percent", 70)
        if gross_margin > 1:  # Convert percentage to decimal if needed
            gross_margin = gross_margin / 100
        metrics["gross_margin"] = gross_margin
        
        # Rule of 40 score (growth % + profit margin %)
        profit_margin = data.get("operating_margin_percent", -20)
        if profit_margin > 1 or profit_margin < -1:  # Convert percentage to decimal if needed
            profit_margin = profit_margin / 100
            
        metrics["rule_of_40_score"] = (growth_rate * 100) + (profit_margin * 100)
        
        # Efficiency score (revenue growth / burn rate)
        if metrics["burn_rate"] > 0:
            metrics["capital_efficiency"] = metrics["monthly_revenue"] * growth_rate / metrics["burn_rate"]
        else:
            metrics["capital_efficiency"] = 0
            
        return metrics

    def _generate_acquisition_recommendations(
        self, 
        data: Dict[str, Any],
        readiness_scores: Dict[str, float],
        synergies: List[AcquisitionSynergy],
        integration_risks: List[IntegrationRisk]
    ) -> List[str]:
        """
        Generate actionable recommendations for improving acquisition readiness.
        
        Args:
            data: Startup data dictionary
            readiness_scores: Dictionary of readiness scores by dimension
            synergies: List of AcquisitionSynergy objects
            integration_risks: List of IntegrationRisk objects
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Identify weakest readiness dimensions
        sorted_dims = sorted(readiness_scores.items(), key=lambda x: x[1])
        weak_dims = sorted_dims[:3] if len(sorted_dims) >= 3 else sorted_dims
        
        # Recommendations for improving weak dimensions
        for dim, score in weak_dims:
            if score < 60:  # Only recommend improvements for dimensions below 60
                if dim == "product_readiness":
                    recommendations.append("Improve product maturity through feature completion and quality improvements")
                elif dim == "technology_readiness":
                    recommendations.append("Reduce technical debt and improve architecture documentation for smoother integration")
                elif dim == "market_position":
                    recommendations.append("Focus on market share growth in a well-defined niche to enhance acquisition appeal")
                elif dim == "financial_performance":
                    recommendations.append("Improve unit economics and demonstrate path to profitability to increase valuation")
                elif dim == "team_completeness":
                    recommendations.append("Build out executive team with key roles to reduce acquirer's post-acquisition burden")
                elif dim == "operational_scalability":
                    recommendations.append("Document key processes and implement automation to ease operational integration")
                elif dim == "ip_strength":
                    recommendations.append("Formalize IP protection strategy through patents or trade secrets to increase strategic value")
                elif dim == "customer_diversification":
                    recommendations.append("Reduce customer concentration risk by expanding customer base or diversifying revenue streams")
                elif dim == "documentation_completeness":
                    recommendations.append("Prepare comprehensive documentation of technology, business, and legal aspects for due diligence")
                elif dim == "regulatory_readiness":
                    recommendations.append("Address regulatory compliance gaps to reduce integration complexity and risk")
        
        # Target acquirer recommendations
        if synergies:
            top_synergy = max(synergies, key=lambda x: x.overall_score)
            recommendations.append(
                f"Focus acquisition positioning toward {top_synergy.acquirer_type} acquirers where strategic fit is strongest"
            )
            
            if top_synergy.overall_score > 70:
                recommendations.append(
                    f"Develop relationships with potential acquirers in the {top_synergy.acquirer_type} category to build familiarity"
                )
        
        # Risk mitigation recommendations
        if integration_risks:
            high_risks = [r for r in integration_risks if r.risk_score > 0.4]
            for risk in high_risks[:2]:  # Address top 2 risks
                recommendations.append(f"Mitigate {risk.risk_type.lower()} risk: {risk.mitigation_strategy}")
        
        # Valuation improvement recommendations
        sector = data.get("sector", "").lower()
        stage = data.get("stage", "").lower()
        
        if stage in ["pre-seed", "seed", "series-a"]:
            recommendations.append("Focus on growth metrics over profitability to maximize acquisition interest at this stage")
            
        if sector in ["saas", "software"]:
            recommendations.append("Improve key SaaS metrics like net retention and LTV:CAC ratio to increase valuation multiples")
        elif sector in ["marketplace", "platform"]:
            recommendations.append("Demonstrate network effects through engagement metrics to justify premium valuation")
        elif sector in ["ai", "ml"]:
            recommendations.append("Quantify data moat and algorithmic advantages to highlight strategic acquisition value")
            
        # Timing recommendations
        if readiness_scores.get("financial_performance", 0) < 60 and data.get("runway_months", 12) > 9:
            recommendations.append("Delay acquisition positioning until reaching higher revenue scale or growth benchmarks")
        
        # Add sector-specific recommendations
        sector_recs = self._get_sector_specific_recommendations(data)
        recommendations.extend(sector_recs)
        
        # Limit to top 10 recommendations
        return recommendations[:10]

    def _generate_fallback_result(self, data: Dict[str, Any], error_msg: str) -> AcquisitionFitResult:
        """
        Generate a fallback result when analysis fails.
        
        Args:
            data: Startup data dictionary
            error_msg: Error message from the exception
            
        Returns:
            Basic AcquisitionFitResult with error information
        """
        logger.warning(f"Generating fallback acquisition fit result due to error: {error_msg}")
        
        sector = data.get("sector", "unknown")
        
        return AcquisitionFitResult(
            overall_acquisition_readiness=50.0,
            readiness_by_dimension={dim: 50.0 for dim in self.acquisition_readiness_dimensions},
            primary_acquisition_appeal=f"Analysis Error - Using {sector} defaults",
            top_synergies=[
                AcquisitionSynergy(
                    acquirer_type="Strategic",
                    strategic_fit_score=50.0,
                    revenue_synergy=50.0,
                    cost_synergy=50.0,
                    time_to_value=12,
                    cultural_fit_score=50.0,
                    overall_score=50.0,
                    justification=f"Default analysis due to error: {error_msg}"
                )
            ],
            valuations=[
                AcquisitionValuation(
                    method="Revenue Multiple (Default)",
                    low_value=data.get("monthly_revenue", 50000) * 12 * 3,
                    base_value=data.get("monthly_revenue", 50000) * 12 * 4,
                    high_value=data.get("monthly_revenue", 50000) * 12 * 5,
                    justification="Default fallback valuation"
                )
            ],
            integration_risks=[
                IntegrationRisk(
                    risk_type="Technical Integration",
                    probability=0.5,
                    impact=0.5,
                    risk_score=0.25,
                    description="Default risk assessment",
                    mitigation_strategy="Conduct thorough due diligence"
                )
            ],
            recommendations=[
                "Error in analysis - review input data for completeness",
                "Ensure financial metrics are properly provided",
                "Rerun analysis with complete data set"
            ],
            key_metrics={
                "monthly_revenue": data.get("monthly_revenue", 0),
                "active_users": data.get("monthly_active_users", 0),
                "employee_count": data.get("employee_count", 0),
                "error": error_msg
            }
        )

    def _get_sector_revenue_multiple(self, sector: str) -> float:
        """
        Get typical revenue multiple for a specific sector.
        
        Args:
            sector: Sector string
            
        Returns:
            Revenue multiple as a float
        """
        sector = sector.lower()
        
        # Define default multiples by sector
        multiples = {
            "saas": 8,
            "enterprise": 7,
            "fintech": 6,
            "marketplace": 5,
            "ecommerce": 2,
            "ai": 10,
            "ml": 10,
            "crypto": 5,
            "biotech": 12,
            "hardware": 3,
            "consumer": 4,
            "mobile": 5,
            "social": 6,
            "gaming": 4,
            "edtech": 5,
            "health": 6,
            "deeptech": 8
        }
        
        # Check for exact match
        if sector in multiples:
            return multiples[sector]
        
        # Check for partial matches
        for s, multiple in multiples.items():
            if s in sector or sector in s:
                return multiple
        
        # Default multiple
        return 5.0

    def _get_comparable_transactions(self, sector: str, stage: str) -> List[Dict[str, Any]]:
        """
        Get comparable acquisition transactions for a sector and stage.
        
        Args:
            sector: Sector string
            stage: Stage string
            
        Returns:
            List of comparable transaction dictionaries
        """
        sector = sector.lower()
        stage = stage.lower()
        
        # Try to find exact sector match
        if sector in self.industry_comparables:
            comps = self.industry_comparables[sector]
        else:
            # Find closest sector
            comps = []
            for s, transactions in self.industry_comparables.items():
                if s in sector or sector in s:
                    comps.extend(transactions)
            
            # If still no matches, use "other" category
            if not comps and "other" in self.industry_comparables:
                comps = self.industry_comparables["other"]
        
        # Filter by stage if available
        if stage:
            stage_comps = [c for c in comps if c.get("stage", "").lower() == stage]
            if stage_comps:
                return stage_comps
                
            # Try broader stage matching
            if stage in ["pre-seed", "seed"]:
                broader = [c for c in comps if c.get("stage", "").lower() in ["pre-seed", "seed"]]
            elif stage in ["series-a", "series-b"]:
                broader = [c for c in comps if c.get("stage", "").lower() in ["series-a", "series-b"]]
            elif stage in ["series-c", "growth", "pre-ipo"]:
                broader = [c for c in comps if c.get("stage", "").lower() in ["series-c", "growth", "pre-ipo"]]
                
            if broader:
                return broader
        
        # Return all comps if we couldn't filter effectively
        return comps

    def _get_relevant_acquirer_types(self, sector: str, business_model: str) -> List[str]:
        """
        Determine relevant acquirer types based on sector and business model.
        
        Args:
            sector: Sector string
            business_model: Business model string
            
        Returns:
            List of relevant acquirer type strings
        """
        sector = sector.lower()
        business_model = business_model.lower()
        
        # Start with all acquirer types
        all_types = list(self.acquirer_profiles.keys())
        
        # Default acquirers that are almost always relevant
        relevant = ["Strategic", "Private Equity"]
        
        # Add sector-specific acquirers
        if sector in ["saas", "enterprise", "software"]:
            relevant.extend(["Big Tech", "Enterprise Software", "Cloud Provider"])
        
        elif sector in ["fintech", "finance", "banking"]:
            relevant.extend(["Financial Institution", "Big Tech"])
        
        elif sector in ["ecommerce", "retail", "d2c"]:
            relevant.extend(["Retail Giant", "Marketplace"])
        
        elif sector in ["ai", "ml", "data"]:
            relevant.extend(["Big Tech", "Enterprise Software", "Cloud Provider"])
        
        elif sector in ["biotech", "pharma", "medical"]:
            relevant.extend(["Pharmaceutical", "Healthcare"])
        
        elif sector in ["marketplace", "platform"]:
            relevant.extend(["Marketplace", "Big Tech"])
        
        elif sector in ["crypto", "blockchain"]:
            relevant.extend(["Crypto/Blockchain", "Financial Institution"])
        
        elif sector in ["media", "content", "entertainment"]:
            relevant.extend(["Media Conglomerate", "Big Tech"])
        
        # Add business model specific acquirers
        if "subscription" in business_model or "saas" in business_model:
            if "Enterprise Software" not in relevant:
                relevant.append("Enterprise Software")
        
        if "marketplace" in business_model or "platform" in business_model:
            if "Marketplace" not in relevant:
                relevant.append("Marketplace")
        
        if "hardware" in business_model:
            if "Manufacturing" not in relevant:
                relevant.append("Manufacturing")
        
        # Return all unique relevant acquirer types that exist in our profiles
        return list(set([r for r in relevant if r in all_types]))

    def _calculate_strategic_fit(self, data: Dict[str, Any], acquirer_profile: Dict[str, Any]) -> float:
        """
        Calculate strategic fit score for a particular acquirer type.
        
        Args:
            data: Startup data dictionary
            acquirer_profile: Acquirer type profile dictionary
            
        Returns:
            Strategic fit score (0-100)
        """
        # Default moderate score
        if not acquirer_profile:
            return 50
        
        # Get strategic priorities and startup characteristics
        priorities = acquirer_profile.get("strategic_priorities", [])
        fit_scores = []
        
        for priority in priorities:
            if priority == "market_expansion":
                # Higher score for startups with market share in new segments
                market_share = data.get("market_share", 0)
                score = min(100, market_share * 500)  # 20% market share = 100 score
                fit_scores.append((score, 0.5))  # High weight
                
            elif priority == "technology_acquisition":
                # Higher score for innovation and IP
                tech_score = data.get("technical_innovation_score", 0)
                patent_count = data.get("patent_count", 0)
                score = min(100, tech_score * 0.6 + patent_count * 10 * 0.4)
                fit_scores.append((score, 0.5))  # High weight
                
            elif priority == "talent_acquisition":
                # Higher score for strong teams
                team_score = data.get("team_score", 0)
                engineering_ratio = data.get("tech_talent_ratio", 0.5)
                score = min(100, team_score * 0.7 + engineering_ratio * 100 * 0.3)
                fit_scores.append((score, 0.4))  # Medium-high weight
                
            elif priority == "product_expansion":
                # Higher score for mature products
                product_score = data.get("product_maturity_score", 0)
                score = product_score
                fit_scores.append((score, 0.4))  # Medium-high weight
                
            elif priority == "vertical_integration":
                # Higher score for complementary value chain
                supply_chain_pos = acquirer_profile.get("supply_chain_position", "middle")
                startup_pos = data.get("supply_chain_position", "middle")
                
                # Higher fit for adjacent positions in supply chain
                if supply_chain_pos == "upstream" and startup_pos == "middle":
                    score = 80
                elif supply_chain_pos == "middle" and startup_pos in ["upstream", "downstream"]:
                    score = 80
                elif supply_chain_pos == "downstream" and startup_pos == "middle":
                    score = 80
                else:
                    score = 40
                    
                fit_scores.append((score, 0.3))  # Medium weight
                
            elif priority == "digital_transformation":
                # Higher score for digital-native startups
                tech_stack = acquirer_profile.get("tech_stack", [])
                startup_tech = data.get("tech_stack", {})
                
                # Calculate tech stack similarity
                if tech_stack and startup_tech:
                    overlaps = sum(1 for tech in tech_stack if tech in startup_tech)
                    similarity = overlaps / len(tech_stack) if tech_stack else 0
                    score = 50 + similarity * 50  # Higher similarity = lower strategic fit (for transformation)
                else:
                    score = 70  # Default good score
                    
                fit_scores.append((score, 0.3))  # Medium weight
        
        # Add general strategic alignment
        acquirer_sectors = acquirer_profile.get("target_sectors", [])
        startup_sector = data.get("sector", "").lower()
        
        if startup_sector in acquirer_sectors or any(s in startup_sector for s in acquirer_sectors):
            fit_scores.append((90, 0.3))  # Good sector alignment
        else:
            fit_scores.append((30, 0.3))  # Poor sector alignment
        
        # Calculate weighted average
        if fit_scores:
            total_score = sum(score * weight for score, weight in fit_scores)
            total_weight = sum(weight for _, weight in fit_scores)
            return total_score / total_weight if total_weight > 0 else 50
        else:
            return 50  # Default moderate score

    def _calculate_revenue_synergy(self, data: Dict[str, Any], acquirer_profile: Dict[str, Any]) -> float:
        """
        Calculate potential revenue synergy percentage.
        
        Args:
            data: Startup data dictionary
            acquirer_profile: Acquirer type profile dictionary
            
        Returns:
            Revenue synergy score (0-100)
        """
        # Default moderate score
        if not acquirer_profile:
            return 50
        
        # Start with base score
        base_score = 50
        
        # Customer overlap
        customer_segments = data.get("customer_segments", [])
        acquirer_segments = acquirer_profile.get("customer_segments", [])
        
        overlap = False
        for segment in customer_segments:
            if segment in acquirer_segments:
                overlap = True
                break
                
        # More synergy from complementary (non-overlapping) customer segments
        if overlap:
            base_score += 10  # Some synergy from cross-selling
        else:
            base_score += 25  # More synergy from market expansion
            
        # Geographic expansion opportunity
        startup_geo = data.get("geographic_presence", ["US"])
        acquirer_geo = acquirer_profile.get("geographic_presence", ["US"])
        
        new_geos = [geo for geo in startup_geo if geo not in acquirer_geo]
        if new_geos:
            base_score += len(new_geos) * 5  # Bonus for each new geography
            
        # Distribution channel synergy
        startup_channels = data.get("distribution_channels", [])
        acquirer_channels = acquirer_profile.get("distribution_channels", [])
        
        new_channels = [ch for ch in startup_channels if ch not in acquirer_channels]
        if new_channels:
            base_score += len(new_channels) * 5  # Bonus for each new channel
            
        # Product complementarity
        startup_category = data.get("product_category", "")
        acquirer_categories = acquirer_profile.get("product_categories", [])
        
        if startup_category and startup_category not in acquirer_categories:
            base_score += 15  # High synergy for new product category
            
        # Pricing model alignment
        startup_pricing = data.get("pricing_model", "")
        acquirer_pricing = acquirer_profile.get("pricing_models", [])
        
        if startup_pricing in acquirer_pricing:
            base_score += 10  # Easier integration of similar pricing models
            
        # Return capped score
        return min(100, base_score)

    def _calculate_cost_synergy(self, data: Dict[str, Any], acquirer_profile: Dict[str, Any]) -> float:
        """
        Calculate potential cost synergy percentage.
        
        Args:
            data: Startup data dictionary
            acquirer_profile: Acquirer type profile dictionary
            
        Returns:
            Cost synergy score (0-100)
        """
        # Default moderate score
        if not acquirer_profile:
            return 50
        
        # Start with base score
        base_score = 50
        
        # Operational overlap (higher overlap = higher cost synergy)
        startup_ops = data.get("operational_areas", [])
        acquirer_ops = acquirer_profile.get("operational_areas", [])
        
        overlap_count = sum(1 for op in startup_ops if op in acquirer_ops)
        if startup_ops:
            overlap_pct = overlap_count / len(startup_ops)
            base_score += overlap_pct * 25  # Up to 25 points for operational overlap
            
        # Team redundancy potential
        employee_count = data.get("employee_count", 10)
        redundancy_ratio = acquirer_profile.get("redundancy_ratio", 0.3)
        
        potential_reduction = employee_count * redundancy_ratio
        base_score += min(20, potential_reduction * 2)  # Up to 20 points for team synergies
        
        # Technology stack similarity (more similar = more cost synergy)
        startup_tech = data.get("tech_stack", {})
        acquirer_tech = acquirer_profile.get("tech_stack", [])
        
        if startup_tech and acquirer_tech:
            similarity = sum(1 for tech in startup_tech if tech in acquirer_tech) / len(startup_tech)
            base_score += similarity * 20  # Up to 20 points for tech similarity
            
        # Vendor consolidation potential
        base_score += 10  # Default vendor synergy value
        
        # Return capped score
        return min(100, base_score)

    def _estimate_time_to_value(self, data: Dict[str, Any], acquirer_profile: Dict[str, Any]) -> int:
        """
        Estimate time to value in months for acquisition.
        
        Args:
            data: Startup data dictionary
            acquirer_profile: Acquirer type profile dictionary
            
        Returns:
            Estimated time to value in months
        """
        # Default estimate
        base_time = 12  # 1 year
        
        # Integration complexity factors
        tech_complexity = 100 - data.get("technology_readiness", 50)  # Invert: lower readiness = higher complexity
        team_size = data.get("employee_count", 10)
        documentation = data.get("documentation_completeness", 50)
        
        # Adjust time based on complexity factors
        tech_factor = tech_complexity / 100  # 0-1 scale
        base_time += tech_factor * 6  # Up to 6 additional months for tech complexity
        
        # Team size affects integration time
        if team_size <= 5:
            base_time -= 2  # Small team is faster to integrate
        elif team_size <= 20:
            base_time += 0  # Neutral
        elif team_size <= 50:
            base_time += 3  # Medium team takes longer
        else:
            base_time += 6  # Large team takes much longer
            
        # Documentation reduces integration time
        doc_factor = (100 - documentation) / 100  # 0-1 scale (higher = worse docs)
        base_time += doc_factor * 4  # Up to 4 additional months for poor documentation
        
        # Acquirer integration capability
        integration_capability = acquirer_profile.get("integration_capability", 0.5)  # 0-1 scale
        base_time = base_time * (1.5 - integration_capability)  # Better capability = less time
        
        # Return minimum 3 months, maximum 36 months
        return max(3, min(36, round(base_time)))

    def _assess_cultural_fit(self, data: Dict[str, Any], acquirer_profile: Dict[str, Any]) -> float:
        """
        Assess cultural fit with acquirer.
        
        Args:
            data: Startup data dictionary
            acquirer_profile: Acquirer type profile dictionary
            
        Returns:
            Cultural fit score (0-100)
        """
        # Default moderate score
        if not acquirer_profile:
            return 50
        
        # Start with moderate score
        base_score = 50
        
        # Company stage alignment
        startup_stage = data.get("stage", "").lower()
        preferred_stages = acquirer_profile.get("preferred_stages", [])
        
        if startup_stage in preferred_stages:
            base_score += 15  # Good stage alignment
        elif startup_stage in ["series-c", "growth"] and "series-b" in preferred_stages:
            base_score += 10  # Close stage alignment
        elif startup_stage in ["seed", "series-a"] and "series-b" in preferred_stages:
            base_score += 10  # Close stage alignment
        else:
            base_score -= 10  # Poor stage alignment
            
        # Team size alignment
        startup_size = data.get("employee_count", 10)
        acquirer_size = acquirer_profile.get("typical_size", 1000)
        
        # Very small companies may struggle in large acquirers
        if startup_size < 10 and acquirer_size > 1000:
            base_score -= 15
        elif startup_size < 20 and acquirer_size > 5000:
            base_score -= 10
        elif startup_size > 100 and "Startup" in acquirer_profile.get("acquirer_type", ""):
            base_score -= 10  # Large startup may not fit with small acquirer
            
        # Work style alignment
        startup_remote = data.get("remote_work", False)
        acquirer_remote = acquirer_profile.get("remote_friendly", False)
        
        if startup_remote == acquirer_remote:
            base_score += 10  # Aligned work styles
        else:
            base_score -= 10  # Misaligned work styles
            
        # Decision-making alignment
        startup_decision = data.get("decision_making_style", "")
        acquirer_decision = acquirer_profile.get("decision_making_style", "")
        
        if startup_decision and acquirer_decision:
            if startup_decision == acquirer_decision:
                base_score += 15  # Aligned decision-making
            elif (startup_decision == "consensus" and acquirer_decision == "collaborative") or \
                (startup_decision == "collaborative" and acquirer_decision == "consensus"):
                base_score += 10  # Similar decision-making
            elif (startup_decision == "top-down" and acquirer_decision == "hierarchical") or \
                (startup_decision == "hierarchical" and acquirer_decision == "top-down"):
                base_score += 10  # Similar decision-making
            else:
                base_score -= 10  # Different decision-making styles
                
        # Return capped score
        return max(10, min(100, base_score))

    def _generate_synergy_justification(
        self, 
        acquirer_type: str, 
        strategic_fit: float,
        revenue_synergy: float,
        cost_synergy: float,
        time_to_value: int,
        cultural_fit: float
    ) -> str:
        """
        Generate justification text for acquisition synergy.
        
        Args:
            acquirer_type: Type of acquirer
            strategic_fit: Strategic fit score
            revenue_synergy: Revenue synergy score
            cost_synergy: Cost synergy score
            time_to_value: Time to value in months
            cultural_fit: Cultural fit score
            
        Returns:
            Justification string
        """
        overall = (strategic_fit * 0.3 + revenue_synergy * 0.3 + cost_synergy * 0.2 + 
                  (100 - min(time_to_value, 36) * 100 / 36) * 0.1 + cultural_fit * 0.1)
        
        if overall >= 80:
            quality = "excellent"
        elif overall >= 65:
            quality = "strong"
        elif overall >= 50:
            quality = "moderate"
        else:
            quality = "weak"
            
        justification = f"{acquirer_type} acquirers offer {quality} fit with "
        
        # Add highlights based on strongest synergies
        highlights = []
        
        if strategic_fit >= 70:
            highlights.append("strong strategic alignment")
        if revenue_synergy >= 70:
            highlights.append("significant revenue synergies")
        if cost_synergy >= 70:
            highlights.append("substantial cost efficiencies")
        if time_to_value <= 6:
            highlights.append("rapid time-to-value")
        if cultural_fit >= 70:
            highlights.append("high cultural compatibility")
            
        # Add challenges for weak areas
        challenges = []
        
        if strategic_fit < 40:
            challenges.append("limited strategic alignment")
        if revenue_synergy < 40:
            challenges.append("modest revenue opportunities")
        if cost_synergy < 40:
            challenges.append("minimal cost synergies")
        if time_to_value > 18:
            challenges.append("extended integration timeline")
        if cultural_fit < 40:
            challenges.append("potential cultural friction")
            
        # Combine highlights and challenges
        if highlights:
            justification += ", ".join(highlights)
        else:
            justification += "acceptable overall fit"
            
        if challenges:
            justification += " despite " + ", ".join(challenges)
            
        return justification

    def _identify_potential_acquirers(self, acquirer_type: str, data: Dict[str, Any]) -> List[str]:
        """
        Identify specific potential acquirers based on acquirer type and startup data.
        
        Args:
            acquirer_type: Type of acquirer
            data: Startup data dictionary
            
        Returns:
            List of potential acquirer names
        """
        sector = data.get("sector", "").lower()
        
        # Default lists by acquirer type and sector
        acquirers_by_type = {
            "Big Tech": ["Google", "Microsoft", "Amazon", "Meta", "Apple"],
            "Enterprise Software": ["Salesforce", "Oracle", "SAP", "Adobe", "ServiceNow"],
            "Cloud Provider": ["AWS", "Microsoft Azure", "Google Cloud", "IBM Cloud"],
            "Financial Institution": ["JPMorgan Chase", "Goldman Sachs", "Morgan Stanley", "Visa", "Mastercard"],
            "Retail Giant": ["Amazon", "Walmart", "Target", "Alibaba"],
            "Marketplace": ["Amazon", "eBay", "Etsy", "Shopify"],
            "Pharmaceutical": ["Johnson & Johnson", "Pfizer", "Roche", "Novartis", "Merck"],
            "Healthcare": ["UnitedHealth Group", "CVS Health", "Kaiser Permanente"],
            "Media Conglomerate": ["Disney", "Comcast", "Warner Bros. Discovery", "Netflix"],
            "Manufacturing": ["GE", "Siemens", "3M", "Honeywell"],
            "Crypto/Blockchain": ["Coinbase", "Binance", "FTX", "Circle"],
            "Private Equity": ["Blackstone", "KKR", "Carlyle Group", "TPG", "Vista Equity Partners"],
            "Strategic": []  # Will be filled based on sector
        }
        
        # Add sector-specific potential acquirers for Strategic category
        if sector in ["saas", "software", "enterprise"]:
            acquirers_by_type["Strategic"] = ["Microsoft", "Oracle", "Salesforce", "Adobe", "SAP"]
        elif sector in ["fintech", "finance"]:
            acquirers_by_type["Strategic"] = ["PayPal", "Stripe", "Block", "Intuit", "Visa"]
        elif sector in ["ai", "ml"]:
            acquirers_by_type["Strategic"] = ["Google", "Microsoft", "Amazon", "IBM", "Nvidia"]
        elif sector in ["ecommerce", "retail"]:
            acquirers_by_type["Strategic"] = ["Amazon", "Shopify", "Walmart", "Target", "eBay"]
        elif sector in ["marketplace", "platform"]:
            acquirers_by_type["Strategic"] = ["Airbnb", "Uber", "Doordash", "Amazon", "eBay"]
        elif sector in ["biotech", "pharma"]:
            acquirers_by_type["Strategic"] = ["Pfizer", "Merck", "Johnson & Johnson", "Roche", "Novartis"]
        elif sector in ["crypto", "blockchain"]:
            acquirers_by_type["Strategic"] = ["Coinbase", "Binance", "FTX", "PayPal", "Block"]
        else:
            acquirers_by_type["Strategic"] = ["Industry Leader", "Vertical Market Leader", "Adjacent Market Player"]
            
        # Return the list for the requested acquirer type, or empty list if not found
        return acquirers_by_type.get(acquirer_type, [])

    def _get_sector_specific_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """
        Get sector-specific acquisition recommendations.
        
        Args:
            data: Startup data dictionary
            
        Returns:
            List of sector-specific recommendation strings
        """
        sector = data.get("sector", "").lower()
        recommendations = []
        
        if sector in ["saas", "software", "enterprise"]:
            recommendations.append("Prioritize customer retention metrics to demonstrate stable MRR and reduced acquisition risk")
            recommendations.append("Document product API and integrations to highlight technical synergy opportunities")
            
        elif sector in ["fintech", "finance"]:
            recommendations.append("Ensure full regulatory compliance documentation to reduce acquisition due diligence friction")
            recommendations.append("Quantify risk management processes and outcomes to increase acquirer confidence")
            
        elif sector in ["ai", "ml", "data"]:
            recommendations.append("Document proprietary algorithms and data moats to justify acquisition premium")
            recommendations.append("Create robust IP protection for AI/ML innovations to increase strategic value")
            
        elif sector in ["biotech", "pharma"]:
            recommendations.append("Advance key assets to meaningful clinical milestones before positioning for acquisition")
            recommendations.append("Document regulatory strategy and pathway to reduce perceived risk for acquirers")
            
        elif sector in ["marketplace", "platform"]:
            recommendations.append("Demonstrate network effects through cohort analysis to justify higher acquisition valuation")
            recommendations.append("Reduce customer and supplier concentration risk to increase platform resilience")
            
        elif sector in ["ecommerce", "d2c"]:
            recommendations.append("Focus on customer acquisition cost efficiency to improve unit economics for acquirers")
            recommendations.append("Build diversified marketing channels to demonstrate scalable growth potential")
            
        return recommendations[:2]  # Return top 2 sector-specific recommendations

    def _initialize_acquirer_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize acquirer profiles with characteristics and preferences.
        
        Returns:
            Dictionary mapping acquirer types to profile dictionaries
        """
        return {
            "Big Tech": {
                "strategic_priorities": ["technology_acquisition", "talent_acquisition", "product_expansion"],
                "target_sectors": ["ai", "ml", "saas", "cloud", "enterprise", "consumer", "mobile"],
                "customer_segments": ["enterprise", "smb", "consumer"],
                "geographic_presence": ["US", "EMEA", "APAC", "LATAM"],
                "product_categories": ["cloud", "productivity", "analytics", "mobile", "advertising"],
                "pricing_models": ["subscription", "freemium", "usage-based"],
                "tech_stack": ["aws", "gcp", "azure", "kubernetes", "react", "python", "tensorflow"],
                "operational_areas": ["r&d", "marketing", "sales", "customer_success", "infrastructure"],
                "integration_capability": 0.8,
                "typical_size": 10000,
                "redundancy_ratio": 0.4,
                "remote_friendly": True,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-a", "series-b", "series-c"]
            },
            "Enterprise Software": {
                "strategic_priorities": ["product_expansion", "market_expansion", "technology_acquisition"],
                "target_sectors": ["saas", "enterprise", "security", "data", "analytics", "vertical_saas"],
                "customer_segments": ["enterprise", "smb", "mid-market"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["crm", "erp", "hcm", "analytics", "collaboration"],
                "pricing_models": ["subscription", "per-seat", "enterprise-license"],
                "tech_stack": ["java", "dotnet", "react", "angular", "sql", "nosql"],
                "operational_areas": ["product", "sales", "marketing", "customer_success"],
                "integration_capability": 0.7,
                "typical_size": 5000,
                "redundancy_ratio": 0.3,
                "remote_friendly": True,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Cloud Provider": {
                "strategic_priorities": ["technology_acquisition", "product_expansion", "market_expansion"],
                "target_sectors": ["saas", "devops", "security", "data", "ai", "ml"],
                "customer_segments": ["enterprise", "smb", "startups"],
                "geographic_presence": ["US", "EMEA", "APAC", "LATAM"],
                "product_categories": ["iaas", "paas", "security", "analytics", "ai", "serverless"],
                "pricing_models": ["usage-based", "subscription", "tiered"],
                "tech_stack": ["kubernetes", "containers", "microservices", "python", "go", "rust"],
                "operational_areas": ["infrastructure", "r&d", "security", "support"],
                "integration_capability": 0.75,
                "typical_size": 8000,
                "redundancy_ratio": 0.35,
                "remote_friendly": True,
                "decision_making_style": "data-driven",
                "preferred_stages": ["series-a", "series-b", "series-c"]
            },
            "Financial Institution": {
                "strategic_priorities": ["digital_transformation", "market_expansion", "technology_acquisition"],
                "target_sectors": ["fintech", "payments", "banking", "insurance", "wealth", "regtech"],
                "customer_segments": ["enterprise", "consumer", "smb"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["payments", "lending", "banking", "wealth", "insurance"],
                "pricing_models": ["subscription", "transaction-fee", "assets-under-management"],
                "tech_stack": ["java", "python", "dotnet", "blockchain", "cloud"],
                "operational_areas": ["compliance", "risk", "operations", "technology", "customer_service"],
                "integration_capability": 0.5,
                "typical_size": 20000,
                "redundancy_ratio": 0.3,
                "remote_friendly": False,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Retail Giant": {
                "strategic_priorities": ["market_expansion", "vertical_integration", "digital_transformation"],
                "target_sectors": ["ecommerce", "logistics", "marketplace", "d2c", "retail-tech"],
                "customer_segments": ["consumer", "marketplace_seller"],
                "geographic_presence": ["US", "EMEA", "APAC", "LATAM"],
                "product_categories": ["consumer_goods", "grocery", "fashion", "electronics"],
                "pricing_models": ["retail", "marketplace-fee", "subscription"],
                "tech_stack": ["web", "mobile", "logistics_software", "inventory_management"],
                "operational_areas": ["logistics", "retail_operations", "merchandising", "marketing"],
                "integration_capability": 0.6,
                "typical_size": 50000,
                "redundancy_ratio": 0.25,
                "remote_friendly": False,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Marketplace": {
                "strategic_priorities": ["market_expansion", "product_expansion", "vertical_integration"],
                "target_sectors": ["marketplace", "ecommerce", "services", "sharing_economy"],
                "customer_segments": ["consumer", "merchant", "service_provider"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["retail", "services", "food", "travel", "real_estate"],
                "pricing_models": ["take-rate", "subscription", "listing-fee"],
                "tech_stack": ["web", "mobile", "matching_algorithms", "payments"],
                "operational_areas": ["marketplace_ops", "seller_acquisition", "trust_safety"],
                "integration_capability": 0.65,
                "typical_size": 3000,
                "redundancy_ratio": 0.2,
                "remote_friendly": True,
                "decision_making_style": "data-driven",
                "preferred_stages": ["series-a", "series-b", "series-c"]
            },
            "Pharmaceutical": {
                "strategic_priorities": ["technology_acquisition", "market_expansion", "product_expansion"],
                "target_sectors": ["biotech", "pharma", "medtech", "diagnostics", "therapeutics"],
                "customer_segments": ["patients", "providers", "payers"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["drugs", "biologics", "devices", "diagnostics"],
                "pricing_models": ["reimbursement", "value-based", "direct-to-consumer"],
                "tech_stack": ["clinical_data_systems", "r&d_platforms", "regulatory_systems"],
                "operational_areas": ["r&d", "clinical_trials", "regulatory", "manufacturing"],
                "integration_capability": 0.5,
                "typical_size": 30000,
                "redundancy_ratio": 0.2,
                "remote_friendly": False,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Healthcare": {
                "strategic_priorities": ["digital_transformation", "market_expansion", "vertical_integration"],
                "target_sectors": ["health-tech", "telemedicine", "diagnostics", "care-delivery", "wellness"],
                "customer_segments": ["patients", "providers", "employers", "payers"],
                "geographic_presence": ["US", "EMEA"],
                "product_categories": ["care_delivery", "insurance", "wellness", "digital_health"],
                "pricing_models": ["subscription", "fee-for-service", "value-based"],
                "tech_stack": ["emr_systems", "health_data_platforms", "telemedicine", "analytics"],
                "operational_areas": ["clinical_operations", "insurance_ops", "care_management"],
                "integration_capability": 0.4,
                "typical_size": 15000,
                "redundancy_ratio": 0.15,
                "remote_friendly": False,
                "decision_making_style": "consensus",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Media Conglomerate": {
                "strategic_priorities": ["digital_transformation", "product_expansion", "market_expansion"],
                "target_sectors": ["media", "content", "streaming", "gaming", "advertising"],
                "customer_segments": ["consumer", "advertiser", "creator"],
                "geographic_presence": ["US", "EMEA", "APAC", "LATAM"],
                "product_categories": ["streaming", "publishing", "games", "advertising", "events"],
                "pricing_models": ["subscription", "advertising", "freemium"],
                "tech_stack": ["streaming_platform", "content_management", "analytics"],
                "operational_areas": ["content_production", "distribution", "marketing", "advertising_sales"],
                "integration_capability": 0.6,
                "typical_size": 10000,
                "redundancy_ratio": 0.3,
                "remote_friendly": True,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Manufacturing": {
                "strategic_priorities": ["digital_transformation", "vertical_integration", "market_expansion"],
                "target_sectors": ["hardware", "industrial-tech", "iot", "robotics", "3d-printing"],
                "customer_segments": ["enterprise", "industrial", "manufacturing"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["equipment", "components", "materials", "software"],
                "pricing_models": ["capital-expense", "recurring-maintenance", "subscription"],
                "tech_stack": ["erp", "scm", "automation", "iot_platforms"],
                "operational_areas": ["manufacturing", "supply_chain", "logistics", "r&d"],
                "integration_capability": 0.4,
                "typical_size": 25000,
                "redundancy_ratio": 0.2,
                "remote_friendly": False,
                "decision_making_style": "hierarchical",
                "preferred_stages": ["series-b", "series-c", "growth"]
            },
            "Crypto/Blockchain": {
                "strategic_priorities": ["technology_acquisition", "talent_acquisition", "market_expansion"],
                "target_sectors": ["crypto", "blockchain", "defi", "web3", "payments"],
                "customer_segments": ["consumer", "institution", "developer"],
                "geographic_presence": ["US", "EMEA", "APAC", "global"],
                "product_categories": ["exchange", "wallet", "protocol", "defi", "nft"],
                "pricing_models": ["transaction-fee", "protocol-fee", "token-economy"],
                "tech_stack": ["blockchain", "smart-contracts", "distributed-systems", "cryptography"],
                "operational_areas": ["protocol_development", "exchange_ops", "compliance", "security"],
                "integration_capability": 0.7,
                "typical_size": 500,
                "redundancy_ratio": 0.3,
                "remote_friendly": True,
                "decision_making_style": "decentralized",
                "preferred_stages": ["seed", "series-a", "series-b"]
            },
            "Private Equity": {
                "strategic_priorities": ["market_expansion", "operational_efficiency", "vertical_integration"],
                "target_sectors": ["saas", "fintech", "healthcare", "industrial", "consumer"],
                "customer_segments": ["enterprise", "smb", "consumer"],
                "geographic_presence": ["US", "EMEA"],
                "product_categories": ["software", "services", "platforms"],
                "pricing_models": ["subscription", "transaction", "recurring-revenue"],
                "tech_stack": ["varied", "enterprise", "cloud"],
                "operational_areas": ["sales", "marketing", "operations", "finance"],
                "integration_capability": 0.8,  # PE firms have playbooks for integration
                "typical_size": 50,  # PE firm itself is small but portfolio companies vary
                "redundancy_ratio": 0.4,  # PE often looks for operational efficiency
                "remote_friendly": True,
                "decision_making_style": "data-driven",
                "preferred_stages": ["series-c", "growth"]  # Later stage focus
            },
            "Strategic": {
                "strategic_priorities": ["market_expansion", "product_expansion", "technology_acquisition"],
                "target_sectors": ["varied", "industry-specific"],
                "customer_segments": ["enterprise", "smb", "consumer"],
                "geographic_presence": ["US", "EMEA", "APAC"],
                "product_categories": ["complementary", "vertical-specific"],
                "pricing_models": ["industry-standard", "subscription", "transaction"],
                "tech_stack": ["varied", "industry-standard"],
                "operational_areas": ["product", "sales", "marketing", "r&d"],
                "integration_capability": 0.6,
                "typical_size": 2000,
                "redundancy_ratio": 0.3,
                "remote_friendly": True,
                "decision_making_style": "collaborative",
                "preferred_stages": ["series-a", "series-b", "series-c", "growth"]
            }
        }

    def _load_default_comparables(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load default industry acquisition comparables.
        
        Returns:
            Dictionary mapping sectors to lists of comparable transactions
        """
        return {
            "saas": [
                {
                    "acquirer": "Microsoft",
                    "target": "GitHub",
                    "value": 7500000000,
                    "date": "2018-06-04",
                    "stage": "growth",
                    "sales_multiple": 25
                },
                {
                    "acquirer": "Salesforce",
                    "target": "Slack",
                    "value": 27700000000,
                    "date": "2020-12-01",
                    "stage": "public",
                    "sales_multiple": 25
                },
                {
                    "acquirer": "Adobe",
                    "target": "Figma",
                    "value": 20000000000,
                    "date": "2022-09-15",
                    "stage": "series-c",
                    "sales_multiple": 50
                },
                {
                    "acquirer": "Cisco",
                    "target": "AppDynamics",
                    "value": 3700000000,
                    "date": "2017-01-24",
                    "stage": "pre-ipo",
                    "sales_multiple": 18
                }
            ],
            "fintech": [
                {
                    "acquirer": "Visa",
                    "target": "Plaid",
                    "value": 5300000000,
                    "date": "2020-01-13",
                    "stage": "series-c",
                    "sales_multiple": 25
                },
                {
                    "acquirer": "PayPal",
                    "target": "Honey",
                    "value": 4000000000,
                    "date": "2019-11-20",
                    "stage": "series-c",
                    "sales_multiple": 20
                },
                {
                    "acquirer": "Block (Square)",
                    "target": "Afterpay",
                    "value": 29000000000,
                    "date": "2021-08-01",
                    "stage": "public",
                    "sales_multiple": 25
                }
            ],
            "ai": [
                {
                    "acquirer": "Microsoft",
                    "target": "Nuance",
                    "value": 19700000000,
                    "date": "2021-04-12",
                    "stage": "public",
                    "sales_multiple": 13
                },
                {
                    "acquirer": "Google",
                    "target": "DeepMind",
                    "value": 500000000,
                    "date": "2014-01-26",
                    "stage": "series-a",
                    "sales_multiple": 0  # Pre-revenue
                },
                {
                    "acquirer": "ServiceNow",
                    "target": "Element AI",
                    "value": 500000000,
                    "date": "2020-11-30",
                    "stage": "series-b",
                    "sales_multiple": 0  # Pre-revenue
                }
            ],
            "ecommerce": [
                {
                    "acquirer": "Walmart",
                    "target": "Jet.com",
                    "value": 3300000000,
                    "date": "2016-08-08",
                    "stage": "series-d",
                    "sales_multiple": 4
                },
                {
                    "acquirer": "Amazon",
                    "target": "Whole Foods",
                    "value": 13700000000,
                    "date": "2017-06-16",
                    "stage": "public",
                    "sales_multiple": 0.9
                },
                {
                    "acquirer": "Target",
                    "target": "Shipt",
                    "value": 550000000,
                    "date": "2017-12-13",
                    "stage": "series-b",
                    "sales_multiple": 8
                }
            ],
            "biotech": [
                {
                    "acquirer": "Pfizer",
                    "target": "Arena Pharmaceuticals",
                    "value": 6700000000,
                    "date": "2021-12-13",
                    "stage": "public",
                    "sales_multiple": 0  # Focused on pipeline value
                },
                {
                    "acquirer": "Gilead",
                    "target": "Forty Seven",
                    "value": 4900000000,
                    "date": "2020-03-02",
                    "stage": "public",
                    "sales_multiple": 0  # Pre-revenue
                },
                {
                    "acquirer": "Roche",
                    "target": "Spark Therapeutics",
                    "value": 4300000000,
                    "date": "2019-02-25",
                    "stage": "public",
                    "sales_multiple": 10  # High multiple for early revenue
                }
            ],
            "other": [
                {
                    "acquirer": "Various",
                    "target": "Seed Stage Average",
                    "value": 30000000,
                    "date": "2023-01-01",
                    "stage": "seed",
                    "sales_multiple": 10
                },
                {
                    "acquirer": "Various",
                    "target": "Series A Average",
                    "value": 100000000,
                    "date": "2023-01-01",
                    "stage": "series-a",
                    "sales_multiple": 12
                },
                {
                    "acquirer": "Various",
                    "target": "Series B Average",
                    "value": 250000000,
                    "date": "2023-01-01",
                    "stage": "series-b",
                    "sales_multiple": 10
                },
                {
                    "acquirer": "Various",
                    "target": "Series C+ Average",
                    "value": 500000000,
                    "date": "2023-01-01",
                    "stage": "series-c",
                    "sales_multiple": 8
                }
            ]
        }
