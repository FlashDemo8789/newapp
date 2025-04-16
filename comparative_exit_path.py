"""
Comparative Exit Path Analysis - FlashDNA

Analyzes potential exit paths for startups, comparing expected outcomes,
timelines, probabilities, and risk-adjusted returns across scenarios.

Integration with FlashDNA's CAMP framework allows exit path recommendations
based on company-specific metrics and market conditions.

Usage:
    from comparative_exit_path import ExitPathAnalyzer
    
    # Create analyzer with startup data
    exit_analyzer = ExitPathAnalyzer(startup_data)
    
    # Run analysis
    exit_analysis = exit_analyzer.analyze_exit_paths()
    
    # Get recommended exit strategy
    recommendation = exit_analyzer.get_exit_recommendations()
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import math

logger = logging.getLogger("exit_path")

@dataclass
class ExitPathScenario:
    """Represents a specific exit path scenario with associated metrics."""
    path_name: str
    time_to_exit: float  # Years
    exit_valuation: float
    probability: float
    risk_factor: float
    requirements: Dict[str, float]
    risk_adjusted_value: float = 0.0
    npv: float = 0.0
    irr: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        self.risk_adjusted_value = self.exit_valuation * self.probability * (1 - self.risk_factor)


@dataclass
class ExitPathAnalysis:
    """Complete exit path analysis results."""
    scenarios: List[ExitPathScenario]
    optimal_path: str
    timeline_data: Dict[str, List[float]]
    success_factors: Dict[str, float]
    recommended_milestones: List[Dict[str, Any]]
    sensitivity_analysis: Dict[str, Any]
    comparable_exits: List[Dict[str, Any]]
    exit_readiness_score: float


class ExitPathAnalyzer:
    """
    Analyzes potential exit paths for startups based on company data,
    market conditions, and historical comparables.
    """

    def __init__(self, company_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer with company and market data.
        
        Args:
            company_data: Dictionary containing company metrics and data
            market_data: Optional market data dictionary (will use defaults if None)
        """
        self.company_data = company_data
        self.market_data = market_data if market_data is not None else self._get_default_market_data()
        self.exit_paths = self._initialize_exit_paths()
        self.sector = company_data.get("sector", "").lower()
        self.stage = company_data.get("stage", "").lower()
        
        # Extract key company metrics
        self._extract_company_metrics()

    def _extract_company_metrics(self):
        """Extract and normalize key company metrics for analysis."""
        # Financial metrics
        self.monthly_revenue = self.company_data.get("monthly_revenue", 0)
        self.annual_revenue = self.monthly_revenue * 12
        
        # Convert percentage values to decimal if needed
        self.growth_rate = self.company_data.get("revenue_growth_rate", 0)
        if self.growth_rate > 1:
            self.growth_rate = self.growth_rate / 100
            
        self.gross_margin = self.company_data.get("gross_margin_percent", 0)
        if self.gross_margin > 1:
            self.gross_margin = self.gross_margin / 100
            
        # Extract other metrics
        self.burn_rate = self.company_data.get("burn_rate", 0)
        self.cash_on_hand = self.company_data.get("current_cash", 0)
        self.team_score = self.company_data.get("team_score", 0)
        self.moat_score = self.company_data.get("moat_score", 0)
        self.founder_exits = self.company_data.get("founder_exits", 0)
        self.user_count = self.company_data.get("current_users", 0)
        
        # Capital efficiency metrics
        self.ltv_cac_ratio = self.company_data.get("ltv_cac_ratio", 0)
        self.cac_payback_months = self.company_data.get("cac_payback_months", 0)
        
        # CAMP framework scores (if available)
        self.capital_score = self.company_data.get("capital_score", 0)
        self.advantage_score = self.company_data.get("advantage_score", 0)
        self.market_score = self.company_data.get("market_score", 0)
        self.people_score = self.company_data.get("people_score", 0)
        
        # Calculated values
        self.arpa = self.annual_revenue / max(1, self.user_count)  # Annual revenue per account
        
        # Ensure stage is normalized
        if self.stage not in ["pre-seed", "seed", "series-a", "series-b", "series-c", "growth"]:
            # Try to normalize known variations
            if "seed" in self.stage:
                self.stage = "seed"
            elif "series a" in self.stage or "series-a" in self.stage:
                self.stage = "series-a"
            elif "series b" in self.stage or "series-b" in self.stage:
                self.stage = "series-b"
            elif "series c" in self.stage or "series-c" in self.stage:
                self.stage = "series-c"
            elif "growth" in self.stage:
                self.stage = "growth"
            else:
                self.stage = "seed"  # Default to seed if can't determine

    def _get_default_market_data(self) -> Dict[str, Any]:
        """
        Get default market data if not provided.
        
        Returns:
            Dictionary with market data defaults
        """
        return {
            "ipo_market": {
                "saas": {
                    "min_arr": 100_000_000,  # Minimum ARR for IPO
                    "arr_multiple": 15,      # Revenue multiple
                    "growth_threshold": 0.4,  # Minimum growth rate
                    "margin_threshold": 0.7,  # Gross margin requirement
                },
                "fintech": {
                    "min_arr": 150_000_000,
                    "arr_multiple": 12,
                    "growth_threshold": 0.5,
                    "margin_threshold": 0.65,
                },
                "ecommerce": {
                    "min_arr": 200_000_000,
                    "arr_multiple": 4,
                    "growth_threshold": 0.3,
                    "margin_threshold": 0.5,
                },
                "biotech": {
                    "min_arr": 50_000_000,
                    "arr_multiple": 10,
                    "growth_threshold": 0.25,
                    "margin_threshold": 0.6,
                },
                "default": {
                    "min_arr": 100_000_000,
                    "arr_multiple": 10,
                    "growth_threshold": 0.3,
                    "margin_threshold": 0.6,
                }
            },
            "acquisition": {
                "saas": {
                    "arr_multiple_range": [6, 15],
                    "growth_impact": 0.5,    # Additional multiple for each 10% growth
                    "margin_impact": 0.3,    # Additional multiple for each 10% margin above threshold
                    "margin_threshold": 0.6, # Gross margin threshold
                },
                "fintech": {
                    "arr_multiple_range": [5, 12],
                    "growth_impact": 0.4,
                    "margin_impact": 0.2,
                    "margin_threshold": 0.5,
                },
                "ecommerce": {
                    "arr_multiple_range": [2, 4],
                    "growth_impact": 0.3,
                    "margin_impact": 0.1,
                    "margin_threshold": 0.4,
                },
                "biotech": {
                    "arr_multiple_range": [8, 20],
                    "growth_impact": 0.6,
                    "margin_impact": 0.2,
                    "margin_threshold": 0.6,
                },
                "default": {
                    "arr_multiple_range": [4, 10],
                    "growth_impact": 0.4,
                    "margin_impact": 0.2,
                    "margin_threshold": 0.5,
                }
            },
            "pe_buyout": {
                "saas": {
                    "arr_multiple_range": [4, 8],
                    "ebitda_multiple_range": [10, 20],
                    "min_arr": 20_000_000,
                    "margin_threshold": 0.7,
                },
                "fintech": {
                    "arr_multiple_range": [3, 7],
                    "ebitda_multiple_range": [8, 18],
                    "min_arr": 30_000_000,
                    "margin_threshold": 0.6,
                },
                "ecommerce": {
                    "arr_multiple_range": [1.5, 3],
                    "ebitda_multiple_range": [6, 12],
                    "min_arr": 50_000_000,
                    "margin_threshold": 0.4,
                },
                "default": {
                    "arr_multiple_range": [3, 6],
                    "ebitda_multiple_range": [8, 15],
                    "min_arr": 30_000_000,
                    "margin_threshold": 0.6,
                }
            },
            "discount_rate": 0.25,  # High discount rate for startups
            "market_conditions": {
                "ipo_favorability": 0.7,      # 0-1 scale of current IPO market conditions
                "acquisition_favorability": 0.8, # 0-1 scale of M&A market activity
                "pe_favorability": 0.6        # 0-1 scale of PE market conditions
            }
        }
    
    def _initialize_exit_paths(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the standard exit paths with baseline values.
        
        Returns:
            Dictionary of exit path configurations
        """
        return {
            "ipo": {
                "description": "Initial Public Offering",
                "typical_timeline": {
                    "pre-seed": 10,
                    "seed": 8,
                    "series-a": 6,
                    "series-b": 4,
                    "series-c": 2.5,
                    "growth": 1.5
                },
                "valuation_factors": {
                    "revenue_multiple": [8, 20],
                    "growth_premium": 0.6,  # Premium for each 10% growth
                    "margin_premium": 0.4,  # Premium for each 10% margin above expected
                },
                "requirements": {
                    "min_revenue": 100_000_000,
                    "min_growth": 0.3,
                    "min_margin": 0.6,
                    "team_maturity": 0.8
                },
                "risk_factors": {
                    "market_volatility": 0.4,
                    "execution_risk": 0.3,
                    "competition_risk": 0.2,
                    "regulatory_risk": 0.1
                }
            },
            "strategic_acquisition": {
                "description": "Acquisition by a Strategic Buyer",
                "typical_timeline": {
                    "pre-seed": 7,
                    "seed": 5,
                    "series-a": 4,
                    "series-b": 2.5,
                    "series-c": 1.5,
                    "growth": 1
                },
                "valuation_factors": {
                    "revenue_multiple": [4, 12], 
                    "growth_premium": 0.4,
                    "margin_premium": 0.3,
                    "strategic_premium": 0.3  # Premium for strategic value
                },
                "requirements": {
                    "min_revenue": 10_000_000,
                    "min_growth": 0.2,
                    "min_margin": 0.5,
                    "product_integration": 0.7
                },
                "risk_factors": {
                    "strategic_fit_risk": 0.3,
                    "integration_risk": 0.3,
                    "culture_risk": 0.2,
                    "negotiation_risk": 0.2
                }
            },
            "financial_acquisition": {
                "description": "Acquisition by a Financial Buyer",
                "typical_timeline": {
                    "pre-seed": 7,
                    "seed": 5,
                    "series-a": 4,
                    "series-b": 3,
                    "series-c": 2,
                    "growth": 1
                },
                "valuation_factors": {
                    "revenue_multiple": [3, 8],
                    "ebitda_multiple": [8, 15],
                    "growth_premium": 0.3,
                    "margin_premium": 0.4
                },
                "requirements": {
                    "min_revenue": 5_000_000,
                    "min_growth": 0.15,
                    "min_margin": 0.5,
                    "positive_ebitda": 0.6
                },
                "risk_factors": {
                    "financial_performance_risk": 0.4,
                    "market_risk": 0.3,
                    "debt_capacity_risk": 0.2,
                    "valuation_risk": 0.1
                }
            },
            "pe_buyout": {
                "description": "Private Equity Buyout",
                "typical_timeline": {
                    "pre-seed": 8,
                    "seed": 7,
                    "series-a": 5,
                    "series-b": 3.5,
                    "series-c": 2,
                    "growth": 1
                },
                "valuation_factors": {
                    "revenue_multiple": [2.5, 6],
                    "ebitda_multiple": [6, 12],
                    "growth_premium": 0.2,
                    "margin_premium": 0.5
                },
                "requirements": {
                    "min_revenue": 20_000_000,
                    "min_growth": 0.1,
                    "min_margin": 0.4,
                    "positive_ebitda": 0.8
                },
                "risk_factors": {
                    "financial_performance_risk": 0.5,
                    "market_risk": 0.2,
                    "debt_capacity_risk": 0.2,
                    "management_risk": 0.1
                }
            },
            "acquihire": {
                "description": "Talent Acquisition",
                "typical_timeline": {
                    "pre-seed": 3,
                    "seed": 2.5,
                    "series-a": 2,
                    "series-b": 1.5,
                    "series-c": 1,
                    "growth": 0.5
                },
                "valuation_factors": {
                    "per_engineer_value": [0.5, 2],  # $M per engineer
                    "team_quality_premium": 0.5,
                    "tech_stack_premium": 0.3,
                    "ip_premium": 0.2
                },
                "requirements": {
                    "min_team_score": 70,
                    "min_tech_talent_ratio": 0.6,
                    "innovative_tech": 0.7
                },
                "risk_factors": {
                    "team_retention_risk": 0.5,
                    "acquirer_interest_risk": 0.3,
                    "technology_relevance_risk": 0.2
                }
            }
        }

    def analyze_exit_paths(self) -> ExitPathAnalysis:
        """
        Perform comprehensive exit path analysis.
        
        Returns:
            ExitPathAnalysis object containing analysis results
        """
        try:
            # Generate scenarios for each exit path
            scenarios = self._generate_exit_scenarios()
            
            # Determine optimal path based on risk-adjusted value
            optimal_path = self._determine_optimal_path(scenarios)
            
            # Generate timeline data
            timeline_data = self._generate_timeline_data(scenarios)
            
            # Calculate success factors for each path
            success_factors = self._calculate_success_factors()
            
            # Generate recommended milestones
            recommended_milestones = self._generate_milestones(scenarios, optimal_path)
            
            # Perform sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(scenarios, optimal_path)
            
            # Find comparable exits
            comparable_exits = self._find_comparable_exits()
            
            # Calculate exit readiness score
            exit_readiness_score = self._calculate_exit_readiness_score()
            
            # Create and return the analysis object
            return ExitPathAnalysis(
                scenarios=scenarios,
                optimal_path=optimal_path,
                timeline_data=timeline_data,
                success_factors=success_factors,
                recommended_milestones=recommended_milestones,
                sensitivity_analysis=sensitivity_analysis,
                comparable_exits=comparable_exits,
                exit_readiness_score=exit_readiness_score
            )
        
        except Exception as e:
            logger.error(f"Error in exit path analysis: {str(e)}")
            # Return a minimal valid object in case of error
            return ExitPathAnalysis(
                scenarios=[],
                optimal_path="",
                timeline_data={},
                success_factors={},
                recommended_milestones=[],
                sensitivity_analysis={},
                comparable_exits=[],
                exit_readiness_score=0.0
            )

    def _generate_exit_scenarios(self) -> List[ExitPathScenario]:
        """
        Generate detailed scenarios for each applicable exit path.
        
        Returns:
            List of ExitPathScenario objects
        """
        scenarios = []
        
        for path_name, path_config in self.exit_paths.items():
            # Skip paths that are not applicable based on requirements
            if not self._is_path_applicable(path_name, path_config):
                continue
            
            # Calculate time to exit based on current stage
            time_to_exit = path_config["typical_timeline"].get(
                self.stage, 
                path_config["typical_timeline"].get("seed", 5)
            )
            
            # Calculate exit valuation
            exit_valuation = self._calculate_exit_valuation(path_name, time_to_exit)
            
            # Calculate probability of success for this path
            probability = self._calculate_success_probability(path_name, path_config)
            
            # Calculate risk factor
            risk_factor = self._calculate_risk_factor(path_name, path_config)
            
            # Extract key requirements
            requirements = self._extract_path_requirements(path_name, path_config)
            
            # Calculate NPV and IRR
            npv = self._calculate_npv(exit_valuation, time_to_exit)
            irr = self._calculate_irr(exit_valuation, time_to_exit)
            
            # Create scenario object
            scenario = ExitPathScenario(
                path_name=path_name,
                time_to_exit=time_to_exit,
                exit_valuation=exit_valuation,
                probability=probability,
                risk_factor=risk_factor,
                requirements=requirements,
                risk_adjusted_value=exit_valuation * probability * (1 - risk_factor),
                npv=npv,
                irr=irr
            )
            
            scenarios.append(scenario)
        
        # Sort scenarios by risk-adjusted value
        scenarios.sort(key=lambda s: s.risk_adjusted_value, reverse=True)
        return scenarios

    def _is_path_applicable(self, path_name: str, path_config: Dict[str, Any]) -> bool:
        """
        Check if an exit path is applicable to the company based on requirements.
        
        Args:
            path_name: Name of the exit path
            path_config: Configuration for the exit path
            
        Returns:
            Boolean indicating if the path is applicable
        """
        # Get requirements
        requirements = path_config.get("requirements", {})
        
        # Check IPO requirements
        if path_name == "ipo":
            min_revenue = requirements.get("min_revenue", 100_000_000)
            min_growth = requirements.get("min_growth", 0.3)
            
            # If company is very far from IPO requirements, path is not applicable
            projected_annual_revenue = self._project_revenue(self.annual_revenue, self.growth_rate, years=5)
            if projected_annual_revenue < min_revenue * 0.25:
                return False
            
            if self.growth_rate < min_growth * 0.5:
                return False
        
        # Check PE buyout requirements
        elif path_name == "pe_buyout":
            min_revenue = requirements.get("min_revenue", 20_000_000)
            positive_ebitda = requirements.get("positive_ebitda", 0.8)
            
            # Check if the company has positive EBITDA or close to it
            if self.burn_rate > self.monthly_revenue and positive_ebitda > 0.5:
                return False
            
            projected_annual_revenue = self._project_revenue(self.annual_revenue, self.growth_rate, years=3)
            if projected_annual_revenue < min_revenue * 0.25:
                return False
        
        # Check acquihire requirements
        elif path_name == "acquihire":
            min_team_score = requirements.get("min_team_score", 70)
            min_tech_talent_ratio = requirements.get("min_tech_talent_ratio", 0.6)
            
            # Check team requirements
            if self.team_score < min_team_score * 0.8:
                return False
            
            tech_talent_ratio = self.company_data.get("tech_talent_ratio", 0)
            if tech_talent_ratio < min_tech_talent_ratio * 0.8:
                return False
        
        # For other paths, assume they're applicable
        return True

    def _calculate_exit_valuation(self, path_name: str, time_to_exit: float) -> float:
        """
        Calculate the projected exit valuation for a specific path.
        
        Args:
            path_name: Name of the exit path
            time_to_exit: Time to exit in years
            
        Returns:
            Projected exit valuation in dollars
        """
        # Get the appropriate path configuration
        path_config = self.exit_paths.get(path_name, {})
        valuation_factors = path_config.get("valuation_factors", {})
        
        # Project future revenue based on growth rate
        future_annual_revenue = self._project_revenue(self.annual_revenue, self.growth_rate, time_to_exit)
        
        if path_name == "ipo":
            # Get sector-specific IPO metrics
            sector_data = self.market_data["ipo_market"].get(
                self.sector, 
                self.market_data["ipo_market"]["default"]
            )
            
            # Calculate revenue multiple based on growth and margin
            revenue_multiple_range = valuation_factors.get("revenue_multiple", [8, 20])
            base_multiple = (revenue_multiple_range[0] + revenue_multiple_range[1]) / 2
            
            # Adjust multiple based on growth rate
            growth_premium = valuation_factors.get("growth_premium", 0.6)
            growth_adjustment = (self.growth_rate - sector_data.get("growth_threshold", 0.3)) / 0.1 * growth_premium
            
            # Adjust multiple based on margin
            margin_premium = valuation_factors.get("margin_premium", 0.4)
            margin_threshold = sector_data.get("margin_threshold", 0.6)
            margin_adjustment = (self.gross_margin - margin_threshold) / 0.1 * margin_premium
            
            # Calculate adjusted multiple
            adjusted_multiple = base_multiple + growth_adjustment + margin_adjustment
            adjusted_multiple = max(revenue_multiple_range[0], min(revenue_multiple_range[1], adjusted_multiple))
            
            # Calculate valuation
            valuation = future_annual_revenue * adjusted_multiple
            
        elif path_name in ["strategic_acquisition", "financial_acquisition"]:
            # Get sector-specific acquisition metrics
            sector_data = self.market_data["acquisition"].get(
                self.sector, 
                self.market_data["acquisition"]["default"]
            )
            
            # Calculate revenue multiple
            revenue_multiple_range = valuation_factors.get("revenue_multiple", sector_data.get("arr_multiple_range", [4, 10]))
            base_multiple = (revenue_multiple_range[0] + revenue_multiple_range[1]) / 2
            
            # Adjust for growth
            growth_premium = valuation_factors.get("growth_premium", 0.4)
            growth_adjustment = (self.growth_rate / 0.1) * growth_premium
            
            # Adjust for margins
            margin_premium = valuation_factors.get("margin_premium", 0.3)
            margin_threshold = sector_data.get("margin_threshold", 0.5)
            margin_adjustment = max(0, (self.gross_margin - margin_threshold) / 0.1 * margin_premium)
            
            # Strategic premium (only for strategic acquisitions)
            strategic_premium = 0
            if path_name == "strategic_acquisition":
                strategic_premium = valuation_factors.get("strategic_premium", 0.3) * base_multiple
                # Add premium for high moat score
                if self.moat_score > 70:
                    strategic_premium *= 1.5
            
            # Calculate adjusted multiple
            adjusted_multiple = base_multiple + growth_adjustment + margin_adjustment + strategic_premium
            adjusted_multiple = max(revenue_multiple_range[0], min(revenue_multiple_range[1], adjusted_multiple))
            
            # Calculate valuation
            valuation = future_annual_revenue * adjusted_multiple
        
        elif path_name == "pe_buyout":
            # Get sector-specific PE metrics
            sector_data = self.market_data["pe_buyout"].get(
                self.sector, 
                self.market_data["pe_buyout"]["default"]
            )
            
            # Calculate projected EBITDA
            projected_ebitda = self._project_ebitda(time_to_exit)
            
            # If projected EBITDA is positive, use EBITDA multiple
            if projected_ebitda > 0:
                ebitda_multiple_range = valuation_factors.get("ebitda_multiple", sector_data.get("ebitda_multiple_range", [8, 15]))
                base_multiple = (ebitda_multiple_range[0] + ebitda_multiple_range[1]) / 2
                
                # Adjust for growth and margin
                growth_premium = valuation_factors.get("growth_premium", 0.2)
                margin_premium = valuation_factors.get("margin_premium", 0.5)
                
                growth_adjustment = (self.growth_rate / 0.1) * growth_premium
                margin_threshold = sector_data.get("margin_threshold", 0.5)
                margin_adjustment = max(0, (self.gross_margin - margin_threshold) / 0.1 * margin_premium)
                
                # Calculate adjusted multiple
                adjusted_multiple = base_multiple + growth_adjustment + margin_adjustment
                adjusted_multiple = max(ebitda_multiple_range[0], min(ebitda_multiple_range[1], adjusted_multiple))
                
                # Calculate valuation
                valuation = projected_ebitda * adjusted_multiple
            else:
                # Use revenue multiple for negative EBITDA companies
                revenue_multiple_range = valuation_factors.get("revenue_multiple", sector_data.get("arr_multiple_range", [2.5, 6]))
                base_multiple = revenue_multiple_range[0]  # Use lower end for negative EBITDA
                
                # Apply conservative multiple
                valuation = future_annual_revenue * base_multiple
        
        elif path_name == "acquihire":
            # Value based on engineering team
            engineer_value_range = valuation_factors.get("per_engineer_value", [0.5, 2])  # $M per engineer
            tech_talent_ratio = self.company_data.get("tech_talent_ratio", 0.4)
            employee_count = self.company_data.get("employee_count", 10)
            
            # Estimate number of engineers
            eng_count = max(1, int(employee_count * tech_talent_ratio))
            
            # Base value per engineer based on team quality
            team_quality = self.team_score / 100
            base_per_eng = engineer_value_range[0] + (engineer_value_range[1] - engineer_value_range[0]) * team_quality
            
            # Premiums based on tech stack and IP
            tech_stack_premium = valuation_factors.get("tech_stack_premium", 0.3)
            ip_premium = valuation_factors.get("ip_premium", 0.2)
            
            tech_innovation = self.company_data.get("technical_innovation_score", 50) / 100
            patent_count = self.company_data.get("patent_count", 0)
            
            # Apply premiums
            tech_adjustment = tech_stack_premium * tech_innovation
            ip_adjustment = ip_premium * min(1, patent_count / 5)
            
            # Calculate total value
            per_eng_value = base_per_eng * (1 + tech_adjustment + ip_adjustment)
            valuation = eng_count * per_eng_value * 1_000_000  # Convert to dollars
        
        else:
            # Default calculation for other exit types
            valuation = future_annual_revenue * 5  # Conservative multiple
        
        return max(valuation, 0)

    def _project_revenue(self, current_revenue: float, growth_rate: float, years: float) -> float:
        """
        Project future revenue based on current revenue and growth rate.
        
        Args:
            current_revenue: Current annual revenue
            growth_rate: Annual growth rate
            years: Number of years to project
            
        Returns:
            Projected annual revenue
        """
        # Adjust growth rate to be more conservative over time
        # Use a decay factor to represent slowing growth as company matures
        if years <= 0:
            return current_revenue
            
        if growth_rate <= 0:
            return current_revenue
        
        # Calculate compound growth with decay
        decay_factor = 0.8  # Reduce growth rate by 20% per year
        cumulative_growth = 1.0
        current_growth = growth_rate
        
        for year in range(int(years)):
            cumulative_growth *= (1 + current_growth)
            current_growth *= decay_factor
        
        # Handle partial year if needed
        fractional_year = years - int(years)
        if fractional_year > 0:
            cumulative_growth *= (1 + current_growth * fractional_year)
        
        return current_revenue * cumulative_growth

    def _project_ebitda(self, years: float) -> float:
        """
        Project future EBITDA based on current financials and growth.
        
        Args:
            years: Number of years to project
            
        Returns:
            Projected annual EBITDA
        """
        # Calculate current EBITDA
        monthly_ebitda = self.monthly_revenue - self.burn_rate
        current_ebitda = monthly_ebitda * 12
        
        # Project revenue
        future_revenue = self._project_revenue(self.annual_revenue, self.growth_rate, years)
        
        # If currently profitable, assume improving margins with scale
        if monthly_ebitda > 0:
            current_ebitda_margin = current_ebitda / self.annual_revenue
            # Assume margins improve with scale
            projected_margin = min(0.3, current_ebitda_margin * (1 + 0.1 * years))
            projected_ebitda = future_revenue * projected_margin
        else:
            # If currently unprofitable, project path to profitability
            # Assume profitability after 2-4 years depending on growth rate
            years_to_profitability = max(1, 4 - self.growth_rate * 10)
            
            if years < years_to_profitability:
                # Still unprofitable at exit
                burn_rate_reduction = years / years_to_profitability
                projected_monthly_burn = self.burn_rate * (1 - burn_rate_reduction)
                projected_monthly_revenue = future_revenue / 12
                projected_monthly_ebitda = projected_monthly_revenue - projected_monthly_burn
                projected_ebitda = projected_monthly_ebitda * 12
            else:
                # Profitable at exit
                years_of_profitability = years - years_to_profitability
                target_margin = min(0.2, 0.05 + 0.03 * years_of_profitability)
                projected_ebitda = future_revenue * target_margin
        
        return projected_ebitda

    def _calculate_success_probability(self, path_name: str, path_config: Dict[str, Any]) -> float:
        """
        Calculate probability of successfully reaching this exit path.
        
        Args:
            path_name: Name of the exit path
            path_config: Configuration for the exit path
            
        Returns:
            Probability as a float (0-1)
        """
        # Base probability depends on current stage
        stage_probabilities = {
            "pre-seed": 0.1,
            "seed": 0.2,
            "series-a": 0.3,
            "series-b": 0.5,
            "series-c": 0.7,
            "growth": 0.8
        }
        
        base_probability = stage_probabilities.get(self.stage, 0.3)
        
        # Get path-specific requirements
        requirements = path_config.get("requirements", {})
        
        # Adjust probability based on path and company specifics
        if path_name == "ipo":
            # Check revenue against minimum
            min_revenue = requirements.get("min_revenue", 100_000_000)
            revenue_factor = min(1, self.annual_revenue / min_revenue)
            
            # Check growth against minimum
            min_growth = requirements.get("min_growth", 0.3)
            growth_factor = min(1, self.growth_rate / min_growth)
            
            # Check team maturity
            team_maturity_req = requirements.get("team_maturity", 0.8)
            team_factor = min(1, self.team_score / 100 / team_maturity_req)
            
            # Calculate adjustment
            adjustment = (revenue_factor * 0.4 + growth_factor * 0.4 + team_factor * 0.2) / 2
            
        elif path_name in ["strategic_acquisition", "financial_acquisition"]:
            # Strategic buyers care about fit, financial about metrics
            if path_name == "strategic_acquisition":
                # Check product integration potential
                integration_req = requirements.get("product_integration", 0.7)
                moat_factor = min(1, self.moat_score / 100 / integration_req)
                
                # Check strategic fit based on market position
                market_factor = min(1, self.market_score / 100)
                
                # Previous founder exits increase probability
                founder_exit_factor = min(1, self.founder_exits / 2)
                
                # Calculate adjustment
                adjustment = (moat_factor * 0.4 + market_factor * 0.3 + founder_exit_factor * 0.3) / 2
                
            else:  # financial_acquisition
                # Check revenue against minimum
                min_revenue = requirements.get("min_revenue", 5_000_000)
                revenue_factor = min(1, self.annual_revenue / min_revenue)
                
                # Check margins
                min_margin = requirements.get("min_margin", 0.5)
                margin_factor = min(1, self.gross_margin / min_margin)
                
                # Check positive EBITDA requirement
                positive_ebitda_req = requirements.get("positive_ebitda", 0.6)
                ebitda_factor = 0.5  # Default
                if self.monthly_revenue > self.burn_rate:
                    ebitda_factor = 1.0
                
                # Calculate adjustment
                adjustment = (revenue_factor * 0.4 + margin_factor * 0.3 + ebitda_factor * 0.3) / 2
                
        elif path_name == "pe_buyout":
            # Check revenue against minimum
            min_revenue = requirements.get("min_revenue", 20_000_000)
            revenue_factor = min(1, self.annual_revenue / min_revenue)
            
            # Check margins
            min_margin = requirements.get("min_margin", 0.4)
            margin_factor = min(1, self.gross_margin / min_margin)
            
            # Check positive EBITDA requirement
            positive_ebitda_req = requirements.get("positive_ebitda", 0.8)
            ebitda_factor = 0.2  # Default low score
            if self.monthly_revenue > self.burn_rate:
                ebitda_margin = (self.monthly_revenue - self.burn_rate) / self.monthly_revenue
                ebitda_factor = min(1, ebitda_margin / 0.2)
            
            # Calculate adjustment
            adjustment = (revenue_factor * 0.3 + margin_factor * 0.3 + ebitda_factor * 0.4) / 2
                
        elif path_name == "acquihire":
            # Check team quality
            min_team_score = requirements.get("min_team_score", 70)
            team_factor = min(1, self.team_score / min_team_score)
            
            # Check tech talent ratio
            min_tech_ratio = requirements.get("min_tech_talent_ratio", 0.6)
            tech_talent_ratio = self.company_data.get("tech_talent_ratio", 0.4)
            tech_factor = min(1, tech_talent_ratio / min_tech_ratio)
            
            # Check tech innovation
            innovative_tech_req = requirements.get("innovative_tech", 0.7)
            tech_innovation = self.company_data.get("technical_innovation_score", 50) / 100
            innovation_factor = min(1, tech_innovation / innovative_tech_req)
            
            # Calculate adjustment
            adjustment = (team_factor * 0.4 + tech_factor * 0.3 + innovation_factor * 0.3) / 1.5
            
        else:
            # Default adjustment
            adjustment = 0.0
        
        # Apply market condition factors
        market_conditions = self.market_data.get("market_conditions", {})
        if path_name == "ipo":
            market_factor = market_conditions.get("ipo_favorability", 0.7)
        elif path_name in ["strategic_acquisition", "financial_acquisition"]:
            market_factor = market_conditions.get("acquisition_favorability", 0.8)
        elif path_name == "pe_buyout":
            market_factor = market_conditions.get("pe_favorability", 0.6)
        else:
            market_factor = 0.7  # Default
        
        # Calculate final probability with market adjustment
        final_probability = base_probability * (1 + adjustment) * market_factor
        
        # Ensure probability is between 0 and 1
        return max(0.05, min(0.95, final_probability))

    def _calculate_risk_factor(self, path_name: str, path_config: Dict[str, Any]) -> float:
        """
        Calculate the risk factor for a specific exit path.
        
        Args:
            path_name: Name of the exit path
            path_config: Configuration for the exit path
            
        Returns:
            Risk factor as a float (0-1)
        """
        # Get risk factors for this path
        risk_factors = path_config.get("risk_factors", {})
        
        # Calculate base risk based on stage
        stage_risks = {
            "pre-seed": 0.8,
            "seed": 0.7,
            "series-a": 0.6,
            "series-b": 0.5,
            "series-c": 0.4,
            "growth": 0.3
        }
        
        base_risk = stage_risks.get(self.stage, 0.6)
        
        # Calculate weighted risk based on path-specific factors
        weighted_risk = 0.0
        total_weight = 0.0
        
        for risk_name, risk_weight in risk_factors.items():
            risk_value = self._calculate_specific_risk(path_name, risk_name)
            weighted_risk += risk_value * risk_weight
            total_weight += risk_weight
        
        # Combine base risk with weighted path-specific risks
        if total_weight > 0:
            path_risk = weighted_risk / total_weight
            final_risk = base_risk * 0.4 + path_risk * 0.6
        else:
            final_risk = base_risk
        
        # Ensure risk is between 0 and 1
        return max(0.1, min(0.9, final_risk))

    def _calculate_specific_risk(self, path_name: str, risk_name: str) -> float:
        """
        Calculate specific risk factor based on company data.
        
        Args:
            path_name: Name of the exit path
            risk_name: Name of the risk factor
            
        Returns:
            Risk value as a float (0-1)
        """
        # Market volatility risk
        if risk_name == "market_volatility":
            # Higher risk for capital-intensive or cyclical sectors
            if self.sector in ["biotech", "hardware", "manufacturing"]:
                return 0.7
            elif self.sector in ["fintech", "ecommerce"]:
                return 0.6
            elif self.sector in ["saas", "software"]:
                return 0.5
            else:
                return 0.6
        
        # Execution risk
        elif risk_name == "execution_risk":
            # Lower risk with experienced team
            if self.team_score > 80:
                return 0.3
            elif self.team_score > 60:
                return 0.5
            else:
                return 0.7
        
        # Competition risk
        elif risk_name == "competition_risk":
            # Lower risk with strong moat
            if self.moat_score > 80:
                return 0.3
            elif self.moat_score > 60:
                return 0.5
            else:
                return 0.7
        
        # Regulatory risk
        elif risk_name == "regulatory_risk":
            # Higher for regulated industries
            if self.sector in ["fintech", "biotech", "healthcare"]:
                return 0.8
            elif self.sector in ["crypto", "transportation"]:
                return 0.7
            else:
                return 0.4
        
        # Strategic fit risk (for acquisitions)
        elif risk_name == "strategic_fit_risk":
            # Lower for companies with strong market position
            if self.market_score > 80:
                return 0.3
            elif self.market_score > 60:
                return 0.5
            else:
                return 0.7
        
        # Integration risk (for acquisitions)
        elif risk_name == "integration_risk":
            # Higher for complex products or technologies
            tech_score = self.company_data.get("product_maturity_score", 50)
            if tech_score > 80:
                return 0.4  # Mature product is easier to integrate
            elif tech_score > 60:
                return 0.6
            else:
                return 0.8  # Immature product has high integration risk
        
        # Culture risk (for acquisitions)
        elif risk_name == "culture_risk":
            # Standard risk unless company has specific culture metrics
            return 0.6
        
        # Financial performance risk
        elif risk_name == "financial_performance_risk":
            # Lower for companies with good financials
            if self.monthly_revenue > self.burn_rate:
                return 0.4  # Profitable
            elif self.monthly_revenue > self.burn_rate * 0.7:
                return 0.6  # Near profitable
            else:
                return 0.8  # Far from profitable
        
        # Team retention risk (for acquihires)
        elif risk_name == "team_retention_risk":
            # Lower for teams with good management satisfaction
            mgmt_satisfaction = self.company_data.get("management_satisfaction_score", 50) / 100
            return 0.8 - (mgmt_satisfaction * 0.5)
        
        # Default risk value
        return 0.5

    def _extract_path_requirements(self, path_name: str, path_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and normalize key requirements for a specific exit path.
        
        Args:
            path_name: Name of the exit path
            path_config: Configuration for the exit path
            
        Returns:
            Dictionary of key requirements with normalized values
        """
        requirements = path_config.get("requirements", {})
        normalized_req = {}
        
        # Convert requirements to a format suitable for UI display
        for req_name, req_value in requirements.items():
            if req_name == "min_revenue":
                normalized_req["Annual Revenue"] = req_value / 1_000_000  # Convert to millions
            elif req_name == "min_growth":
                normalized_req["Annual Growth Rate"] = req_value * 100  # Convert to percentage
            elif req_name == "min_margin":
                normalized_req["Gross Margin"] = req_value * 100  # Convert to percentage
            elif req_name == "positive_ebitda":
                if req_value > 0.5:
                    normalized_req["Positive EBITDA"] = "Required"
                else:
                    normalized_req["Positive EBITDA"] = "Preferred"
            elif req_name == "min_team_score":
                normalized_req["Team Quality"] = req_value
            elif req_name == "min_tech_talent_ratio":
                normalized_req["Tech Talent Ratio"] = req_value * 100  # Convert to percentage
            elif req_name == "team_maturity":
                normalized_req["Team Maturity"] = req_value * 100  # Convert to percentage
            elif req_name == "product_integration":
                normalized_req["Product Integration Potential"] = req_value * 100  # Convert to percentage
            elif req_name == "innovative_tech":
                normalized_req["Technical Innovation"] = req_value * 100  # Convert to percentage
            else:
                normalized_req[req_name.replace("_", " ").title()] = req_value
        
        return normalized_req

    def _calculate_npv(self, future_value: float, years: float) -> float:
        """
        Calculate Net Present Value.
        
        Args:
            future_value: Future exit valuation
            years: Time to exit in years
            
        Returns:
            Net Present Value
        """
        discount_rate = self.market_data.get("discount_rate", 0.25)
        if years <= 0 or discount_rate >= 1:
            return future_value
        
        return future_value / ((1 + discount_rate) ** years)

    def _calculate_irr(self, exit_valuation: float, years: float) -> float:
        """
        Calculate Internal Rate of Return.
        
        Args:
            exit_valuation: Exit valuation
            years: Time to exit in years
            
        Returns:
            IRR as a percentage
        """
        try:
            # Estimate initial investment based on current valuation
            current_valuation = self._estimate_current_valuation()
            
            # If we can't calculate a meaningful current valuation, use a fallback
            if current_valuation <= 0:
                if self.annual_revenue > 0:
                    current_valuation = self.annual_revenue * 3  # Conservative multiple
                else:
                    current_valuation = max(1_000_000, self.company_data.get("current_cash", 1_000_000) * 2)
            
            # Avoid division by zero or negative values
            if years <= 0 or current_valuation <= 0 or exit_valuation <= 0:
                return 0.0
            
            # Calculate IRR
            irr = (exit_valuation / current_valuation) ** (1 / years) - 1
            
            # Convert to percentage and ensure reasonable range
            return max(0, min(2, irr)) * 100  # Cap at 200% IRR
            
        except Exception as e:
            logger.error(f"Error calculating IRR: {str(e)}")
            return 0.0

    def _estimate_current_valuation(self) -> float:
        """
        Estimate current company valuation based on available metrics.
        
        Returns:
            Estimated current valuation
        """
        # Use different methods depending on company stage and metrics
        
        # If company is profitable, use EBITDA multiple
        monthly_ebitda = self.monthly_revenue - self.burn_rate
        if monthly_ebitda > 0:
            annual_ebitda = monthly_ebitda * 12
            ebitda_multiple = 10  # Conservative multiple
            return annual_ebitda * ebitda_multiple
        
        # If company has revenue, use revenue multiple based on sector and growth
        if self.annual_revenue > 0:
            # Get appropriate multiple based on sector
            sector_multiples = {
                "saas": [4, 10],
                "fintech": [3, 8],
                "ecommerce": [1, 3],
                "biotech": [5, 12],
                "default": [2, 6]
            }
            
            multiple_range = sector_multiples.get(self.sector, sector_multiples["default"])
            
            # Adjust multiple based on growth rate
            base_multiple = multiple_range[0]
            max_multiple = multiple_range[1]
            
            growth_adjustment = min(1, self.growth_rate / 0.5)  # Normalize growth to 0-1 scale
            adjusted_multiple = base_multiple + (max_multiple - base_multiple) * growth_adjustment
            
            return self.annual_revenue * adjusted_multiple
        
        # For pre-revenue companies, use a combination of methods
        # 1. Team value
        team_value = self.team_score * 100_000  # $100K per team score point
        
        # 2. Cash on hand
        cash_value = self.cash_on_hand * 1.5  # 1.5x cash on hand
        
        # 3. IP/tech value
        patent_count = self.company_data.get("patent_count", 0)
        tech_score = self.company_data.get("technical_innovation_score", 50)
        ip_value = (patent_count * 500_000) + (tech_score * 50_000)
        
        # Combine methods
        return max(1_000_000, team_value + cash_value + ip_value)

    def _determine_optimal_path(self, scenarios: List[ExitPathScenario]) -> str:
        """
        Determine the optimal exit path based on scenarios.
        
        Args:
            scenarios: List of exit path scenarios
            
        Returns:
            Name of the optimal exit path
        """
        if not scenarios:
            return ""
        
        # Sort by risk-adjusted value (already done in _generate_exit_scenarios)
        optimal = scenarios[0].path_name
        
        return optimal

    def _generate_timeline_data(self, scenarios: List[ExitPathScenario]) -> Dict[str, List[float]]:
        """
        Generate timeline data for visualization.
        
        Args:
            scenarios: List of exit path scenarios
            
        Returns:
            Dictionary with timeline data
        """
        timeline = {
            "years": [],
            "valuations": [],
            "paths": []
        }
        
        # Create year range from 0 to max exit time
        if not scenarios:
            return timeline
            
        max_years = max(scenario.time_to_exit for scenario in scenarios)
        years = np.linspace(0, max_years, num=20)
        
        # Add years to timeline
        timeline["years"] = years.tolist()
        
        # Generate valuation trajectory for each path
        for scenario in scenarios:
            path_valuations = []
            
            # Current valuation estimate
            current_valuation = self._estimate_current_valuation()
            
            for year in years:
                if year == 0:
                    path_valuations.append(current_valuation)
                elif year >= scenario.time_to_exit:
                    path_valuations.append(scenario.exit_valuation)
                else:
                    # Interpolate valuation between current and exit
                    progress = year / scenario.time_to_exit
                    # Use non-linear growth for more realistic trajectory
                    adjusted_progress = progress ** 0.8  # Slightly front-loaded growth
                    interpolated_val = current_valuation + adjusted_progress * (scenario.exit_valuation - current_valuation)
                    path_valuations.append(interpolated_val)
            
            timeline[scenario.path_name] = path_valuations
            timeline["paths"].append(scenario.path_name)
        
        return timeline

    def _calculate_success_factors(self) -> Dict[str, float]:
        """
        Calculate success factors for different exit paths.
        
        Returns:
            Dictionary mapping exit paths to success factor scores
        """
        factors = {}
        
        # IPO readiness factors
        ipo_factor = 0.0
        if self.annual_revenue > 0:
            min_ipo_revenue = 100_000_000
            revenue_factor = min(1, self.annual_revenue / min_ipo_revenue)
            growth_factor = min(1, self.growth_rate / 0.3)
            margin_factor = min(1, self.gross_margin / 0.6)
            team_factor = self.team_score / 100
            
            ipo_factor = revenue_factor * 0.4 + growth_factor * 0.3 + margin_factor * 0.2 + team_factor * 0.1
            factors["ipo"] = max(0, min(1, ipo_factor)) * 100
        else:
            factors["ipo"] = 0
        
        # Strategic acquisition factors
        strategic_factor = 0.0
        if self.annual_revenue > 0:
            market_factor = self.market_score / 100
            moat_factor = self.moat_score / 100
            growth_factor = min(1, self.growth_rate / 0.2)
            
            strategic_factor = market_factor * 0.3 + moat_factor * 0.4 + growth_factor * 0.3
            factors["strategic_acquisition"] = max(0, min(1, strategic_factor)) * 100
        else:
            # Even pre-revenue companies can have strategic value
            moat_factor = self.moat_score / 100
            innovation_factor = self.company_data.get("technical_innovation_score", 50) / 100
            patent_factor = min(1, self.company_data.get("patent_count", 0) / 3)
            
            strategic_factor = moat_factor * 0.4 + innovation_factor * 0.4 + patent_factor * 0.2
            factors["strategic_acquisition"] = max(0, min(1, strategic_factor)) * 100
        
        # PE buyout factors
        pe_factor = 0.0
        if self.annual_revenue > 1_000_000:
            revenue_factor = min(1, self.annual_revenue / 20_000_000)
            margin_factor = min(1, self.gross_margin / 0.4)
            ebitda_factor = 0
            if self.monthly_revenue > self.burn_rate:
                ebitda_margin = (self.monthly_revenue - self.burn_rate) / self.monthly_revenue
                ebitda_factor = min(1, ebitda_margin / 0.2)
            
            pe_factor = revenue_factor * 0.3 + margin_factor * 0.3 + ebitda_factor * 0.4
            factors["pe_buyout"] = max(0, min(1, pe_factor)) * 100
        else:
            factors["pe_buyout"] = 0
        
        # Acquihire factors
        team_factor = self.team_score / 100
        tech_talent_ratio = self.company_data.get("tech_talent_ratio", 0.4)
        tech_talent_factor = min(1, tech_talent_ratio / 0.6)
        tech_innovation = self.company_data.get("technical_innovation_score", 50) / 100
        
        acquihire_factor = team_factor * 0.5 + tech_talent_factor * 0.3 + tech_innovation * 0.2
        factors["acquihire"] = max(0, min(1, acquihire_factor)) * 100
        
        return factors

    def _generate_milestones(self, scenarios: List[ExitPathScenario], optimal_path: str) -> List[Dict[str, Any]]:
        """
        Generate recommended milestones based on exit scenarios.
        
        Args:
            scenarios: List of exit path scenarios
            optimal_path: Name of the optimal exit path
            
        Returns:
            List of milestone dictionaries
        """
        milestones = []
        
        if not scenarios:
            return milestones
            
        # Find the optimal scenario
        optimal_scenario = next((s for s in scenarios if s.path_name == optimal_path), None)
        if not optimal_scenario:
            return milestones
        
        # Generate milestones based on the optimal path
        if optimal_path == "ipo":
            # Financial milestones
            target_arr = 100_000_000
            if self.annual_revenue < target_arr:
                revenue_milestone = {
                    "title": f"Reach ${target_arr / 1_000_000:.0f}M ARR",
                    "description": "Achieve the minimum revenue scale typically required for a successful IPO",
                    "timeline": f"{max(1, int(self._years_to_target_revenue(target_arr)))} years",
                    "category": "financial"
                }
                milestones.append(revenue_milestone)
            
            # Growth milestone
            if self.growth_rate < 0.3:
                growth_milestone = {
                    "title": "Maintain 30%+ YoY Growth",
                    "description": "Public markets typically value high-growth companies at premium multiples",
                    "timeline": "Ongoing",
                    "category": "financial"
                }
                milestones.append(growth_milestone)
            
            # Team milestone
            team_milestone = {
                "title": "Build Public Company Executive Team",
                "description": "Hire key executives with public company experience (CFO, CRO, etc.)",
                "timeline": f"{max(1, int(optimal_scenario.time_to_exit) - 2)} years",
                "category": "team"
            }
            milestones.append(team_milestone)
            
            # Governance milestone
            governance_milestone = {
                "title": "Establish Board and Governance",
                "description": "Form independent board, audit committee, and public company governance structure",
                "timeline": f"{max(1, int(optimal_scenario.time_to_exit) - 1)} years",
                "category": "governance"
            }
            milestones.append(governance_milestone)
        
        elif optimal_path in ["strategic_acquisition", "financial_acquisition"]:
            # Determine acquisition target timeline
            acq_timeline = max(1, int(optimal_scenario.time_to_exit))
            
            # Strategic positioning milestone
            positioning_milestone = {
                "title": "Develop Strategic Partnership Relationships",
                "description": "Build relationships with potential acquirers through partnerships and integrations",
                "timeline": f"{max(1, acq_timeline - 2)} years",
                "category": "strategic"
            }
            milestones.append(positioning_milestone)
            
            # Growth milestone
            target_arr = 20_000_000
            if self.annual_revenue < target_arr:
                revenue_milestone = {
                    "title": f"Reach ${target_arr / 1_000_000:.0f}M ARR",
                    "description": "Achieve revenue scale attractive to potential acquirers",
                    "timeline": f"{max(1, int(self._years_to_target_revenue(target_arr)))} years",
                    "category": "financial"
                }
                milestones.append(revenue_milestone)
            
            # Profitability milestone (for financial acquisitions)
            if optimal_path == "financial_acquisition" and self.monthly_revenue <= self.burn_rate:
                profit_milestone = {
                    "title": "Achieve Profitability",
                    "description": "Financial acquirers typically value profitable companies more highly",
                    "timeline": f"{max(1, acq_timeline - 1)} years",
                    "category": "financial"
                }
                milestones.append(profit_milestone)
            
            # Product milestone (for strategic acquisitions)
            if optimal_path == "strategic_acquisition":
                product_milestone = {
                    "title": "Strengthen Core IP and Integrations",
                    "description": "Develop unique IP and integration capabilities valuable to strategic acquirers",
                    "timeline": "Ongoing",
                    "category": "product"
                }
                milestones.append(product_milestone)
        
        elif optimal_path == "pe_buyout":
            # Financial milestones
            target_arr = 30_000_000
            if self.annual_revenue < target_arr:
                revenue_milestone = {
                    "title": f"Reach ${target_arr / 1_000_000:.0f}M ARR",
                    "description": "Achieve minimum revenue scale attractive to PE firms",
                    "timeline": f"{max(1, int(self._years_to_target_revenue(target_arr)))} years",
                    "category": "financial"
                }
                milestones.append(revenue_milestone)
            
            # Profitability milestone
            if self.monthly_revenue <= self.burn_rate:
                profit_milestone = {
                    "title": "Achieve Profitability",
                    "description": "PE firms typically require consistent profitability for buyout candidates",
                    "timeline": f"{max(1, int(optimal_scenario.time_to_exit) - 2)} years",
                    "category": "financial"
                }
                milestones.append(profit_milestone)
            
            # Debt capacity milestone
            debt_milestone = {
                "title": "Build Debt Capacity",
                "description": "Develop strong balance sheet and predictable cash flows to support leverage",
                "timeline": f"{max(1, int(optimal_scenario.time_to_exit) - 1)} years",
                "category": "financial"
            }
            milestones.append(debt_milestone)
        
        elif optimal_path == "acquihire":
            # Team milestone
            team_milestone = {
                "title": "Build Elite Technical Team",
                "description": "Recruit top technical talent to increase team value",
                "timeline": "12 months",
                "category": "team"
            }
            milestones.append(team_milestone)
            
            # Technology milestone
            tech_milestone = {
                "title": "Develop Innovative IP",
                "description": "Create differentiated technology and IP that demonstrates team capabilities",
                "timeline": "18 months",
                "category": "product"
            }
            milestones.append(tech_milestone)
            
            # Networking milestone
            network_milestone = {
                "title": "Build Relationships with Potential Acquirers",
                "description": "Develop connections at target acquirer companies",
                "timeline": "Ongoing",
                "category": "strategic"
            }
            milestones.append(network_milestone)
        
        return milestones

    def _years_to_target_revenue(self, target_revenue: float) -> float:
        """
        Calculate years to reach target revenue based on current growth.
        
        Args:
            target_revenue: Target annual revenue
            
        Returns:
            Years to reach target
        """
        if self.annual_revenue >= target_revenue:
            return 0
        
        if self.growth_rate <= 0:
            return float('inf')
        
        # Using compound growth formula: FV = PV(1+r)^t
        # Solving for t: t = log(FV/PV)/log(1+r)
        return math.log(target_revenue / self.annual_revenue) / math.log(1 + self.growth_rate)

    def _perform_sensitivity_analysis(self, scenarios: List[ExitPathScenario], optimal_path: str) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on key factors affecting exit valuation.
        
        Args:
            scenarios: List of exit path scenarios
            optimal_path: Name of the optimal exit path
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        result = {
            "optimal_path": optimal_path,
            "parameters": {},
            "valuation_impact": {}
        }
        
        if not scenarios:
            return result
            
        # Find the optimal scenario
        optimal_scenario = next((s for s in scenarios if s.path_name == optimal_path), None)
        if not optimal_scenario:
            return result
        
        # Define parameters to test and their variation ranges
        parameters = {
            "growth_rate": [self.growth_rate * 0.5, self.growth_rate, self.growth_rate * 1.5],
            "gross_margin": [max(0.1, self.gross_margin - 0.1), self.gross_margin, min(0.9, self.gross_margin + 0.1)],
            "time_to_exit": [max(1, optimal_scenario.time_to_exit - 1), optimal_scenario.time_to_exit, optimal_scenario.time_to_exit + 1]
        }
        
        result["parameters"] = parameters
        
        # Calculate valuation impact for each parameter variation
        for param, values in parameters.items():
            # Skip if we only have one value (can't calculate sensitivity)
            if len(set(values)) <= 1:
                continue
                
            valuations = []
            
            # Create a temporary copy of the analyzer to modify parameters
            temp_analyzer = self
            
            for value in values:
                # Set the parameter value
                if param == "growth_rate":
                    temp_growth = value
                    exit_val = self._calculate_with_modified_growth(optimal_path, temp_growth, optimal_scenario.time_to_exit)
                elif param == "gross_margin":
                    temp_margin = value
                    exit_val = self._calculate_with_modified_margin(optimal_path, temp_margin, optimal_scenario.time_to_exit)
                elif param == "time_to_exit":
                    exit_val = self._calculate_exit_valuation(optimal_path, value)
                else:
                    exit_val = optimal_scenario.exit_valuation
                
                valuations.append(exit_val)
            
            # Calculate percentage impact from base case
            base_index = 1  # Middle value is the base case
            base_val = valuations[base_index]
            
            pct_changes = []
            for val in valuations:
                if base_val == 0:
                    pct_changes.append(0)
                else:
                    pct_change = (val - base_val) / base_val * 100
                    pct_changes.append(pct_change)
            
            result["valuation_impact"][param] = {
                "values": values,
                "valuations": valuations,
                "pct_changes": pct_changes
            }
        
        return result

    def _calculate_with_modified_growth(self, path_name: str, growth_rate: float, time_to_exit: float) -> float:
        """
        Calculate exit valuation with a modified growth rate.
        
        Args:
            path_name: Name of the exit path
            growth_rate: Modified growth rate
            time_to_exit: Time to exit in years
            
        Returns:
            Modified exit valuation
        """
        # Save original growth rate
        original_growth = self.growth_rate
        
        # Temporarily modify growth rate
        self.growth_rate = growth_rate
        
        # Calculate valuation with modified growth
        valuation = self._calculate_exit_valuation(path_name, time_to_exit)
        
        # Restore original growth rate
        self.growth_rate = original_growth
        
        return valuation

    def _calculate_with_modified_margin(self, path_name: str, gross_margin: float, time_to_exit: float) -> float:
        """
        Calculate exit valuation with a modified gross margin.
        
        Args:
            path_name: Name of the exit path
            gross_margin: Modified gross margin
            time_to_exit: Time to exit in years
            
        Returns:
            Modified exit valuation
        """
        # Save original margin
        original_margin = self.gross_margin
        
        # Temporarily modify margin
        self.gross_margin = gross_margin
        
        # Calculate valuation with modified margin
        valuation = self._calculate_exit_valuation(path_name, time_to_exit)
        
        # Restore original margin
        self.gross_margin = original_margin
        
        return valuation

    def _find_comparable_exits(self) -> List[Dict[str, Any]]:
        """
        Find comparable company exits based on sector and metrics.
        
        Returns:
            List of comparable exit dictionaries
        """
        # In a real implementation, this would query a database of exits
        # Here we'll generate synthetic comparables based on company data
        
        comparables = []
        
        # Generate 3-5 relevant comparables based on sector
        count = min(5, max(3, self.company_data.get("category_leadership_score", 50) // 20 + 3))
        
        for i in range(count):
            # Base exit amount on company size with some variation
            size_factor = 0.5 + (i / count) * 2  # Varies from 0.5x to 2.5x
            
            if self.annual_revenue > 0:
                exit_amount = self.annual_revenue * 10 * size_factor  # Roughly 10x ARR with variation
            else:
                exit_amount = 5_000_000 * size_factor  # Base amount for pre-revenue
            
            # Vary the exit type
            exit_types = ["acquisition", "ipo", "pe_buyout", "merger", "acquisition"]
            exit_type = exit_types[i % len(exit_types)]
            
            # Generate comparable details
            comparable = {
                "company_name": f"Example {self.sector.capitalize()} {i+1}",
                "sector": self.sector,
                "exit_year": 2021 - i,  # Recent exits
                "exit_type": exit_type,
                "exit_amount": exit_amount,
                "revenue_at_exit": exit_amount / (8 + i % 5),  # Varies multiple from 8-12x
                "years_from_founding": 5 + i,
                "venture_rounds": 1 + i % 4,
                "notes": self._generate_comparable_notes(exit_type, exit_amount)
            }
            
            comparables.append(comparable)
        
        return comparables

    def _generate_comparable_notes(self, exit_type: str, exit_amount: float) -> str:
        """
        Generate notes for comparable exits.
        
        Args:
            exit_type: Type of exit
            exit_amount: Exit amount
            
        Returns:
            Notes string
        """
        if exit_type == "acquisition":
            acquirers = ["Google", "Microsoft", "Amazon", "Meta", "Apple", "Salesforce", "Adobe"]
            acquirer = acquirers[hash(str(exit_amount)) % len(acquirers)]
            return f"Acquired by {acquirer} to strengthen their {self.sector} offerings"
        
        elif exit_type == "ipo":
            exchanges = ["NYSE", "NASDAQ"]
            exchange = exchanges[hash(str(exit_amount)) % len(exchanges)]
            return f"Went public on {exchange} with strong institutional interest"
        
        elif exit_type == "pe_buyout":
            pe_firms = ["KKR", "Blackstone", "Vista Equity", "TPG", "Thoma Bravo"]
            firm = pe_firms[hash(str(exit_amount)) % len(pe_firms)]
            return f"Acquired by {firm} to accelerate growth and profitability"
        
        elif exit_type == "merger":
            return f"Merged with a complementary {self.sector} company to gain scale"
        
        return "Successful exit with strong investor returns"

    def _calculate_exit_readiness_score(self) -> float:
        """
        Calculate a score representing the company's readiness for exit.
        
        Returns:
            Exit readiness score (0-100)
        """
        # Base score depends on stage
        stage_scores = {
            "pre-seed": 10,
            "seed": 20,
            "series-a": 40,
            "series-b": 60,
            "series-c": 75,
            "growth": 85
        }
        
        base_score = stage_scores.get(self.stage, 30)
        
        # Adjust based on key metrics
        adjustments = 0
        
        # Financial health
        if self.monthly_revenue > self.burn_rate:
            adjustments += 15  # Profitability is crucial for exit readiness
        elif self.monthly_revenue > self.burn_rate * 0.7:
            adjustments += 7  # Near profitability
        
        # Growth rate
        if self.growth_rate > 0.5:
            adjustments += 10  # High growth
        elif self.growth_rate > 0.3:
            adjustments += 7  # Good growth
        elif self.growth_rate > 0.15:
            adjustments += 3  # Moderate growth
        
        # Team quality
        if self.team_score > 80:
            adjustments += 10  # Strong team
        elif self.team_score > 60:
            adjustments += 5  # Good team
        
        # Founder experience (prior exits is a strong signal)
        if self.founder_exits > 0:
            adjustments += 10
        
        # Revenue scale
        if self.annual_revenue > 50_000_000:
            adjustments += 15  # Significant scale
        elif self.annual_revenue > 10_000_000:
            adjustments += 10  # Good scale
        elif self.annual_revenue > 1_000_000:
            adjustments += 5  # Some scale
        
        # Calculate final score
        final_score = base_score + adjustments
        
        # Ensure score is within bounds
        return max(0, min(100, final_score))

    def get_exit_recommendations(self) -> Dict[str, Any]:
        """
        Generate strategic recommendations based on exit path analysis.
        
        Returns:
            Dictionary with exit path recommendations
        """
        try:
            # Run analysis if not already done
            analysis = self.analyze_exit_paths()
            
            if not analysis or not analysis.scenarios:
                return {
                    "optimal_path": "",
                    "recommendations": [
                        "Insufficient data for exit path recommendations"
                    ],
                    "readiness": 0,
                    "timeline": {}
                }
            
            # Get key data from analysis
            optimal_path = analysis.optimal_path
            readiness = analysis.exit_readiness_score
            
            # Find optimal scenario
            optimal_scenario = next((s for s in analysis.scenarios if s.path_name == optimal_path), None)
            
            # Generate recommendations based on optimal path and readiness score
            general_recs = []
            path_specific_recs = []
            
            # General recommendations based on readiness
            if readiness < 30:
                general_recs.append("Focus on core business metrics before considering exit planning")
                general_recs.append("Establish product-market fit and sustainable growth trajectory")
            elif readiness < 50:
                general_recs.append("Begin exit planning while focusing on growth and scaling operations")
                general_recs.append("Develop relationships with potential acquirers or investors")
            elif readiness < 70:
                general_recs.append("Formalize exit strategy and timeline")
                general_recs.append("Strengthen financial reporting and governance")
            else:
                general_recs.append("Actively pursue optimal exit path with formal process")
                general_recs.append("Engage advisors to maximize exit value and manage process")
            
            # Path-specific recommendations
            if optimal_path == "ipo":
                path_specific_recs.append("Build public company infrastructure and reporting capabilities")
                path_specific_recs.append("Strengthen board with public company experience")
                path_specific_recs.append("Develop consistent growth and margin expansion narrative")
                
                if readiness < 60:
                    path_specific_recs.append("Consider expanding management team with public company experience")
                
            elif optimal_path == "strategic_acquisition":
                path_specific_recs.append("Identify and cultivate relationships with 3-5 potential strategic acquirers")
                path_specific_recs.append("Develop strategic partnerships that could lead to acquisition interest")
                path_specific_recs.append("Focus product development on integration capabilities with potential acquirers")
                
                if self.moat_score < 70:
                    path_specific_recs.append("Strengthen competitive moat and unique IP to increase strategic value")
                
            elif optimal_path == "financial_acquisition":
                path_specific_recs.append("Prioritize path to profitability and consistent financial performance")
                path_specific_recs.append("Develop strong financial controls and reporting")
                path_specific_recs.append("Focus on unit economics and operational efficiency")
                
                if self.monthly_revenue <= self.burn_rate:
                    path_specific_recs.append("Accelerate timeline to profitability to increase attractiveness to financial buyers")
                
            elif optimal_path == "pe_buyout":
                path_specific_recs.append("Demonstrate scalable, profitable business model attractive to PE firms")
                path_specific_recs.append("Develop predictable recurring revenue streams and clear growth levers")
                path_specific_recs.append("Strengthen management team with executives PE firms will back")
                
                if self.annual_revenue < 20_000_000:
                    path_specific_recs.append(f"Focus on scaling revenue to exceed minimum PE threshold (${20_000_000/1_000_000:.0f}M)")
            
            elif optimal_path == "acquihire":
                path_specific_recs.append("Focus on recruiting and retaining top technical talent")
                path_specific_recs.append("Develop innovative technology to showcase team capabilities")
                path_specific_recs.append("Network with larger companies seeking talent in your domain")
                
                if self.team_score < 70:
                    path_specific_recs.append("Strengthen team quality and technical credentials to increase acquihire value")
            
            # Build timeline data
            timeline = {}
            if optimal_scenario:
                timeline["years_to_exit"] = optimal_scenario.time_to_exit
                timeline["exit_valuation"] = optimal_scenario.exit_valuation
                timeline["exit_year"] = int(2024 + optimal_scenario.time_to_exit)
                timeline["milestones"] = analysis.recommended_milestones
            
            # Combine all recommendations
            recommendations = general_recs + path_specific_recs
            
            return {
                "optimal_path": optimal_path,
                "path_details": {
                    "name": optimal_path,
                    "description": self.exit_paths.get(optimal_path, {}).get("description", "")
                } if optimal_path else {},
                "recommendations": recommendations,
                "readiness": readiness,
                "timeline": timeline
            }
            
        except Exception as e:
            logger.error(f"Error generating exit recommendations: {str(e)}")
            return {
                "optimal_path": "",
                "recommendations": [
                    "Error generating exit recommendations"
                ],
                "readiness": 0,
                "timeline": {}
            }

# Add this function at the end of the file
def analyze_exit_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze comparative exit paths
    
    Parameters:
    data (dict): Input data about the company and market conditions
    
    Returns:
    dict: Comprehensive exit path analysis results
    """
    try:
        # Extract market data if provided
        market_data = data.get("market_data")
        
        # Create analyzer with company data
        analyzer = ExitPathAnalyzer(company_data=data, market_data=market_data)
        
        # Run analysis
        analysis_result = analyzer.analyze_exit_paths()
        
        # Convert dataclass objects to dictionaries
        if not isinstance(analysis_result, dict):
            result = {
                "scenarios": [vars(scenario) for scenario in analysis_result.scenarios],
                "optimal_path": analysis_result.optimal_path,
                "timeline_data": analysis_result.timeline_data,
                "success_factors": analysis_result.success_factors,
                "recommended_milestones": analysis_result.recommended_milestones,
                "sensitivity_analysis": analysis_result.sensitivity_analysis,
                "comparable_exits": analysis_result.comparable_exits,
                "exit_readiness_score": analysis_result.exit_readiness_score,
                "success": True
            }
        else:
            result = analysis_result
            result["success"] = True
            
        return result
    except Exception as e:
        # Return error with traceback for debugging
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }
