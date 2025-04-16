"""
Analysis Orchestrator Service for FlashDNA Startup Analysis Platform
Coordinates analysis across all specialized services and ensures proper data flow
"""
import sys
import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Import specialized services
from app.services.competitive_intelligence_service import competitive_intelligence_service
from app.services.ecosystem_map_service import ecosystem_map_service
from app.services.funding_trajectory_service import funding_trajectory_service
from app.services.monte_carlo_service import monte_carlo_service
from app.services.exit_path_service import exit_path_service
from app.services.benchmarking_service import benchmarking_service

# Set up logging
logger = logging.getLogger(__name__)

class AnalysisOrchestratorService:
    """Service that orchestrates analysis across all specialized services"""
    
    def __init__(self):
        """Initialize the analysis orchestrator service"""
        logger.info("AnalysisOrchestratorService initialized")
        
        # Register specialized services
        self.specialized_services = {
            "competitive_intelligence": competitive_intelligence_service,
            "ecosystem_map": ecosystem_map_service,
            "funding_trajectory": funding_trajectory_service,
            "monte_carlo": monte_carlo_service,
            "exit_path": exit_path_service,
            "benchmarking": benchmarking_service
        }
    
    async def perform_comprehensive_analysis(
        self, 
        startup_data: Dict[str, Any],
        include_specialized: bool = True,
        specialized_analyses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis including core and specialized analyses
        
        Args:
            startup_data: Core startup data
            include_specialized: Whether to include specialized analyses
            specialized_analyses: List of specific specialized analyses to include
                                  (if None, all are included)
            
        Returns:
            Dict containing complete analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for: {startup_data.get('name', 'Unknown Company')}")
            
            # Extract core metrics for basic analysis
            core_analysis = self._perform_core_analysis(startup_data)
            
            # Create results dictionary with core analysis
            results = {
                "core_analysis": core_analysis,
                "timestamp": self._get_timestamp(),
                "version": "1.0.0"
            }
            
            # Return early if specialized analyses are not needed
            if not include_specialized:
                return results
            
            # Determine which specialized analyses to perform
            if specialized_analyses is None:
                specialized_to_run = list(self.specialized_services.keys())
            else:
                specialized_to_run = [
                    analysis for analysis in specialized_analyses 
                    if analysis in self.specialized_services
                ]
            
            # Run specialized analyses in parallel
            specialized_results = await self._run_specialized_analyses(
                startup_data, 
                specialized_to_run
            )
            
            # Add specialized results to the main results
            results.update(specialized_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            # Return partial results if available, otherwise basic error response
            if 'results' in locals() and isinstance(results, dict):
                results["error"] = str(e)
                return results
            else:
                return {
                    "error": str(e),
                    "timestamp": self._get_timestamp(),
                    "version": "1.0.0"
                }
    
    def _perform_core_analysis(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform core analysis on the startup data
        
        Args:
            startup_data: Core startup data
            
        Returns:
            Dict containing core analysis results
        """
        try:
            # Extract metrics categories
            capital_metrics = self._extract_metrics(startup_data, "capital")
            advantage_metrics = self._extract_metrics(startup_data, "advantage")
            market_metrics = self._extract_metrics(startup_data, "market")
            people_metrics = self._extract_metrics(startup_data, "people")
            
            # Calculate core scores based on metrics
            capital_score = self._calculate_capital_score(capital_metrics)
            advantage_score = self._calculate_advantage_score(advantage_metrics)
            market_score = self._calculate_market_score(market_metrics)
            people_score = self._calculate_people_score(people_metrics)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                capital_score, advantage_score, market_score, people_score
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(startup_data, composite_score)
            recommendations = self._generate_recommendations(
                startup_data, capital_score, advantage_score, market_score, people_score
            )
            
            return {
                "scores": {
                    "capital": capital_score,
                    "advantage": advantage_score,
                    "market": market_score,
                    "people": people_score,
                    "composite": composite_score
                },
                "insights": insights,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in core analysis: {str(e)}")
            return {
                "scores": {
                    "capital": 0.5,
                    "advantage": 0.5,
                    "market": 0.5,
                    "people": 0.5,
                    "composite": 0.5
                },
                "insights": ["Error generating insights"],
                "recommendations": ["Error generating recommendations"],
                "error": str(e)
            }
    
    async def _run_specialized_analyses(
        self, 
        startup_data: Dict[str, Any],
        analyses_to_run: List[str]
    ) -> Dict[str, Any]:
        """
        Run specified specialized analyses in parallel
        
        Args:
            startup_data: Core startup data
            analyses_to_run: List of specialized analyses to run
            
        Returns:
            Dict containing specialized analysis results
        """
        try:
            # Create tasks for each specialized analysis
            tasks = {}
            for analysis_name in analyses_to_run:
                if analysis_name in self.specialized_services:
                    service = self.specialized_services[analysis_name]
                    
                    # Map service methods to their expected function names
                    method_mapping = {
                        "competitive_intelligence": "analyze_competitive_landscape",
                        "ecosystem_map": "analyze_ecosystem",
                        "funding_trajectory": "analyze_funding_trajectory",
                        "monte_carlo": "run_monte_carlo_simulation",
                        "exit_path": "analyze_exit_paths",
                        "benchmarking": "analyze_benchmarks"
                    }
                    
                    # Get the appropriate method for this service
                    method_name = method_mapping.get(analysis_name)
                    
                    if method_name and hasattr(service, method_name):
                        method = getattr(service, method_name)
                        # Create task for this analysis
                        tasks[analysis_name] = asyncio.create_task(method(startup_data))
                    else:
                        logger.warning(f"Method {method_name} not found for service {analysis_name}")
            
            # Wait for all tasks to complete
            results = {}
            for analysis_name, task in tasks.items():
                try:
                    # Rename keys to match frontend expectations
                    key_mapping = {
                        "competitive_intelligence": "competitiveIntelligence",
                        "ecosystem_map": "ecosystemMap",
                        "funding_trajectory": "fundingTrajectoryAnalysis",
                        "monte_carlo": "monteCarloAnalysis",
                        "exit_path": "exitPathAnalysis",
                        "benchmarking": "benchmarkingAnalysis"
                    }
                    
                    frontend_key = key_mapping.get(analysis_name, analysis_name)
                    results[frontend_key] = await task
                    
                except Exception as e:
                    logger.error(f"Error in specialized analysis {analysis_name}: {str(e)}")
                    results[key_mapping.get(analysis_name, analysis_name)] = {
                        "error": str(e),
                        "is_fallback": True
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running specialized analyses: {str(e)}")
            return {
                "error": f"Failed to run specialized analyses: {str(e)}"
            }
    
    def _extract_metrics(self, startup_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Extract metrics for a specific category from startup data
        
        Args:
            startup_data: Core startup data
            category: Category to extract (capital, advantage, market, people)
            
        Returns:
            Dict containing metrics for the specified category
        """
        # Direct access by category name
        if category in startup_data and isinstance(startup_data[category], dict):
            return startup_data[category]
        
        # Check for metrics naming pattern
        metrics_key = f"{category}_metrics"
        if metrics_key in startup_data and isinstance(startup_data[metrics_key], dict):
            return startup_data[metrics_key]
        
        # Return empty dict if no metrics found
        return {}
    
    def _calculate_capital_score(self, capital_metrics: Dict[str, Any]) -> float:
        """Calculate capital score from metrics"""
        try:
            # Extract key metrics with defaults
            annual_revenue = capital_metrics.get("annual_revenue", 0)
            burn_rate = capital_metrics.get("burn_rate", 1)
            gross_margin = capital_metrics.get("gross_margin", 0) / 100
            cac = capital_metrics.get("customer_acquisition_cost", 1000)
            ltv = capital_metrics.get("lifetime_value", 1000)
            runway_months = capital_metrics.get("runway_months", 3)
            
            # Calculate component scores
            revenue_score = min(1.0, annual_revenue / 10000000)
            burn_efficiency = min(1.0, (annual_revenue / burn_rate) / 5) if burn_rate > 0 else 0.5
            margin_score = gross_margin
            ltv_cac_score = min(1.0, (ltv / cac) / 5) if cac > 0 else 0.5
            runway_score = min(1.0, runway_months / 24)
            
            # Calculate weighted capital score
            capital_score = (
                revenue_score * 0.25 +
                burn_efficiency * 0.2 +
                margin_score * 0.2 +
                ltv_cac_score * 0.25 +
                runway_score * 0.1
            )
            
            return round(capital_score, 2)
        except Exception as e:
            logger.error(f"Error calculating capital score: {str(e)}")
            return 0.5
    
    def _calculate_advantage_score(self, advantage_metrics: Dict[str, Any]) -> float:
        """Calculate advantage score from metrics"""
        try:
            # Extract key metrics with defaults (assuming 0-10 scale)
            product_innovation = advantage_metrics.get("product_innovation", 5) / 10
            tech_diff = advantage_metrics.get("tech_differentiation", 5) / 10
            ip_strength = advantage_metrics.get("intellectual_property", 5) / 10
            switching_costs = advantage_metrics.get("switching_costs", 5) / 10
            network_effects = advantage_metrics.get("network_effects", 5) / 10
            
            # Calculate weighted advantage score
            advantage_score = (
                product_innovation * 0.25 +
                tech_diff * 0.25 +
                ip_strength * 0.15 +
                switching_costs * 0.15 +
                network_effects * 0.2
            )
            
            return round(advantage_score, 2)
        except Exception as e:
            logger.error(f"Error calculating advantage score: {str(e)}")
            return 0.5
    
    def _calculate_market_score(self, market_metrics: Dict[str, Any]) -> float:
        """Calculate market score from metrics"""
        try:
            # Extract key metrics with defaults
            tam = market_metrics.get("tam", 100000000)
            market_growth = market_metrics.get("market_growth_rate", 10) / 100
            retention = market_metrics.get("customer_retention_rate", 70) / 100
            market_share = market_metrics.get("market_share", 1) / 100
            revenue_growth = market_metrics.get("revenue_growth_rate", 20) / 100
            
            # Calculate component scores
            tam_score = min(1.0, tam / 10000000000)
            growth_score = min(1.0, market_growth * 5)
            retention_score = retention
            share_score = min(1.0, market_share * 20)
            rev_growth_score = min(1.0, revenue_growth * 2)
            
            # Calculate weighted market score
            market_score = (
                tam_score * 0.2 +
                growth_score * 0.2 +
                retention_score * 0.2 +
                share_score * 0.15 +
                rev_growth_score * 0.25
            )
            
            return round(market_score, 2)
        except Exception as e:
            logger.error(f"Error calculating market score: {str(e)}")
            return 0.5
    
    def _calculate_people_score(self, people_metrics: Dict[str, Any]) -> float:
        """Calculate people score from metrics"""
        try:
            # Extract key metrics with defaults (assuming 0-10 scale)
            founder_exp = people_metrics.get("founder_experience", 5) / 10
            tech_talent = people_metrics.get("technical_talent", 5) / 10
            sales_execution = people_metrics.get("sales_execution", 5) / 10
            leadership = people_metrics.get("leadership_team", 5) / 10
            retention = people_metrics.get("employee_retention", 80) / 100
            
            # Calculate weighted people score
            people_score = (
                founder_exp * 0.3 +
                tech_talent * 0.2 +
                sales_execution * 0.2 +
                leadership * 0.2 +
                retention * 0.1
            )
            
            return round(people_score, 2)
        except Exception as e:
            logger.error(f"Error calculating people score: {str(e)}")
            return 0.5
    
    def _calculate_composite_score(
        self, 
        capital_score: float, 
        advantage_score: float, 
        market_score: float, 
        people_score: float
    ) -> float:
        """Calculate composite score from component scores"""
        try:
            # Weighted composite score based on CAMP framework
            composite_score = (
                capital_score * 0.25 +
                advantage_score * 0.3 +
                market_score * 0.25 +
                people_score * 0.2
            )
            
            return round(composite_score, 2)
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0.5
    
    def _generate_insights(self, startup_data: Dict[str, Any], composite_score: float) -> List[str]:
        """Generate insights based on analysis"""
        # Basic insights as a fallback
        insights = [
            "Consider focusing on expanding your product's competitive advantage",
            "Your market growth rate suggests significant opportunity for expansion",
            "Improving your customer acquisition costs could increase capital efficiency"
        ]
        
        return insights
    
    def _generate_recommendations(
        self, 
        startup_data: Dict[str, Any],
        capital_score: float,
        advantage_score: float,
        market_score: float,
        people_score: float
    ) -> List[str]:
        """Generate recommendations based on component scores"""
        recommendations = []
        
        # Capital recommendations
        if capital_score < 0.4:
            recommendations.append("Focus on improving unit economics and extending runway")
        
        # Advantage recommendations
        if advantage_score < 0.4:
            recommendations.append("Strengthen your product differentiation and competitive moat")
        
        # Market recommendations
        if market_score < 0.4:
            recommendations.append("Identify higher-growth market segments and optimize customer retention")
        
        # People recommendations
        if people_score < 0.4:
            recommendations.append("Invest in team development and hiring key expertise gaps")
        
        # Add general recommendation if nothing specific
        if not recommendations:
            recommendations.append("Continue executing on your current strategy while monitoring key metrics")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

# Singleton instance
analysis_orchestrator_service = AnalysisOrchestratorService()
