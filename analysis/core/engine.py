"""
Core Analysis Engine
Manages the main analysis workflow with proper error handling and dependency management
for the Flash DNA Analysis System
"""
import logging
import os
import sys
import importlib
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger("analysis.core.engine")

class AnalysisError(Exception):
    """Base exception for analysis errors with proper context"""
    def __init__(self, message: str, component: str = None, details: Dict = None):
        self.component = component
        self.details = details or {}
        self.message = message
        super().__init__(f"{component}: {message}" if component else message)

class AnalysisEngine:
    """
    Core analysis engine that manages the workflow and dependencies
    
    This replaces the monolithic run_analysis function with a proper class
    that handles dependencies and component isolation
    """
    
    def __init__(self, load_dependencies: bool = True):
        """
        Initialize the analysis engine
        
        Args:
            load_dependencies: Whether to load dependencies on initialization
        """
        self.components = {}
        self.loaded = False
        
        if load_dependencies:
            self.load_dependencies()
    
    def load_dependencies(self) -> Dict[str, bool]:
        """
        Load all required dependencies with proper error handling
        
        Returns:
            Dict mapping component names to their load status (True/False)
        """
        status = {}
        
        # Core analysis components
        status["competitive_intelligence"] = self._safe_import("competitive_intelligence", "CompetitiveIntelligence")
        status["benchmarking"] = self._safe_import("benchmarking", "BenchmarkEngine", "BenchmarkResult")
        status["network_analysis"] = self._safe_import("network_analysis", "NetworkEffectAnalyzer")
        status["pattern_detector"] = self._safe_import("pattern_detector", "detect_patterns", "generate_pattern_insights")
        
        # Monte Carlo simulation
        status["monte_carlo"] = self._safe_import_with_fallback(
            primary=("monte_carlo", ["MonteCarloSimulator", "run_simulation", "analyze_simulation_results"]),
            fallbacks=[
                ("monte_carlo_sim", ["MonteCarloSimulator", "run_simulation", "analyze_simulation_results"]),
                ("monte_carlo_engine", ["MonteCarloSimulator", "run_simulation", "analyze_simulation_results"])
            ]
        )
        
        # Team and moat analysis
        status["team_moat"] = self._safe_import_with_fallback(
            primary=("team_moat", ["TeamAnalyzer", "analyze_team", "MoatEvaluator"]),
            fallbacks=[
                ("team_analysis", ["TeamAnalyzer", "analyze_team"]),
                ("moat_analysis", ["MoatEvaluator"])
            ]
        )
        
        # DNA analysis
        status["dna_analysis"] = self._safe_import_with_fallback(
            primary=("dna_analyzer", ["DnaAnalyzer", "generate_dna_profile", "visualize_dna"]),
            fallbacks=[
                ("startup_dna", ["DnaAnalyzer", "generate_dna_profile", "visualize_dna"]),
                ("dna_profiler", ["DnaAnalyzer", "generate_dna_profile", "visualize_dna"])
            ]
        )
        
        # PMF analysis
        status["pmf_analysis"] = self._safe_import_with_fallback(
            primary=("pmf_analyzer", ["PmfAnalyzer", "calculate_pmf_score", "determine_pmf_stage"]),
            fallbacks=[
                ("product_market_fit", ["PmfAnalyzer", "calculate_pmf_score", "determine_pmf_stage"]),
                ("pmf_assessment", ["PmfAnalyzer", "calculate_pmf_score", "determine_pmf_stage"])
            ]
        )
        
        # Cohort analysis
        status["cohort_analysis"] = self._safe_import("cohort_analysis", "CohortAnalyzer")
        
        # Exit path analysis
        status["exit_path_analysis"] = self._safe_import_with_fallback(
            primary=("exit_path_analyzer", ["ExitPathAnalyzer", "analyze_exit_options"]),
            fallbacks=[
                ("exit_strategies", ["ExitPathAnalyzer", "analyze_exit_options"]),
                ("exit_analyzer", ["ExitPathAnalyzer", "analyze_exit_options"])
            ]
        )
        
        # Acquisition analysis
        status["acquisition_analysis"] = self._safe_import_with_fallback(
            primary=("acquisition_analyzer", ["AcquisitionAnalyzer", "analyze_acquisition_fit"]),
            fallbacks=[
                ("acquisition_fit", ["AcquisitionAnalyzer", "analyze_acquisition_fit"]),
                ("ma_analyzer", ["AcquisitionAnalyzer", "analyze_acquisition_fit"])
            ]
        )
        
        # Clustering analysis
        status["clustering_analysis"] = self._safe_import_with_fallback(
            primary=("clustering", ["StartupClusterer", "perform_kmeans_clustering"]),
            fallbacks=[
                ("cluster_analysis", ["StartupClusterer", "perform_kmeans_clustering"]),
                ("kmeans_clustering", ["StartupClusterer", "perform_kmeans_clustering"])
            ]
        )
        
        # System dynamics simulation
        status["system_dynamics"] = self._safe_import_with_fallback(
            primary=("system_dynamics", ["SystemDynamicsModel", "simulate_growth"]),
            fallbacks=[
                ("growth_simulator", ["SystemDynamicsModel", "simulate_growth"]),
                ("growth_model", ["SystemDynamicsModel", "simulate_growth"])
            ]
        )
        
        # Financial components
        status["financial_modeling"] = self._safe_import(
            "financial_modeling", 
            "scenario_runway", 
            "calculate_unit_economics", 
            "forecast_financials",
            "calculate_valuation_metrics"
        )
        
        # Technical analysis
        status["technical_due_diligence"] = self._safe_import_with_fallback(
            primary=("technical_due_diligence", ["TechnicalDueDiligence", "TechnicalAssessment"]),
            fallbacks=[
                ("technical_due_diligence_wrapper", ["TechnicalDueDiligence", "TechnicalAssessment"]),
                ("technical_due_diligence_minimal", ["TechnicalDueDiligence", "TechnicalAssessment"])
            ],
            error_handler=self._create_mock_technical_classes
        )
        
        # Intangible assessment (pitch and sentiment analysis)
        status["intangible_assessment"] = self._safe_import_with_fallback(
            primary=("intangible_analyzer", ["analyze_pitch", "evaluate_sentiment"]),
            fallbacks=[
                ("pitch_analyzer", ["analyze_pitch", "evaluate_sentiment"]),
                ("sentiment_analysis", ["analyze_pitch", "evaluate_sentiment"])
            ]
        )
        
        # Domain-specific analysis
        status["domain_expansion"] = self._safe_import_with_fallback(
            primary=("domain_expansion", ["expand_industry_metrics", "get_industry_factors"]),
            fallbacks=[
                ("industry_metrics", ["expand_industry_metrics", "get_industry_factors"]),
                ("industry_expansion", ["expand_industry_metrics", "get_industry_factors"])
            ]
        )
        
        # Insights generation
        status["insights_generator"] = self._safe_import_with_fallback(
            primary=("insights_generator", ["generate_insights", "prioritize_recommendations"]),
            fallbacks=[
                ("recommendation_engine", ["generate_insights", "prioritize_recommendations"]),
                ("action_items", ["generate_insights", "prioritize_recommendations"])
            ]
        )
        
        # PDF Generation
        status["pdf_generator"] = self._safe_import_with_fallback(
            primary=("global_pdf_functions", ["generate_enhanced_pdf", "generate_emergency_pdf", "generate_investor_report"]),
            fallbacks=[
                ("unified_pdf_generator", ["generate_enhanced_pdf", "generate_emergency_pdf", "generate_investor_report"]),
                ("robust_pdf", ["generate_enhanced_pdf", "generate_emergency_pdf", "generate_investor_report"])
            ]
        )
        
        # Special handling for run_analysis_fix if it exists
        try:
            from run_analysis_fix import run_analysis_safe
            self.components["run_analysis_safe"] = run_analysis_safe
            status["run_analysis_fix"] = True
            logger.info("Successfully imported safer run_analysis implementation")
        except ImportError:
            status["run_analysis_fix"] = False
            logger.warning("Could not import run_analysis_fix, using built-in analysis")
        
        self.loaded = all(status.values())
        if not self.loaded:
            logger.warning(f"Some components failed to load: {[k for k, v in status.items() if not v]}")
        
        return status
    
    def _safe_import(self, module_name: str, *objects) -> bool:
        """
        Safely import objects from a module
        
        Args:
            module_name: Name of the module to import from
            *objects: Names of objects to import
            
        Returns:
            True if import succeeded, False otherwise
        """
        try:
            module = importlib.import_module(module_name)
            for obj_name in objects:
                if hasattr(module, obj_name):
                    self.components[obj_name] = getattr(module, obj_name)
                else:
                    logger.warning(f"Object {obj_name} not found in module {module_name}")
                    return False
            return True
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
            return False
    
    def _safe_import_with_fallback(self, primary: Tuple[str, List[str]], 
                                  fallbacks: List[Tuple[str, List[str]]] = None,
                                  error_handler = None) -> bool:
        """
        Try to import from primary module, fall back to alternatives if needed
        
        Args:
            primary: Tuple of (module_name, [object_names])
            fallbacks: List of fallback (module_name, [object_names]) tuples
            error_handler: Function to call if all imports fail
            
        Returns:
            True if any import succeeded, False otherwise
        """
        # Try primary import
        primary_module, primary_objects = primary
        if self._safe_import(primary_module, *primary_objects):
            return True
        
        # Try fallbacks
        if fallbacks:
            for module_name, objects in fallbacks:
                if self._safe_import(module_name, *objects):
                    return True
        
        # Apply error handler if all imports failed
        if error_handler:
            error_handler()
            return True
            
        return False
    
    def _create_mock_technical_classes(self):
        """Create minimal implementations of technical classes if real ones fail to load"""
        class MockTechnicalDueDiligence:
            def __init__(self, *args, **kwargs):
                pass
                
            def assess_technical_architecture(self, tech_data, generate_report=False):
                logger.warning("Using mock TechnicalDueDiligence - results will be placeholders")
                return MockTechnicalAssessment()
        
        class MockTechnicalAssessment:
            def __init__(self, *args, **kwargs):
                self.score = 50.0
                self.risk_level = "Medium"
                self.architecture_rating = "C"
                self.scalability_assessment = "Unknown"
                
            def to_dict(self):
                return {
                    "score": self.score,
                    "risk_level": self.risk_level,
                    "architecture_rating": self.architecture_rating,
                    "scalability_assessment": self.scalability_assessment,
                    "is_mock": True
                }
        
        self.components["TechnicalDueDiligence"] = MockTechnicalDueDiligence
        self.components["TechnicalAssessment"] = MockTechnicalAssessment
        logger.warning("Using mock TechnicalDueDiligence and TechnicalAssessment classes")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete analysis with proper error handling
        
        Args:
            data: Input data dictionary with all required fields
            
        Returns:
            Complete analysis results
        
        Raises:
            AnalysisError: If analysis fails
        """
        if not self.loaded:
            self.load_dependencies()
        
        results = {"input": data, "analyses": {}, "status": {}}
        
        try:
            # Check if we should use the safer implementation if available
            if "run_analysis_safe" in self.components:
                logger.info("Using safer run_analysis implementation")
                return self.components["run_analysis_safe"](data)
            
            # Otherwise proceed with normal analysis sequence
            self._validate_input(data)
            
            # Apply domain-specific expansions
            expanded_data = self._expand_domain_metrics(data)
            results["expanded_data"] = expanded_data
            
            # Run core CAMP framework analysis
            results["analyses"]["camp_scores"] = self._analyze_camp_framework(expanded_data)
            
            # Run each specialized analysis module with error isolation
            results["analyses"]["technical"] = self._analyze_technical(expanded_data)
            results["analyses"]["financial"] = self._analyze_financial(expanded_data)
            results["analyses"]["market"] = self._analyze_market(expanded_data)
            results["analyses"]["competitive"] = self._analyze_competitive(expanded_data)
            results["analyses"]["dna"] = self._analyze_dna(expanded_data)
            results["analyses"]["pmf"] = self._analyze_pmf(expanded_data)
            results["analyses"]["monte_carlo"] = self._analyze_monte_carlo(expanded_data)
            results["analyses"]["cohort"] = self._analyze_cohort(expanded_data)
            results["analyses"]["patterns"] = self._analyze_patterns(expanded_data)
            results["analyses"]["clustering"] = self._analyze_clustering(expanded_data)
            results["analyses"]["exit_path"] = self._analyze_exit_path(expanded_data)
            results["analyses"]["acquisition"] = self._analyze_acquisition(expanded_data)
            results["analyses"]["system_dynamics"] = self._analyze_system_dynamics(expanded_data)
            results["analyses"]["intangible"] = self._analyze_intangible(expanded_data)
            
            # Calculate overall scores and success probability
            self._calculate_summary_metrics(results)
            
            # Generate insights and recommendations
            results["insights"] = self._generate_insights(results)
            
            results["status"]["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            results["status"] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Try to recover partial results where possible
            return results
    
    def _validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data and raise descriptive errors if invalid
        
        Args:
            data: Input data dictionary
            
        Raises:
            AnalysisError: If validation fails
        """
        required_fields = [
            "name", "industry", "funding_stage", 
            "founding_date", "monthly_revenue", "monthly_active_users"
        ]
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise AnalysisError(
                f"Missing required fields: {', '.join(missing)}", 
                component="validation"
            )
    
    def _expand_domain_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand input data with domain-specific metrics and factors
        
        Args:
            data: Original input data
            
        Returns:
            Expanded data with industry-specific metrics
        """
        if "expand_industry_metrics" not in self.components:
            return data  # No expansion available, return original
        
        try:
            industry = data.get("industry", "")
            expanded_data = data.copy()
            
            # Get industry-specific factors
            if "get_industry_factors" in self.components:
                industry_factors = self.components["get_industry_factors"](industry)
                expanded_data["industry_factors"] = industry_factors
            
            # Expand metrics based on industry
            expanded_metrics = self.components["expand_industry_metrics"](data)
            
            # Merge expanded metrics into appropriate categories
            for category in ["capital_metrics", "advantage_metrics", "market_metrics", "team_metrics"]:
                if category in expanded_metrics and category in expanded_data:
                    expanded_data[category].update(expanded_metrics.get(category, {}))
                elif category in expanded_metrics:
                    expanded_data[category] = expanded_metrics.get(category, {})
            
            return expanded_data
        except Exception as e:
            logger.error(f"Domain expansion failed: {e}")
            return data  # Return original data on failure
    
    def _analyze_camp_framework(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run CAMP framework analysis with support for enhanced CAMP input format"""
        results = {
            "capital_score": 0,
            "advantage_score": 0,
            "market_score": 0,
            "people_score": 0,
            "overall_score": 0,
        }
        
        # Capital analysis
        try:
            # Handle the updated capital metrics structure
            capital_metrics = data.get("capital_metrics", {})
            
            if "calculate_unit_economics" in self.components:
                # Ensure compatibility with calculate_unit_economics method
                unit_econ_data = {
                    "monthly_revenue": capital_metrics.get("monthly_revenue", 0),
                    "gross_margin": capital_metrics.get("gross_margin_percent", 0.75),
                    "burn_rate": capital_metrics.get("burn_rate", 0),
                    "ltv": capital_metrics.get("lifetime_value_ltv", 0),
                    "cac": capital_metrics.get("customer_acquisition_cost", 0),
                    "ltv_cac_ratio": capital_metrics.get("ltv_cac_ratio", 0),
                    "runway_months": capital_metrics.get("runway_months", 0),
                }
                
                unit_economics = self.components["calculate_unit_economics"](unit_econ_data)
                results["unit_economics"] = unit_economics
                results["capital_score"] = min(100, max(0, unit_economics.get("capital_efficiency_score", 0)))
            else:
                # Fallback calculation if the component is not available
                efficiency_indicators = [
                    capital_metrics.get("gross_margin_percent", 0.5),
                    capital_metrics.get("ltv_cac_ratio", 3.0) / 10.0, # Normalize to 0-1 scale
                    min(1.0, capital_metrics.get("runway_months", 12) / 24.0) # Normalize to 0-1 scale
                ]
                results["capital_score"] = min(100, max(0, sum(efficiency_indicators) / len(efficiency_indicators) * 100))
                
        except Exception as e:
            logger.error(f"Capital analysis failed: {e}", exc_info=True)
            results["capital_error"] = str(e)
        
        # Advantage analysis
        try:
            # Handle the updated advantage metrics structure
            advantage_metrics = data.get("advantage_metrics", {})
            
            if "NetworkEffectAnalyzer" in self.components:
                network_analyzer = self.components["NetworkEffectAnalyzer"]()
                
                # Prepare data for the network analyzer
                network_data = {
                    "moat_score": advantage_metrics.get("moat_score", 0.5), 
                    "network_effects": advantage_metrics.get("data_moat_strength", 0.5),
                    "data_advantage": advantage_metrics.get("data_moat_strength", 0.5),
                    "tech_differentiation": advantage_metrics.get("technical_innovation_score", 0.5),
                    "ip_strength": advantage_metrics.get("patent_count", 0) / 10.0  # Normalize patents to 0-1 scale
                }
                
                moat_analysis = network_analyzer.analyze_moat(network_data)
                results["advantage_score"] = min(100, max(0, moat_analysis.get("moat_score", 0)))
            else:
                # Fallback calculation if the component is not available
                moat_indicators = [
                    advantage_metrics.get("moat_score", 0.5),
                    advantage_metrics.get("technical_innovation_score", 0.5),
                    advantage_metrics.get("data_moat_strength", 0.5),
                    advantage_metrics.get("business_model_strength", 0.5),
                    advantage_metrics.get("technical_debt_score", 0.5)
                ]
                results["advantage_score"] = min(100, max(0, sum(moat_indicators) / len(moat_indicators) * 100))
                
        except Exception as e:
            logger.error(f"Advantage analysis failed: {e}", exc_info=True)
            results["advantage_error"] = str(e)
        
        # Market analysis
        try:
            # Handle the updated market metrics structure
            market_metrics = data.get("market_metrics", {})
            
            if "BenchmarkEngine" in self.components:
                benchmark_engine = self.components["BenchmarkEngine"]()
                
                # Prepare data for the benchmark engine
                market_data = {
                    "market_growth_rate": market_metrics.get("market_growth_rate", 0.15),
                    "tam_size_billion": market_metrics.get("market_size", 1.0),
                    "market_share": market_metrics.get("market_share", 0.01),
                    "category_leadership_score": market_metrics.get("category_leadership_score", 0.5)
                }
                
                market_analysis = benchmark_engine.analyze_market_position(market_data)
                results["market_score"] = min(100, max(0, market_analysis.get("market_score", 0)))
            else:
                # Fallback calculation if the component is not available
                market_indicators = [
                    market_metrics.get("market_growth_rate", 0.15) * 2.0,  # Weight growth rate higher
                    market_metrics.get("category_leadership_score", 0.5),
                    1.0 - market_metrics.get("churn_rate", 0.2) * 5.0,  # Lower churn is better
                    market_metrics.get("viral_coefficient", 0.3) / 2.0  # Normalize viral coefficient
                ]
                results["market_score"] = min(100, max(0, sum(market_indicators) / len(market_indicators) * 100))
                
        except Exception as e:
            logger.error(f"Market analysis failed: {e}", exc_info=True)
            results["market_error"] = str(e)
        
        # People analysis
        try:
            # Handle the updated team metrics structure
            team_metrics = data.get("team_metrics", {})
            
            # Use TeamAnalyzer if available (from team_moat.py)
            if "TeamAnalyzer" in self.components and "analyze_team" in self.components:
                team_analyzer = self.components["TeamAnalyzer"]()
                team_analysis = self.components["analyze_team"](team_metrics)
                results["people_score"] = min(100, max(0, team_analysis.get("team_score", 0)))
            else:
                # Fallback calculation if the component is not available
                team_indicators = [
                    team_metrics.get("team_score", 0.5),
                    team_metrics.get("founder_domain_exp_yrs", 5) / 10.0,  # Normalize years experience
                    team_metrics.get("management_satisfaction_score", 0.6),
                    team_metrics.get("founder_diversity_score", 0.5),
                    1.0 - team_metrics.get("employee_turnover_rate", 0.15),  # Lower turnover is better
                    (team_metrics.get("has_cto", False) + 
                     team_metrics.get("has_cmo", False) + 
                     team_metrics.get("has_cfo", False)) / 3.0  # Average of executive presence
                ]
                results["people_score"] = min(100, max(0, sum(team_indicators) / len(team_indicators) * 100))
                
        except Exception as e:
            logger.error(f"People analysis failed: {e}", exc_info=True)
            results["people_error"] = str(e)
        
        # Calculate overall CAMP score as weighted average using the weights from CAMP_CATEGORIES if possible
        try:
            # Try to import CAMP_CATEGORIES from constants
            from constants import CAMP_CATEGORIES
            weights = {
                "capital_score": float(CAMP_CATEGORIES["capital"]["weight"]), 
                "advantage_score": float(CAMP_CATEGORIES["advantage"]["weight"]), 
                "market_score": float(CAMP_CATEGORIES["market"]["weight"]), 
                "people_score": float(CAMP_CATEGORIES["people"]["weight"])
            }
        except (ImportError, KeyError):
            # Fallback to hardcoded weights if import fails
            weights = {"capital_score": 0.3, "advantage_score": 0.2, "market_score": 0.25, "people_score": 0.25}
        
        # Calculate weighted sum
        weighted_sum = sum(results[key] * weights[key] for key in weights.keys())
        results["overall_score"] = min(100, max(0, weighted_sum))
        
        # Include individual component weights in results
        results["component_weights"] = weights
        
        return results
    
    def _analyze_technical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run technical due diligence analysis"""
        if "TechnicalDueDiligence" not in self.components:
            return {"error": "Technical due diligence component not available"}
        
        try:
            tech_dd = self.components["TechnicalDueDiligence"]()
            assessment = tech_dd.assess_technical_architecture(data.get("tech_data", {}))
            
            # Convert assessment to dictionary if it has a to_dict method
            if hasattr(assessment, "to_dict"):
                return assessment.to_dict()
            
            # Otherwise try to extract attributes directly
            return {
                "score": getattr(assessment, "score", 50),
                "risk_level": getattr(assessment, "risk_level", "Medium"),
                "architecture_rating": getattr(assessment, "architecture_rating", "C"),
                "scalability_assessment": getattr(assessment, "scalability_assessment", "Unknown"),
                "technical_debt": getattr(assessment, "technical_debt", "Medium"),
                "security_rating": getattr(assessment, "security_rating", "B"),
                "development_practices": getattr(assessment, "development_practices", "Average")
            }
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_financial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run financial analysis"""
        results = {}
        
        # Runway analysis
        try:
            if "scenario_runway" in self.components:
                runway_results = self.components["scenario_runway"](
                    data.get("monthly_revenue", 0),
                    data.get("burn_rate", 0),
                    data.get("growth_rate", 0)
                )
                results["runway"] = runway_results
        except Exception as e:
            logger.error(f"Runway analysis failed: {e}")
            results["runway_error"] = str(e)
        
        # Financial forecasting if component exists
        try:
            if "forecast_financials" in self.components:
                forecast = self.components["forecast_financials"](data)
                results["forecast"] = forecast
        except Exception as e:
            logger.error(f"Financial forecasting failed: {e}")
            results["forecast_error"] = str(e)
        
        # Valuation metrics if component exists
        try:
            if "calculate_valuation_metrics" in self.components:
                valuation = self.components["calculate_valuation_metrics"](data)
                results["valuation"] = valuation
        except Exception as e:
            logger.error(f"Valuation calculation failed: {e}")
            results["valuation_error"] = str(e)
        
        return results
    
    def _analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run market analysis"""
        results = {}
        
        # Market size and growth
        try:
            if "BenchmarkEngine" in self.components:
                benchmark_engine = self.components["BenchmarkEngine"]()
                market_analysis = benchmark_engine.analyze_market_size_and_growth(data)
                results.update(market_analysis)
        except Exception as e:
            logger.error(f"Market size analysis failed: {e}")
            results["market_size_error"] = str(e)
        
        # Network effect analysis
        try:
            if "NetworkEffectAnalyzer" in self.components:
                network_analyzer = self.components["NetworkEffectAnalyzer"]()
                network_data = {
                    "viral_coefficient": data.get("market_metrics", {}).get("viral_coefficient", 0.3),
                    "user_count": data.get("monthly_active_users", 0),
                    "growth_rate": data.get("growth_rate", 0.1),
                    "network_type": data.get("network_type", "marketplace")
                }
                network_analysis = network_analyzer.analyze_network_effects(network_data)
                results["network_effects"] = network_analysis
        except Exception as e:
            logger.error(f"Network effect analysis failed: {e}")
            results["network_effect_error"] = str(e)
        
        # Cohort analysis
        try:
            if "CohortAnalyzer" in self.components:
                cohort_analyzer = self.components["CohortAnalyzer"](data)
                cohort_results = cohort_analyzer.analyze()
                results["cohort_analysis"] = cohort_results
        except Exception as e:
            logger.error(f"Cohort analysis failed: {e}")
            results["cohort_error"] = str(e)
        
        return results
    
    def _analyze_competitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run competitive analysis"""
        if "CompetitiveIntelligence" not in self.components:
            return {"error": "Competitive intelligence component not available"}
        
        try:
            ci = self.components["CompetitiveIntelligence"](data.get("industry", ""))
            competitive_analysis = ci.analyze_competition(
                data.get("competitors", []),
                data.get("features", [])
            )
            
            # Add differentiation recommendations if method exists
            if hasattr(ci, "recommend_differentiation"):
                differentiation = ci.recommend_differentiation(
                    data.get("features", []),
                    data.get("competitors", [])
                )
                competitive_analysis["differentiation_recommendations"] = differentiation
            
            return competitive_analysis
        except Exception as e:
            logger.error(f"Competitive analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_dna(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run DNA analysis"""
        if "DnaAnalyzer" not in self.components or "generate_dna_profile" not in self.components:
            return {"error": "DNA analysis component not available"}
        
        try:
            # Create analyzer if it's a class, otherwise assume it's a function
            dna_analyzer = self.components["DnaAnalyzer"]() if callable(self.components["DnaAnalyzer"]) else None
            
            # Prepare DNA metrics from input data
            dna_metrics = {
                "growth_rate": data.get("growth_rate", 0.1),
                "technical_innovation": data.get("advantage_metrics", {}).get("technical_innovation_score", 0.5),
                "market_fit": data.get("market_metrics", {}).get("category_leadership_score", 0.5),
                "team_quality": data.get("team_metrics", {}).get("team_score", 0.5),
                "capital_efficiency": data.get("capital_metrics", {}).get("capital_efficiency", 0.5),
                "moat_strength": data.get("advantage_metrics", {}).get("moat_score", 0.5),
                "revenue_growth": data.get("revenue_growth_rate", 0.2),
                "user_retention": 1.0 - data.get("market_metrics", {}).get("churn_rate", 0.2),
            }
            
            # Generate DNA profile using either the analyzer object or the function directly
            if dna_analyzer and hasattr(dna_analyzer, "generate_dna_profile"):
                profile = dna_analyzer.generate_dna_profile(dna_metrics)
            else:
                profile = self.components["generate_dna_profile"](dna_metrics)
            
            # Generate visualizations if the component exists
            visualizations = {}
            if "visualize_dna" in self.components:
                if dna_analyzer and hasattr(dna_analyzer, "visualize_dna"):
                    visualizations = dna_analyzer.visualize_dna(profile)
                else:
                    visualizations = self.components["visualize_dna"](profile)
            
            return {
                "profile": profile,
                "visualizations": visualizations,
                "dna_score": profile.get("overall_score", 0),
                "dominant_traits": profile.get("dominant_traits", []),
                "recessive_traits": profile.get("recessive_traits", []),
                "growth_potential": profile.get("growth_potential", 0)
            }
        except Exception as e:
            logger.error(f"DNA analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_pmf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run product-market fit analysis"""
        if "PmfAnalyzer" not in self.components or "calculate_pmf_score" not in self.components:
            return {"error": "PMF analysis component not available"}
        
        try:
            # Create analyzer if it's a class, otherwise assume it's a function
            pmf_analyzer = self.components["PmfAnalyzer"]() if callable(self.components["PmfAnalyzer"]) else None
            
            # Prepare PMF metrics from input data
            pmf_metrics = {
                "retention_rate": 1.0 - data.get("market_metrics", {}).get("churn_rate", 0.2),
                "nps_score": data.get("market_metrics", {}).get("nps_score", 0),
                "user_growth_rate": data.get("growth_rate", 0.1),
                "usage_frequency": data.get("market_metrics", {}).get("usage_frequency", 0),
                "paying_customer_ratio": data.get("capital_metrics", {}).get("paying_customer_ratio", 0),
                "market_validation": data.get("market_metrics", {}).get("market_validation_score", 0)
            }
            
            # Calculate PMF score using either the analyzer object or the function directly
            if pmf_analyzer and hasattr(pmf_analyzer, "calculate_pmf_score"):
                pmf_score = pmf_analyzer.calculate_pmf_score(pmf_metrics)
            else:
                pmf_score = self.components["calculate_pmf_score"](pmf_metrics)
            
            # Determine PMF stage
            if pmf_analyzer and hasattr(pmf_analyzer, "determine_pmf_stage"):
                pmf_stage = pmf_analyzer.determine_pmf_stage(pmf_score)
            else:
                pmf_stage = self.components["determine_pmf_stage"](pmf_score)
            
            # Generate dimensional analysis if method exists
            dimensions = {}
            if pmf_analyzer and hasattr(pmf_analyzer, "analyze_dimensions"):
                dimensions = pmf_analyzer.analyze_dimensions(pmf_metrics)
            
            # Generate strengths and weaknesses if method exists
            strengths_weaknesses = {}
            if pmf_analyzer and hasattr(pmf_analyzer, "identify_strengths_weaknesses"):
                strengths_weaknesses = pmf_analyzer.identify_strengths_weaknesses(pmf_metrics)
            
            return {
                "pmf_score": pmf_score,
                "pmf_stage": pmf_stage,
                "metrics": pmf_metrics,
                "dimensions": dimensions,
                "strengths": strengths_weaknesses.get("strengths", []),
                "weaknesses": strengths_weaknesses.get("weaknesses", []),
                "recommendations": strengths_weaknesses.get("recommendations", [])
            }
        except Exception as e:
            logger.error(f"PMF analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_monte_carlo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulations"""
        if "MonteCarloSimulator" not in self.components or "run_simulation" not in self.components:
            return {"error": "Monte Carlo simulation component not available"}
        
        try:
            # Create simulator if it's a class, otherwise assume it's a function
            simulator = self.components["MonteCarloSimulator"]() if callable(self.components["MonteCarloSimulator"]) else None
            
            # Prepare simulation parameters
            sim_params = {
                "initial_revenue": data.get("monthly_revenue", 0) * 12,  # Annualized
                "initial_users": data.get("monthly_active_users", 0),
                "growth_rate_mean": data.get("growth_rate", 0.1),
                "growth_rate_stddev": 0.05,  # Uncertainty in growth
                "burn_rate": data.get("burn_rate", 0) * 12,  # Annualized
                "simulation_periods": 36,  # 3 years
                "num_simulations": 1000
            }
            
            # Run simulation using either the simulator object or the function directly
            if simulator and hasattr(simulator, "run_simulation"):
                sim_results = simulator.run_simulation(sim_params)
            else:
                sim_results = self.components["run_simulation"](sim_params)
            
            # Analyze results
            if simulator and hasattr(simulator, "analyze_simulation_results"):
                analysis = simulator.analyze_simulation_results(sim_results)
            else:
                analysis = self.components["analyze_simulation_results"](sim_results)
            
            return {
                "simulation_parameters": sim_params,
                "simulation_analysis": analysis,
                "success_probability": analysis.get("success_probability", 0),
                "expected_runway": analysis.get("expected_runway", 0),
                "revenue_projections": analysis.get("revenue_projections", {}),
                "user_projections": analysis.get("user_projections", {}),
                "burn_projections": analysis.get("burn_projections", {}),
                "scenario_breakdown": analysis.get("scenario_breakdown", {})
            }
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_cohort(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run cohort analysis"""
        if "CohortAnalyzer" not in self.components:
            return {"error": "Cohort analysis component not available"}
        
        try:
            cohort_analyzer = self.components["CohortAnalyzer"](data)
            
            # Run analysis
            cohort_results = cohort_analyzer.analyze()
            
            # Extract retention matrices and metrics
            retention_matrix = cohort_results.get("retention_matrix", [])
            ltv_by_cohort = cohort_results.get("ltv_by_cohort", [])
            
            # Generate insights if method exists
            insights = {}
            if hasattr(cohort_analyzer, "generate_insights"):
                insights = cohort_analyzer.generate_insights()
            
            return {
                "retention_matrix": retention_matrix,
                "ltv_by_cohort": ltv_by_cohort,
                "insights": insights,
                "average_retention": cohort_results.get("average_retention", 0),
                "best_cohort": cohort_results.get("best_cohort", None),
                "worst_cohort": cohort_results.get("worst_cohort", None),
                "cohort_trends": cohort_results.get("cohort_trends", {}),
                "retention_visualization": cohort_results.get("visualization", {})
            }
        except Exception as e:
            logger.error(f"Cohort analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pattern detection"""
        if "detect_patterns" not in self.components:
            return {"error": "Pattern detection component not available"}
        
        try:
            # Extract metrics for pattern detection
            metrics = {
                "revenue": data.get("monthly_revenue", 0),
                "users": data.get("monthly_active_users", 0),
                "churn": data.get("market_metrics", {}).get("churn_rate", 0.2),
                "growth_rate": data.get("growth_rate", 0.1),
                "cac": data.get("capital_metrics", {}).get("customer_acquisition_cost", 0),
                "ltv": data.get("capital_metrics", {}).get("lifetime_value_ltv", 0),
                "burn_rate": data.get("burn_rate", 0),
                "runway": data.get("capital_metrics", {}).get("runway_months", 0),
                "team_size": data.get("team_metrics", {}).get("employee_count", 0)
            }
            
            # Detect patterns
            patterns = self.components["detect_patterns"](metrics)
            
            # Generate insights based on patterns if component exists
            insights = []
            if "generate_pattern_insights" in self.components:
                insights = self.components["generate_pattern_insights"](patterns)
            
            # Compare with success patterns if method exists
            success_correlation = {}
            if "compare_with_success_patterns" in self.components:
                success_correlation = self.components["compare_with_success_patterns"](patterns)
            
            return {
                "detected_patterns": patterns,
                "insights": insights,
                "success_correlation": success_correlation,
                "metrics_used": metrics
            }
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {"error": str(e)}
    
    def _analyze_clustering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run clustering analysis"""
        if "StartupClusterer" not in self.components and "perform_kmeans_clustering" not in self.components:
            return {"error": "Clustering component not available"}
        
        try:
            # Create clusterer if it's a class, otherwise assume it's a function
            clusterer = self.components["StartupClusterer"]() if "StartupClusterer" in self.components and callable(self.components["StartupClusterer"]) else None
            
            # Extract features for clustering
            features = {
                "monthly_revenue": data.get("monthly_revenue", 0),
                "monthly_active_users": data.get("monthly_active_users", 0),
                "growth_rate": data.get("growth_rate", 0.1),
                "churn_rate": data.get("market_metrics", {}).get("churn_rate", 0.2),
                "burn_rate": data.get("burn_rate", 0),
                "runway_months": data.get("capital_metrics", {}).get("runway_months", 0),
                "ltv": data.get("capital_metrics", {}).get("lifetime_value_ltv", 0),
                "cac": data.get("capital_metrics", {}).get("customer_acquisition_cost", 0),
                "team_size": data.get("team_metrics", {}).get("employee_count", 0),
                "funding_raised": data.get("capital_metrics", {}).get("total_funding", 0)
            }
            
            # Perform clustering
            if clusterer and hasattr(clusterer, "perform_kmeans_clustering"):
                clustering_results = clusterer.perform_kmeans_clustering(features)
            else:
                clustering_results = self.components["perform_kmeans_clustering"](features)
            
            # Generate visualizations if method exists
            visualizations = {}
            if clusterer and hasattr(clusterer, "generate_visualizations"):
                visualizations = clusterer.generate_visualizations(clustering_results)
            
            # Analyze cluster centers if method exists
            cluster_centers = {}
            if clusterer and hasattr(clusterer, "analyze_cluster_centers"):
                cluster_centers = clusterer.analyze_cluster_centers(clustering_results)
            
            return {
                "assigned_cluster": clustering_results.get("assigned_cluster", 0),
                "clusters": clustering_results.get("clusters", []),
                "cluster_centers": cluster_centers,
                "visualizations": visualizations,
                "similar_startups": clustering_results.get("similar_startups", []),
                "cluster_characteristics": clustering_results.get("cluster_characteristics", {})
            }
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_exit_path(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run exit path analysis"""
        if "ExitPathAnalyzer" not in self.components and "analyze_exit_options" not in self.components:
            return {"error": "Exit path analysis component not available"}
        
        try:
            # Create analyzer if it's a class, otherwise assume it's a function
            analyzer = self.components["ExitPathAnalyzer"]() if "ExitPathAnalyzer" in self.components and callable(self.components["ExitPathAnalyzer"]) else None
            
            # Extract metrics for exit analysis
            exit_data = {
                "industry": data.get("industry", ""),
                "monthly_revenue": data.get("monthly_revenue", 0),
                "growth_rate": data.get("growth_rate", 0.1),
                "market_share": data.get("market_metrics", {}).get("market_share", 0.01),
                "funding_stage": data.get("funding_stage", "Seed"),
                "total_funding": data.get("capital_metrics", {}).get("total_funding", 0),
                "team_size": data.get("team_metrics", {}).get("employee_count", 0),
                "founding_date": data.get("founding_date", "")
            }
            
            # Analyze exit options
            if analyzer and hasattr(analyzer, "analyze_exit_options"):
                exit_analysis = analyzer.analyze_exit_options(exit_data)
            else:
                exit_analysis = self.components["analyze_exit_options"](exit_data)
            
            # Get valuation multiples if method exists
            valuation_multiples = {}
            if analyzer and hasattr(analyzer, "calculate_valuation_multiples"):
                valuation_multiples = analyzer.calculate_valuation_multiples(exit_data)
            
            # Get timing recommendations if method exists
            timing = {}
            if analyzer and hasattr(analyzer, "recommend_timing"):
                timing = analyzer.recommend_timing(exit_data)
            
            return {
                "recommended_path": exit_analysis.get("recommended_path", ""),
                "ipo_feasibility": exit_analysis.get("ipo_feasibility", 0),
                "acquisition_attractiveness": exit_analysis.get("acquisition_attractiveness", 0),
                "potential_acquirers": exit_analysis.get("potential_acquirers", []),
                "valuation_multiples": valuation_multiples,
                "timing_recommendations": timing,
                "exit_strategy_details": exit_analysis.get("strategy_details", {})
            }
        except Exception as e:
            logger.error(f"Exit path analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_acquisition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run acquisition fit analysis"""
        if "AcquisitionAnalyzer" not in self.components and "analyze_acquisition_fit" not in self.components:
            return {"error": "Acquisition analysis component not available"}
        
        try:
            # Create analyzer if it's a class, otherwise assume it's a function
            analyzer = self.components["AcquisitionAnalyzer"]() if "AcquisitionAnalyzer" in self.components and callable(self.components["AcquisitionAnalyzer"]) else None
            
            # Extract acquisition data
            acquisition_data = {
                "industry": data.get("industry", ""),
                "monthly_revenue": data.get("monthly_revenue", 0),
                "growth_rate": data.get("growth_rate", 0.1),
                "market_share": data.get("market_metrics", {}).get("market_share", 0.01),
                "team_size": data.get("team_metrics", {}).get("employee_count", 0),
                "technologies": data.get("tech_data", {}).get("technologies", []),
                "intellectual_property": data.get("advantage_metrics", {}).get("intellectual_property", []),
                "customer_base": data.get("market_metrics", {}).get("customer_base", [])
            }
            
            # Add potential acquirers if provided
            if "potential_acquirers" in data:
                acquisition_data["potential_acquirers"] = data["potential_acquirers"]
            
            # Analyze acquisition fit
            if analyzer and hasattr(analyzer, "analyze_acquisition_fit"):
                acquisition_analysis = analyzer.analyze_acquisition_fit(acquisition_data)
            else:
                acquisition_analysis = self.components["analyze_acquisition_fit"](acquisition_data)
            
            # Evaluate strategic fit if method exists
            strategic_fit = {}
            if analyzer and hasattr(analyzer, "evaluate_strategic_fit"):
                strategic_fit = analyzer.evaluate_strategic_fit(acquisition_data)
            
            # Calculate financial synergies if method exists
            financial_synergies = {}
            if analyzer and hasattr(analyzer, "calculate_financial_synergies"):
                financial_synergies = analyzer.calculate_financial_synergies(acquisition_data)
            
            return {
                "acquisition_attractiveness": acquisition_analysis.get("acquisition_attractiveness", 0),
                "best_fit_acquirers": acquisition_analysis.get("best_fit_acquirers", []),
                "strategic_fit": strategic_fit,
                "financial_synergies": financial_synergies,
                "cultural_alignment": acquisition_analysis.get("cultural_alignment", {}),
                "acquisition_timeline": acquisition_analysis.get("acquisition_timeline", {}),
                "deal_structure_recommendations": acquisition_analysis.get("deal_structure", {})
            }
        except Exception as e:
            logger.error(f"Acquisition analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_system_dynamics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run system dynamics simulation"""
        if "SystemDynamicsModel" not in self.components and "simulate_growth" not in self.components:
            return {"error": "System dynamics component not available"}
        
        try:
            # Create model if it's a class, otherwise assume it's a function
            model = self.components["SystemDynamicsModel"]() if "SystemDynamicsModel" in self.components and callable(self.components["SystemDynamicsModel"]) else None
            
            # Extract model parameters
            simulation_params = {
                "initial_users": data.get("monthly_active_users", 0),
                "viral_coefficient": data.get("market_metrics", {}).get("viral_coefficient", 0.3),
                "churn_rate": data.get("market_metrics", {}).get("churn_rate", 0.2),
                "acquisition_rate": data.get("market_metrics", {}).get("user_acquisition_rate", 0.1),
                "time_periods": 36  # 3 years
            }
            
            # Run simulation
            if model and hasattr(model, "simulate_growth"):
                simulation_results = model.simulate_growth(simulation_params)
            else:
                simulation_results = self.components["simulate_growth"](simulation_params)
            
            # Run parameter sensitivity if method exists
            sensitivity = {}
            if model and hasattr(model, "analyze_parameter_sensitivity"):
                sensitivity = model.analyze_parameter_sensitivity(simulation_params)
            
            # Optimize scenarios if method exists
            optimized_scenarios = {}
            if model and hasattr(model, "optimize_scenarios"):
                optimized_scenarios = model.optimize_scenarios(simulation_params)
            
            return {
                "growth_trajectory": simulation_results.get("growth_trajectory", []),
                "equilibrium_users": simulation_results.get("equilibrium_users", 0),
                "growth_rate_over_time": simulation_results.get("growth_rate_over_time", []),
                "parameter_sensitivity": sensitivity,
                "optimized_scenarios": optimized_scenarios,
                "viral_growth_contribution": simulation_results.get("viral_growth_contribution", 0),
                "user_acquisition_efficiency": simulation_results.get("user_acquisition_efficiency", 0)
            }
        except Exception as e:
            logger.error(f"System dynamics simulation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_intangible(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run intangible assessment (pitch and sentiment analysis)"""
        if "analyze_pitch" not in self.components:
            return {"error": "Intangible assessment component not available"}
        
        try:
            # Extract pitch text and sentiment data
            pitch_data = {
                "pitch_text": data.get("pitch_text", ""),
                "company_description": data.get("company_description", ""),
                "mission_statement": data.get("mission_statement", ""),
                "value_proposition": data.get("value_proposition", "")
            }
            
            # Skip if no pitch text is available
            if not any(pitch_data.values()):
                return {"error": "No pitch text available for analysis"}
            
            # Analyze pitch
            pitch_analysis = self.components["analyze_pitch"](pitch_data)
            
            # Evaluate sentiment if component exists
            sentiment = {}
            if "evaluate_sentiment" in self.components:
                sentiment = self.components["evaluate_sentiment"](pitch_data)
            
            return {
                "pitch_clarity": pitch_analysis.get("clarity_score", 0),
                "pitch_uniqueness": pitch_analysis.get("uniqueness_score", 0),
                "key_differentiators": pitch_analysis.get("key_differentiators", []),
                "sentiment_analysis": sentiment,
                "topic_distribution": pitch_analysis.get("topic_distribution", {}),
                "pitch_strengths": pitch_analysis.get("strengths", []),
                "pitch_weaknesses": pitch_analysis.get("weaknesses", []),
                "recommendations": pitch_analysis.get("recommendations", [])
            }
        except Exception as e:
            logger.error(f"Intangible assessment failed: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations from analysis results"""
        if "generate_insights" not in self.components:
            return {"error": "Insights generator component not available"}
        
        try:
            # Prioritize recommendations if component exists
            recommendations = []
            if "prioritize_recommendations" in self.components:
                recommendations = self.components["prioritize_recommendations"](results)
            
            # Generate insights from all analysis results
            insights = self.components["generate_insights"](results)
            
            return {
                "key_insights": insights.get("key_insights", []),
                "strengths": insights.get("strengths", []),
                "weaknesses": insights.get("weaknesses", []),
                "opportunities": insights.get("opportunities", []),
                "threats": insights.get("threats", []),
                "prioritized_recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return {
                "error": str(e),
                "key_insights": ["Error generating insights"],
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "prioritized_recommendations": []
            }
    
    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> None:
        """Calculate summary metrics including success probability"""
        try:
            camp_scores = results["analyses"]["camp_scores"]
            
            # Get DNA score if available
            dna_score = 0
            if "dna" in results["analyses"] and "dna_score" in results["analyses"]["dna"]:
                dna_score = results["analyses"]["dna"]["dna_score"]
            
            # Get PMF score if available
            pmf_score = 0
            if "pmf" in results["analyses"] and "pmf_score" in results["analyses"]["pmf"]:
                pmf_score = results["analyses"]["pmf"]["pmf_score"]
            
            # Get Monte Carlo success probability if available
            monte_carlo_prob = 0
            if "monte_carlo" in results["analyses"] and "success_probability" in results["analyses"]["monte_carlo"]:
                monte_carlo_prob = results["analyses"]["monte_carlo"]["success_probability"]
            
            # Calculate weighted success probability from multiple sources
            camp_score = camp_scores.get("overall_score", 0)
            
            # Use Monte Carlo if available, otherwise use weighted formula
            if monte_carlo_prob > 0:
                success_prob = monte_carlo_prob * 0.7 + (camp_score / 100) * 0.3  # Weight Monte Carlo higher
            else:
                # Weighted formula combining CAMP, DNA, and PMF scores
                success_prob = (
                    camp_score * 0.5 +
                    (dna_score or camp_score * 0.8) * 0.3 +
                    (pmf_score or camp_score * 0.7) * 0.2
                ) * 0.01  # Convert to 0-1 scale
            
            # Ensure probability is in 0-100 range
            success_prob = min(100, max(0, success_prob * 100))
            
            # Add to results
            results["summary"] = {
                "camp_score": camp_score,
                "success_probability": success_prob,
            }
            
            # Add financial summary metrics
            if "financial" in results["analyses"]:
                financial = results["analyses"]["financial"]
                if "runway" in financial:
                    results["summary"]["runway_months"] = financial["runway"].get("runway_months", 0)
                    
                if "valuation" in financial:
                    results["summary"]["estimated_valuation"] = financial["valuation"].get("estimated_valuation", 0)
            
            # Add market metrics
            if "market" in results["analyses"]:
                market = results["analyses"]["market"]
                results["summary"]["market_size"] = market.get("market_size", 0)
                results["summary"]["market_growth_rate"] = market.get("market_growth_rate", 0)
            
            # Add technical assessment
            if "technical" in results["analyses"]:
                tech = results["analyses"]["technical"]
                results["summary"]["tech_score"] = tech.get("score", 0)
            
            # Add PMF stage if available
            if "pmf" in results["analyses"] and "pmf_stage" in results["analyses"]["pmf"]:
                results["summary"]["pmf_stage"] = results["analyses"]["pmf"]["pmf_stage"]
        
        except Exception as e:
            logger.error(f"Summary metric calculation failed: {e}")
            results["summary"] = {
                "error": str(e),
                "camp_score": 50,  # Fallback values
                "success_probability": 50
            }
    
    def generate_pdf_report(self, results: Dict[str, Any], report_type: str = "full") -> bytes:
        """
        Generate a PDF report from analysis results
        
        Args:
            results: Analysis results dictionary
            report_type: Type of report to generate (full, executive, etc.)
            
        Returns:
            PDF report as bytes
        """
        if "generate_enhanced_pdf" not in self.components:
            raise AnalysisError("PDF generation component not available", "pdf_generation")
        
        try:
            # Create a complete document from analysis results
            doc = self._create_pdf_document(results)
            
            # Generate PDF using the loaded component
            pdf_data = self.components["generate_enhanced_pdf"](doc, report_type)
            return pdf_data
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            
            # Try emergency PDF if available
            if "generate_emergency_pdf" in self.components:
                try:
                    doc = self._create_pdf_document(results)
                    pdf_data = self.components["generate_emergency_pdf"](doc)
                    return pdf_data
                except Exception as e2:
                    logger.error(f"Emergency PDF generation also failed: {e2}")
            
            # Re-raise the original error if all PDF generation fails
            raise AnalysisError(f"PDF generation failed: {str(e)}", "pdf_generation")
    
    def _create_pdf_document(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a document dictionary suitable for PDF generation
        
        Args:
            results: Analysis results
            
        Returns:
            Document dictionary compatible with PDF generator
        """
        # Extract all relevant fields from results
        doc = {}
        
        # Basic information
        if "input" in results:
            doc["name"] = results["input"].get("name", "Unnamed Startup")
            doc["industry"] = results["input"].get("industry", "Technology")
            doc["funding_stage"] = results["input"].get("funding_stage", "Seed")
            doc["founding_date"] = results["input"].get("founding_date", "Unknown")
        
        # Summary metrics
        if "summary" in results:
            doc["camp_score"] = results["summary"].get("camp_score", 0)
            doc["success_prob"] = results["summary"].get("success_probability", 0)
            doc["runway_months"] = results["summary"].get("runway_months", 0)
            doc["estimated_valuation"] = results["summary"].get("estimated_valuation", 0)
            
            # Add PMF stage if available
            if "pmf_stage" in results["summary"]:
                doc["pmf_stage"] = results["summary"]["pmf_stage"]
        
        # CAMP scores
        if "analyses" in results and "camp_scores" in results["analyses"]:
            camp = results["analyses"]["camp_scores"]
            doc["capital_score"] = camp.get("capital_score", 0)
            doc["advantage_score"] = camp.get("advantage_score", 0)
            doc["market_score"] = camp.get("market_score", 0)
            doc["people_score"] = camp.get("people_score", 0)
            
            # Unit economics
            if "unit_economics" in camp:
                doc["ltv_cac_ratio"] = camp["unit_economics"].get("ltv_cac_ratio", 0)
                doc["cac"] = camp["unit_economics"].get("cac", 0)
                doc["ltv"] = camp["unit_economics"].get("ltv", 0)
                doc["gross_margin"] = camp["unit_economics"].get("gross_margin", 0)
        
        # DNA scores if available
        if "analyses" in results and "dna" in results["analyses"]:
            dna = results["analyses"]["dna"]
            doc["dna_score"] = dna.get("dna_score", 0)
            doc["dominant_traits"] = dna.get("dominant_traits", [])
            doc["growth_potential"] = dna.get("growth_potential", 0)
        
        # PMF metrics if available
        if "analyses" in results and "pmf" in results["analyses"]:
            pmf = results["analyses"]["pmf"]
            doc["pmf_score"] = pmf.get("pmf_score", 0)
            doc["pmf_strengths"] = pmf.get("strengths", [])
            doc["pmf_weaknesses"] = pmf.get("weaknesses", [])
        
        # Flatten financial data
        if "analyses" in results and "financial" in results["analyses"]:
            financial = results["analyses"]["financial"]
            
            # Monthly metrics
            if "input" in results:
                doc["monthly_revenue"] = results["input"].get("monthly_revenue", 0)
                doc["burn_rate"] = results["input"].get("burn_rate", 0)
                doc["monthly_active_users"] = results["input"].get("monthly_active_users", 0)
            
            # Growth metrics
            if "forecast" in financial:
                doc["revenue_growth_rate"] = financial["forecast"].get("revenue_growth_rate", 0)
                doc["user_growth_rate"] = financial["forecast"].get("user_growth_rate", 0)
        
        # Technical assessment
        if "analyses" in results and "technical" in results["analyses"]:
            tech = results["analyses"]["technical"]
            doc["tech_score"] = tech.get("score", 0)
            doc["tech_risk_level"] = tech.get("risk_level", "Medium")
            doc["scalability_assessment"] = tech.get("scalability_assessment", "Unknown")
            doc["technical_debt"] = tech.get("technical_debt", "Medium")
        
        # Market analysis
        if "analyses" in results and "market" in results["analyses"]:
            market = results["analyses"]["market"]
            doc["market_size"] = market.get("market_size", 0)
            doc["market_growth_rate"] = market.get("market_growth_rate", 0)
            doc["market_share"] = market.get("market_share", 0)
        
        # Competitive analysis
        if "analyses" in results and "competitive" in results["analyses"]:
            competitive = results["analyses"]["competitive"]
            doc["competitive_position"] = competitive.get("position", "")
            doc["key_competitors"] = competitive.get("key_competitors", [])
            
            if "differentiation_recommendations" in competitive:
                doc["differentiation_strategy"] = competitive["differentiation_recommendations"]
        
        # Exit path analysis
        if "analyses" in results and "exit_path" in results["analyses"]:
            exit_path = results["analyses"]["exit_path"]
            doc["recommended_exit"] = exit_path.get("recommended_path", "")
            doc["exit_timing"] = exit_path.get("timing_recommendations", {})
        
        # Team data if available
        if "input" in results and "team_metrics" in results["input"]:
            team = results["input"]["team_metrics"]
            doc["employee_count"] = team.get("employee_count", 0)
            doc["founder_count"] = team.get("founder_count", 0)
            doc["tech_talent_ratio"] = team.get("tech_talent_ratio", 0)
            doc["founder_diversity_score"] = team.get("founder_diversity_score", 0)
        
        # Insights and recommendations
        if "insights" in results:
            insights = results["insights"]
            doc["key_insights"] = insights.get("key_insights", [])
            doc["prioritized_recommendations"] = insights.get("prioritized_recommendations", [])
            doc["swot"] = {
                "strengths": insights.get("strengths", []),
                "weaknesses": insights.get("weaknesses", []),
                "opportunities": insights.get("opportunities", []),
                "threats": insights.get("threats", [])
            }
        
        return doc

    def export_to_excel(self, results: Dict[str, Any], filename: str = None) -> bytes:
        """
        Export analysis results to Excel format
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename to use
            
        Returns:
            Excel file as bytes
        """
        try:
            import io
            import pandas as pd
            from datetime import datetime
            
            # Create a buffer to hold the Excel file
            output = io.BytesIO()
            
            # Create Excel writer
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Basic information sheet
                basic_info = {
                    "Metric": ["Name", "Industry", "Funding Stage", "Founding Date", 
                              "Monthly Revenue", "Monthly Active Users", "Analysis Date"],
                    "Value": [
                        results["input"].get("name", "Unnamed Startup"),
                        results["input"].get("industry", "Technology"),
                        results["input"].get("funding_stage", "Seed"),
                        results["input"].get("founding_date", "Unknown"),
                        results["input"].get("monthly_revenue", 0),
                        results["input"].get("monthly_active_users", 0),
                        datetime.now().strftime("%Y-%m-%d")
                    ]
                }
                pd.DataFrame(basic_info).to_excel(writer, sheet_name="Overview", index=False)
                
                # Summary metrics sheet
                if "summary" in results:
                    summary_data = []
                    for key, value in results["summary"].items():
                        if key != "error":  # Skip error messages
                            summary_data.append({"Metric": key, "Value": value})
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                
                # CAMP framework sheet
                if "analyses" in results and "camp_scores" in results["analyses"]:
                    camp = results["analyses"]["camp_scores"]
                    camp_data = []
                    for key, value in camp.items():
                        if not isinstance(value, dict) and key != "component_weights":
                            camp_data.append({"Metric": key, "Score": value})
                    pd.DataFrame(camp_data).to_excel(writer, sheet_name="CAMP Framework", index=False)
                
                # Financial analysis sheet
                if "analyses" in results and "financial" in results["analyses"]:
                    financial = results["analyses"]["financial"]
                    
                    # Extract runway data
                    if "runway" in financial:
                        runway_data = []
                        for key, value in financial["runway"].items():
                            runway_data.append({"Metric": key, "Value": value})
                        pd.DataFrame(runway_data).to_excel(writer, sheet_name="Runway Analysis", index=False)
                    
                    # Extract forecast data if it's time series
                    if "forecast" in financial and isinstance(financial["forecast"].get("revenue_forecast"), list):
                        forecast_data = financial["forecast"].get("revenue_forecast", [])
                        pd.DataFrame(forecast_data).to_excel(writer, sheet_name="Financial Forecast", index=False)
                
                # Add additional specialized sheets
                for analysis_type in ["dna", "pmf", "cohort", "technical", "market", "competitive", "monte_carlo", "exit_path"]:
                    if "analyses" in results and analysis_type in results["analyses"]:
                        analysis = results["analyses"][analysis_type]
                        
                        # Convert dict to dataframe-friendly format
                        sheet_data = []
                        for key, value in analysis.items():
                            if not isinstance(value, (dict, list)):
                                sheet_data.append({"Metric": key, "Value": value})
                        
                        if sheet_data:  # Only create sheet if we have data
                            sheet_name = analysis_type.replace("_", " ").title()
                            pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Insights sheet
                if "insights" in results:
                    insights = results["insights"]
                    
                    # Key insights
                    key_insights = insights.get("key_insights", [])
                    if key_insights:
                        insights_df = pd.DataFrame({"Key Insights": key_insights})
                        insights_df.to_excel(writer, sheet_name="Insights", index=False)
                    
                    # Recommendations
                    recommendations = insights.get("prioritized_recommendations", [])
                    if recommendations:
                        recom_df = pd.DataFrame({"Recommendations": recommendations})
                        recom_df.to_excel(writer, sheet_name="Recommendations", index=False)
                    
                    # SWOT analysis
                    swot_data = {
                        "Strengths": insights.get("strengths", []),
                        "Weaknesses": insights.get("weaknesses", []),
                        "Opportunities": insights.get("opportunities", []),
                        "Threats": insights.get("threats", [])
                    }
                    
                    # Get max length for padding
                    max_len = max(len(v) for v in swot_data.values())
                    for k, v in swot_data.items():
                        swot_data[k] = v + [""] * (max_len - len(v))
                    
                    swot_df = pd.DataFrame(swot_data)
                    swot_df.to_excel(writer, sheet_name="SWOT Analysis", index=False)
            
            # Get the value from the BytesIO buffer
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise AnalysisError(f"Excel export failed: {str(e)}", "excel_export")
    
    def export_to_json(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to JSON format
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename to use
            
        Returns:
            JSON string
        """
        try:
            import json
            from datetime import datetime, date
            
            # Create a custom JSON encoder to handle dates and other complex types
            class AnalysisEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    elif hasattr(obj, 'to_dict'):
                        return obj.to_dict()
                    elif pd.isna(obj):
                        return None
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    elif isinstance(obj, pd.Period):
                        return str(obj)
                    elif hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return super().default(obj)
            
            # Add export timestamp
            export_results = results.copy()
            export_results["export_timestamp"] = datetime.now().isoformat()
            
            # Convert to JSON
            json_data = json.dumps(export_results, cls=AnalysisEncoder, indent=2)
            return json_data
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise AnalysisError(f"JSON export failed: {str(e)}", "json_export")
    
    def export_to_csv(self, results: Dict[str, Any], filename: str = None) -> Dict[str, bytes]:
        """
        Export analysis results to multiple CSV files (one per major section)
        
        Args:
            results: Analysis results dictionary
            filename: Optional base filename to use
            
        Returns:
            Dictionary mapping filenames to CSV content as bytes
        """
        try:
            import io
            import pandas as pd
            from datetime import datetime
            
            csv_files = {}
            
            # Basic information CSV
            basic_info = {
                "Metric": ["Name", "Industry", "Funding Stage", "Founding Date", 
                          "Monthly Revenue", "Monthly Active Users", "Analysis Date"],
                "Value": [
                    results["input"].get("name", "Unnamed Startup"),
                    results["input"].get("industry", "Technology"),
                    results["input"].get("funding_stage", "Seed"),
                    results["input"].get("founding_date", "Unknown"),
                    results["input"].get("monthly_revenue", 0),
                    results["input"].get("monthly_active_users", 0),
                    datetime.now().strftime("%Y-%m-%d")
                ]
            }
            overview_buffer = io.StringIO()
            pd.DataFrame(basic_info).to_csv(overview_buffer, index=False)
            csv_files["overview.csv"] = overview_buffer.getvalue().encode('utf-8')
            
            # Summary metrics CSV
            if "summary" in results:
                summary_data = []
                for key, value in results["summary"].items():
                    if key != "error":  # Skip error messages
                        summary_data.append({"Metric": key, "Value": value})
                
                summary_buffer = io.StringIO()
                pd.DataFrame(summary_data).to_csv(summary_buffer, index=False)
                csv_files["summary.csv"] = summary_buffer.getvalue().encode('utf-8')
            
            # CAMP framework CSV
            if "analyses" in results and "camp_scores" in results["analyses"]:
                camp = results["analyses"]["camp_scores"]
                camp_data = []
                for key, value in camp.items():
                    if not isinstance(value, dict) and key != "component_weights":
                        camp_data.append({"Metric": key, "Score": value})
                
                camp_buffer = io.StringIO()
                pd.DataFrame(camp_data).to_csv(camp_buffer, index=False)
                csv_files["camp_framework.csv"] = camp_buffer.getvalue().encode('utf-8')
            
            # Add additional specialized CSVs
            for analysis_type in ["dna", "pmf", "technical", "market", "competitive", "monte_carlo", "exit_path"]:
                if "analyses" in results and analysis_type in results["analyses"]:
                    analysis = results["analyses"][analysis_type]
                    
                    # Convert dict to dataframe-friendly format
                    sheet_data = []
                    for key, value in analysis.items():
                        if not isinstance(value, (dict, list)):
                            sheet_data.append({"Metric": key, "Value": value})
                    
                    if sheet_data:  # Only create CSV if we have data
                        analysis_buffer = io.StringIO()
                        pd.DataFrame(sheet_data).to_csv(analysis_buffer, index=False)
                        file_name = f"{analysis_type.replace('_', '-')}.csv"
                        csv_files[file_name] = analysis_buffer.getvalue().encode('utf-8')
            
            # Insights CSV
            if "insights" in results and "key_insights" in results["insights"]:
                insights = results["insights"]
                key_insights = insights.get("key_insights", [])
                
                if key_insights:
                    insights_buffer = io.StringIO()
                    pd.DataFrame({"Key Insights": key_insights}).to_csv(insights_buffer, index=False)
                    csv_files["insights.csv"] = insights_buffer.getvalue().encode('utf-8')
            
            return csv_files
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise AnalysisError(f"CSV export failed: {str(e)}", "csv_export")
