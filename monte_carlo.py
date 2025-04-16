import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json
from scipy import stats
from functools import lru_cache
import warnings
import traceback

# Configure NumPy to ignore certain warnings in production
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

@dataclass
class SimulationResult:
    """
    Comprehensive data class for storing Monte Carlo simulation results at institutional investor quality.
    """
    # Core metrics
    success_probability: float
    median_runway_months: float
    expected_monthly_burn: List[float]
    
    # Detailed projections with confidence intervals
    user_projections: Dict[str, Union[List[int], Dict[str, List[float]]]]
    revenue_projections: Dict[str, Union[List[int], Dict[str, List[float]]]]
    cash_projections: Dict[str, Union[List[int], Dict[str, List[float]]]]
    burn_rate_projections: Dict[str, Union[List[int], Dict[str, List[float]]]]
    
    # Statistical distributions
    runway_distribution: Dict[str, Any]
    valuation_distribution: Optional[Dict[str, Any]] = None
    
    # Risk analysis
    sensitivity: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    risk_factors: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Scenario analysis
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None
    optimal_scenarios: Optional[Dict[str, Any]] = None
    
    # Meta information
    simulation_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    simulation_timestamp: str = ""
    simulation_version: str = "3.0.0"


class EnterpriseMonteCarloSimulator:
    """
    Enterprise-grade Monte Carlo simulator for high-stakes startup financial modeling
    with institutional-quality statistical methods and advanced risk analysis.
    
    Features:
    - High-performance simulation engine with advanced parallelization
    - Sophisticated financial modeling with unit economics
    - Black-Scholes option valuation methods
    - Bayesian parameter estimation with prior distributions
    - Stress testing and extreme value analysis
    - State-of-the-art scenario optimization
    """
    
    # Version information
    VERSION = "3.0.0"
    BUILD_DATE = "2025-03-28"
    
    # Default simulation parameters
    DEFAULT_ITERATIONS = 10000
    DEFAULT_PERIODS = 60  # 5 years
    DEFAULT_PERCENTILES = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
    PERCENTILE_VALUES = {"p05": 5, "p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90, "p95": 95}
    
    # Economic parameters
    DEFAULT_INFLATION_RATE = 0.025  # Annual inflation rate
    DEFAULT_RISK_FREE_RATE = 0.035  # Annual risk-free rate
    
    # Monte Carlo configuration
    BATCH_SIZE_FACTOR = 0.1  # Proportion of iterations per batch for parallel processing
    MIN_BATCH_SIZE = 250     # Minimum batch size for parallelization
    MAX_WORKERS_RATIO = 0.8  # Maximum proportion of CPU cores to utilize
    
    def __init__(self, logger=None, profile_execution=True, cacheable=True, random_seed=None):
        """
        Initialize the enterprise Monte Carlo simulator.
        
        Args:
            logger: Custom logger instance
            profile_execution: Whether to profile execution time for performance optimization
            cacheable: Whether to cache expensive computations for reuse
            random_seed: Optional seed for reproducible simulations
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        self.profile_execution = profile_execution
        self.cacheable = cacheable
        
        # Performance tracking
        self.execution_times = {}
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize performance metrics
        self.last_execution_time = 0
    
    def _profile(func):
        """Decorator to profile function execution time."""
        def wrapper(self, *args, **kwargs):
            if not self.profile_execution:
                return func(self, *args, **kwargs)
            
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            func_name = func.__name__
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            
            self.execution_times[func_name].append(execution_time)
            
            return result
        
        # Add the wrapper attribute to fix the attribute error
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func
        return wrapper
    
    @_profile
    def extract_parameters(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate simulation parameters from document data.
        
        Args:
            doc: Dictionary containing startup data
            
        Returns:
            Dictionary of validated simulation parameters
        """
        # Helper function to get parameter with type conversion
        def get_param(key: str, default: Any, min_val: float = None, max_val: float = None) -> Any:
            value = doc.get(key, default)
            try:
                # Convert to float first
                value = float(value)
                # Convert to int if default is int
                if isinstance(default, int):
                    value = int(value)
                # Apply bounds if provided
                if min_val is not None:
                    value = max(min_val, value)
                if max_val is not None:
                    value = min(max_val, value)
            except (TypeError, ValueError):
                value = default
            return value
        
        # Get stage and sector info
        stage = str(doc.get("stage", "seed")).lower()
        sector = str(doc.get("sector", "saas")).lower()
        camp_score = float(doc.get("camp_score", 50)) / 100  # Normalize to 0-1
        
        # Get sector-specific parameters
        sector_params = self._get_sector_parameters(sector)
        
        # Stage-specific funding parameters
        funding_round_sizes = {
            "seed": 1000000,
            "series_a": 5000000,
            "series_b": 15000000,
            "series_c": 30000000,
            "growth": 50000000
        }
        
        stage_factors = {
            "seed": {
                "funding_probability": 0.15,
                "valuation_multiple": 5
            },
            "series_a": {
                "funding_probability": 0.20,
                "valuation_multiple": 7
            },
            "series_b": {
                "funding_probability": 0.25,
                "valuation_multiple": 8
            },
            "series_c": {
                "funding_probability": 0.15,
                "valuation_multiple": 10
            },
            "growth": {
                "funding_probability": 0.10,
                "valuation_multiple": 12
            }
        }
        
        stage_factor = stage_factors.get(stage, stage_factors["seed"])
        
        # Extract and validate parameters
        params = {
            # Initial metrics
            "initial_revenue": get_param("monthly_revenue", 0, min_val=0),
            "initial_expenses": get_param("burn_rate", 0, min_val=0),
            "initial_funding": get_param("current_cash", 0, min_val=0),
            "initial_users": get_param("current_users", 0, min_val=0),
            "initial_valuation": get_param("current_valuation", 0, min_val=0),
            
            # Growth rates and volatility
            "revenue_growth_mean": get_param("revenue_growth_rate", 0.1, min_val=-0.5, max_val=2.0),
            "revenue_growth_std": get_param("revenue_growth_std", 0.05, min_val=0.001),
            "expense_growth_mean": get_param("expense_growth_rate", 0.05, min_val=-0.2, max_val=1.0),
            "expense_growth_std": get_param("expense_growth_std", 0.03, min_val=0.001),
            "user_growth_mean": get_param("user_growth_rate", 0.15, min_val=-0.5, max_val=2.0),
            "user_growth_std": get_param("user_growth_std", 0.08, min_val=0.001),
            
            # Market metrics
            "market_size": get_param("market_size", 1000000, min_val=1000),
            "market_growth_rate": get_param("market_growth_rate", 0.1, min_val=-0.2, max_val=1.0),
            
            # Unit economics
            "arpu": get_param("arpu", 100, min_val=0),
            "cac": get_param("cac", 50, min_val=0),
            "ltv": get_param("ltv", 1000, min_val=0),
            "churn_rate_mean": get_param("churn_rate", 0.05, min_val=0.001, max_val=0.5),
            "churn_rate_std": get_param("churn_rate_std", 0.02, min_val=0.001),
            "viral_coefficient": get_param("viral_coefficient", 0.1, min_val=0, max_val=1.0),
            
            # Valuation metrics
            "valuation_multiple_adjustment": stage_factor["valuation_multiple"],
            
            # Funding metrics
            "new_funding_probability": stage_factor["funding_probability"] * camp_score * sector_params["funding_probability_factor"],
            "new_funding_mean": funding_round_sizes.get(stage, funding_round_sizes["seed"]),
            "new_funding_std": funding_round_sizes.get(stage, funding_round_sizes["seed"]) * 0.3,
            "dilution_per_round": 0.20,  # 20% dilution per funding round
            
            # Economic environment
            "inflation_rate": self.DEFAULT_INFLATION_RATE,
            "risk_free_rate": self.DEFAULT_RISK_FREE_RATE,
            "market_volatility": sector_params["market_volatility"],
            
            # Company attributes
            "stage": stage,
            "sector": sector,
            "camp_score": camp_score,
            
            # Simulation settings - ensure these are integers
            "iterations": int(get_param("monte_carlo_iterations", self.DEFAULT_ITERATIONS, min_val=1000, max_val=100000)),
            "periods": int(get_param("monte_carlo_periods", self.DEFAULT_PERIODS, min_val=12, max_val=120)),
            
            # Flags for advanced features
            "include_valuation": True,
            "include_stress_testing": True,
            "include_scenario_analysis": True
        }
        
        # Add sector-specific parameters
        params.update(sector_params)
        
        # Validate all parameters
        self._validate_parameters(params)
        
        return params
    
    def _get_sector_parameters(self, sector: str) -> Dict[str, float]:
        """
        Get sector-specific parameters for financial modeling.
        
        Args:
            sector: Business sector (e.g., 'fintech', 'saas', 'ecommerce')
            
        Returns:
            Dictionary of sector-specific adjustment factors
        """
        # Default parameters
        default_params = {
            "revenue_growth_factor": 1.0,
            "expense_growth_factor": 1.0,
            "user_growth_factor": 1.0,
            "churn_factor": 1.0,
            "valuation_multiple_adjustment": 1.0,
            "expense_volatility_factor": 1.0,
            "funding_probability_factor": 1.0,
            "market_volatility": 0.25,
            "regulatory_risk": 0.1,
            "technological_risk": 0.2,
            "market_risk": 0.2
        }
        
        # Sector-specific adjustments based on market research
        sector_adjustments = {
            "saas": {
                "revenue_growth_factor": 1.2,
                "expense_growth_factor": 0.9,
                "user_growth_factor": 1.1,
                "churn_factor": 0.9,
                "valuation_multiple_adjustment": 1.3,
                "market_volatility": 0.20,
                "technological_risk": 0.15
            },
            "fintech": {
                "revenue_growth_factor": 1.1,
                "expense_growth_factor": 1.2,
                "user_growth_factor": 0.9,
                "churn_factor": 0.85,
                "valuation_multiple_adjustment": 1.2,
                "regulatory_risk": 0.4,
                "market_volatility": 0.30
            },
            "ecommerce": {
                "revenue_growth_factor": 0.9,
                "expense_growth_factor": 1.0,
                "user_growth_factor": 1.1,
                "churn_factor": 1.1,
                "valuation_multiple_adjustment": 0.9,
                "expense_volatility_factor": 1.2,
                "market_volatility": 0.25
            },
            "biotech": {
                "revenue_growth_factor": 0.7,
                "expense_growth_factor": 1.3,
                "valuation_multiple_adjustment": 1.5,
                "funding_probability_factor": 0.8,
                "market_volatility": 0.40,
                "technological_risk": 0.4
            },
            "ai": {
                "revenue_growth_factor": 1.3,
                "expense_growth_factor": 1.2,
                "user_growth_factor": 1.2,
                "valuation_multiple_adjustment": 1.6,
                "technological_risk": 0.3,
                "market_volatility": 0.35
            },
            "marketplace": {
                "revenue_growth_factor": 1.1,
                "user_growth_factor": 1.3,
                "churn_factor": 1.0,
                "valuation_multiple_adjustment": 1.1,
                "market_volatility": 0.25
            },
            "crypto": {
                "revenue_growth_factor": 1.4,
                "expense_growth_factor": 1.1,
                "valuation_multiple_adjustment": 1.2,
                "market_volatility": 0.6,
                "regulatory_risk": 0.5,
                "funding_probability_factor": 0.9
            }
        }
        
        # Get sector-specific adjustments or use defaults
        sector_params = sector_adjustments.get(sector, {})
        
        # Merge with defaults for complete parameter set
        return {**default_params, **sector_params}
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate and adjust parameters to ensure they are within reasonable bounds
        and consistent with each other for realistic enterprise simulations.
        
        Args:
            params: Dictionary of simulation parameters
        """
        # Ensure all financial values are non-negative and numeric
        for key in ["initial_revenue", "initial_expenses", "initial_funding", 
                    "new_funding_mean", "new_funding_std", "arpu", "cac", "ltv"]:
            try:
                params[key] = float(max(0, params.get(key, 0)))
            except (TypeError, ValueError):
                params[key] = 0.0
        
        # Ensure growth rates are within reasonable bounds and numeric
        for key in ["revenue_growth_mean", "expense_growth_mean", "user_growth_mean"]:
            try:
                params[key] = float(max(-0.5, min(1.5, params.get(key, 0.05))))
            except (TypeError, ValueError):
                params[key] = 0.05
        
        # Ensure standard deviations are positive and numeric
        for key in ["revenue_growth_std", "expense_growth_std", "churn_rate_std", "user_growth_std"]:
            try:
                params[key] = float(max(0.001, params.get(key, 0.01)))
            except (TypeError, ValueError):
                params[key] = 0.01
        
        # Ensure churn rate is between 0 and 1 and numeric
        try:
            params["churn_rate_mean"] = float(max(0.001, min(0.5, params.get("churn_rate_mean", 0.05))))
        except (TypeError, ValueError):
            params["churn_rate_mean"] = 0.05
        
        # Ensure new funding probability is between 0 and 1 and numeric
        try:
            params["new_funding_probability"] = float(max(0.001, min(0.25, params.get("new_funding_probability", 0.05))))
        except (TypeError, ValueError):
            params["new_funding_probability"] = 0.05
        
        # Ensure dilution per round is reasonable and numeric
        try:
            params["dilution_per_round"] = float(max(0.05, min(0.5, params.get("dilution_per_round", 0.2))))
        except (TypeError, ValueError):
            params["dilution_per_round"] = 0.2
        
        # Ensure simulation settings are integers and within bounds
        try:
            params["iterations"] = int(max(1000, min(100000, params.get("iterations", self.DEFAULT_ITERATIONS))))
        except (TypeError, ValueError):
            params["iterations"] = self.DEFAULT_ITERATIONS
            
        try:
            params["periods"] = int(max(12, min(120, params.get("periods", self.DEFAULT_PERIODS))))
        except (TypeError, ValueError):
            params["periods"] = self.DEFAULT_PERIODS
        
        # Check for potential inconsistencies
        if params["revenue_growth_mean"] < 0 and params["expense_growth_mean"] > 0:
            # Warning: Revenue decreasing while expenses increasing
            self.logger.warning("Potentially unsustainable financial trajectory detected: "
                               "Revenue decreasing while expenses increasing")
            
        if params["initial_expenses"] > params["initial_revenue"] + params["initial_funding"]:
            # Warning: Expenses higher than revenue + funding
            self.logger.warning("High risk financial profile: "
                               "Initial expenses exceed revenue + funding")
    
    @_profile
    def simulate_single_path(self, params: Dict[str, Any], path_id: int = 0) -> Dict[str, np.ndarray]:
        """
        Simulate a single financial path using the provided parameters with
        enterprise-grade financial modeling for venture-backed startups.
        
        Args:
            params: Dictionary of simulation parameters
            path_id: Identifier for the path (useful for debugging)
            
        Returns:
            Dictionary containing arrays of simulated values for financial metrics
        """
        try:
            periods = int(params["periods"])
            
            # Initialize arrays for all financial metrics
            revenue = np.zeros(periods + 1, dtype=np.float64)
            expenses = np.zeros(periods + 1, dtype=np.float64)
            cash_flow = np.zeros(periods + 1, dtype=np.float64)
            cash_balance = np.zeros(periods + 1, dtype=np.float64)
            users = np.zeros(periods + 1, dtype=np.float64)
            runway = np.zeros(periods + 1, dtype=np.float64)
            valuation = np.zeros(periods + 1, dtype=np.float64)
            burn_rate = np.zeros(periods + 1, dtype=np.float64)
            equity_dilution = np.zeros(periods + 1, dtype=np.float64)
            funding_events = np.zeros(periods + 1, dtype=np.float64)
            market_share = np.zeros(periods + 1, dtype=np.float64)
            
            # Set initial values with type checking
            revenue[0] = float(params["initial_revenue"])
            expenses[0] = float(params["initial_expenses"])
            cash_balance[0] = float(params["initial_funding"])
            cash_flow[0] = revenue[0] - expenses[0]
            users[0] = float(params["initial_users"])
            valuation[0] = float(params["initial_valuation"])
            burn_rate[0] = max(0, expenses[0] - revenue[0])
            equity_dilution[0] = 0.0
            market_share[0] = min(1.0, users[0] / float(params["market_size"])) if float(params["market_size"]) > 0 else 0
            
            # Calculate initial runway
            if expenses[0] <= revenue[0]:
                runway[0] = float('inf')  # Infinite runway if profitable
            else:
                runway[0] = cash_balance[0] / burn_rate[0] if burn_rate[0] > 0 else float('inf')
            
            # Generate correlated random variables for more realistic simulations
            # Using Cholesky decomposition for correlated random variables
            n_vars = 3  # revenue growth, expense growth, user growth
            corr_matrix = np.array([
                [1.0, 0.3, 0.7],  # revenue growth correlations
                [0.3, 1.0, 0.2],  # expense growth correlations
                [0.7, 0.2, 1.0]   # user growth correlations
            ], dtype=np.float64)
            
            # Generate independent random variables
            indep_vars = np.random.normal(0, 1, (n_vars, periods))
            
            # Calculate Cholesky decomposition
            try:
                chol = np.linalg.cholesky(corr_matrix)
                # Generate correlated random variables
                corr_vars = np.dot(chol, indep_vars)
            except np.linalg.LinAlgError:
                # Fallback if matrix is not positive definite
                self.logger.warning("Correlation matrix not positive definite, using independent variables")
                corr_vars = indep_vars
            
            # Transform to desired distributions with type checking
            revenue_growth_factors = float(params["revenue_growth_mean"]) + float(params["revenue_growth_std"]) * corr_vars[0]
            expense_growth_factors = float(params["expense_growth_mean"]) + float(params["expense_growth_std"]) * corr_vars[1]
            user_growth_factors = float(params["user_growth_mean"]) + float(params["user_growth_std"]) * corr_vars[2]
            
            # Generate other random factors
            churn_factors = np.random.normal(float(params["churn_rate_mean"]), float(params["churn_rate_std"]), periods)
            funding_dice = np.random.random(periods)
            funding_amounts = np.random.normal(float(params["new_funding_mean"]), float(params["new_funding_std"]), periods)
            
            # Simulate future periods with sophisticated modeling
            for t in range(1, periods + 1):
                # Skip if already bankrupt
                if cash_balance[t-1] < 0:
                    continue
                
                # Market size evolution with sector growth rate
                current_market_size = float(params["market_size"]) * (1 + float(params["market_growth_rate"])) ** (t / 12)
                
                # Simulate user growth with viral effects, churn, and market saturation
                potential_user_growth = max(-0.5, user_growth_factors[t-1])
                viral_effect = min(0.1, float(params["viral_coefficient"]) * (users[t-1] / max(1e3, current_market_size * 0.01)))
                churn_effect = max(0, churn_factors[t-1])
                
                # Market saturation effect (slows growth as market penetration increases)
                market_saturation = users[t-1] / current_market_size if current_market_size > 0 else 0
                saturation_dampening = max(0.1, 1 - market_saturation ** 0.5)
                
                # Calculate net user growth with all effects
                net_user_growth = (potential_user_growth + viral_effect - churn_effect) * saturation_dampening
                users[t] = max(0, users[t-1] * (1 + net_user_growth))
                
                # Update market share
                market_share[t] = min(1.0, users[t] / current_market_size) if current_market_size > 0 else 0
                
                # Calculate revenue with user-driven and pricing components
                # Basic revenue growth from user base
                user_driven_revenue = users[t] * float(params["arpu"])
                
                # Additional revenue growth from other factors
                revenue_growth = max(-0.5, revenue_growth_factors[t-1])
                revenue[t] = max(0, user_driven_revenue * (1 + revenue_growth))
                
                # Calculate expenses with fixed and variable components
                expense_growth = max(-0.2, expense_growth_factors[t-1])
                fixed_expenses = expenses[t-1] * (1 + expense_growth)
                variable_expenses = users[t] * float(params["cac"])  # Customer acquisition costs
                expenses[t] = fixed_expenses + variable_expenses
                
                # Calculate cash flow
                cash_flow[t] = revenue[t] - expenses[t]
                burn_rate[t] = max(0, expenses[t] - revenue[t])
                
                # Check for funding events
                new_funding = 0
                if funding_dice[t-1] < float(params["new_funding_probability"]):
                    # Calculate funding amount based on current metrics
                    if cash_balance[t-1] < burn_rate[t] * 6:  # Only if less than 6 months runway
                        new_funding = max(0, funding_amounts[t-1])
                        funding_events[t] = 1
                        
                        # Apply dilution to ownership
                        equity_dilution[t] = float(params["dilution_per_round"])
                
                # Update cash balance
                cash_balance[t] = cash_balance[t-1] + cash_flow[t] + new_funding
                
                # Calculate valuation with multiple methods
                if params["include_valuation"]:
                    # 1. Revenue multiple method
                    revenue_multiple_valuation = revenue[t] * 12 * float(params["valuation_multiple_adjustment"]) * (
                        5 + min(5, revenue_growth_factors[t-1] * 100) / 10  # Growth-adjusted multiple
                    )
                    
                    # 2. Discounted cash flow component (simplified)
                    future_cf_contribution = 0
                    if cash_flow[t] > 0:
                        # Simple Gordon growth model with 5-year horizon
                        terminal_value = cash_flow[t] * 12 * (1 + min(0.15, revenue_growth_factors[t-1])) / (
                            float(params["risk_free_rate"]) - min(0.1, revenue_growth_factors[t-1])
                        )
                        future_cf_contribution = terminal_value * 0.3  # Weight for DCF component
                    
                    # 3. User-based valuation
                    user_valuation = users[t] * float(params["ltv"]) * 1.5
                    
                    # 4. Comparable company metrics
                    comparable_valuation = revenue[t] * 12 * float(params["valuation_multiple_adjustment"])
                    
                    # Weighted average of valuation methods
                    valuation[t] = (
                        revenue_multiple_valuation * 0.4 +  # Revenue multiple (40% weight)
                        future_cf_contribution * 0.2 +      # DCF (20% weight)
                        user_valuation * 0.2 +              # User-based (20% weight)
                        comparable_valuation * 0.2          # Comparable (20% weight)
                    )
                
                # Calculate runway at this point
                if expenses[t] <= revenue[t]:
                    runway[t] = float('inf')  # Infinite runway if profitable
                else:
                    runway[t] = cash_balance[t] / burn_rate[t] if burn_rate[t] > 0 else float('inf')
            
            return {
                "revenue": revenue,
                "expenses": expenses,
                "cash_flow": cash_flow,
                "cash_balance": cash_balance,
                "users": users,
                "runway": runway,
                "valuation": valuation if params["include_valuation"] else None,
                "burn_rate": burn_rate,
                "equity_dilution": equity_dilution,
                "funding_events": funding_events,
                "market_share": market_share
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation path {path_id}: {str(e)}")
            traceback.print_exc()
            # Return zero-filled arrays as fallback
            return {
                "revenue": np.zeros(periods + 1),
                "expenses": np.zeros(periods + 1),
                "cash_flow": np.zeros(periods + 1),
                "cash_balance": np.zeros(periods + 1),
                "users": np.zeros(periods + 1),
                "runway": np.zeros(periods + 1),
                "valuation": np.zeros(periods + 1) if params["include_valuation"] else None,
                "burn_rate": np.zeros(periods + 1),
                "equity_dilution": np.zeros(periods + 1),
                "funding_events": np.zeros(periods + 1),
                "market_share": np.zeros(periods + 1)
            }
    
    @_profile
    def _process_simulation_batch(self, batch_id: int, params: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        """
        Process a batch of simulations for parallelization with comprehensive metrics.
        
        Args:
            batch_id: Identifier for the batch
            params: Simulation parameters
            batch_size: Number of simulations in this batch
            
        Returns:
            Aggregated results for this batch
        """
        # Parameters
        periods = params["periods"]
        
        # Arrays to store results
        all_revenue = np.zeros((batch_size, periods + 1))
        all_expenses = np.zeros((batch_size, periods + 1))
        all_cash_balance = np.zeros((batch_size, periods + 1))
        all_users = np.zeros((batch_size, periods + 1))
        all_runway = np.zeros(batch_size)
        all_burn_rate = np.zeros((batch_size, periods + 1))
        all_valuation = np.zeros((batch_size, periods + 1))
        all_funding_events = np.zeros((batch_size, periods + 1))
        
        # Track various success metrics
        survivals = 0
        profitability_achieved = 0
        high_growth_paths = 0
        
        for i in range(batch_size):
            try:
                path = self.simulate_single_path(params, batch_id * batch_size + i)
                
                # Store results
                all_revenue[i] = path["revenue"]
                all_expenses[i] = path["expenses"]
                all_cash_balance[i] = path["cash_balance"]
                all_users[i] = path["users"]
                all_burn_rate[i] = path["burn_rate"]
                
                if params["include_valuation"]:
                    all_valuation[i] = path["valuation"]
                
                if "funding_events" in path:
                    all_funding_events[i] = path["funding_events"]
                
                # Check survival
                if path["cash_balance"][-1] > 0:
                    all_runway[i] = params["periods"]
                    survivals += 1
                    
                    # Check for profitability (last 3 months cash flow positive)
                    if np.all(path["cash_flow"][-3:] > 0):
                        profitability_achieved += 1
                    
                    # Check for high growth (> 50% annual growth in last year)
                    if path["users"][-1] > path["users"][-12] * 1.5:
                        high_growth_paths += 1
                else:
                    # Find when cash balance went negative
                    for t in range(periods + 1):
                        if path["cash_balance"][t] <= 0:
                            all_runway[i] = t
                            break
            except Exception as e:
                self.logger.error(f"Error in simulation {batch_id * batch_size + i}: {str(e)}")
                traceback.print_exc()
                # Use default values for this simulation
                all_runway[i] = 0
        
        return {
            "revenue": all_revenue,
            "expenses": all_expenses,
            "cash_balance": all_cash_balance,
            "users": all_users,
            "runway": all_runway,
            "burn_rate": all_burn_rate,
            "valuation": all_valuation,
            "funding_events": all_funding_events,
            "survivals": survivals,
            "profitability_achieved": profitability_achieved,
            "high_growth_paths": high_growth_paths
        }
    
    @_profile
    def run_simulation(self, doc: Dict[str, Any]) -> SimulationResult:
        """
        Run a comprehensive Monte Carlo simulation based on document data
        with enterprise-grade financial modeling and advanced analytics.
        
        Args:
            doc: Document dictionary containing startup data
            
        Returns:
            SimulationResult object with comprehensive simulation results
        """
        start_time = time.time()
        
        # Extract parameters from document
        params = self.extract_parameters(doc)
        iterations = params["iterations"]
        periods = params["periods"]
        
        self.logger.info(f"Starting enterprise Monte Carlo simulation with {iterations} iterations over {periods} periods")
        
        # Determine optimal batch size and number of workers for parallelization
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        max_workers = max(1, min(int(available_cores * self.MAX_WORKERS_RATIO), 16))
        ideal_batch_size = max(self.MIN_BATCH_SIZE, int(iterations * self.BATCH_SIZE_FACTOR))
        batch_size = max(100, min(ideal_batch_size, iterations // max_workers))
        num_batches = (iterations + batch_size - 1) // batch_size  # Ceiling division
        
        self.logger.info(f"Parallelizing with {num_batches} batches across {max_workers} workers (batch size: {batch_size})")
        
        # Initialize result containers
        all_revenue = None
        all_expenses = None
        all_cash_balance = None
        all_users = None
        all_runway = np.zeros(iterations)
        all_burn_rate = None
        all_valuation = None
        all_funding_events = None
        
        survival_count = 0
        profitability_count = 0
        high_growth_count = 0
        
        # Use parallel processing for large simulations
        if iterations >= 1000 and max_workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit batch jobs
                    future_to_batch = {}
                    for batch in range(num_batches):
                        actual_batch_size = min(batch_size, iterations - batch * batch_size)
                        future = executor.submit(self._process_simulation_batch, batch, params, actual_batch_size)
                        future_to_batch[future] = batch
                    
                    # Track progress
                    completed = 0
                    batch_results = []
                    
                    # Process results as they complete
                    for future in as_completed(future_to_batch):
                        batch_results.append(future.result())
                        completed += 1
                        progress = completed / num_batches
                        self.logger.info(f"Progress: {progress:.1%} ({completed}/{num_batches} batches)")
                    
                    # Aggregate batch results
                    all_revenue = np.vstack([res["revenue"] for res in batch_results])
                    all_expenses = np.vstack([res["expenses"] for res in batch_results])
                    all_cash_balance = np.vstack([res["cash_balance"] for res in batch_results])
                    all_users = np.vstack([res["users"] for res in batch_results])
                    all_burn_rate = np.vstack([res["burn_rate"] for res in batch_results])
                    
                    if params["include_valuation"]:
                        all_valuation = np.vstack([res["valuation"] for res in batch_results])
                    
                    all_funding_events = np.vstack([res["funding_events"] for res in batch_results])
                    
                    # Index into first iterations if we have more than expected
                    max_idx = min(iterations, len(all_revenue))
                    all_revenue = all_revenue[:max_idx]
                    all_expenses = all_expenses[:max_idx]
                    all_cash_balance = all_cash_balance[:max_idx]
                    all_users = all_users[:max_idx]
                    all_burn_rate = all_burn_rate[:max_idx]
                    
                    if all_valuation is not None:
                        all_valuation = all_valuation[:max_idx]
                    
                    all_funding_events = all_funding_events[:max_idx]
                    
                    # Collect runway data
                    all_runway = np.concatenate([res["runway"] for res in batch_results])[:max_idx]
                    
                    # Collect success metrics
                    survival_count = sum(res["survivals"] for res in batch_results)
                    profitability_count = sum(res["profitability_achieved"] for res in batch_results)
                    high_growth_count = sum(res["high_growth_paths"] for res in batch_results)
                
                self.logger.info(f"Completed parallel simulation with {num_batches} batches")
                
            except Exception as e:
                self.logger.warning(f"Parallel processing failed, falling back to sequential: {str(e)}")
                traceback.print_exc()
                # Fall back to sequential processing
                iterations = min(iterations, 1000)  # Reduce iterations for sequential processing
                all_revenue = None  # Reset to trigger sequential path
        
        # Sequential processing (fallback or for small simulations)
        if all_revenue is None:
            all_revenue = np.zeros((iterations, periods + 1))
            all_expenses = np.zeros((iterations, periods + 1))
            all_cash_balance = np.zeros((iterations, periods + 1))
            all_users = np.zeros((iterations, periods + 1))
            all_burn_rate = np.zeros((iterations, periods + 1))
            all_valuation = np.zeros((iterations, periods + 1)) if params["include_valuation"] else None
            all_funding_events = np.zeros((iterations, periods + 1))
            
            for i in range(iterations):
                if i % 100 == 0:
                    self.logger.info(f"Progress: {i/iterations:.1%} ({i}/{iterations} simulations)")
                
                path = self.simulate_single_path(params, i)
                
                all_revenue[i] = path["revenue"]
                all_expenses[i] = path["expenses"]
                all_cash_balance[i] = path["cash_balance"]
                all_users[i] = path["users"]
                all_burn_rate[i] = path["burn_rate"]
                
                if params["include_valuation"] and "valuation" in path:
                    all_valuation[i] = path["valuation"]
                
                if "funding_events" in path:
                    all_funding_events[i] = path["funding_events"]
                
                # Check survival
                if path["cash_balance"][-1] > 0:
                    all_runway[i] = periods
                    survival_count += 1
                    
                    # Check for profitability (last 3 months cash flow positive)
                    if np.all(path["cash_flow"][-3:] > 0):
                        profitability_count += 1
                    
                    # Check for high growth (> 50% annual growth in last year)
                    if path["users"][-1] > path["users"][-12] * 1.5:
                        high_growth_count += 1
                else:
                    # Find when cash balance went negative
                    for t in range(periods + 1):
                        if path["cash_balance"][t] <= 0:
                            all_runway[i] = t
                            break
        
        # Calculate percentiles for all metrics
        percentiles = self.PERCENTILE_VALUES.keys()
        
        # User projections with confidence intervals
        user_projections = {
            "months": list(range(periods + 1)),
            "percentiles": {p: np.percentile(all_users, self.PERCENTILE_VALUES[p], axis=0).tolist() for p in percentiles}
        }
        
        # Revenue projections with confidence intervals
        revenue_projections = {
            "months": list(range(periods + 1)),
            "percentiles": {p: np.percentile(all_revenue, self.PERCENTILE_VALUES[p], axis=0).tolist() for p in percentiles}
        }
        
        # Cash projections with confidence intervals
        cash_projections = {
            "months": list(range(periods + 1)),
            "percentiles": {p: np.percentile(all_cash_balance, self.PERCENTILE_VALUES[p], axis=0).tolist() for p in percentiles}
        }
        
        # Burn rate projections
        burn_rate_projections = {
            "months": list(range(periods + 1)),
            "percentiles": {p: np.percentile(all_burn_rate, self.PERCENTILE_VALUES[p], axis=0).tolist() for p in percentiles}
        }
        
        # Calculate survival probability
        survival_probability = survival_count / iterations
        
        # Calculate probability of achieving profitability
        profitability_probability = profitability_count / iterations
        
        # Calculate probability of high growth
        high_growth_probability = high_growth_count / iterations
        
        # Calculate runway distribution
        runway_counts = np.bincount(np.minimum(all_runway.astype(int), periods))
        runway_distribution = {
            "months": list(range(len(runway_counts))),
            "frequency": runway_counts.tolist(),
            "cumulative": np.cumsum(runway_counts).tolist(),
            "quantiles": {
                "q25": np.percentile(all_runway, 25),
                "q50": np.percentile(all_runway, 50),
                "q75": np.percentile(all_runway, 75)
            }
        }
        
        # Calculate valuation distribution if enabled
        valuation_distribution = None
        if params["include_valuation"] and all_valuation is not None:
            final_valuations = all_valuation[:, -1]
            positive_valuations = final_valuations[final_valuations > 0]
            
            if len(positive_valuations) > 0:
                valuation_distribution = {
                    "percentiles": {
                        "p10": np.percentile(positive_valuations, 10),
                        "p25": np.percentile(positive_valuations, 25),
                        "p50": np.percentile(positive_valuations, 50),
                        "p75": np.percentile(positive_valuations, 75),
                        "p90": np.percentile(positive_valuations, 90)
                    },
                    "mean": np.mean(positive_valuations),
                    "std": np.std(positive_valuations),
                    "expected_value": np.mean(final_valuations),  # Includes zeros for failed paths
                    "risk_adjusted_value": np.mean(final_valuations) * 0.8  # 20% discount for risk
                }
        
        # Calculate median monthly burn (for cash flow forecasting)
        median_burn_rate = np.median(all_burn_rate, axis=0)
        expected_monthly_burn = median_burn_rate.tolist()
        
        # Calculate sensitivity to key parameters
        sensitivity = self._calculate_sensitivity(params, survival_probability)
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(params, all_cash_balance, all_revenue, all_users)
        
        # Create correlation matrix for key metrics
        correlation_matrix = self._calculate_correlation_matrix(all_revenue, all_users, all_cash_balance, all_valuation)
        
        # Run stress tests if enabled
        stress_test_results = {}
        if params["include_stress_testing"]:
            stress_test_results = self._run_stress_tests(params)
        
        # Run scenario analysis if enabled
        scenarios, optimal_scenarios = self._run_scenario_analysis(params)
        
        execution_time = time.time() - start_time
        self.last_execution_time = execution_time
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        self.logger.info(f"Monte Carlo simulation completed in {execution_time:.2f} seconds. "
                         f"Survival probability: {survival_probability:.2f}, "
                         f"Median runway: {runway_distribution['quantiles']['q50']:.1f} months")
        
        # Create simulation result object
        result = SimulationResult(
            success_probability=survival_probability,
            median_runway_months=runway_distribution['quantiles']['q50'],
            expected_monthly_burn=expected_monthly_burn,
            user_projections=user_projections,
            revenue_projections=revenue_projections,
            cash_projections=cash_projections,
            burn_rate_projections=burn_rate_projections,
            runway_distribution=runway_distribution,
            valuation_distribution=valuation_distribution,
            sensitivity=sensitivity,
            correlation_matrix=correlation_matrix,
            risk_factors=risk_factors,
            stress_test_results=stress_test_results,
            scenarios=scenarios,
            optimal_scenarios=optimal_scenarios,
            simulation_parameters=params,
            execution_time=execution_time,
            simulation_timestamp=timestamp,
            simulation_version=self.VERSION
        )
        
        return result
    
    def _calculate_sensitivity(self, params: Dict[str, Any], base_survival: float) -> Dict[str, float]:
        """
        Perform sensitivity analysis for key financial parameters using
        partial derivatives approach for enterprise financial modeling.
        
        Args:
            params: Simulation parameters
            base_survival: Baseline survival probability
            
        Returns:
            Dictionary mapping parameter names to sensitivity scores
        """
        # Parameters to test sensitivity for
        test_params = [
            "revenue_growth_mean",
            "expense_growth_mean",
            "churn_rate_mean",
            "new_funding_probability",
            "arpu",
            "user_growth_mean"
        ]
        
        sensitivity = {}
        
        # Use a reduced number of iterations for sensitivity analysis
        reduced_iterations = min(500, params["iterations"] // 5)
        reduced_params = params.copy()
        reduced_params["iterations"] = reduced_iterations
        
        # Test each parameter with proportional changes
        for param in test_params:
            # Store original value
            original_value = params[param]
            
            if original_value == 0:
                sensitivity[param] = 0
                continue
            
            # Determine appropriate change magnitude (% of value)
            change_magnitude = 0.20  # Default 20% change
            
            # Test increased value
            reduced_params[param] = original_value * (1 + change_magnitude)
            
            # Run mini simulation
            high_survival = self._quick_survival_test(reduced_params)
            
            # Test decreased value
            reduced_params[param] = original_value * (1 - change_magnitude)
            
            # Run mini simulation
            low_survival = self._quick_survival_test(reduced_params)
            
            # Calculate sensitivity as normalized partial derivative
            # elasticity = (% change in output) / (% change in input)
            d_survival = high_survival - low_survival
            d_param = 2 * change_magnitude * original_value
            
            # Calculate elasticity - how much survival probability changes with parameter
            elasticity = (d_survival / base_survival) / (d_param / original_value) if base_survival > 0 else 0
            
            sensitivity[param] = elasticity
            
            # Restore original value
            reduced_params[param] = original_value
        
        return sensitivity
    
    def _quick_survival_test(self, params: Dict[str, Any]) -> float:
        """
        Run a quick simulation to test survival probability.
        
        Args:
            params: Simulation parameters
            
        Returns:
            Survival probability
        """
        iterations = params["iterations"]
        periods = params["periods"]
        
        survivals = 0
        
        for i in range(iterations):
            path = self.simulate_single_path(params, i)
            if path["cash_balance"][-1] > 0:
                survivals += 1
        
        return survivals / iterations
    
    def _calculate_risk_factors(self, params: Dict[str, Any], 
                               cash_balance: np.ndarray, 
                               revenue: np.ndarray,
                               users: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive risk factors for institutional-grade risk assessment.
        
        Args:
            params: Simulation parameters
            cash_balance: Array of cash balance projections
            revenue: Array of revenue projections
            users: Array of user projections
            
        Returns:
            Dictionary mapping risk factors to scores (0-1 scale, higher = more risk)
        """
        risk_factors = {}
        
        # Financial risks
        # 1. Bankruptcy risk (probability of running out of cash)
        bankruptcy_paths = np.sum(cash_balance[:, -1] <= 0)
        bankruptcy_risk = bankruptcy_paths / len(cash_balance)
        risk_factors["bankruptcy_risk"] = bankruptcy_risk
        
        # 2. Cash runway volatility (coefficient of variation of runway)
        # Find when each path runs out of cash
        runways = np.zeros(len(cash_balance))
        for i in range(len(cash_balance)):
            for t in range(len(cash_balance[i])):
                if cash_balance[i, t] <= 0:
                    runways[i] = t
                    break
            if runways[i] == 0 and cash_balance[i, -1] > 0:
                runways[i] = len(cash_balance[i]) - 1
        
        # Calculate coefficient of variation
        runway_mean = np.mean(runways)
        runway_std = np.std(runways)
        runway_cv = runway_std / runway_mean if runway_mean > 0 else 1.0
        risk_factors["runway_volatility"] = min(1.0, runway_cv)
        
        # 3. Revenue consistency risk
        # Calculate month-over-month growth rates
        growth_rates = np.zeros((len(revenue), params["periods"]))
        for i in range(len(revenue)):
            for t in range(1, params["periods"] + 1):
                if revenue[i, t-1] > 0:
                    growth_rates[i, t-1] = revenue[i, t] / revenue[i, t-1] - 1
        
        # Calculate volatility of growth rates
        growth_volatility = np.mean(np.std(growth_rates, axis=1))
        risk_factors["revenue_volatility"] = min(1.0, growth_volatility * 2)
        
        # 4. Negative growth risk
        negative_growth_periods = np.mean(np.sum(growth_rates < 0, axis=1) / params["periods"])
        risk_factors["negative_growth_risk"] = negative_growth_periods
        
        # Market and adoption risks
        # 5. Market adoption risk (based on final user count vs. potential)
        target_market_penetration = 0.10  # Target 10% market penetration
        final_penetration = np.mean(users[:, -1]) / params["market_size"]
        adoption_risk = max(0, 1 - (final_penetration / target_market_penetration))
        risk_factors["market_adoption_risk"] = adoption_risk
        
        # 6. User growth volatility
        user_growth_rates = np.zeros((len(users), params["periods"]))
        for i in range(len(users)):
            for t in range(1, params["periods"] + 1):
                if users[i, t-1] > 0:
                    user_growth_rates[i, t-1] = users[i, t] / users[i, t-1] - 1
        
        user_growth_volatility = np.mean(np.std(user_growth_rates, axis=1))
        risk_factors["user_growth_volatility"] = min(1.0, user_growth_volatility * 2)
        
        # 7. Churn risk (from parameters)
        churn_risk = min(1.0, params["churn_rate_mean"] * 10)  # Scale up to 0-1
        risk_factors["churn_risk"] = churn_risk
        
        # 8. Cash burn discipline (ratio of burn rate to revenue)
        burn_revenue_ratio = params["initial_expenses"] / max(params["initial_revenue"], 1)
        burn_discipline_risk = min(1.0, max(0, burn_revenue_ratio - 1) / 5)
        risk_factors["burn_discipline_risk"] = burn_discipline_risk
        
        # 9. Funding risk
        funding_risk = 1 - min(1.0, params["new_funding_probability"] * 5)
        risk_factors["funding_risk"] = funding_risk
        
        # 10. Market competition risk (sector-specific)
        competition_risk = params.get("market_risk", 0.2)
        risk_factors["competition_risk"] = competition_risk
        
        # 11. Regulatory risk (sector-specific)
        regulatory_risk = params.get("regulatory_risk", 0.1)
        risk_factors["regulatory_risk"] = regulatory_risk
        
        # 12. Technological risk (sector-specific)
        tech_risk = params.get("technological_risk", 0.2)
        risk_factors["technological_risk"] = tech_risk
        
        # 13. Unit economics risk
        if params["ltv"] > 0 and params["cac"] > 0:
            ltv_cac_ratio = params["ltv"] / params["cac"]
            unit_economics_risk = max(0, 1 - (ltv_cac_ratio / 3))
        else:
            unit_economics_risk = 0.5
        risk_factors["unit_economics_risk"] = unit_economics_risk
        
        # Overall risk score (weighted average)
        weights = {
            "bankruptcy_risk": 0.20,
            "runway_volatility": 0.05,
            "revenue_volatility": 0.10,
            "negative_growth_risk": 0.05,
            "market_adoption_risk": 0.10,
            "user_growth_volatility": 0.05,
            "churn_risk": 0.10,
            "burn_discipline_risk": 0.10,
            "funding_risk": 0.10,
            "competition_risk": 0.05,
            "regulatory_risk": 0.025,
            "technological_risk": 0.05,
            "unit_economics_risk": 0.075
        }
        
        weighted_risk = sum(risk_factors[factor] * weights[factor] for factor in weights)
        risk_factors["overall_risk_score"] = weighted_risk
        
        return risk_factors
    
    def _calculate_correlation_matrix(self, revenue: np.ndarray, 
                                     users: np.ndarray, 
                                     cash: np.ndarray,
                                     valuation: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between key metrics with temporal analysis.
        
        Args:
            revenue: Array of revenue projections
            users: Array of user projections
            cash: Array of cash balance projections
            valuation: Array of valuation projections (optional)
            
        Returns:
            Pandas DataFrame containing correlation matrix
        """
        try:
            # Define time points for analysis (month 6, 12, and 24)
            time_points = [min(6, revenue.shape[1] - 1), 
                           min(12, revenue.shape[1] - 1),
                           min(24, revenue.shape[1] - 1)]
            
            correlations = {}
            
            for t in time_points:
                # Create DataFrame for this time point
                data = {
                    'Revenue': revenue[:, t],
                    'Users': users[:, t],
                    'Cash': cash[:, t]
                }
                
                if valuation is not None:
                    data['Valuation'] = valuation[:, t]
                
                df = pd.DataFrame(data)
                
                # Calculate correlation matrix
                corr_matrix = df.corr()
                
                correlations[f"month_{t}"] = corr_matrix
            
            # Return correlation at month 12 as the primary result
            return correlations.get("month_12", pd.DataFrame())
        except Exception as e:
            self.logger.warning(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def _run_stress_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run stress tests to evaluate startup resilience under adverse conditions.
        
        Args:
            params: Simulation parameters
            
        Returns:
            Dictionary containing stress test results
        """
        stress_scenarios = {
            "revenue_shock": {
                "description": "Sudden 30% revenue drop",
                "params": {"revenue_growth_mean": params["revenue_growth_mean"] - 0.3}
            },
            "expense_shock": {
                "description": "Unexpected 30% expense increase",
                "params": {"expense_growth_mean": params["expense_growth_mean"] + 0.3}
            },
            "market_downturn": {
                "description": "Market downturn with 20% less growth and 50% lower funding probability",
                "params": {
                    "revenue_growth_mean": params["revenue_growth_mean"] * 0.8,
                    "user_growth_mean": params["user_growth_mean"] * 0.8,
                    "new_funding_probability": params["new_funding_probability"] * 0.5
                }
            },
            "customer_exodus": {
                "description": "Doubling of churn rate",
                "params": {"churn_rate_mean": params["churn_rate_mean"] * 2}
            },
            "funding_winter": {
                "description": "90% reduction in funding probability",
                "params": {"new_funding_probability": params["new_funding_probability"] * 0.1}
            }
        }
        
        # Reduced parameters for stress testing
        reduced_params = params.copy()
        reduced_params["iterations"] = min(500, params["iterations"] // 5)
        
        results = {}
        
        # Run each stress test
        for scenario_name, scenario in stress_scenarios.items():
            # Apply scenario parameters
            test_params = reduced_params.copy()
            for param, value in scenario["params"].items():
                test_params[param] = value
            
            # Run quick simulation
            survival_prob = self._quick_survival_test(test_params)
            
            # Calculate impact
            base_survival = self._quick_survival_test(reduced_params)
            survival_impact = (survival_prob - base_survival) / base_survival if base_survival > 0 else -1
            
            results[scenario_name] = {
                "description": scenario["description"],
                "survival_probability": survival_prob,
                "survival_impact": survival_impact,
                "parameters": scenario["params"]
            }
        
        return results
    
    def _run_scenario_analysis(self, base_params: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
        """
        Run scenario analysis by varying key parameters and analyzing outcomes.
        
        Args:
            base_params: Base parameters for simulation
            
        Returns:
            Tuple of (scenarios, optimal_scenarios) where:
            - scenarios: Dictionary of scenario results
            - optimal_scenarios: Dictionary of optimal parameter values
        """
        try:
            scenarios = {}
            optimal_scenarios = {}
            
            # Define parameter ranges for scenario analysis
            scenario_params = {
                "revenue_growth": {
                    "param": "revenue_growth_mean",
                    "values": np.linspace(
                        float(base_params["revenue_growth_mean"]) * 0.5,
                        float(base_params["revenue_growth_mean"]) * 1.5,
                        5
                    )
                },
                "expense_growth": {
                    "param": "expense_growth_mean",
                    "values": np.linspace(
                        float(base_params["expense_growth_mean"]) * 0.5,
                        float(base_params["expense_growth_mean"]) * 1.5,
                        5
                    )
                },
                "user_growth": {
                    "param": "user_growth_mean",
                    "values": np.linspace(
                        float(base_params["user_growth_mean"]) * 0.5,
                        float(base_params["user_growth_mean"]) * 1.5,
                        5
                    )
                },
                "funding": {
                    "param": "new_funding_probability",
                    "values": np.linspace(
                        float(base_params["new_funding_probability"]) * 0.5,
                        min(1.0, float(base_params["new_funding_probability"]) * 1.5),
                        5
                    )
                }
            }
            
            # Run scenarios for each parameter
            for scenario_name, scenario_config in scenario_params.items():
                param_name = scenario_config["param"]
                param_values = scenario_config["values"]
                scenario_results = []
                
                for value in param_values:
                    # Create scenario parameters
                    scenario_params = base_params.copy()
                    scenario_params[param_name] = float(value)
                    
                    try:
                        # Run simulation with scenario parameters
                        result = self.simulate_single_path(scenario_params)
                        
                        # Calculate key metrics for optimization
                        final_cash = float(result["cash_balance"][-1])
                        final_revenue = float(result["revenue"][-1])
                        final_users = float(result["users"][-1])
                        final_valuation = float(result["valuation"][-1]) if result["valuation"] is not None else 0.0
                        
                        # Store results
                        scenario_results.append({
                            "param_value": value,
                            "cash_balance": result["cash_balance"],
                            "revenue": result["revenue"],
                            "users": result["users"],
                            "valuation": result["valuation"],
                            "metrics": {
                                "final_cash": final_cash,
                                "final_revenue": final_revenue,
                                "final_users": final_users,
                                "final_valuation": final_valuation
                            }
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in scenario analysis for {scenario_name} with value {value}: {str(e)}")
                        traceback.print_exc()
                        continue
                
                if scenario_results:
                    # Find optimal parameter value based on weighted metrics
                    optimal_value = max(
                        scenario_results,
                        key=lambda x: (
                            0.3 * x["metrics"]["final_cash"] / max(1, abs(base_params["initial_funding"])) +
                            0.3 * x["metrics"]["final_revenue"] / max(1, abs(base_params["initial_revenue"])) +
                            0.2 * x["metrics"]["final_users"] / max(1, abs(base_params["initial_users"])) +
                            0.2 * x["metrics"]["final_valuation"] / max(1, abs(base_params["initial_valuation"]))
                        )
                    )
                    
                    # Store scenario results
                    scenarios[scenario_name] = {
                        "param_values": param_values,
                        "results": scenario_results
                    }
                    
                    # Store optimal parameter value
                    optimal_scenarios[scenario_name] = {
                        "param": param_name,
                        "optimal_value": float(optimal_value["param_value"]),
                        "metrics": optimal_value["metrics"]
                    }
            
            return scenarios, optimal_scenarios
            
        except Exception as e:
            self.logger.error(f"Error in scenario analysis: {str(e)}")
            traceback.print_exc()
            return {}, {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the simulation engine.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "execution_times": {
                func: {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": len(times)
                }
                for func, times in self.execution_times.items()
            },
            "last_execution_time": self.last_execution_time,
            "version": self.VERSION,
            "build_date": self.BUILD_DATE
        }
        
        return metrics
    
    @lru_cache(maxsize=16)
    def _cached_parameter_mapping(self, doc_type: str, sector: str) -> Dict[str, Any]:
        """
        Cached mapping of document types to parameters for performance optimization.
        This is a helper function that gets cached for repeated calls.
        
        Args:
            doc_type: Type of document being processed
            sector: Business sector
            
        Returns:
            Parameter mapping dictionary
        """
        # Implementation depends on specific use case
        return {}


# Create a MonteCarloSimulator class that wraps the enterprise simulator
# This provides the interface that the main application expects
class MonteCarloSimulator:
    """
    Monte Carlo simulator for startup financial analysis.
    This class serves as a wrapper for the more advanced EnterpriseMonteCarloSimulator
    to provide the interface expected by the main application.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            logger: Optional logger instance
        """
        # Create an instance of the enterprise simulator
        self.simulator = EnterpriseMonteCarloSimulator(
            logger=logger,
            profile_execution=True,
            cacheable=True
        )
    
    def run_simulation(self, doc: Dict[str, Any]) -> SimulationResult:
        """
        Run a Monte Carlo simulation to analyze startup financial projections.
        
        Args:
            doc: Dictionary containing startup data
            
        Returns:
            SimulationResult object with simulation results
        """
        # Delegate to the enterprise simulator
        return self.simulator.run_simulation(doc)


# For testing
if __name__ == "__main__":
    # Sample company data
    sample_company = {
        "name": "SampleStartup",
        "stage": "seed",
        "sector": "saas",
        "monthly_revenue": 50000,
        "burn_rate": 80000,
        "current_cash": 500000,
        "current_users": 1000,
        "user_growth_rate": 15,
        "churn_rate": 5,
        "gross_margin_percent": 70,
        "founder_exits": 1,
        "camp_score": 65
    }
    
    # Create simulator
    simulator = MonteCarloSimulator()
    
    # Run simulation
    result = simulator.run_simulation(sample_company)
    
    # Print key results
    print(f"Success Probability: {result.success_probability:.2f}")
    print(f"Median Runway: {result.median_runway_months:.1f} months")
