"""
Data Helper Utilities for FlashDNA
Shared functions for data processing, validation, and transformation
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("analysis.utils.data_helpers")

def safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary with a fallback default
    
    Args:
        data: Dictionary to extract value from
        keys: List of keys forming the path to the desired value
        default: Default value if path doesn't exist
        
    Returns:
        Value at the specified path or default if not found
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def safe_numeric(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float with fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float or default
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_percent(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a percentage value to a decimal ratio
    
    Args:
        value: Percentage value (0-100) to convert
        default: Default value if conversion fails
        
    Returns:
        Decimal ratio (0-1) or default
    """
    try:
        if value is None:
            return default
        
        float_value = float(value)
        # If value is likely already a ratio (between 0-1)
        if 0 <= float_value <= 1:
            return float_value
        # If value is a percentage (0-100)
        elif 0 <= float_value <= 100:
            return float_value / 100.0
        else:
            logger.warning(f"Percentage value {value} out of range, using default")
            return default
    except (ValueError, TypeError):
        return default

def calculate_runway(monthly_revenue: float, burn_rate: float, 
                   growth_rate: float = 0.0, 
                   cash_balance: float = 0.0) -> Dict[str, Any]:
    """
    Calculate runway scenarios based on current financials
    
    Args:
        monthly_revenue: Current monthly revenue
        burn_rate: Current monthly burn rate
        growth_rate: Monthly revenue growth rate as decimal (0.05 = 5%)
        cash_balance: Current cash balance
        
    Returns:
        Dictionary with runway calculations and scenarios
    """
    # Validate inputs
    monthly_revenue = safe_numeric(monthly_revenue, 0)
    burn_rate = safe_numeric(burn_rate, 1)  # Avoid division by zero
    growth_rate = safe_numeric(growth_rate, 0)
    cash_balance = safe_numeric(cash_balance, 0)
    
    # Calculate monthly net burn (burn_rate - revenue)
    net_burn = max(0, burn_rate - monthly_revenue)
    
    # Basic runway calculation (assuming no growth)
    if net_burn <= 0:
        basic_runway = float('inf')  # Infinite runway if profitable
        basic_depletion_date = "N/A (Profitable)"
    else:
        basic_runway = cash_balance / net_burn if cash_balance > 0 else 0
        today = datetime.now()
        depletion_date = today + timedelta(days=30 * basic_runway)
        basic_depletion_date = depletion_date.strftime("%Y-%m-%d")
    
    # Conservative scenario (half the growth rate)
    conservative_growth = max(0, growth_rate / 2)
    
    # Optimistic scenario (150% of growth rate)
    optimistic_growth = growth_rate * 1.5
    
    # Calculate runway with growth using numeric approximation
    if growth_rate > 0:
        # Conservative scenario calculation
        conservative_runway = _calculate_growth_runway(
            monthly_revenue, burn_rate, conservative_growth, cash_balance)
        
        # Optimistic scenario calculation
        optimistic_runway = _calculate_growth_runway(
            monthly_revenue, burn_rate, optimistic_growth, cash_balance)
        
        # Calculate depletion dates
        today = datetime.now()
        if conservative_runway == float('inf'):
            conservative_depletion_date = "N/A (Profitable)"
        else:
            conservative_date = today + timedelta(days=30 * conservative_runway)
            conservative_depletion_date = conservative_date.strftime("%Y-%m-%d")
            
        if optimistic_runway == float('inf'):
            optimistic_depletion_date = "N/A (Profitable)"
        else:
            optimistic_date = today + timedelta(days=30 * optimistic_runway)
            optimistic_depletion_date = optimistic_date.strftime("%Y-%m-%d")
    else:
        # If no growth, all scenarios are the same
        conservative_runway = basic_runway
        optimistic_runway = basic_runway
        conservative_depletion_date = basic_depletion_date
        optimistic_depletion_date = basic_depletion_date
    
    return {
        "runway_months": round(basic_runway, 1),
        "depletion_date": basic_depletion_date,
        "conservative_runway": round(conservative_runway, 1),
        "conservative_depletion_date": conservative_depletion_date,
        "optimistic_runway": round(optimistic_runway, 1),
        "optimistic_depletion_date": optimistic_depletion_date,
        "net_burn": net_burn,
        "is_profitable": net_burn <= 0
    }

def _calculate_growth_runway(monthly_revenue: float, burn_rate: float, 
                           growth_rate: float, cash_balance: float, 
                           max_months: int = 60) -> float:
    """
    Calculate runway with revenue growth using numeric approximation
    
    Args:
        monthly_revenue: Initial monthly revenue
        burn_rate: Monthly burn rate (assumed constant)
        growth_rate: Monthly revenue growth rate as decimal
        cash_balance: Initial cash balance
        max_months: Maximum number of months to calculate
        
    Returns:
        Runway in months (limited to max_months)
    """
    remaining_cash = cash_balance
    revenue = monthly_revenue
    month = 0
    
    while remaining_cash > 0 and month < max_months:
        net_burn = max(0, burn_rate - revenue)
        
        # If already profitable or becomes profitable
        if net_burn <= 0:
            return float('inf')  # Infinite runway
        
        remaining_cash -= net_burn
        revenue *= (1 + growth_rate)
        month += 1
        
        # If revenue exceeds burn rate, infinite runway
        if revenue >= burn_rate:
            return float('inf')
    
    return month

def calculate_unit_economics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate unit economics metrics
    
    Args:
        data: Input data dictionary with required fields
        
    Returns:
        Dictionary with unit economics calculations
    """
    # Extract required values with safe defaults
    monthly_revenue = safe_numeric(data.get('monthly_revenue'), 0)
    monthly_users = safe_numeric(data.get('monthly_active_users'), 1)  # Avoid division by zero
    customer_acquisition_cost = safe_numeric(data.get('cac'), None)
    customer_lifetime_value = safe_numeric(data.get('ltv'), None)
    churn_rate = safe_percent(data.get('churn_rate'), 0.05)  # Default 5%
    gross_margin = safe_percent(data.get('gross_margin'), 0.7)  # Default 70%
    
    # If CAC not provided, estimate from marketing spend if available
    if customer_acquisition_cost is None:
        marketing_spend = safe_numeric(data.get('marketing_spend'), monthly_revenue * 0.2)  # Estimate 20% of revenue
        new_users = safe_numeric(data.get('new_users_per_month'), monthly_users * 0.1)  # Estimate 10% new users
        customer_acquisition_cost = marketing_spend / max(1, new_users)  # Avoid division by zero
    
    # Calculate average revenue per user
    arpu = monthly_revenue / monthly_users if monthly_users > 0 else 0
    
    # If LTV not provided, calculate using ARPU and churn rate
    if customer_lifetime_value is None:
        # LTV = ARPU * Gross Margin / Churn Rate
        # If churn is zero, cap at 60 months of revenue
        if churn_rate <= 0:
            customer_lifetime_value = arpu * gross_margin * 60
        else:
            customer_lifetime_value = arpu * gross_margin / churn_rate
    
    # Calculate LTV:CAC ratio
    ltv_cac_ratio = customer_lifetime_value / customer_acquisition_cost if customer_acquisition_cost > 0 else 0
    
    # Calculate months to recover CAC
    if arpu * gross_margin > 0:
        months_to_recover_cac = customer_acquisition_cost / (arpu * gross_margin)
    else:
        months_to_recover_cac = float('inf')
    
    # Calculate capital efficiency score (0-100)
    # Based on LTV:CAC ratio benchmarks:
    # - Excellent: LTV:CAC > 3 (score 80-100)
    # - Good: LTV:CAC 2-3 (score 60-80)
    # - Fair: LTV:CAC 1-2 (score 40-60)
    # - Poor: LTV:CAC < 1 (score 0-40)
    if ltv_cac_ratio >= 3:
        capital_efficiency_score = 80 + min(20, (ltv_cac_ratio - 3) * 10)
    elif ltv_cac_ratio >= 2:
        capital_efficiency_score = 60 + (ltv_cac_ratio - 2) * 20
    elif ltv_cac_ratio >= 1:
        capital_efficiency_score = 40 + (ltv_cac_ratio - 1) * 20
    else:
        capital_efficiency_score = max(0, ltv_cac_ratio * 40)
    
    return {
        "arpu": round(arpu, 2),
        "ltv": round(customer_lifetime_value, 2),
        "cac": round(customer_acquisition_cost, 2),
        "ltv_cac_ratio": round(ltv_cac_ratio, 2),
        "months_to_recover_cac": round(months_to_recover_cac, 1),
        "gross_margin": gross_margin,
        "churn_rate": churn_rate,
        "capital_efficiency_score": min(100, max(0, capital_efficiency_score))
    }

def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an object to a dictionary
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, list):
        return [to_dict(i) for i in obj]
    else:
        return obj
