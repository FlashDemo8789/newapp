import json
import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_loader")

def load_startup_data(data_path: str = None) -> pd.DataFrame:
    """
    Load startup data from the big_startups.json file and transform it
    into a format suitable for model training.
    
    Args:
        data_path: Path to the data file. If None, uses the default path.
        
    Returns:
        DataFrame with startup data prepared for model training
    """
    if data_path is None:
        # Default path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        data_path = os.path.join(project_root, "data/big_startups.json")
    
    logger.info(f"Loading startup data from {data_path}")
    
    try:
        # Load JSON data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Check the structure - looking for a 'startups' array
        startups_data = data.get('startups', [])
        if not startups_data and isinstance(data, list):
            startups_data = data
            
        logger.info(f"Loaded {len(startups_data)} startup records")
        
        # Transform data into model-ready format
        processed_data = []
        
        for startup in startups_data:
            try:
                # Skip if not a dictionary
                if not isinstance(startup, dict):
                    continue
                    
                # Create a flat record with appropriate feature naming
                record = {}
                
                # Map fields to CAMP framework metrics
                map_startup_to_camp_metrics(record, startup)
                
                # Only add records with sufficient data
                if is_valid_record(record):
                    processed_data.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing startup record: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Normalize numerical values
        normalize_numerical_features(df)
        
        logger.info(f"Processed {len(df)} valid startup records with {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Error loading startup data: {str(e)}")
        logger.error(f"Returning empty DataFrame")
        return pd.DataFrame()

def map_startup_to_camp_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map startup data to CAMP framework metrics"""
    # Company info
    record["industry"] = startup.get("sector", startup.get("industry", "Unknown")).lower()
    record["stage"] = startup.get("stage", "Unknown").lower()
    record["business_model"] = map_business_model(
        startup.get("business_model", startup.get("model", "Unknown"))
    )
    
    # Capital metrics
    map_capital_metrics(record, startup)
    
    # Advantage metrics
    map_advantage_metrics(record, startup)
    
    # Market metrics
    map_market_metrics(record, startup)
    
    # People metrics
    map_people_metrics(record, startup)
    
    # Target variables (success indicators)
    map_success_metrics(record, startup)

def map_business_model(model_str: str) -> str:
    """Map various business model strings to standardized categories"""
    model_str = str(model_str).lower()
    
    if any(x in model_str for x in ["b2b", "enterprise", "business to business"]):
        return "B2B"
    elif any(x in model_str for x in ["b2c", "consumer", "business to consumer"]):
        return "B2C"
    elif any(x in model_str for x in ["b2b2c", "business to business to consumer"]):
        return "B2B2C"
    elif any(x in model_str for x in ["saas", "software as a service"]):
        return "SaaS"
    elif any(x in model_str for x in ["marketplace", "platform"]):
        return "Marketplace"
    else:
        return "Unknown"

def map_capital_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map startup data to capital metrics"""
    # Burn rate
    record["capital_burn_rate"] = float(startup.get("burn_rate", 0))
    
    # Runway
    record["capital_runway_months"] = float(startup.get("runway_months", 0))
    
    # LTV/CAC ratio
    ltv = float(startup.get("lifetime_value_ltv", startup.get("ltv", 0)))
    cac = float(startup.get("customer_acquisition_cost", startup.get("cac", 1)))  # Default to 1 to avoid div by 0
    record["capital_ltv_cac_ratio"] = ltv / cac if cac > 0 else 0
    
    # Gross margin
    record["capital_gross_margin"] = float(startup.get("gross_margin_percent", 0)) / 100.0
    
    # MRR
    record["capital_mrr"] = float(startup.get("monthly_revenue", startup.get("mrr", 0)))
    
    # CAC payback months
    record["capital_cac_payback_months"] = float(startup.get("cac_payback_months", 0))
    
    # Additional capital metrics
    record["capital_annual_recurring_revenue"] = float(startup.get("annual_recurring_revenue", 0))
    record["capital_cash_on_hand"] = float(startup.get("cash_on_hand_million", 0))
    record["capital_debt_ratio"] = float(startup.get("debt_ratio", 0))
    record["capital_financing_rounds"] = float(startup.get("financing_round_count", 0))

def map_advantage_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map startup data to advantage metrics"""
    # Product uniqueness
    record["advantage_product_uniqueness"] = float(startup.get("product_uniqueness_score", 0)) / 100.0
    
    # Tech innovation score
    record["advantage_tech_innovation_score"] = float(startup.get("tech_innovation_score", 0)) / 100.0
    
    # Network effects
    record["advantage_network_effects"] = float(startup.get("network_effects_score", 0)) / 100.0
    
    # Product maturity
    record["advantage_product_maturity"] = float(startup.get("product_maturity", 0)) / 100.0
    
    # Uptime percentage
    record["advantage_uptime_percentage"] = float(startup.get("uptime_percentage", 99.0)) / 100.0
    
    # Algorithm uniqueness
    record["advantage_algorithm_uniqueness"] = float(startup.get("algorithm_uniqueness", 0)) / 100.0
    
    # Moat score - calculate based on other metrics if not available
    moat_score = float(startup.get("moat_score", 0))
    if moat_score == 0:
        # Calculate from available metrics
        available_scores = [
            record["advantage_product_uniqueness"],
            record["advantage_tech_innovation_score"], 
            record["advantage_network_effects"]
        ]
        non_zero_scores = [s for s in available_scores if s > 0]
        moat_score = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0.5
    else:
        moat_score = moat_score / 100.0
        
    record["advantage_moat_score"] = moat_score

def map_market_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map startup data to market metrics"""
    # TAM size
    tam_str = str(startup.get("total_addressable_market", startup.get("tam", "0")))
    # Handle strings like "$10B" or "10 billion"
    tam_value = 0
    try:
        if "billion" in tam_str.lower() or "b" in tam_str.lower():
            # Extract numeric portion and multiply by billion
            tam_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', tam_str))) * 1_000_000_000
        elif "million" in tam_str.lower() or "m" in tam_str.lower():
            # Extract numeric portion and multiply by million
            tam_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', tam_str))) * 1_000_000
        else:
            # Try to convert directly
            tam_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', tam_str)))
    except:
        tam_value = 0
    
    record["market_tam_size"] = tam_value
    
    # Monthly active users
    record["market_monthly_active_users"] = float(startup.get("monthly_active_users", 0))
    
    # Daily active users
    record["market_daily_active_users"] = float(startup.get("daily_active_users", 0))
    
    # User growth rate
    record["market_user_growth_rate"] = float(startup.get("user_growth_rate", 0)) / 100.0
    
    # Churn rate
    record["market_churn_rate"] = float(startup.get("churn_rate", 0)) / 100.0
    
    # Net revenue retention
    record["market_net_revenue_retention"] = float(startup.get("net_revenue_retention", 100)) / 100.0
    
    # Conversion rate
    record["market_conversion_rate"] = float(startup.get("conversion_rate", 0)) / 100.0

def map_people_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map startup data to people metrics"""
    # Founder experience
    record["people_founder_experience"] = float(startup.get("founder_experience_years", 0)) / 20.0  # Normalize by assuming 20 years is max
    
    # Tech talent ratio
    record["people_tech_talent_ratio"] = float(startup.get("tech_talent_ratio", 0)) / 100.0
    
    # Team completeness
    record["people_team_completeness"] = float(startup.get("team_completeness", 0)) / 100.0
    
    # Team size
    record["people_team_size"] = float(startup.get("team_size", startup.get("employee_count", 0)))
    
    # Founder exits
    record["people_founder_exits"] = float(startup.get("founder_prior_exits", 0))
    
    # Management satisfaction
    record["people_management_satisfaction"] = float(startup.get("management_satisfaction", 0)) / 100.0
    
    # Founder network reach
    record["people_founder_network_reach"] = float(startup.get("founder_network_score", 0)) / 100.0

def map_success_metrics(record: Dict[str, Any], startup: Dict[str, Any]) -> None:
    """Map success metrics to target variables"""
    # Overall success indicator (binary outcome mapped to probability)
    outcome = str(startup.get("outcome", "")).lower()
    if outcome in ["success", "pass", "funded", "acquired", "ipo"]:
        success_probability = 0.8  # High probability for successful outcomes
    elif outcome in ["failure", "fail", "closed", "defunct"]:
        success_probability = 0.2  # Low probability for failed outcomes
    else:
        # No clear outcome - try to calculate from other metrics
        success_probability = calculate_success_probability(startup)
    
    # Store as target variables
    record["overall_success"] = success_probability
    
    # Generate dimension-specific success scores
    # Capital dimension success
    capital_metrics = [
        record.get("capital_runway_months", 0) / 24.0,  # Normalize by assuming 24 months is excellent
        record.get("capital_gross_margin", 0),
        record.get("capital_ltv_cac_ratio", 0) / 5.0,  # Normalize by assuming 5.0 is excellent
    ]
    record["capital_success"] = calculate_dimension_score(capital_metrics)
    
    # Advantage dimension success
    advantage_metrics = [
        record.get("advantage_product_uniqueness", 0),
        record.get("advantage_tech_innovation_score", 0),
        record.get("advantage_moat_score", 0)
    ]
    record["advantage_success"] = calculate_dimension_score(advantage_metrics)
    
    # Market dimension success
    market_metrics = [
        min(1.0, record.get("market_user_growth_rate", 0) * 3.0),  # Normalize by assuming 33% growth is excellent
        1.0 - record.get("market_churn_rate", 0) * 5.0,  # Normalize - low churn is good
        record.get("market_net_revenue_retention", 0) - 0.5  # Normalize - >100% is good
    ]
    record["market_success"] = calculate_dimension_score(market_metrics)
    
    # People dimension success
    people_metrics = [
        record.get("people_founder_experience", 0),
        record.get("people_team_completeness", 0),
        min(1.0, record.get("people_founder_exits", 0) / 2.0)  # Normalize by assuming 2 exits is excellent
    ]
    record["people_success"] = calculate_dimension_score(people_metrics)

def calculate_success_probability(startup: Dict[str, Any]) -> float:
    """Calculate a success probability based on available metrics"""
    # Use a combination of factors to estimate success
    scores = []
    
    # Financing factors
    if startup.get("financing_round_count", 0) > 1:
        scores.append(0.7)  # Multiple rounds indicate some success
    
    # Revenue factors
    mrr = float(startup.get("monthly_revenue", startup.get("mrr", 0)))
    if mrr > 100000:
        scores.append(0.8)  # Significant revenue
    elif mrr > 10000:
        scores.append(0.6)  # Some revenue
    
    # User factors
    users = float(startup.get("monthly_active_users", 0))
    if users > 100000:
        scores.append(0.7)  # Significant user base
    elif users > 10000:
        scores.append(0.5)  # Some users
    
    # Default if no signals
    if not scores:
        return 0.5
    
    return sum(scores) / len(scores)

def calculate_dimension_score(metrics: List[float]) -> float:
    """Calculate a dimension score from a list of metrics"""
    valid_metrics = [m for m in metrics if m > 0]
    if not valid_metrics:
        return 0.5  # Default to neutral
    
    # Cap at 1.0 for normalization
    return min(1.0, sum(valid_metrics) / len(valid_metrics))

def is_valid_record(record: Dict[str, Any]) -> bool:
    """Check if a record has enough valid data to be useful for training"""
    # Need at least some data in each dimension
    capital_fields = [k for k in record.keys() if k.startswith("capital_")]
    advantage_fields = [k for k in record.keys() if k.startswith("advantage_")]
    market_fields = [k for k in record.keys() if k.startswith("market_")]
    people_fields = [k for k in record.keys() if k.startswith("people_")]
    
    # Check if we have at least some non-zero values in each dimension
    has_capital = any(record.get(k, 0) > 0 for k in capital_fields)
    has_advantage = any(record.get(k, 0) > 0 for k in advantage_fields)
    has_market = any(record.get(k, 0) > 0 for k in market_fields)
    has_people = any(record.get(k, 0) > 0 for k in people_fields)
    
    # Require at least 2 dimensions to have data
    dimensions_with_data = sum([has_capital, has_advantage, has_market, has_people])
    return dimensions_with_data >= 2

def normalize_numerical_features(df: pd.DataFrame) -> None:
    """Normalize extreme values in numerical features"""
    # For each numerical column, cap extreme values
    for col in df.columns:
        if col.startswith(("capital_", "advantage_", "market_", "people_")) and not col.endswith("_success"):
            # Skip already normalized values (between 0-1)
            if df[col].max() <= 1.0 and df[col].min() >= 0:
                continue
                
            if "tam_size" in col:
                # Special handling for TAM - use log scale for billions
                df[col] = df[col].apply(lambda x: min(1.0, np.log10(max(1, x)) / 10))
            elif "mrr" in col or "revenue" in col:
                # Special handling for revenue metrics - normalize to millions
                df[col] = df[col].apply(lambda x: min(1.0, x / 10000000))
            elif "team_size" in col or "employee" in col:
                # Special handling for team size - normalize to hundreds
                df[col] = df[col].apply(lambda x: min(1.0, x / 100))
            elif "active_users" in col:
                # Special handling for user counts - normalize to millions
                df[col] = df[col].apply(lambda x: min(1.0, x / 1000000))
            elif df[col].max() > 10:
                # General case - cap at reasonable values
                q99 = df[col].quantile(0.99)
                df[col] = df[col].apply(lambda x: min(q99, x) / q99 if q99 > 0 else 0)

if __name__ == "__main__":
    # Test data loading
    df = load_startup_data()
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
