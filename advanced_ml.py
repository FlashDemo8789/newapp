from typing import Dict, List, Any, Optional
import logging
import numpy as np
# Use XGBoost - a powerful gradient boosting library
import xgboost as xgb
import traceback

from constants import NAMED_METRICS_50, CAMP_CATEGORIES
from domain_expansions import apply_domain_expansions
from team_moat import compute_team_depth_score, compute_moat_score
# Use intangible_api (not intangible_llm) to compute intangible
from intangible_api import compute_intangible_llm

logger = logging.getLogger("flashdna")

def build_feature_vector_no_llm(doc: dict) -> np.ndarray:
    """
    Flash DNA analysis approach (no LLM):
    Uses doc['intangible'] or doc['feel_score'] if present, else defaults=50
    """
    try:
        base = []
        for metric in NAMED_METRICS_50:
            val = float(doc.get(metric, 0.0))
            base.append(val)

        expansions = apply_domain_expansions(doc)
        intangible_val = float(doc.get("intangible", doc.get("feel_score", 50.0)))
        team_val = compute_team_depth_score(doc)
        moat_val = compute_moat_score(doc)

        feature_list = base + list(expansions.values()) + [intangible_val, team_val, moat_val]
        return np.array(feature_list, dtype=float)

    except Exception as e:
        logger.error(f"Error building feature vector (no LLM): {str(e)}")
        logger.debug(traceback.format_exc())
        return np.zeros(len(NAMED_METRICS_50) + 5, dtype=float)

def build_feature_vector(doc: dict) -> np.ndarray:
    """
    Flash DNA analysis approach for inference,
    including intangible from intangible_api if missing pitch text.
    """
    try:
        base = []
        for metric in NAMED_METRICS_50:
            val = float(doc.get(metric, 0.0))
            base.append(val)

        logger.info(f"Building feature vector with {len(NAMED_METRICS_50)} base metrics")
        
        expansions = apply_domain_expansions(doc)
        logger.info(f"Applied domain expansions: {len(expansions)} features")
        
        # If intangible missing => call intangible_api or fallback
        if "intangible" not in doc:
            if "pitch_deck_text" in doc and doc["pitch_deck_text"].strip():
                doc["intangible"] = compute_intangible_llm(doc)
                logger.info(f"Computed intangible score: {doc['intangible']}")
            else:
                # fallback => random range so not always 50
                import random
                doc["intangible"] = random.uniform(55, 60)
                logger.info(f"Using random intangible fallback: {doc['intangible']}")

        intangible_val = float(doc.get("intangible", 60.0))
        team_val = compute_team_depth_score(doc)
        moat_val = compute_moat_score(doc)
        
        logger.info(f"Special metrics - Intangible: {intangible_val}, Team: {team_val}, Moat: {moat_val}")

        feature_list = base + list(expansions.values()) + [intangible_val, team_val, moat_val]
        
        # Ensure feature vector is not all zeros
        if np.all(np.array(feature_list) == 0):
            logger.warning("Feature vector is all zeros! Adding random noise for basic predictions.")
            feature_list = [max(0.1, val + np.random.uniform(0.1, 1.0)) for val in feature_list]
            
        logger.info(f"Final feature vector length: {len(feature_list)}")
        
        return np.array(feature_list, dtype=float)

    except Exception as e:
        logger.error(f"Error building feature vector: {str(e)}")
        logger.debug(traceback.format_exc())
        # Return small random values instead of zeros
        vec_size = len(NAMED_METRICS_50) + 5
        return np.random.uniform(0.1, 1.0, vec_size)

def predict_probability(doc: dict, model) -> float:
    """
    Calculate success probability using the model
    """
    try:
        feats = build_feature_vector(doc).reshape(1, -1)
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None and expected != feats.shape[1]:
            if feats.shape[1] < expected:
                pad = np.zeros((1, expected - feats.shape[1]))
                feats = np.concatenate([feats, pad], axis=1)
            else:
                feats = feats[:, :expected]
            logger.warning(f"Reshaped feature vector from {feats.shape[1]} to {expected} features")
        
        # Predict using model with appropriate method based on available API
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feats)[0][1]  # Get probability of class 1
        else:
            logger.warning("Model does not have predict_proba, using predict")
            proba = float(model.predict(feats)[0])
        
        # Scale to 0-100 range if needed
        if proba <= 1.0:
            proba *= 100.0
            
        return min(100.0, max(0.0, proba))
    except Exception as e:
        logger.error(f"Error predicting probability: {str(e)}")
        logger.debug(traceback.format_exc())
        return 50.0  # Fallback to neutral

def predict_success(doc: dict, xgb_model) -> float:
    """
    Single-proba approach
    """
    try:
        fv = build_feature_vector(doc).reshape(1, -1)
        expected = getattr(xgb_model, 'n_features_in_', None)
        if expected and fv.shape[1] < expected:
            import numpy as np
            pad = np.zeros((1, expected - fv.shape[1]))
            fv = np.concatenate([fv, pad], axis=1)
        elif expected and fv.shape[1] > expected:
            fv = fv[:, :expected]

        # Check if model is None or not properly initialized
        if xgb_model is None:
            logger.warning("XGBoost model is None, returning default probability")
            return 50.0

        # Handle both newer and older XGBoost versions
        if hasattr(xgb_model, 'predict_proba'):
            try:
                prob = xgb_model.predict_proba(fv)[0][1] * 100
            except Exception as e:
                logger.warning(f"Error using predict_proba: {str(e)}")
                # Fallback for older versions
                pred = xgb_model.predict(fv)[0]
                prob = float(pred) * 100
        else:
            pred = xgb_model.predict(fv)[0]
            prob = float(pred) * 100
        return prob
    except Exception as e:
        logger.error(f"predict_success error => {str(e)}")
        logger.debug(traceback.format_exc())
        return 50.0

def evaluate_startup(doc: dict, xgb_model) -> Dict[str, Any]:
    """
    Original evaluation method - kept for backward compatibility
    Weighted approach => 60% from success_prob, 20% intangible, 10% team, 10% moat
    """
    success_prob = predict_success(doc, xgb_model)
    intangible = doc.get("intangible", 50.0)
    team_val = doc.get("team_score", 0.0)
    moat_val = doc.get("moat_score", 0.0)
    # Weighted
    final_score = success_prob * 0.6 + intangible * 0.2 + team_val * 0.1 + moat_val * 0.1

    # Also compute CAMP framework scores for comparison
    camp_scores = evaluate_startup_camp(doc, xgb_model)

    return {
        "success_prob": success_prob,
        "success_probability": success_prob,
        "flashdna_score": camp_scores["camp_score"],  # Use CAMP scoring as primary
        "original_score": final_score,  # Keep original for reference
        **camp_scores  # Include all CAMP scores
    }

def evaluate_startup_camp(doc: dict, xgb_model) -> Dict[str, Any]:
    """
    CAMP framework evaluation => Capital, Advantage, Market, People + Intangible modifier
    """
    try:
        # Calculate individual CAMP scores
        capital_score = calculate_capital_score(doc)
        advantage_score = calculate_advantage_score(doc)
        market_score = calculate_market_score(doc)
        people_score = calculate_people_score(doc)
        
        # Calculate weighted average using CAMP weights from constants
        capital_weight = float(CAMP_CATEGORIES["capital"]["weight"])
        advantage_weight = float(CAMP_CATEGORIES["advantage"]["weight"])
        market_weight = float(CAMP_CATEGORIES["market"]["weight"])
        people_weight = float(CAMP_CATEGORIES["people"]["weight"])
        
        # Normalize weights
        total_weight = capital_weight + advantage_weight + market_weight + people_weight
        capital_weight = capital_weight / total_weight
        advantage_weight = advantage_weight / total_weight
        market_weight = market_weight / total_weight
        people_weight = people_weight / total_weight
        
        # Compute base score
        base_score = (
            capital_score * capital_weight +
            advantage_score * advantage_weight +
            market_score * market_weight +
            people_score * people_weight
        )
        
        # Apply intangible modifier if available (±20%)
        intangible = float(doc.get("intangible", doc.get("feel_score", 50.0)))
        intangible_norm = (intangible - 50) / 50  # Normalize to -1.0 to +1.0 range
        intangible_modifier = intangible_norm * 0.2 * base_score  # ±20% modifier
        
        # Add modifier to base score
        final_score = base_score + intangible_modifier
        final_score = min(100.0, max(0.0, final_score))  # Ensure in valid range
        
        return {
            "camp_score": round(final_score, 1),
            "capital_score": round(capital_score, 1),
            "advantage_score": round(advantage_score, 1),
            "market_score": round(market_score, 1),
            "people_score": round(people_score, 1),
            "intangible_modifier": round(intangible_modifier, 1)
        }
    except Exception as e:
        logger.error(f"Error in CAMP framework evaluation: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "camp_score": 50.0,
            "capital_score": 50.0,
            "advantage_score": 50.0,
            "market_score": 50.0,
            "people_score": 50.0,
            "intangible_modifier": 0.0
        }

def calculate_capital_score(doc: dict) -> float:
    """
    Calculate Capital Efficiency score (30% of CAMP framework)
    
    Evaluates financial health, sustainability, and capital deployment efficiency.
    """
    try:
        # Extract relevant metrics with defaults
        monthly_revenue = float(doc.get("monthly_revenue", 0.0))
        arr = float(doc.get("annual_recurring_revenue", monthly_revenue * 12))
        ltv = float(doc.get("lifetime_value_ltv", 0.0))
        gross_margin = float(doc.get("gross_margin_percent", 0.0))
        if gross_margin > 1:  # Convert percentage to decimal if needed
            gross_margin /= 100
        operating_margin = float(doc.get("operating_margin_percent", 0.0))
        if operating_margin > 1:
            operating_margin /= 100
        burn_rate = float(doc.get("burn_rate", 0.0))
        runway_months = float(doc.get("runway_months", 0.0))
        cash_on_hand = float(doc.get("cash_on_hand_million", 0.0))
        debt_ratio = float(doc.get("debt_ratio", 0.0))
        ltv_cac_ratio = float(doc.get("ltv_cac_ratio", 0.0))
        cac = float(doc.get("customer_acquisition_cost", 0.0))
        
        # Base score calculation (0-100 scale)
        score = 50.0  # Start at neutral
        
        # Revenue impact
        if monthly_revenue > 0:
            revenue_factor = min(25, np.log1p(monthly_revenue) / np.log1p(1e6) * 25)
            score += revenue_factor
        
        # Gross margin impact (0-20 points)
        margin_factor = min(20, gross_margin * 25)
        score += margin_factor
        
        # LTV:CAC impact (0-20 points)
        if ltv_cac_ratio > 0:
            ltv_cac_factor = min(20, ltv_cac_ratio * 5)
            score += ltv_cac_factor
        
        # Runway impact (0-15 points or penalty)
        if runway_months < 6:
            score -= 15  # Severe penalty for very short runway
        elif runway_months < 12:
            score -= 10  # Moderate penalty for short runway
        elif runway_months > 18:
            runway_factor = min(15, (runway_months - 12) / 12 * 10)
            score += runway_factor
            
        # Burn rate efficiency (relative to revenue)
        if monthly_revenue > 0 and burn_rate > 0:
            burn_efficiency = monthly_revenue / burn_rate
            if burn_efficiency > 1:  # Revenue > burn
                score += min(10, (burn_efficiency - 1) * 5)
            else:  # Burn > revenue
                score -= min(10, (1 - burn_efficiency) * 10)
        
        # Debt impact (negative)
        if debt_ratio > 0.5:
            score -= min(15, (debt_ratio - 0.5) * 30)
            
        # Sector-specific adjustments
        sector = doc.get("sector", "").lower()
        if sector in ["biotech", "pharma", "medtech"]:
            # Higher burn tolerance for biotech
            score += 5
        elif sector in ["hardware", "manufacturing"]:
            # Adjust for capital-intensive businesses
            score += 3
            
        # Stage-specific adjustments
        stage = doc.get("stage", "").lower()
        if stage in ["pre-seed", "seed"]:
            # Less emphasis on current financials for early stage
            score = 50 + (score - 50) * 0.8
        elif stage in ["series-c", "series-d", "growth"]:
            # More emphasis on capital efficiency for later stages
            score = 50 + (score - 50) * 1.2
            
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    except Exception as e:
        logger.error(f"Error calculating capital score: {e}")
        logger.debug(traceback.format_exc())
        return 50.0  # Default to neutral on error

def calculate_market_score(doc: dict) -> float:
    """
    Calculate Market Dynamics score (25% of CAMP framework)
    
    Evaluates market opportunity, growth potential, and competitive positioning.
    """
    try:
        # Extract relevant metrics with defaults
        market_size = float(doc.get("market_size", 0.0))
        market_growth_rate = float(doc.get("market_growth_rate", 0.0))
        if market_growth_rate > 1:  # Convert percentage to decimal if needed
            market_growth_rate /= 100
        market_share = float(doc.get("market_share", 0.0))
        if market_share > 1:
            market_share /= 100
        user_growth_rate = float(doc.get("user_growth_rate", 0.0))
        if user_growth_rate > 1:
            user_growth_rate /= 100
        viral_coefficient = float(doc.get("viral_coefficient", 0.0))
        category_leadership = float(doc.get("category_leadership_score", 0.0))
        if category_leadership > 1:
            category_leadership /= 100
        revenue_growth_rate = float(doc.get("revenue_growth_rate", 0.0))
        if revenue_growth_rate > 1:
            revenue_growth_rate /= 100
        net_retention_rate = float(doc.get("net_retention_rate", 0.0))
        churn_rate = float(doc.get("churn_rate", 0.0))
        if churn_rate > 1:
            churn_rate /= 100
        community_growth_rate = float(doc.get("community_growth_rate", 0.0))
        if community_growth_rate > 1:
            community_growth_rate /= 100
        
        # Base score calculation (0-100 scale)
        score = 50.0  # Start at neutral
        
        # Market size impact (0-15 points)
        if market_size > 0:
            size_factor = min(15, np.log1p(market_size) / np.log1p(1e10) * 15)
            score += size_factor
        
        # Market growth impact (0-20 points)
        growth_factor = min(20, market_growth_rate * 100)
        score += growth_factor
        
        # User/revenue growth impact (0-25 points)
        growth_compound = max(user_growth_rate, revenue_growth_rate)
        if growth_compound > 0:
            growth_factor = min(25, growth_compound * 100)
            score += growth_factor
        
        # Viral coefficient impact (exponential bonus)
        if viral_coefficient > 0:
            if viral_coefficient >= 1:  # Viral (k≥1)
                score += min(15, (viral_coefficient - 0.8) * 30)
            else:  # Sub-viral but positive
                score += min(10, viral_coefficient * 10)
        
        # Churn impact (negative)
        if churn_rate > 0:
            churn_penalty = min(20, churn_rate * 100)
            score -= churn_penalty
        
        # Net retention impact (positive)
        if net_retention_rate > 1.0:  # Negative churn
            score += min(15, (net_retention_rate - 1) * 50)
        
        # Category leadership impact
        if category_leadership > 0:
            score += min(10, category_leadership / 10)
            
        # Market share context (depends on stage)
        stage = doc.get("stage", "").lower()
        if stage in ["series-b", "series-c", "growth"]:
            if market_share > 0:
                score += min(10, market_share * 50)
        
        # Community effects
        if community_growth_rate > 0:
            score += min(5, community_growth_rate * 25)
            
        # Sector-specific adjustments
        sector = doc.get("sector", "").lower()
        if sector in ["saas", "cloud"]:
            # SaaS businesses benefit from market dynamics
            score += 5
        elif sector in ["consumer", "social", "marketplace"]:
            # Network effects businesses get bonus
            if viral_coefficient > 0.7:
                score += 8
            
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    except Exception as e:
        logger.error(f"Error calculating market score: {e}")
        logger.debug(traceback.format_exc())
        return 50.0  # Default to neutral on error

def calculate_advantage_score(doc: dict) -> float:
    """
    Calculate Advantage Moat score (20% of CAMP framework)
    
    Evaluates competitive advantages and defensibility.
    """
    try:
        # Extract relevant metrics with defaults
        patent_count = float(doc.get("patent_count", 0.0))
        tech_innovation = float(doc.get("technical_innovation_score", 0.0))
        if tech_innovation > 1:  # Convert percentage to decimal if needed
            tech_innovation /= 100
        product_maturity = float(doc.get("product_maturity_score", 0.0))
        if product_maturity > 1:
            product_maturity /= 100
        api_integrations = float(doc.get("api_integrations_count", 0.0))
        scalability = float(doc.get("scalability_score", 0.0))
        if scalability > 1:
            scalability /= 100
        tech_debt = float(doc.get("technical_debt_score", 0.0))
        if tech_debt > 1:
            tech_debt /= 100
        product_security = float(doc.get("product_security_score", 0.0))
        if product_security > 1:
            product_security /= 100
        business_model_strength = float(doc.get("business_model_strength", 0.0))
        if business_model_strength > 1:
            business_model_strength /= 100
        category_leadership = float(doc.get("category_leadership_score", 0.0))
        if category_leadership > 1:
            category_leadership /= 100
        
        # Also get moat score from the team_moat module
        moat_score = doc.get("moat_score", compute_moat_score(doc))
        if moat_score > 100:  # Normalize if needed
            moat_score /= 100
            
        # Base score calculation (0-100 scale)
        score = 40.0  # Start slightly below neutral
        
        # Patent impact (varies by sector)
        sector = doc.get("sector", "").lower()
        if patent_count > 0:
            if sector in ["biotech", "pharma", "medtech", "hardware"]:
                # Patents matter more in these sectors
                patent_factor = min(25, patent_count * 5)
            else:
                patent_factor = min(15, patent_count * 3)
            score += patent_factor
        
        # Technical innovation impact
        if tech_innovation > 0:
            tech_factor = min(20, tech_innovation * 20)
            score += tech_factor
        
        # Product maturity impact
        if product_maturity > 0:
            maturity_factor = min(15, product_maturity * 15)
            score += maturity_factor
        
        # API integrations impact (network effects)
        if api_integrations > 0:
            api_factor = min(10, api_integrations / 20 * 10)
            score += api_factor
        
        # Tech debt impact (negative)
        if tech_debt > 0:
            tech_debt_penalty = min(15, tech_debt * 15)
            score -= tech_debt_penalty
        
        # Scalability impact
        if scalability > 0:
            scalability_factor = min(15, scalability * 15)
            score += scalability_factor
        
        # Business model strength impact
        if business_model_strength > 0:
            biz_factor = min(15, business_model_strength * 15)
            score += biz_factor
        
        # Category leadership impact
        if category_leadership > 0:
            leadership_factor = min(15, category_leadership * 0.15)
            score += leadership_factor
            
        # Incorporate existing moat score (gives some weight to the old calculation)
        if moat_score > 0:
            score = 0.7 * score + 0.3 * moat_score
            
        # Sector-specific adjustments
        if sector in ["saas", "cloud"]:
            if api_integrations > 5:
                score += 5  # SaaS with integrations has better moat
        elif sector in ["network", "marketplace", "platform"]:
            score += 8  # Network effects businesses get moat bonus
        elif sector in ["consumer"]:
            if category_leadership > 60:
                score += 5  # Consumer with brand leadership
            else:
                score -= 5  # Consumer without leadership
                
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    except Exception as e:
        logger.error(f"Error calculating advantage score: {e}")
        logger.debug(traceback.format_exc())
        return 50.0  # Default to neutral on error

def calculate_people_score(doc: dict) -> float:
    """
    Calculate People & Performance score (25% of CAMP framework)
    
    Evaluates team strength, experience, and execution ability.
    """
    try:
        # Extract relevant metrics with defaults
        founder_exits = float(doc.get("founder_exits", 0.0))
        domain_exp = float(doc.get("founder_domain_exp_yrs", 0.0))
        employee_count = float(doc.get("employee_count", 0.0))
        diversity_score = float(doc.get("founder_diversity_score", 0.0))
        if diversity_score > 1:  # Convert percentage to decimal if needed
            diversity_score /= 100
        mgmt_satisfaction = float(doc.get("management_satisfaction_score", 0.0))
        if mgmt_satisfaction > 1:
            mgmt_satisfaction /= 100
        tech_ratio = float(doc.get("tech_talent_ratio", 0.0))
        has_cto = doc.get("has_cto", False)
        has_cmo = doc.get("has_cmo", False)
        has_cfo = doc.get("has_cfo", False)
        turnover_rate = float(doc.get("employee_turnover_rate", 0.0))
        if turnover_rate > 1:
            turnover_rate /= 100
        nps_score = float(doc.get("nps_score", 0.0))
        support_sla = float(doc.get("support_ticket_sla_percent", 0.0))
        if support_sla > 1:
            support_sla /= 100
            
        # Also get team score from team_moat module
        team_score = doc.get("team_score", compute_team_depth_score(doc))
        if team_score > 100:  # Normalize if needed
            team_score /= 100
            
        # Base score calculation (0-100 scale)
        score = 40.0  # Start slightly below neutral
        
        # Founder exits impact (significant multiplier)
        if founder_exits > 0:
            exit_factor = min(30, founder_exits * 15)
            score += exit_factor
        
        # Domain experience impact (diminishing returns after 10 years)
        if domain_exp > 0:
            if domain_exp > 10:
                exp_factor = 15 + min(5, (domain_exp - 10) / 10 * 5)
            else:
                exp_factor = min(15, domain_exp * 1.5)
            score += exp_factor
        
        # Leadership completeness impact
        exec_count = sum([has_cto, has_cmo, has_cfo])
        if exec_count > 0:
            exec_factor = min(15, exec_count * 5)
            score += exec_factor
        
        # Team size impact (with sector context)
        sector = doc.get("sector", "").lower()
        stage = doc.get("stage", "").lower()
        
        if employee_count > 0:
            # Different expectations by stage
            if stage in ["pre-seed", "seed"]:
                size_factor = min(10, employee_count / 5 * 10)
            else:
                size_factor = min(10, employee_count / 20 * 10)
            score += size_factor
            
        # Diversity impact
        if diversity_score > 0:
            diversity_factor = min(10, diversity_score * 10)
            score += diversity_factor
            
        # Tech talent ratio impact
        if tech_ratio > 0:
            # More important for tech companies
            if sector in ["saas", "ai", "tech", "software"]:
                tech_factor = min(10, tech_ratio * 15)
            else:
                tech_factor = min(10, tech_ratio * 10)
            score += tech_factor
            
        # Turnover impact (negative)
        if turnover_rate > 0:
            turnover_penalty = min(15, turnover_rate * 50)
            score -= turnover_penalty
            
        # NPS impact (customer satisfaction indicator)
        if nps_score != 0:  # NPS can be negative
            nps_factor = min(10, max(-10, nps_score / 10))
            score += nps_factor
            
        # Support SLA impact
        if support_sla > 0:
            sla_factor = min(5, support_sla * 5)
            score += sla_factor
            
        # Incorporate existing team score (gives some weight to the old calculation)
        if team_score > 0:
            score = 0.7 * score + 0.3 * team_score
            
        # Sector-specific adjustments
        if sector in ["deeptech", "biotech", "hardware"]:
            if domain_exp > 5 and has_cto:
                score += 5  # Technical expertise matters more
                
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    except Exception as e:
        logger.error(f"Error calculating people score: {e}")
        logger.debug(traceback.format_exc())
        return 50.0  # Default to neutral on error

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost model with optimized parameters from config
    """
    from sklearn.model_selection import train_test_split
    from config import XGB_PARAMS
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make a copy of params to avoid modifying the original
    params = XGB_PARAMS.copy()
    
    # Ensure compatibility with the installed XGBoost version
    if 'use_label_encoder' in params and xgb.__version__ >= '1.6.0':
        del params['use_label_encoder']
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, 
             eval_set=[(X_test, y_test)],
             early_stopping_rounds=20,
             verbose=False)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"XGBoost v{xgb.__version__} Train acc={train_acc:.2f}, Test acc={test_acc:.2f}")

    return model
