import os
import json
import pickle
import numpy as np
import logging
from config import XGB_PARAMS
from constants import NAMED_METRICS_50
from advanced_ml import build_feature_vector_no_llm, train_model
from sklearn.model_selection import train_test_split

logger= logging.getLogger("train_model")
logging.basicConfig(level=logging.INFO)

def load_training_docs():
    """
    optionally load big_startups.json with pass/fail outcome,
    or fallback with multiple sample docs.
    """
    if os.path.exists("data/big_startups.json"):
        try:
            with open("data/big_startups.json","r",encoding="utf-8", errors="replace") as f:
                data= json.load(f)
            docs= data.get("startups",[])
            # only keep docs that have outcome= pass or fail
            pf_docs= [d for d in docs if d.get("outcome") in ["pass","fail"]]
            if pf_docs:
                logger.info(f"Loaded {len(pf_docs)} pass/fail docs from big_startups.json")
                return pf_docs
        except Exception as e:
            logger.error(f"Error loading big_startups.json => {str(e)}")

    # If no file or no pass/fail doc found => fallback to broader sample data
    logger.warning("Using extended sample dataset with diverse cases")

    data_docs= [
        # Success cases
        {
            "outcome":"pass",
            "monthly_revenue":100000,
            "user_growth_rate":0.15,
            "burn_rate":60000,
            "churn_rate":0.03,
            "ltv_cac_ratio":3.5,
            "founder_exits":2,
            "founder_domain_exp_yrs":8,
            "pitch_deck_text": "We have a strong user base and well-defined product..."
        },
        {
            "outcome":"pass",
            "monthly_revenue":20000,
            "user_growth_rate":0.2,
            "burn_rate":10000,
            "churn_rate":0.02,
            "ltv_cac_ratio":4.0,
            "founder_exits":1,
            "founder_domain_exp_yrs":5,
            "pitch_deck_text": "Our solution addresses a big market with a strong team..."
        },
        {
            "outcome":"pass",
            "monthly_revenue":50000,
            "user_growth_rate":0.25,
            "burn_rate":30000,
            "churn_rate":0.01,
            "ltv_cac_ratio":3.0,
            "founder_exits":1,
            "founder_domain_exp_yrs":7,
            "runway_months":18,
            "gross_margin":0.65
        },
        {
            "outcome":"pass",
            "monthly_revenue":150000,
            "user_growth_rate":0.10,
            "burn_rate":80000,
            "churn_rate":0.04,
            "ltv_cac_ratio":2.8,
            "founder_exits":0,
            "founder_domain_exp_yrs":10,
            "runway_months":24,
            "gross_margin":0.70
        },
        # Failure cases
        {
            "outcome":"fail",
            "monthly_revenue":3000,
            "user_growth_rate":0.01,
            "burn_rate":20000,
            "churn_rate":0.15,
            "ltv_cac_ratio":1.2,
            "founder_exits":0,
            "founder_domain_exp_yrs":1
        },
        {
            "outcome":"fail",
            "monthly_revenue":500,
            "user_growth_rate":0.0,
            "burn_rate":5000,
            "churn_rate":0.2,
            "ltv_cac_ratio":0.8,
            "founder_exits":0,
            "founder_domain_exp_yrs":0
        },
        {
            "outcome":"fail",
            "monthly_revenue":10000,
            "user_growth_rate":0.05,
            "burn_rate":40000,
            "churn_rate":0.12,
            "ltv_cac_ratio":1.5,
            "founder_exits":0,
            "founder_domain_exp_yrs":3,
            "runway_months":3,
            "gross_margin":0.40
        },
        {
            "outcome":"fail",
            "monthly_revenue":30000,
            "user_growth_rate":-0.05,
            "burn_rate":50000,
            "churn_rate":0.18,
            "ltv_cac_ratio":1.0,
            "founder_exits":1,
            "founder_domain_exp_yrs":2,
            "runway_months":6,
            "gross_margin":0.30
        }
    ]
    return data_docs

def load_training_data():
    """
    build X, y from pass/fail outcome
    """
    docs= load_training_docs()
    X_list= []
    y_list= []
    for d in docs:
        fv= build_feature_vector_no_llm(d)
        X_list.append(fv)
        outcome = d.get("outcome","fail")
        y_list.append(1 if outcome=="pass" else 0)
    return np.array(X_list), np.array(y_list,dtype=int)

def train_model_xgb():
    X, y = load_training_data()
    if len(set(y)) < 2:
        logger.error("Need >=2 classes => add more pass/fail docs.")
        return
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")
    logger.info(f"Class distribution in training: Pass={sum(y_train)}, Fail={len(y_train) - sum(y_train)}")
    
    model = train_model(X, y)  # Train on all data for final model
    
    # Validate model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Validation set predictions
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Validation metrics - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    
    # Save feature importance information
    if hasattr(model, 'feature_importances_'):
        # Get feature names (base features + domain expansions + special metrics)
        feature_names = NAMED_METRICS_50.copy()
        # Add additional feature names (domain expansions + others)
        feature_names.extend(['intangible', 'team_score', 'moat_score'])
        
        # Ensure feature_names matches feature_importances_ length
        if len(feature_names) > len(model.feature_importances_):
            feature_names = feature_names[:len(model.feature_importances_)]
        elif len(feature_names) < len(model.feature_importances_):
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), len(model.feature_importances_))])
        
        # Get sorted indices of feature importance
        sorted_idx = model.feature_importances_.argsort()[::-1]
        
        # Print top features
        logger.info("Top 10 important features:")
        for i in sorted_idx[:10]:
            if i < len(feature_names):
                logger.info(f"  {feature_names[i]}: {model.feature_importances_[i]:.4f}")
    
    # Save the model
    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model_xgb.pkl => done")
    
    # Test prediction on a sample case to verify it works
    sample_case = {
        "monthly_revenue": 75000,
        "user_growth_rate": 0.12,
        "burn_rate": 50000,
        "churn_rate": 0.05,
        "ltv_cac_ratio": 2.5
    }
    
    test_features = build_feature_vector_no_llm(sample_case).reshape(1, -1)
    
    if hasattr(model, 'predict_proba'):
        test_prob = model.predict_proba(test_features)[0, 1]
        logger.info(f"Test prediction probability on sample case: {test_prob:.4f} ({test_prob*100:.1f}%)")
    else:
        test_pred = model.predict(test_features)[0]
        logger.info(f"Test prediction on sample case: {test_pred:.4f}")

if __name__=="__main__":
    train_model_xgb()