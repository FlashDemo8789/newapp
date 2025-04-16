import os
import logging
import pandas as pd
import numpy as np
from .camp_model import CAMPModel
from app.ml.data_loader import load_startup_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_training")

def train_ml_model():
    """Train the CAMP framework ML model using real startup data."""
    logger.info("Starting model training process with real startup data")
    
    # Initialize model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model = CAMPModel(model_dir=model_dir)
    
    # Check if models are already trained
    if model.is_trained:
        logger.info("Models are already trained, forcing retraining with real data")
    
    # Load real startup data
    logger.info("Loading real startup data for training")
    training_data = load_startup_data()
    
    if training_data.empty:
        logger.error("Failed to load real startup data, cannot train models")
        return {
            "status": "error",
            "message": "Failed to load real startup data"
        }
    
    logger.info(f"Successfully loaded {len(training_data)} real startup records for training")
    
    # Train model with real data
    result = model.train(data=training_data)
    
    if result['status'] == 'success':
        logger.info(f"Model training completed successfully with {result['samples']} real startup samples")
        
        # Log feature importance for each dimension
        for dimension, metrics in result['metrics'].items():
            logger.info(f"Top features for {dimension} dimension:")
            importance = metrics['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in top_features:
                logger.info(f"  {feature}: {score:.4f}")
        
        return {
            "status": "success",
            "message": "Models trained successfully with real startup data",
            "details": result
        }
    else:
        logger.error(f"Model training failed: {result['message']}")
        return {
            "status": "error",
            "message": result['message']
        }

def test_model():
    """Test the trained model with a sample startup to verify it's working."""
    logger.info("Testing model with sample startup data")
    
    # Initialize model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model = CAMPModel(model_dir=model_dir)
    
    # Sample startup data following the expected schema
    sample_data = {
        'company_info': {
            'industry': 'SaaS',
            'stage': 'seed',
            'business_model': 'B2B'
        },
        'capital_metrics': {
            'burn_rate': 120000,
            'runway_months': 18,
            'ltv_cac_ratio': 3.5,
            'gross_margin': 0.75,
            'mrr': 80000,
            'cac_payback_months': 10
        },
        'advantage_metrics': {
            'product_uniqueness': 0.8,
            'tech_innovation_score': 0.7,
            'network_effects': 0.6,
            'product_maturity': 0.65,
            'uptime_percentage': 0.995,
            'algorithm_uniqueness': 0.75,
            'moat_score': 0.75
        },
        'market_metrics': {
            'tam_size': 5000000000,
            'monthly_active_users': 50000,
            'daily_active_users': 25000,
            'user_growth_rate': 0.2,
            'churn_rate': 0.05,
            'net_revenue_retention': 1.15,
            'conversion_rate': 0.08
        },
        'people_metrics': {
            'founder_experience': 0.8,
            'tech_talent_ratio': 0.6,
            'team_completeness': 0.7,
            'team_size': 15,
            'founder_exits': 1,
            'management_satisfaction': 0.85,
            'founder_network_reach': 0.7
        }
    }
    
    try:
        # Make predictions
        predictions = model.predict(sample_data)
        
        logger.info("Model predictions for sample startup:")
        for dimension, score in predictions.items():
            logger.info(f"  {dimension}: {score:.4f}")
        
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error in model testing: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "predictions": {
                "overall_score": 0.5,
                "capital_score": 0.5,
                "advantage_score": 0.5,
                "market_score": 0.5,
                "people_score": 0.5
            }
        }

if __name__ == "__main__":
    # Train the model
    train_result = train_ml_model()
    
    if train_result["status"] == "success":
        # Test the model
        test_result = test_model()
        print(f"Test predictions: {test_result['predictions']}")
    else:
        print(f"Model training failed: {train_result['message']}")
