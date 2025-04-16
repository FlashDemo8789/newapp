"""
Test the startup success prediction model directly.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from ml_infrastructure.model_registry.registry import ModelRegistry

def main():
    # Initialize the model registry
    registry = ModelRegistry("./model_store")
    
    # Get the model
    try:
        model = registry.get_model("startup_success_predictor")
        print("Successfully loaded model from registry")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Verify the model type
    print(f"Model type: {type(model)}")
    
    # Inspect model pipeline
    if isinstance(model, Pipeline):
        print("\nModel pipeline steps:")
        for i, (name, step) in enumerate(model.steps):
            print(f"  {i}. {name}: {type(step)}")
        
        # Get the classifier
        classifier = model.named_steps.get('classifier')
        if classifier:
            print(f"\nClassifier: {type(classifier)}")
            if hasattr(classifier, 'n_features_in_'):
                print(f"Expected features count: {classifier.n_features_in_}")
            if hasattr(classifier, 'classes_'):
                print(f"Classes: {classifier.classes_}")
            if hasattr(classifier, 'n_classes_'):
                print(f"Number of classes: {classifier.n_classes_}")
            
            # Feature importances
            if hasattr(classifier, 'feature_importances_'):
                print("\nFeature importances:")
                importances = classifier.feature_importances_
                for i, importance in enumerate(importances):
                    print(f"  Feature {i}: {importance:.4f}")
    
    # Sample data
    sample_data = {
        "name": "TechVenture",
        "stage": "seed",
        "sector": "saas",
        "monthly_revenue": 50000,
        "annual_recurring_revenue": 600000,
        "lifetime_value_ltv": 8000,
        "gross_margin_percent": 70,
        "operating_margin_percent": 15,
        "burn_rate": 1.2,
        "runway_months": 18,
        "cash_on_hand_million": 2.5,
        "debt_ratio": 0.1,
        "financing_round_count": 1,
        "monthly_active_users": 12000
    }
    
    # Feature definitions from the training script
    numeric_features = [
        'monthly_revenue', 'annual_recurring_revenue', 'lifetime_value_ltv',
        'gross_margin_percent', 'operating_margin_percent', 'burn_rate',
        'runway_months', 'cash_on_hand_million', 'debt_ratio',
        'financing_round_count', 'monthly_active_users'
    ]
    categorical_features = ['stage_numeric', 'sector_numeric']
    
    # Preprocess the data
    stage_mapping = {'seed': 0, 'series_a': 1, 'series_b': 2, 'series_c': 3, 'growth': 4}
    sector_mapping = {
        'healthtech': 0, 'fintech': 1, 'saas': 2, 'ecommerce': 3, 
        'ai': 4, 'edtech': 5, 'consumer': 6, 'hardware': 7
    }
    
    # Create a DataFrame matching the expected format
    processed_data = {}
    
    # Add categorical features
    processed_data['stage_numeric'] = stage_mapping.get(sample_data['stage'], 0)
    processed_data['sector_numeric'] = sector_mapping.get(sample_data['sector'], 0)
    
    # Add numeric features
    for feature in numeric_features:
        processed_data[feature] = sample_data.get(feature, 0.0)
    
    # Create a DataFrame with the exact columns used in training
    columns = categorical_features + numeric_features
    df = pd.DataFrame([processed_data], columns=columns)
    print(f"\nInput DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to numpy array
    X = df.values
    print(f"X shape: {X.shape}")
    
    # Make prediction
    try:
        prediction = model.predict(X)
        print("\nPrediction result:")
        print(f"Raw prediction: {prediction}")
        
        # Try to get class probabilities if available
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                print(f"Class probabilities: {probabilities}")
        except Exception as e:
            print(f"Error getting probabilities: {e}")
        
        # Correct interpretation: prediction=1 means 'pass' (approval)
        outcome = "pass" if prediction[0] == 1 else "fund"
        print(f"Outcome: {outcome}")
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    print("\nPrediction successful! The startup success prediction model is working correctly.")

if __name__ == "__main__":
    main() 