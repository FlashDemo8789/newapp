"""
Startup Success Prediction Model Training

This script trains a machine learning model to predict startup success
based on financial and operational metrics. It uses the startup dataset
and registers the trained model with the model registry.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Import the model registry
from ml_infrastructure.model_registry.registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_startup_data(file_path: str) -> pd.DataFrame:
    """
    Load startup data from JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame with startup data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['startups'])
    
    logger.info(f"Loaded {len(df)} startups from {file_path}")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the startup data
    
    Args:
        df: Raw startup data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert categorical variables to numeric
    stage_mapping = {'seed': 0, 'series_a': 1, 'series_b': 2, 'series_c': 3, 'growth': 4}
    if 'stage' in processed_df.columns:
        processed_df['stage_numeric'] = processed_df['stage'].map(stage_mapping)
    
    sector_mapping = {
        'healthtech': 0, 'fintech': 1, 'saas': 2, 'ecommerce': 3, 
        'ai': 4, 'edtech': 5, 'consumer': 6, 'hardware': 7
    }
    if 'sector' in processed_df.columns:
        processed_df['sector_numeric'] = processed_df['sector'].map(sector_mapping)
    
    # Convert outcome to binary target
    # In our data 'pass' means approval (positive outcome)
    if 'outcome' in processed_df.columns:
        processed_df['success'] = processed_df['outcome'].apply(lambda x: 1 if x == 'pass' else 0)
    
    logger.info("Preprocessed startup data")
    return processed_df

def extract_features_and_target(df: pd.DataFrame):
    """
    Extract features and target from DataFrame
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Define features to use
    numeric_features = [
        'monthly_revenue', 'annual_recurring_revenue', 'lifetime_value_ltv',
        'gross_margin_percent', 'operating_margin_percent', 'burn_rate',
        'runway_months', 'cash_on_hand_million', 'debt_ratio',
        'financing_round_count', 'monthly_active_users'
    ]
    
    categorical_features = ['stage_numeric', 'sector_numeric']
    
    # Combine all features
    feature_columns = [col for col in numeric_features + categorical_features if col in df.columns]
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=feature_columns + ['success'])
    
    X = df_clean[feature_columns].values
    y = df_clean['success'].values
    
    logger.info(f"Extracted features and target from {len(df_clean)} startups")
    return X, y, feature_columns

def train_model(X, y):
    """
    Train a model to predict startup success
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Trained model pipeline
    """
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = pipeline.score(X_train, y_train)
    val_accuracy = pipeline.score(X_val, y_val)
    
    logger.info(f"Model trained with train accuracy: {train_accuracy:.4f}, validation accuracy: {val_accuracy:.4f}")
    
    return pipeline, {'train_accuracy': train_accuracy, 'validation_accuracy': val_accuracy}

def register_model(model, metrics, feature_columns):
    """
    Register the trained model with the model registry
    
    Args:
        model: Trained model pipeline
        metrics: Performance metrics
        feature_columns: List of feature column names
        
    Returns:
        Model version ID
    """
    # Initialize the model registry
    registry = ModelRegistry("./model_store")
    
    # Register the model
    version_id = registry.register_model(
        model_name="startup_success_predictor",
        model_object=model,
        metrics=metrics,
        metadata={
            'description': 'Startup success prediction model',
            'feature_columns': feature_columns,
            'target': 'success (1=pass, 0=fund)'
        },
        framework="scikit-learn",
        tags=["startup", "prediction", "random_forest"],
        activate=True  # Make this the active version
    )
    
    logger.info(f"Registered model with version ID: {version_id}")
    return version_id

def main():
    """Main function to train and register the model"""
    # Create model store directory if it doesn't exist
    os.makedirs("./model_store", exist_ok=True)
    
    # Load the data
    data_path = "data/big_startups.json"
    df = load_startup_data(data_path)
    
    # Preprocess the data
    processed_df = preprocess_data(df)
    
    # Extract features and target
    X, y, feature_columns = extract_features_and_target(processed_df)
    
    # Train the model
    model, metrics = train_model(X, y)
    
    # Register the model
    version_id = register_model(model, metrics, feature_columns)
    
    print(f"Successfully trained and registered startup success prediction model (version {version_id})")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    
    return version_id

if __name__ == "__main__":
    main() 