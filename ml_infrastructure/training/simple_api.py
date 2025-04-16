"""
Simple Startup Success Prediction API

A simple Flask API for serving the startup success prediction model
without using the complex model server framework.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from flask import Flask, request, jsonify

# Import model registry
from ml_infrastructure.model_registry.registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the feature columns in correct order
CATEGORICAL_FEATURES = ['stage_numeric', 'sector_numeric']
NUMERIC_FEATURES = [
    'monthly_revenue', 'annual_recurring_revenue', 'lifetime_value_ltv',
    'gross_margin_percent', 'operating_margin_percent', 'burn_rate',
    'runway_months', 'cash_on_hand_million', 'debt_ratio',
    'financing_round_count', 'monthly_active_users'
]

# Initialize Flask app
app = Flask(__name__)

# Load the model once at startup
registry = ModelRegistry("./model_store")
model = None

def load_model():
    """Load the model from the registry"""
    global model
    try:
        model = registry.get_model("startup_success_predictor")
        logger.info("Loaded startup success prediction model")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_input(input_data):
    """Preprocess input data for the model"""
    # Convert categorical variables to numeric
    stage_mapping = {'seed': 0, 'series_a': 1, 'series_b': 2, 'series_c': 3, 'growth': 4}
    sector_mapping = {
        'healthtech': 0, 'fintech': 1, 'saas': 2, 'ecommerce': 3, 
        'ai': 4, 'edtech': 5, 'consumer': 6, 'hardware': 7
    }
    
    # Create a structured data dictionary
    processed_data = {}
    
    # Process stage
    if 'stage' in input_data:
        processed_data['stage_numeric'] = stage_mapping.get(input_data['stage'], 0)
    else:
        processed_data['stage_numeric'] = 0
    
    # Process sector
    if 'sector' in input_data:
        processed_data['sector_numeric'] = sector_mapping.get(input_data['sector'], 0)
    else:
        processed_data['sector_numeric'] = 0
    
    # Process numeric features
    for feature in NUMERIC_FEATURES:
        if feature in input_data:
            processed_data[feature] = float(input_data[feature])
        else:
            processed_data[feature] = 0.0
    
    # Create DataFrame with columns in the exact order expected by the model
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    df = pd.DataFrame([processed_data], columns=columns)
    
    return df.values

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction with the model"""
    # Make sure model is loaded
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not available'}), 500
    
    # Get input data
    try:
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400
    
    # Preprocess input
    try:
        X = preprocess_input(input_data)
    except Exception as e:
        return jsonify({'error': f'Error preprocessing input: {e}'}), 400
    
    # Make prediction
    try:
        prediction = model.predict(X)
        
        # Try to get probability if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X)
            except:
                pass
        
        # Format result - prediction=1 means 'pass' (approval)
        result = {
            'prediction': int(prediction[0]),
            'outcome': 'pass' if prediction[0] == 1 else 'fail'
        }
        
        # Add probability if available
        if probabilities is not None and len(probabilities) > 0:
            if probabilities.shape[1] > 1:  # Binary classification
                result['success_probability'] = float(probabilities[0, 1])
            else:  # Single class
                result['success_probability'] = 1.0 if prediction[0] == 1 else 0.0
        else:
            result['success_probability'] = 1.0 if prediction[0] == 1 else 0.0
        
        result['confidence'] = (
            result['success_probability'] 
            if result['prediction'] == 1 
            else 1 - result['success_probability']
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

def start_server(host='0.0.0.0', port=5001):
    """Start the Flask server"""
    # Create the model store directory if it doesn't exist
    os.makedirs("./model_store", exist_ok=True)
    
    # Load the model
    load_model()
    
    # Run the app
    app.run(host=host, port=port)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Startup Success Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind the server")
    
    args = parser.parse_args()
    
    logger.info(f"Starting simple API server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port) 