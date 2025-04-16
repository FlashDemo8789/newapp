"""
Startup Success Prediction Model Serving

This script loads and serves the trained startup success prediction model
using the ML Infrastructure serving framework.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Import model serving components
from ml_infrastructure.serving.model_serving import ModelServer
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

def preprocess_input(input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Preprocess input data for the model
    
    Args:
        input_data: Raw input data
        
    Returns:
        Preprocessed data ready for the model
    """
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
            # Handle the case where the value might be a dictionary or other complex type
            value = input_data[feature]
            if isinstance(value, dict):
                # If it's a dictionary, try to extract a numeric value
                # This is a common pattern in APIs where values might come with metadata
                for k, v in value.items():
                    if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                        value = v
                        break
            
            # Convert to float
            try:
                processed_data[feature] = float(value)
            except (TypeError, ValueError):
                # If conversion fails, use default value
                logger.warning(f"Could not convert {feature} value '{value}' to float, using default 0.0")
                processed_data[feature] = 0.0
        else:
            processed_data[feature] = 0.0
    
    # Create DataFrame with columns in the exact order expected by the model
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    df = pd.DataFrame([processed_data], columns=columns)
    
    # Convert to numpy array
    X = df.values
    
    return {'input': X}

def postprocess_output(output: Any) -> Dict[str, Any]:
    """
    Postprocess model output
    
    Args:
        output: Raw model output
        
    Returns:
        Processed output with additional information
    """
    # Extract prediction result from model output
    if isinstance(output, np.ndarray):
        prediction = int(output[0])
    elif isinstance(output, List) and len(output) > 0:
        prediction = int(output[0])
    elif isinstance(output, Dict) and 'result' in output:
        if isinstance(output['result'], List) and len(output['result']) > 0:
            prediction = int(output['result'][0])
        else:
            prediction = int(output['result'])
    else:
        # Default case
        prediction = 0
    
    # Set probability (would come from model.predict_proba in a real case)
    probability = 1.0 if prediction == 1 else 0.0
    
    # Format the result - prediction=1 means 'pass' (approval)
    result = {
        'prediction': prediction,
        'outcome': 'pass' if prediction == 1 else 'fail',
        'confidence': probability if prediction == 1 else (1 - probability),
        'success_probability': probability
    }
    
    return result

def create_input_schema() -> Dict[str, Any]:
    """
    Create JSON schema for input validation
    
    Returns:
        JSON schema for input validation
    """
    return {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'stage': {'type': 'string', 'enum': ['seed', 'series_a', 'series_b', 'series_c', 'growth']},
            'sector': {'type': 'string'},
            'monthly_revenue': {'type': ['number', 'string']},
            'annual_recurring_revenue': {'type': ['number', 'string']},
            'lifetime_value_ltv': {'type': ['number', 'string']},
            'gross_margin_percent': {'type': ['number', 'string']},
            'operating_margin_percent': {'type': ['number', 'string']},
            'burn_rate': {'type': ['number', 'string']},
            'runway_months': {'type': ['number', 'string']},
            'cash_on_hand_million': {'type': ['number', 'string']},
            'debt_ratio': {'type': ['number', 'string']},
            'financing_round_count': {'type': ['number', 'string']},
            'monthly_active_users': {'type': ['number', 'string']}
        },
        'required': [
            'stage', 'sector', 'monthly_revenue', 'annual_recurring_revenue',
            'gross_margin_percent', 'burn_rate', 'runway_months'
        ]
    }

def start_model_server(host: str = '0.0.0.0', port: int = 5000):
    """
    Start the model server
    
    Args:
        host: Host to bind the server
        port: Port to bind the server
    """
    # Create the model store directory if it doesn't exist
    os.makedirs("./model_store", exist_ok=True)
    
    # Initialize model server
    model_server = ModelServer(host=host, port=port)
    
    # Load startup success prediction model
    model_server.load_model(
        model_name="startup_success_predictor",
        preprocessor=preprocess_input,
        postprocessor=postprocess_output,
        schema=create_input_schema()
    )
    
    # Start the server
    logger.info(f"Starting model server on {host}:{port}")
    try:
        model_server.start()
    except OSError as e:
        if "Address already in use" in str(e):
            # Try with a different port
            alt_port = port + 1
            logger.warning(f"Port {port} is already in use. Trying port {alt_port}")
            model_server.port = alt_port
            try:
                model_server.start()
            except OSError:
                # If that fails too, try one more port
                alt_port = port + 2
                logger.warning(f"Port {port+1} is also in use. Trying port {alt_port}")
                model_server.port = alt_port
                model_server.start()
        else:
            # If it's a different error, re-raise it
            raise

def test_prediction(port: int = 5000):
    """Test prediction with sample data"""
    import requests
    
    # Sample startup data
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
    
    # Try ports in sequence if needed
    ports_to_try = [port, port+1, port+2]
    response = None
    
    for current_port in ports_to_try:
        try:
            # Make prediction request
            url = f"http://localhost:{current_port}/models/startup_success_predictor/predict"
            response = requests.post(url, json=sample_data, timeout=2)
            if response.status_code == 200:
                logger.info(f"Successfully connected to server on port {current_port}")
                break
        except requests.exceptions.RequestException:
            logger.warning(f"Could not connect to server on port {current_port}")
            response = None
    
    # Print result
    if response and response.status_code == 200:
        result = response.json()
        print("\nSample Prediction Result:")
        print(f"Startup: {sample_data['name']}")
        
        # Handle different response formats
        if 'result' in result and isinstance(result['result'], dict) and 'outcome' in result['result']:
            # New format with nested result
            outcome = result['result']['outcome']
            success_prob = result['result'].get('success_probability', 'N/A')
            confidence = result['result'].get('confidence', 'N/A')
        elif 'result' in result:
            # Direct result format
            if isinstance(result['result'], (int, float)):
                # Numeric prediction
                outcome = 'pass' if int(result['result']) == 1 else 'fail'
                success_prob = 1.0 if int(result['result']) == 1 else 0.0
                confidence = success_prob
            else:
                # Unknown format
                outcome = str(result['result'])
                success_prob = 'N/A'
                confidence = 'N/A'
        else:
            # Completely different format
            outcome = str(result.get('outcome', 'unknown'))
            success_prob = result.get('success_probability', 'N/A')
            confidence = result.get('confidence', 'N/A')
            
        print(f"Outcome: {outcome} (pass = success, fail = rejection)")
        print(f"Success Probability: {success_prob}")
        print(f"Confidence: {confidence}")
        
        # Print model info if available
        if 'model' in result:
            print(f"\nModel info:")
            print(f"Name: {result['model'].get('name', 'unknown')}")
            print(f"Version: {result['model'].get('version', 'unknown')}")
        
        print("\nPrediction successful!")
    else:
        error_text = response.text if response else "Could not connect to server"
        status = response.status_code if response else "N/A"
        print(f"Error: {status} - {error_text}")
        print("Make sure the model server is running.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Startup Success Prediction Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server")
    parser.add_argument("--test", action="store_true", help="Test prediction with sample data")
    
    args = parser.parse_args()
    
    if args.test:
        test_prediction(port=args.port)
    else:
        start_model_server(host=args.host, port=args.port) 