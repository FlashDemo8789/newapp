"""
Test script for the simple startup success prediction API.
"""

import requests
import json

def test_prediction():
    """Test prediction with the simple API"""
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
    
    # Make prediction request
    try:
        response = requests.post(
            "http://localhost:5001/predict",
            json=sample_data
        )
        
        # Check result
        if response.status_code == 200:
            result = response.json()
            print("\nSample Prediction Result:")
            print(f"Startup: {sample_data['name']}")
            print(f"Outcome: {result['outcome']} (pass = approved, fail = rejected)")
            
            if 'success_probability' in result:
                print(f"Success Probability: {result['success_probability']:.2f}")
            
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2f}")
            
            print("\nPrediction successful!")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request error: {e}")

def test_health():
    """Test health check endpoint"""
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            health = response.json()
            print("\nHealth check:")
            print(f"Status: {health['status']}")
            print(f"Model loaded: {health['model_loaded']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    print("Testing simple API...")
    print("Note: In our model, 'pass' means the startup was approved (positive outcome).")
    print("      'fail' means the startup was rejected (negative outcome).")
    
    # First check health
    test_health()
    
    # Then test prediction
    test_prediction() 