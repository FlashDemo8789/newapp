"""
Test script for FlashDNA API endpoints using memory repository
This ensures the API works correctly even without MongoDB
"""
import sys
import os
import asyncio
import logging
from fastapi.testclient import TestClient
from datetime import datetime, date

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force memory repository
os.environ["MONGO_URI"] = "mongodb://non-existent-host:27017/flash_dna"
os.environ["ALLOW_MEMORY_FALLBACK"] = "true"

# Import app after setting env vars
from app.main import app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_api")

# Create test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns proper response"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()
    logger.info("Root endpoint test passed")

def test_metadata_endpoints():
    """Test the metadata endpoints"""
    # Test business models endpoint
    response = client.get("/api/business-models")
    assert response.status_code == 200
    assert len(response.json()) > 0
    logger.info("Business models endpoint test passed")
    
    # Test startup stages endpoint
    response = client.get("/api/startup-stages")
    assert response.status_code == 200
    assert len(response.json()) > 0
    logger.info("Startup stages endpoint test passed")
    
    # Test metrics tooltips endpoint
    response = client.get("/api/metrics-tooltips")
    assert response.status_code == 200
    assert "metrics" in response.json()
    assert len(response.json()["metrics"]) > 0
    logger.info("Metrics tooltips endpoint test passed")

def test_analysis_workflow():
    """Test the full analysis workflow"""
    # First get valid business models and startup stages
    response = client.get("/api/business-models")
    business_models = response.json()
    logger.info(f"Business models response: {business_models}")
    valid_industry = "SaaS"  # Hard-code a valid value from the constants
    
    response = client.get("/api/startup-stages")
    startup_stages = response.json()
    logger.info(f"Startup stages response: {startup_stages}")
    valid_stage = "Seed"  # Hard-code a valid value from the constants
    
    logger.info(f"Using industry: {valid_industry}, stage: {valid_stage}")
    
    # Create a test startup data payload
    startup_data = {
        "name": "Test Startup",
        "industry": valid_industry,
        "stage": valid_stage,
        "founded_date": date(2023, 1, 1).isoformat(),
        "description": "A test startup for API validation",
        "capital_metrics": {
            "monthly_revenue": 100000,
            "annual_recurring_revenue": 1200000,
            "burn_rate": 50000,
            "runway_months": 18,
            "customer_acquisition_cost": 1000,
            "ltv_cac_ratio": 3.5,
            "gross_margin": 0.65
        },
        "advantage_metrics": {
            "competition_level": 7,
            "tech_innovation_score": 0.8,
            "network_effects_score": 0.75,
            "business_model_strength": 0.85
        },
        "market_metrics": {
            "tam_size": 5000000000,
            "market_growth_rate": 0.25,
            "active_users": 10000,
            "user_growth_rate": 0.15
        },
        "people_metrics": {
            "founder_domain_exp_yrs": 8,
            "team_size": 30,
            "technical_skill_score": 0.85,
            "leadership_score": 0.75,
            "founder_exits": 1
        }
    }
    
    # Submit analysis
    logger.info("Submitting test analysis...")
    response = client.post("/api/analyze", json=startup_data)
    if response.status_code != 200:
        logger.error(f"API error: {response.status_code} - {response.json()}")
        sys.exit(1)
    
    logger.info("Analysis API response successful!")
    analysis_result = response.json()
    
    # Print full response for debugging
    logger.info(f"Full analysis response: {analysis_result}")
    
    logger.info(f"Analysis result keys: {list(analysis_result.keys())}")
    assert "id" in analysis_result
    assert analysis_result["id"] is not None, "Analysis ID should not be None"
    assert "overall_score" in analysis_result
    assert "capital_score" in analysis_result
    assert "advantage_score" in analysis_result
    assert "market_score" in analysis_result
    assert "people_score" in analysis_result
    
    analysis_id = analysis_result["id"]
    logger.info(f"Analysis created with ID: {analysis_id}")
    
    # Test passes for first analysis creation
    logger.info("Basic analysis test complete. Moving to analysis details...")
    
    # Get analysis details
    logger.info("Retrieving analysis details...")
    response = client.get(f"/api/analysis-details/{analysis_id}")
    if response.status_code != 200:
        logger.error(f"Analysis details API error: {response.status_code} - {response.json()}")
        sys.exit(1)
        
    logger.info("Analysis details API response successful!")
    analysis_details = response.json()
    logger.info(f"Analysis details keys: {list(analysis_details.keys())}")
    assert analysis_details["id"] == analysis_id
    
    # Test passes for analysis details
    logger.info("Analysis details test complete. Moving to all analyses list...")
    
    # Get all analyses
    logger.info("Retrieving all analyses...")
    response = client.get("/api/analyses")
    if response.status_code != 200:
        logger.error(f"All analyses API error: {response.status_code} - {response.json()}")
        sys.exit(1)
        
    logger.info("All analyses API response successful!")
    analyses_list = response.json()
    logger.info(f"Analyses list keys: {list(analyses_list.keys())}")
    assert "analyses" in analyses_list
    assert len(analyses_list["analyses"]) > 0
    
    # Test passes for all analyses
    logger.info("All analyses test complete. Moving to second analysis creation...")
    
    # Test complete
    logger.info("All tests passed successfully!")
    print("\n✅ API tests PASSED")
    sys.exit(0)

if __name__ == "__main__":
    try:
        test_root_endpoint()
        test_metadata_endpoints()
        test_analysis_workflow()
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        print("\n❌ API tests FAILED")
        sys.exit(1)
