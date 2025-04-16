"""
Test script to verify MongoDB connections and repository operations.
This is a simple verification script to ensure our database connection works properly.
"""
import sys
import os
import asyncio
import logging
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import repository
from app.services.repository import CAMPRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mongo")

async def test_mongo_connection():
    """Test basic MongoDB connection and operations"""
    try:
        # Initialize repository
        logger.info("Initializing CAMPRepository...")
        repo = CAMPRepository()
        
        # Test connection
        if not repo.client:
            logger.error("MongoDB connection failed!")
            return False
        
        logger.info("MongoDB connection successful!")
        
        # Create test document for analyses collection
        test_analysis = {
            "company_info": {
                "name": f"Test Startup {datetime.now().strftime('%Y%m%d%H%M%S')}",
                "industry": "Technology",
                "stage": "Seed"
            },
            "overall_score": 0.75,
            "capital_score": 0.8,
            "advantage_score": 0.7,
            "market_score": 0.75,
            "people_score": 0.72,
            "success_probability": 0.65,
            "recommendations": ["Focus on product market fit", "Increase marketing spend"],
            "top_strengths": ["Strong founding team", "Innovative technology"],
            "improvement_areas": ["Customer acquisition", "Operational efficiency"]
        }
        
        test_startup_data = {
            "name": test_analysis["company_info"]["name"],
            "industry": test_analysis["company_info"]["industry"],
            "stage": test_analysis["company_info"]["stage"]
        }
        
        # Save test analysis
        logger.info("Saving test analysis...")
        analysis_id = await repo.save_camp_analysis(test_analysis, test_startup_data)
        
        if not analysis_id:
            logger.error("Failed to save test analysis")
            return False
            
        logger.info(f"Test analysis saved with ID: {analysis_id}")
        
        # Retrieve the analysis
        logger.info("Retrieving test analysis...")
        retrieved_analysis = await repo.get_analysis_by_id(analysis_id)
        
        if not retrieved_analysis:
            logger.error("Failed to retrieve test analysis")
            return False
            
        logger.info(f"Retrieved analysis for: {retrieved_analysis.get('company_info', {}).get('name')}")
        
        # Clean up - delete the test analysis
        logger.info("Cleaning up test data...")
        delete_result = await repo.delete_analysis(analysis_id)
        
        if not delete_result:
            logger.warning("Failed to delete test analysis")
        else:
            logger.info("Test analysis deleted successfully")
        
        # Close connection
        repo.close()
        logger.info("MongoDB connection closed")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during MongoDB test: {str(e)}")
        return False

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(test_mongo_connection())
    
    if success:
        print("\n✅ MongoDB connection test PASSED")
        sys.exit(0)
    else:
        print("\n❌ MongoDB connection test FAILED")
        sys.exit(1)
