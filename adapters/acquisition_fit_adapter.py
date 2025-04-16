"""
Adapter module for acquisition_fit.py
This provides a standardized interface to the Acquisition Fit analysis module
"""
from backend.adapters.base_adapter import BaseAnalysisAdapter
import json
import os
from datetime import datetime
import sys
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Add root directory to path to ensure imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, parent_dir)

# Try to import module_exports directly
try:
    import module_exports
    HAS_MODULE_EXPORTS = True
    logger.info("Module exports imported successfully in acquisition adapter")
except ImportError:
    logger.warning("Failed to import module_exports directly in acquisition adapter")
    HAS_MODULE_EXPORTS = False
    module_exports = None

class AcquisitionFitAdapter(BaseAnalysisAdapter):
    """Adapter for acquisition_fit analysis module"""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the adapter"""
        super().__init__('acquisition_fit')
        self.main_function = "analyze_fit"
        self.fallback_functions = ["analyze_acquisition", "evaluate_acquisition_potential"]
        self.use_mock = use_mock
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run acquisition fit analysis
        
        Args:
            data: Input data for the analysis
            
        Returns:
            Dict containing acquisition fit analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
            
        # Try to use the module_exports version first
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'analyze_acquisition_fit'):
            try:
                logger.info("Using module_exports.analyze_acquisition_fit")
                return module_exports.analyze_acquisition_fit(data)
            except Exception as e:
                logger.error(f"Error using module_exports.analyze_acquisition_fit: {e}")
                # Continue to fallback methods
        else:
            logger.warning("module_exports.analyze_acquisition_fit not available, falling back to direct module import")
        
        # Use the base adapter implementation if module_exports failed
        return super().run_analysis(data)
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for testing"""
        # First try to use module_exports get_mock_data if available
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'get_mock_data'):
            try:
                logger.info("Using module_exports.get_mock_data for acquisition_fit")
                return module_exports.get_mock_data("acquisition_fit")
            except Exception as e:
                logger.error(f"Error using module_exports.get_mock_data: {e}")
                # Fall through to local implementation
        
        # Use local implementation as fallback
        try:
            mock_file = os.path.join(os.path.dirname(__file__), '../data/mock/acquisition_fit_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data: {e}")
            
        # Fallback basic mock data
        return {
            "overallScore": 76,
            "categories": [
                {
                    "name": "Strategic Fit",
                    "score": 82,
                    "description": "Alignment with acquirer's strategic goals",
                    "factors": [
                        {
                            "name": "Market Expansion",
                            "score": 85,
                            "description": "Potential to expand into new markets"
                        },
                        {
                            "name": "Product Portfolio",
                            "score": 78,
                            "description": "Complementary nature of products"
                        }
                    ]
                },
                {
                    "name": "Cultural Compatibility",
                    "score": 71,
                    "description": "Alignment of company cultures and values",
                    "factors": [
                        {
                            "name": "Management Style",
                            "score": 68,
                            "description": "Similarity in leadership approaches"
                        },
                        {
                            "name": "Work Environment",
                            "score": 74,
                            "description": "Compatibility of work environments"
                        }
                    ]
                },
                {
                    "name": "Technology Integration",
                    "score": 79,
                    "description": "Ease of integrating technology stacks",
                    "factors": [
                        {
                            "name": "Tech Stack Compatibility",
                            "score": 82,
                            "description": "Compatibility of technology platforms"
                        },
                        {
                            "name": "Data Integration",
                            "score": 76,
                            "description": "Ease of integrating data systems"
                        }
                    ]
                }
            ],
            "potentialAcquirers": [
                {
                    "name": "TechGiant Inc.",
                    "matchScore": 85,
                    "reasonsForAcquisition": [
                        "Fill product gap in their portfolio",
                        "Access to target market segment",
                        "Talent acquisition"
                    ],
                    "estimatedValueRange": {
                        "min": 80000000,
                        "max": 120000000,
                        "currency": "USD"
                    }
                },
                {
                    "name": "InnovateNow Corp",
                    "matchScore": 76,
                    "reasonsForAcquisition": [
                        "Expand market share",
                        "Acquire proprietary technology",
                        "Eliminate competition"
                    ],
                    "estimatedValueRange": {
                        "min": 65000000,
                        "max": 90000000,
                        "currency": "USD"
                    }
                }
            ],
            "integrationConsiderations": [
                {
                    "aspect": "Customer Transition",
                    "riskLevel": "Medium",
                    "mitigation": "Develop comprehensive customer communication plan"
                },
                {
                    "aspect": "Employee Retention",
                    "riskLevel": "High",
                    "mitigation": "Implement retention bonuses and clear career paths"
                },
                {
                    "aspect": "Brand Integration",
                    "riskLevel": "Low",
                    "mitigation": "Gradual transition to parent brand over 12 months"
                }
            ],
            "timestamp": datetime.now().isoformat()
        }

# Create a singleton instance
acquisition_fit_adapter = AcquisitionFitAdapter()

# Export the main function
def analyze_acquisition_fit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze acquisition fit potential
    
    Parameters:
    data (dict): Input data about the company
    
    Returns:
    dict: Formatted acquisition analysis results matching the React frontend structure
    """
    return acquisition_fit_adapter.run_analysis(data)
