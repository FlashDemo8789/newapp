"""
Adapter module for product_market_fit.py
This provides a standardized interface to the Product Market Fit analysis module
"""
from backend.adapters.base_adapter import BaseAnalysisAdapter
import json
import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Add root directory to path to ensure imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, parent_dir)

# Try to import module_exports directly
try:
    import module_exports
    HAS_MODULE_EXPORTS = True
    logger.info("Module exports imported successfully in PMF adapter")
except ImportError:
    logger.warning("Failed to import module_exports directly in PMF adapter")
    HAS_MODULE_EXPORTS = False
    module_exports = None

class PmfAdapter(BaseAnalysisAdapter):
    """Adapter for product_market_fit analysis module"""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the adapter"""
        super().__init__('product_market_fit')
        self.main_function = "analyze_pmf"
        self.fallback_functions = ["evaluate_pmf", "calculate_pmf"]
        self.use_mock = use_mock
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run product market fit analysis
        
        Args:
            data: Input data for the analysis
            
        Returns:
            Dict containing product market fit analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
            
        # Try to use the module_exports version first
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'analyze_pmf'):
            try:
                logger.info("Using module_exports.analyze_pmf")
                return module_exports.analyze_pmf(data)
            except Exception as e:
                logger.error(f"Error using module_exports.analyze_pmf: {e}")
                # Continue to fallback methods
        else:
            logger.warning("module_exports.analyze_pmf not available, falling back to direct module import")
        
        # Use the base adapter implementation if module_exports failed
        return super().run_analysis(data)
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for Product Market Fit analysis"""
        # First try to use module_exports get_mock_data if available
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'get_mock_data'):
            try:
                logger.info("Using module_exports.get_mock_data for product_market_fit")
                return module_exports.get_mock_data("product_market_fit")
            except Exception as e:
                logger.error(f"Error using module_exports.get_mock_data: {e}")
                # Fall through to local implementation
        
        # Try to load mock data from the React frontend
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            frontend_mock_path = os.path.join(root_dir, 'frontend/src/services/analysis/mockAnalysisData.js')
            if os.path.exists(frontend_mock_path):
                # This is a simple extraction approach - in production you'd use a proper parser
                with open(frontend_mock_path, 'r') as f:
                    content = f.read()
                    pmf_section = self._extract_section(content, 'pmfMockData')
                    if pmf_section:
                        return json.loads(pmf_section)
        except Exception as e:
            logger.error(f"Error loading mock data from frontend: {e}")
        
        # Try to load from data/mock directory
        try:
            mock_file = os.path.join(os.path.dirname(__file__), '../data/mock/product_market_fit_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data from file: {e}")
        
        # Fallback basic mock data
        return {
            "overallScore": 72,
            "pmfStage": "Early Validation",
            "marketPull": {
                "score": 68,
                "description": "Degree to which market is actively seeking a solution",
                "factors": [
                    {
                        "name": "Problem Urgency",
                        "score": 74,
                        "description": "How urgent the problem is for customers"
                    },
                    {
                        "name": "Market Demand",
                        "score": 65,
                        "description": "Level of existing demand for solution"
                    },
                    {
                        "name": "Willingness to Pay",
                        "score": 68,
                        "description": "Customer willingness to pay for solution"
                    }
                ]
            },
            "productSolution": {
                "score": 76,
                "description": "How well the product solves the target problem",
                "factors": [
                    {
                        "name": "Solution Effectiveness",
                        "score": 82,
                        "description": "How effectively the product solves the problem"
                    },
                    {
                        "name": "Usability",
                        "score": 78,
                        "description": "Ease of use for target customers"
                    },
                    {
                        "name": "Differentiation",
                        "score": 70,
                        "description": "Uniqueness compared to alternatives"
                    }
                ]
            },
            "userEngagement": {
                "score": 73,
                "description": "Level of user engagement with the product",
                "metrics": [
                    {
                        "name": "Activation Rate",
                        "value": 0.68,
                        "benchmark": 0.62,
                        "description": "Percentage of users who activate key features"
                    },
                    {
                        "name": "Retention Rate",
                        "value": 0.42,
                        "benchmark": 0.38,
                        "description": "Percentage of users who remain active after 30 days"
                    },
                    {
                        "name": "NPS",
                        "value": 35,
                        "benchmark": 28,
                        "description": "Net Promoter Score"
                    }
                ]
            },
            "growthPotential": {
                "score": 70,
                "description": "Potential for rapid market adoption and growth",
                "factors": [
                    {
                        "name": "Market Size",
                        "score": 75,
                        "description": "Size of the addressable market"
                    },
                    {
                        "name": "Acquisition Channels",
                        "score": 65,
                        "description": "Effectiveness of customer acquisition channels"
                    },
                    {
                        "name": "Viral Coefficient",
                        "score": 72,
                        "description": "Potential for organic/viral growth"
                    }
                ]
            },
            "recommendations": [
                {
                    "area": "Product",
                    "recommendation": "Improve onboarding flow to increase activation rate",
                    "priority": "High",
                    "impact": "Medium"
                },
                {
                    "area": "Market",
                    "recommendation": "Refine messaging to emphasize unique value proposition",
                    "priority": "Medium",
                    "impact": "High"
                },
                {
                    "area": "Pricing",
                    "recommendation": "Test higher price points with premium feature bundle",
                    "priority": "Medium",
                    "impact": "Medium"
                }
            ]
        }
    
    def _extract_section(self, content, section_name):
        """Extract a JavaScript object literal from a JavaScript file"""
        try:
            # This is a simplistic approach - in production, use a proper JS parser
            start_marker = f"export const {section_name} = "
            start = content.find(start_marker)
            if start == -1:
                return None
            
            start = content.find('{', start)
            if start == -1:
                return None
            
            # Track nested braces to find the end of the object
            brace_count = 1
            end = start + 1
            while brace_count > 0 and end < len(content):
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            if brace_count != 0:
                return None
            
            # Extract the object literal
            js_object = content[start:end]
            
            # Convert JavaScript to JSON
            # This is a very simplistic approach - in production use a proper converter
            js_object = js_object.replace('true', 'true')
            js_object = js_object.replace('false', 'false')
            js_object = js_object.replace('null', 'null')
            
            # Handle trailing commas (valid in JS, invalid in JSON)
            js_object = js_object.replace(',}', '}')
            js_object = js_object.replace(',]', ']')
            
            return js_object
        except Exception as e:
            logger.error(f"Error extracting section: {e}")
            return None

# Create a singleton instance
pmf_adapter = PmfAdapter()

# Export the main function
def analyze_pmf(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze product-market fit
    
    Parameters:
    data (dict): Input data about the product and market
    
    Returns:
    dict: Formatted PMF analysis results matching the React frontend structure
    """
    return pmf_adapter.run_analysis(data)
