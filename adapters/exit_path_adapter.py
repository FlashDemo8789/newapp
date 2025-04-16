"""
Adapter module for exit_path_tab.py
This provides a standardized interface to the Exit Path analysis module
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
    logger.info("Module exports imported successfully in exit path adapter")
except ImportError:
    logger.warning("Failed to import module_exports directly in exit path adapter")
    HAS_MODULE_EXPORTS = False
    module_exports = None

class ExitPathAdapter(BaseAnalysisAdapter):
    """Adapter for the Exit Path Analysis module"""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the Exit Path Analysis adapter"""
        super().__init__(
            module_name="comparative_exit_path"
        )
        # Set additional properties after init
        self.main_function = "analyze_exit_paths"
        self.fallback_functions = [
            "run_exit_path_analysis", 
            "generate_exit_scenarios", 
            "compute_exit_metrics"
        ]
        self.use_mock = use_mock
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run exit path analysis
        
        Args:
            data: Input data for the analysis
            
        Returns:
            Dict containing exit path analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
            
        # Try to use the module_exports version first
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'analyze_exit_paths'):
            try:
                logger.info("Using module_exports.analyze_exit_paths")
                return module_exports.analyze_exit_paths(data)
            except Exception as e:
                logger.error(f"Error using module_exports.analyze_exit_paths: {e}")
                # Continue to fallback methods
        else:
            logger.warning("module_exports.analyze_exit_paths not available, falling back to direct module import")
        
        # Use the base adapter implementation if module_exports failed
        return super().run_analysis(data)
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for Exit Path analysis"""
        # First try to use module_exports get_mock_data if available
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'get_mock_data'):
            try:
                logger.info("Using module_exports.get_mock_data for comparative_exit_path")
                return module_exports.get_mock_data("comparative_exit_path")
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
                    exit_path_section = self._extract_section(content, 'exitPathMockData')
                    if exit_path_section:
                        return json.loads(exit_path_section)
        except Exception as e:
            logger.error(f"Error loading mock data from frontend: {e}")
        
        # Try to load from data/mock directory
        try:
            mock_file = os.path.join(os.path.dirname(__file__), '../data/mock/comparative_exit_path_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data from file: {e}")
        
        # Fallback basic mock data
        return {
            "summary": {
                "optimalTimeframe": "3-5 years",
                "recommendedPath": "Strategic Acquisition",
                "estimatedValuation": {
                    "min": 75000000,
                    "max": 120000000,
                    "currency": "USD"
                },
                "keyFactors": [
                    "Strong IP portfolio",
                    "Established market position",
                    "Industry consolidation trend",
                    "Complementary fit with potential acquirers"
                ]
            },
            "exitOptions": [
                {
                    "type": "Strategic Acquisition",
                    "probability": 0.65,
                    "timeframe": "3-5 years",
                    "potentialValue": {
                        "min": 75000000,
                        "max": 120000000,
                        "currency": "USD"
                    },
                    "potentialAcquirers": [
                        {
                            "name": "TechGiant Inc.",
                            "match": 0.85,
                            "strategic_rationale": "Fill product gap in enterprise portfolio",
                            "estimated_offer": {
                                "min": 90000000,
                                "max": 120000000,
                                "currency": "USD"
                            }
                        },
                        {
                            "name": "GrowthCorp",
                            "match": 0.72,
                            "strategic_rationale": "Expand into adjacent market",
                            "estimated_offer": {
                                "min": 75000000,
                                "max": 95000000,
                                "currency": "USD"
                            }
                        }
                    ],
                    "prerequisites": [
                        "Achieve $10M ARR",
                        "Expand enterprise customer base",
                        "Establish international presence"
                    ],
                    "enhancementActions": [
                        {
                            "action": "Strengthen patent portfolio",
                            "impact": "High",
                            "timeline": "12-18 months"
                        },
                        {
                            "action": "Develop strategic partnerships",
                            "impact": "Medium",
                            "timeline": "6-12 months"
                        }
                    ]
                },
                {
                    "type": "IPO",
                    "probability": 0.25,
                    "timeframe": "5-7 years",
                    "potentialValue": {
                        "min": 150000000,
                        "max": 250000000,
                        "currency": "USD"
                    },
                    "marketConditions": {
                        "current": "Moderate",
                        "forecast": "Improving",
                        "relevantIndices": [
                            {
                                "name": "Tech IPO Index",
                                "trend": "Upward",
                                "projected_change": "+12% over 2 years"
                            }
                        ]
                    },
                    "prerequisites": [
                        "Achieve $30M ARR",
                        "Demonstrate consistent profitability",
                        "Build strong executive team",
                        "Establish governance structures"
                    ],
                    "enhancementActions": [
                        {
                            "action": "Implement SOX-compliant processes",
                            "impact": "High",
                            "timeline": "18-24 months"
                        },
                        {
                            "action": "Develop relationships with investment banks",
                            "impact": "Medium",
                            "timeline": "12-18 months"
                        }
                    ]
                },
                {
                    "type": "Private Equity",
                    "probability": 0.45,
                    "timeframe": "2-4 years",
                    "potentialValue": {
                        "min": 60000000,
                        "max": 90000000,
                        "currency": "USD"
                    },
                    "potentialInvestors": [
                        {
                            "name": "Growth Partners",
                            "match": 0.78,
                            "investment_thesis": "Consolidation platform in fragmented market",
                            "estimated_valuation": {
                                "min": 70000000,
                                "max": 90000000,
                                "currency": "USD"
                            }
                        },
                        {
                            "name": "TechVenture Capital",
                            "match": 0.65,
                            "investment_thesis": "Growth acceleration through international expansion",
                            "estimated_valuation": {
                                "min": 60000000,
                                "max": 75000000,
                                "currency": "USD"
                            }
                        }
                    ],
                    "prerequisites": [
                        "Achieve $8M ARR",
                        "Demonstrate clear growth levers",
                        "Streamline operations for efficiency"
                    ],
                    "enhancementActions": [
                        {
                            "action": "Improve unit economics documentation",
                            "impact": "High",
                            "timeline": "6-9 months"
                        },
                        {
                            "action": "Develop detailed growth model",
                            "impact": "Medium",
                            "timeline": "3-6 months"
                        }
                    ]
                }
            ],
            "valuationFactors": [
                {
                    "factor": "Revenue Growth Rate",
                    "current": "45% YoY",
                    "target": "50%+ YoY",
                    "multiplierImpact": "High",
                    "improvement_actions": [
                        "Expand sales team",
                        "Launch new pricing tiers"
                    ]
                },
                {
                    "factor": "Gross Margin",
                    "current": "72%",
                    "target": "80%+",
                    "multiplierImpact": "Medium",
                    "improvement_actions": [
                        "Optimize cloud infrastructure",
                        "Automate customer onboarding"
                    ]
                },
                {
                    "factor": "Customer Retention",
                    "current": "85% annual",
                    "target": "90%+ annual",
                    "multiplierImpact": "High",
                    "improvement_actions": [
                        "Implement customer success program",
                        "Develop deeper product integration capabilities"
                    ]
                }
            ],
            "timestamp": datetime.now().isoformat()
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
exit_path_adapter = ExitPathAdapter()

# Export the main function
def analyze_exit_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze exit path options and recommendations
    
    Parameters:
    data (dict): Input data about the company and market
    
    Returns:
    dict: Formatted exit path analysis results matching the React frontend structure
    """
    return exit_path_adapter.run_analysis(data)
