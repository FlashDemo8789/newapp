"""
Adapter module for competitive_intelligence.py
This provides a standardized interface to the Competitive Intelligence analysis module
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
    logger.info("Module exports imported successfully in competitive intelligence adapter")
except ImportError:
    logger.warning("Failed to import module_exports directly in competitive intelligence adapter")
    HAS_MODULE_EXPORTS = False
    module_exports = None

class CompetitiveIntelligenceAdapter(BaseAnalysisAdapter):
    """Adapter for competitive_intelligence analysis module"""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the adapter"""
        super().__init__('competitive_intelligence')
        self.main_function = "analyze_competition"
        self.fallback_functions = ["competitive_analysis", "evaluate_competitors"]
        self.use_mock = use_mock
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run competitive intelligence analysis
        
        Args:
            data: Input data for the analysis
            
        Returns:
            Dict containing competitive intelligence analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
            
        # Try to use the module_exports version first
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'analyze_competitive_landscape'):
            try:
                logger.info("Using module_exports.analyze_competitive_landscape")
                return module_exports.analyze_competitive_landscape(data)
            except Exception as e:
                logger.error(f"Error using module_exports.analyze_competitive_landscape: {e}")
                # Continue to fallback methods
        else:
            logger.warning("module_exports.analyze_competitive_landscape not available, falling back to direct module import")
        
        # Use the base adapter implementation if module_exports failed
        return super().run_analysis(data)
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for Competitive Intelligence analysis"""
        # First try to use module_exports get_mock_data if available
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'get_mock_data'):
            try:
                logger.info("Using module_exports.get_mock_data for competitive_intelligence")
                return module_exports.get_mock_data("competitive_intelligence")
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
                    competitive_section = self._extract_section(content, 'competitiveMockData')
                    if competitive_section:
                        return json.loads(competitive_section)
        except Exception as e:
            logger.error(f"Error loading mock data from frontend: {e}")
        
        # Try to load from data/mock directory
        try:
            mock_file = os.path.join(os.path.dirname(__file__), '../data/mock/competitive_intelligence_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data from file: {e}")
        
        # Fallback basic mock data
        return {
            "competitivePosition": {
                "overall": {
                    "score": 72,
                    "ranking": 3,
                    "description": "Strong position with specific advantages in technology and UX"
                },
                "marketShare": {
                    "value": 0.14,
                    "trend": "increasing",
                    "description": "Growing market share, up from 11% last year"
                }
            },
            "competitors": [
                {
                    "name": "MarketLeader Inc.",
                    "overallScore": 85,
                    "marketShare": 0.32,
                    "founded": 2012,
                    "funding": "$75M",
                    "strengths": [
                        "Established brand recognition",
                        "Comprehensive feature set",
                        "Strong enterprise relationships"
                    ],
                    "weaknesses": [
                        "Aging technology stack",
                        "Higher pricing",
                        "Slower innovation cycle"
                    ],
                    "recentMoves": [
                        {
                            "action": "Acquired DataViz Inc.",
                            "date": "2024-02-15",
                            "impact": "Strengthened analytics capabilities"
                        },
                        {
                            "action": "Expanded to European market",
                            "date": "2024-01-10",
                            "impact": "15% increase in total addressable market"
                        }
                    ]
                },
                {
                    "name": "InnovateTech",
                    "overallScore": 78,
                    "marketShare": 0.22,
                    "founded": 2015,
                    "funding": "$42M",
                    "strengths": [
                        "Cutting-edge technology",
                        "Rapid innovation",
                        "Strong developer community"
                    ],
                    "weaknesses": [
                        "Limited enterprise features",
                        "Smaller sales team",
                        "Less established brand"
                    ],
                    "recentMoves": [
                        {
                            "action": "Launched AI-powered features",
                            "date": "2024-03-01",
                            "impact": "Significant positive press coverage"
                        }
                    ]
                }
            ],
            "marketDynamics": {
                "growthRate": 0.18,
                "competitiveIntensity": "High",
                "barriers": [
                    {
                        "type": "Technology",
                        "strength": "Medium",
                        "description": "Moderate technical complexity to enter"
                    },
                    {
                        "type": "Brand Recognition",
                        "strength": "High",
                        "description": "Significant investment required to build brand"
                    },
                    {
                        "type": "User Lock-in",
                        "strength": "Medium",
                        "description": "Some switching costs for established users"
                    }
                ],
                "trends": [
                    {
                        "name": "AI Integration",
                        "impact": "High",
                        "timeline": "0-12 months",
                        "description": "Rapid adoption of AI capabilities across competitors"
                    },
                    {
                        "name": "Vertical Specialization",
                        "impact": "Medium",
                        "timeline": "12-24 months",
                        "description": "Increasing focus on industry-specific solutions"
                    }
                ]
            },
            "swotAnalysis": {
                "strengths": [
                    {
                        "factor": "User Experience",
                        "description": "Superior ease of use compared to competitors",
                        "impact": "High"
                    },
                    {
                        "factor": "API Flexibility",
                        "description": "More extensive API capabilities",
                        "impact": "Medium"
                    }
                ],
                "weaknesses": [
                    {
                        "factor": "Limited Marketing",
                        "description": "Smaller marketing budget and reach",
                        "impact": "High"
                    },
                    {
                        "factor": "Geographic Coverage",
                        "description": "Limited presence outside North America",
                        "impact": "Medium"
                    }
                ],
                "opportunities": [
                    {
                        "factor": "International Expansion",
                        "description": "Growing demand in EMEA and APAC regions",
                        "impact": "High"
                    },
                    {
                        "factor": "Enterprise Partnerships",
                        "description": "Potential for strategic enterprise alliances",
                        "impact": "Medium"
                    }
                ],
                "threats": [
                    {
                        "factor": "New Market Entrants",
                        "description": "Well-funded startups entering the space",
                        "impact": "Medium"
                    },
                    {
                        "factor": "Pricing Pressure",
                        "description": "Increasing competitive pressure on pricing",
                        "impact": "High"
                    }
                ]
            },
            "recommendations": [
                {
                    "focus": "Product Differentiation",
                    "action": "Accelerate AI feature roadmap",
                    "priority": "High",
                    "expected_impact": "Significant competitive advantage in 6-12 months"
                },
                {
                    "focus": "Market Expansion",
                    "action": "Establish EMEA presence through partnership",
                    "priority": "Medium",
                    "expected_impact": "15-20% revenue growth within 18 months"
                },
                {
                    "focus": "Marketing Strategy",
                    "action": "Increase content marketing for organic growth",
                    "priority": "Medium",
                    "expected_impact": "30% increase in inbound leads within 12 months"
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
competitive_intelligence_adapter = CompetitiveIntelligenceAdapter()

# Export the main function
def analyze_competition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze competitive landscape and market positioning
    
    Parameters:
    data (dict): Input data about the company and market
    
    Returns:
    dict: Formatted competitive analysis results matching the React frontend structure
    """
    return competitive_intelligence_adapter.run_analysis(data)
