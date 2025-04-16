"""
Adapter module for team_moat.py
This provides a standardized interface to the Team Moat analysis module
"""
from backend.adapters.base_adapter import BaseAnalysisAdapter
from backend.utils.path_utils import import_analysis_module
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
    logger.info("Module exports imported successfully in team moat adapter")
except ImportError:
    logger.warning("Failed to import module_exports directly in team moat adapter")
    HAS_MODULE_EXPORTS = False
    module_exports = None

class TeamMoatAdapter(BaseAnalysisAdapter):
    """Adapter for team_moat analysis module"""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the adapter"""
        super().__init__('team_moat')
        self.main_function = "analyze_team"
        self.fallback_functions = ["evaluate_team", "analyze_team_moat", "analyze"]
        self.use_mock = use_mock
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run team moat analysis
        
        Args:
            data: Input data for the analysis
            
        Returns:
            Dict containing team moat analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
            
        # Try to use the module_exports version first
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'analyze_team'):
            try:
                logger.info("Using module_exports.analyze_team")
                return module_exports.analyze_team(data)
            except Exception as e:
                logger.error(f"Error using module_exports.analyze_team: {e}")
                # Continue to fallback methods
        else:
            logger.warning("module_exports.analyze_team not available, falling back to direct module import")
        
        # Use the base adapter implementation if module_exports failed
        return super().run_analysis(data)
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for testing"""
        # First try to use module_exports get_mock_data if available
        if HAS_MODULE_EXPORTS and hasattr(module_exports, 'get_mock_data'):
            try:
                logger.info("Using module_exports.get_mock_data for team_moat")
                return module_exports.get_mock_data("team_moat")
            except Exception as e:
                logger.error(f"Error using module_exports.get_mock_data: {e}")
                # Fall through to local implementation
        
        # Use local implementation as fallback
        try:
            mock_file = os.path.join(os.path.dirname(__file__), '../data/mock/team_moat_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data: {e}")
            
        # Fallback basic mock data
        return {
            "profile": {
                "teamSize": 12,
                "coreTeamSize": 5,
                "averageExperience": 8.5,
                "priorStartups": 7,
                "overallScore": 7.8,
                "benchmarks": {
                    "industry": 6.2,
                    "successful": 7.5,
                    "unicorns": 8.4,
                    "competitors": 6.8
                },
                "competencies": [
                    {
                        "name": "Technical Expertise",
                        "score": 8.5,
                        "description": "Deep technical knowledge in core domain",
                        "benchmarks": {
                            "industry": 6.5,
                            "successful": 7.8,
                            "unicorns": 8.7,
                            "competitors": 7.0
                        }
                    },
                    {
                        "name": "Industry Experience",
                        "score": 7.2,
                        "description": "Previous roles in the target industry",
                        "benchmarks": {
                            "industry": 6.8,
                            "successful": 7.2,
                            "unicorns": 7.5,
                            "competitors": 7.5
                        }
                    }
                ],
                "diversity": {
                    "background": 7.5,
                    "education": 6.8,
                    "experience": 8.2,
                    "thinking": 7.6,
                    "demographics": 6.2
                }
            },
            "advantage": {
                "overallScore": 7.5,
                "benchmarks": {
                    "industry": 5.8,
                    "successful": 7.2,
                    "unicorns": 8.6,
                    "competitors": 6.3
                },
                "dimensions": [
                    {
                        "name": "Domain Expertise",
                        "score": 8.3,
                        "description": "Specialized knowledge in core domain",
                        "benchmarks": {
                            "industry": 6.2,
                            "successful": 7.5,
                            "unicorns": 8.7,
                            "competitors": 6.5
                        }
                    }
                ]
            },
            "execution": {
                "metrics": {
                    "efficiency": [
                        {
                            "name": "Time to Market",
                            "value": 8.2,
                            "benchmark": {
                                "industry": 6.5,
                                "successful": 7.8,
                                "unicorns": 8.5,
                                "competitors": 6.8
                            }
                        }
                    ]
                }
            },
            "adaptability": {
                "overallScore": 7.2,
                "benchmarks": {
                    "industry": 5.5,
                    "successful": 7.0,
                    "unicorns": 8.5,
                    "competitors": 6.0
                },
                "dimensions": [
                    {
                        "name": "Learning Agility",
                        "score": 8.0,
                        "description": "Ability to quickly acquire new knowledge",
                        "benchmarks": {
                            "industry": 5.8,
                            "successful": 7.2,
                            "unicorns": 8.5,
                            "competitors": 6.2
                        },
                        "trend": 1
                    }
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

# Create a singleton instance
team_moat_adapter = TeamMoatAdapter()

# Export the main function
def analyze_team(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze team competitive advantage
    
    Parameters:
    data (dict): Input data about the team
    
    Returns:
    dict: Formatted team analysis results matching the React frontend structure
    """
    return team_moat_adapter.run_analysis(data)
