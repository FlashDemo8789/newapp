"""
Base adapter class for standardizing module interfaces
"""
import importlib
import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from backend.utils.path_utils import import_analysis_module

# Configure logging
logger = logging.getLogger(__name__)

# Add root directory to path to ensure imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, parent_dir)

# Try to import module_exports directly
try:
    import module_exports
    HAS_MODULE_EXPORTS = True
    logger.info("Module exports imported successfully in base adapter")
except ImportError:
    HAS_MODULE_EXPORTS = False
    logger.warning("Failed to import module_exports directly in base adapter")

class BaseAnalysisAdapter:
    """Base adapter class for analysis modules"""
    
    def __init__(self, module_name):
        """
        Initialize the base adapter
        
        Parameters:
        module_name (str): Name of the module this adapter is for
        """
        self.module_name = module_name
        self.use_mock = False
        
        # Import the target module
        self.module = import_analysis_module(module_name)
        
        # Store default function names for later use
        self.main_function = "analyze"
        self.fallback_functions = ["run_analysis", "evaluate", "simulate"]
        
        # Store export function name
        self.export_function = self._get_export_function_name(module_name)
        
    def _get_export_function_name(self, module_name):
        """Map module name to expected export function name"""
        function_map = {
            "monte_carlo": "run_analysis",
            "acquisition_fit": "analyze_acquisition_fit",
            "cohort_analysis": "analyze_cohorts",
            "comparative_exit_path": "analyze_exit_paths",
            "product_market_fit": "analyze_pmf",
            "technical_due_diligence": "assess_technical_architecture",
            "competitive_intelligence": "analyze_competitive_landscape",
            "team_moat": "analyze_team",
            "pattern_detector": "analyze_patterns"
        }
        
        # Strip potential path prefixes
        short_name = module_name.split('.')[-1]
        return function_map.get(short_name)
        
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analysis using the configured module
        
        Parameters:
        data (dict): Input data for the analysis
        
        Returns:
        dict: Analysis results
        """
        if self.use_mock:
            logger.info(f"Using mock data for {self.module_name}")
            return self.get_mock_data()
        
        # Try module_exports if available
        if HAS_MODULE_EXPORTS and self.export_function and hasattr(module_exports, self.export_function):
            try:
                logger.info(f"Using module_exports.{self.export_function} for {self.module_name}")
                return getattr(module_exports, self.export_function)(data)
            except Exception as e:
                logger.error(f"Error using module_exports.{self.export_function}: {e}")
                # Continue to fallback methods
            
        # If module_exports not available or failed, try direct module import
        if not self.module:
            logger.warning(f"Module {self.module_name} not loaded properly")
            return self.get_mock_data()
            
        # Try to call the main function
        if hasattr(self.module, self.main_function):
            try:
                logger.info(f"Calling {self.main_function} on {self.module_name}")
                return getattr(self.module, self.main_function)(data)
            except Exception as e:
                logger.error(f"Error calling {self.main_function} on {self.module_name}: {e}")
        
        # Try fallback functions
        for func_name in self.fallback_functions:
            if hasattr(self.module, func_name):
                try:
                    logger.info(f"Calling {func_name} on {self.module_name}")
                    return getattr(self.module, func_name)(data)
                except Exception as e:
                    logger.error(f"Error calling {func_name} on {self.module_name}: {e}")
                    continue
        
        # If we get here, all methods failed
        logger.warning(f"No suitable function found in {self.module_name}")
        return self.get_mock_data()
    
    def get_mock_data(self) -> Dict[str, Any]:
        """Get mock data for testing"""
        try:
            # Try to load from data/mock directory
            mock_file = os.path.join(os.path.dirname(__file__), f'../data/mock/{self.module_name}_result.json')
            if os.path.exists(mock_file):
                with open(mock_file, 'r') as f:
                    logger.info(f"Loaded mock data from {mock_file}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mock data: {e}")
        
        # Fallback mock data
        return {
            "success": True,
            "message": "Mock data for " + self.module_name,
            "module": self.module_name,
            "timestamp": "2023-01-01T00:00:00Z",
            "data": {}
        }
