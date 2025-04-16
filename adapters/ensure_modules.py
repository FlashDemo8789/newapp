"""
Module Availability Checker

This script attempts to import and verify all required Python modules
to ensure they are available for the backend adapters.
"""

import os
import sys
import importlib
import logging
from typing import List, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("module_checker")

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, parent_dir)

# Add backend directory to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, backend_dir)

# Import path utilities
from backend.utils.path_utils import ensure_path_setup

# Set up all paths
ensure_path_setup()

# Modules to check with their main classes or functions
MODULES_TO_CHECK = [
    ("monte_carlo", ["EnterpriseMonteCarloSimulator", "SimulationResult"]),
    ("acquisition_fit", ["AcquisitionFitAnalyzer", "AcquisitionFitResult"]),
    ("cohort_analysis", ["CohortAnalyzer", "CohortMetrics"]),
    ("comparative_exit_path", ["ExitPathAnalyzer", "ExitPathAnalysis"]),
    ("product_market_fit", ["ProductMarketFitAnalyzer", "PMFMetrics", "PMFStage"]),
    ("technical_due_diligence", ["EnterpriseGradeTechnicalDueDiligence", "TechnicalAssessment"]),
    ("competitive_intelligence", ["CompetitiveIntelligence"]),
    ("team_moat", ["TeamMoatAnalyzer"]),
    ("pattern_detector", ["detect_patterns"]),
]

def check_module(module_name: str, components: List[str]) -> Tuple[bool, str]:
    """
    Check if a module and its components can be imported
    
    Args:
        module_name (str): Name of the module to check
        components (List[str]): List of components (classes, functions) to check in the module
        
    Returns:
        Tuple[bool, str]: Success status and message
    """
    try:
        # Try different import paths
        module = None
        
        # List of paths to try
        paths_to_try = [
            module_name,  # Direct import
            f"analysis.{module_name}",  # From analysis package
            f"analysis.core.{module_name}",  # From analysis.core package
            f"analysis.ui.{module_name}",  # From analysis.ui package
            f"backend.app.services.analysis.{module_name}",  # From backend services
        ]
        
        # Try each path
        for path in paths_to_try:
            try:
                module = importlib.import_module(path)
                logger.info(f"Successfully imported {module_name} as {path}")
                break
            except ImportError:
                continue
        
        if module is None:
            return False, f"Could not import {module_name} from any path"
        
        # Check for components
        missing_components = []
        for component in components:
            if not hasattr(module, component):
                missing_components.append(component)
        
        if missing_components:
            return False, f"Module {module_name} is missing components: {', '.join(missing_components)}"
        
        return True, f"Module {module_name} and all components are available"
        
    except Exception as e:
        return False, f"Error checking {module_name}: {str(e)}"

def main():
    """Main function to check all modules"""
    logger.info("Starting module availability check")
    
    results = []
    all_passed = True
    
    for module_name, components in MODULES_TO_CHECK:
        success, message = check_module(module_name, components)
        results.append((module_name, success, message))
        if not success:
            all_passed = False
    
    # Print results
    print("\n=== MODULE AVAILABILITY CHECK RESULTS ===\n")
    
    for module_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {module_name}: {message}")
    
    print("\n=== SUMMARY ===")
    if all_passed:
        print("✅ All modules available and ready!")
    else:
        print("❌ Some modules have issues. Please resolve before running the application.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 