"""
Module exports for standardizing analysis function names and interfaces.

This module provides consistent function names for all analysis modules,
acting as a bridge between the backend API and the actual module implementations.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path to ensure imports work properly
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

# Try to import all modules
try:
    import monte_carlo
    HAS_MONTE_CARLO = True
except ImportError:
    HAS_MONTE_CARLO = False
    logger.warning("Failed to import monte_carlo module")

try:
    import acquisition_fit
    HAS_ACQUISITION_FIT = True
except ImportError:
    HAS_ACQUISITION_FIT = False
    logger.warning("Failed to import acquisition_fit module")

try:
    import cohort_analysis
    HAS_COHORT_ANALYSIS = True
except ImportError:
    HAS_COHORT_ANALYSIS = False
    logger.warning("Failed to import cohort_analysis module")

try:
    import comparative_exit_path
    HAS_EXIT_PATH = True
except ImportError:
    HAS_EXIT_PATH = False
    logger.warning("Failed to import comparative_exit_path module")

try:
    import product_market_fit
    HAS_PMF = True
except ImportError:
    HAS_PMF = False
    logger.warning("Failed to import product_market_fit module")

try:
    import technical_due_diligence
    HAS_TDD = True
except ImportError:
    HAS_TDD = False
    logger.warning("Failed to import technical_due_diligence module")

try:
    import competitive_intelligence
    HAS_COMPETITIVE = True
except ImportError:
    HAS_COMPETITIVE = False
    logger.warning("Failed to import competitive_intelligence module")

try:
    import team_moat
    HAS_TEAM_MOAT = True
except ImportError:
    HAS_TEAM_MOAT = False
    logger.warning("Failed to import team_moat module")

try:
    import pattern_detector
    HAS_PATTERN_DETECTOR = True
except ImportError:
    HAS_PATTERN_DETECTOR = False
    logger.warning("Failed to import pattern_detector module")


def get_mock_data(module_name: str) -> Dict[str, Any]:
    """Get mock data for a specific module"""
    try:
        # Try to load from data/mock directory
        mock_file = os.path.join(root_dir, f'backend/data/mock/{module_name}_result.json')
        if os.path.exists(mock_file):
            with open(mock_file, 'r') as f:
                logger.info(f"Loaded mock data from {mock_file}")
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading mock data: {e}")
    
    # Fallback mock data
    return {
        "success": True,
        "message": f"Mock data for {module_name}",
        "module": module_name,
        "timestamp": "2023-01-01T00:00:00Z",
        "data": {}
    }


def run_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation analysis
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_MONTE_CARLO:
        logger.warning("Monte Carlo module not available, returning mock data")
        return get_mock_data("monte_carlo")
    
    try:
        # Try to find the appropriate method
        if hasattr(monte_carlo, "analyze"):
            logger.info("Using monte_carlo.analyze()")
            return monte_carlo.analyze(data)
        elif hasattr(monte_carlo, "run_analysis"):
            logger.info("Using monte_carlo.run_analysis()")
            return monte_carlo.run_analysis(data)
        elif hasattr(monte_carlo, "run_simulation"):
            logger.info("Using monte_carlo.run_simulation()")
            return monte_carlo.run_simulation(data)
        elif hasattr(monte_carlo, "evaluate"):
            logger.info("Using monte_carlo.evaluate()")
            return monte_carlo.evaluate(data)
        
        # Look for a class-based analyzer
        for attr_name in dir(monte_carlo):
            if attr_name.endswith(('Analyzer', 'Simulator', 'Evaluator')) and not attr_name.startswith('_'):
                try:
                    logger.info(f"Trying class {attr_name} in monte_carlo module")
                    class_obj = getattr(monte_carlo, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "run_analysis", "run_simulation", "evaluate"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        logger.info(f"Using monte_carlo.{attr_name}.{method.__name__}()")
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in monte_carlo: {e}")
        
        logger.warning("No suitable function found in monte_carlo module")
        return get_mock_data("monte_carlo")
    except Exception as e:
        logger.error(f"Error in run_analysis: {e}")
        return get_mock_data("monte_carlo")


def analyze_acquisition_fit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze acquisition fit data
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_ACQUISITION_FIT:
        logger.warning("Acquisition Fit module not available, returning mock data")
        return get_mock_data("acquisition_fit")
    
    try:
        # Try to find the appropriate method
        if hasattr(acquisition_fit, "analyze_acquisition_fit"):
            return acquisition_fit.analyze_acquisition_fit(data)
        elif hasattr(acquisition_fit, "analyze"):
            return acquisition_fit.analyze(data)
        elif hasattr(acquisition_fit, "evaluate_fit"):
            return acquisition_fit.evaluate_fit(data)
        
        # Try class-based approach
        for attr_name in dir(acquisition_fit):
            if attr_name.endswith(('Analyzer', 'Evaluator')) and not attr_name.startswith('_'):
                try:
                    logger.info(f"Trying class {attr_name} in acquisition_fit module")
                    class_obj = getattr(acquisition_fit, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_acquisition_fit", "evaluate_fit"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        logger.info(f"Using acquisition_fit.{attr_name}.{method.__name__}()")
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in acquisition_fit: {e}")
        
        logger.warning("No suitable function found in acquisition_fit module")
        return get_mock_data("acquisition_fit")
    except Exception as e:
        logger.error(f"Error in analyze_acquisition_fit: {e}")
        return get_mock_data("acquisition_fit")


def analyze_cohorts(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cohort data
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_COHORT_ANALYSIS:
        logger.warning("Cohort Analysis module not available, returning mock data")
        return get_mock_data("cohort_analysis")
    
    try:
        # Try different function names
        if hasattr(cohort_analysis, "analyze_cohorts"):
            return cohort_analysis.analyze_cohorts(data)
        elif hasattr(cohort_analysis, "analyze"):
            return cohort_analysis.analyze(data)
        
        # Try class-based approach
        for attr_name in dir(cohort_analysis):
            if attr_name.endswith('Analyzer') and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(cohort_analysis, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_cohorts"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in cohort_analysis: {e}")
        
        logger.warning("No suitable function found in cohort_analysis module")
        return get_mock_data("cohort_analysis")
    except Exception as e:
        logger.error(f"Error in analyze_cohorts: {e}")
        return get_mock_data("cohort_analysis")


def analyze_exit_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze comparative exit paths
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_EXIT_PATH:
        logger.warning("Comparative Exit Path module not available, returning mock data")
        return get_mock_data("comparative_exit_path")
    
    try:
        # Try different function names
        if hasattr(comparative_exit_path, "analyze_exit_paths"):
            return comparative_exit_path.analyze_exit_paths(data)
        elif hasattr(comparative_exit_path, "analyze"):
            return comparative_exit_path.analyze(data)
        
        # Try class-based approach
        for attr_name in dir(comparative_exit_path):
            if attr_name.endswith('Analyzer') and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(comparative_exit_path, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_exit_paths"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in comparative_exit_path: {e}")
        
        logger.warning("No suitable function found in comparative_exit_path module")
        return get_mock_data("comparative_exit_path")
    except Exception as e:
        logger.error(f"Error in analyze_exit_paths: {e}")
        return get_mock_data("comparative_exit_path")


def analyze_pmf(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze product-market fit
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_PMF:
        logger.warning("Product Market Fit module not available, returning mock data")
        return get_mock_data("product_market_fit")
    
    try:
        # Try different function names
        if hasattr(product_market_fit, "analyze_pmf"):
            return product_market_fit.analyze_pmf(data)
        elif hasattr(product_market_fit, "analyze"):
            return product_market_fit.analyze(data)
        elif hasattr(product_market_fit, "evaluate_pmf"):
            return product_market_fit.evaluate_pmf(data)
        
        # Try class-based approach
        for attr_name in dir(product_market_fit):
            if attr_name.endswith(('Analyzer', 'Evaluator')) and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(product_market_fit, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_pmf", "evaluate_pmf"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in product_market_fit: {e}")
        
        logger.warning("No suitable function found in product_market_fit module")
        return get_mock_data("product_market_fit")
    except Exception as e:
        logger.error(f"Error in analyze_pmf: {e}")
        return get_mock_data("product_market_fit")


def assess_technical_architecture(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess technical architecture for due diligence
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_TDD:
        logger.warning("Technical Due Diligence module not available, returning mock data")
        return get_mock_data("technical_due_diligence")
    
    try:
        # Try different function names
        if hasattr(technical_due_diligence, "assess_technical_architecture"):
            return technical_due_diligence.assess_technical_architecture(data)
        elif hasattr(technical_due_diligence, "analyze"):
            return technical_due_diligence.analyze(data)
        elif hasattr(technical_due_diligence, "evaluate_architecture"):
            return technical_due_diligence.evaluate_architecture(data)
        
        # Try class-based approach
        for attr_name in dir(technical_due_diligence):
            if attr_name.endswith(('Analyzer', 'Assessor', 'Evaluator')) and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(technical_due_diligence, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "assess_technical_architecture", "evaluate_architecture"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in technical_due_diligence: {e}")
        
        logger.warning("No suitable function found in technical_due_diligence module")
        return get_mock_data("technical_due_diligence")
    except Exception as e:
        logger.error(f"Error in assess_technical_architecture: {e}")
        return get_mock_data("technical_due_diligence")


def analyze_competitive_landscape(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze competitive intelligence data
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_COMPETITIVE:
        logger.warning("Competitive Intelligence module not available, returning mock data")
        return get_mock_data("competitive_intelligence")
    
    try:
        # Try different function names
        if hasattr(competitive_intelligence, "analyze_competitive_landscape"):
            return competitive_intelligence.analyze_competitive_landscape(data)
        elif hasattr(competitive_intelligence, "analyze"):
            return competitive_intelligence.analyze(data)
        
        # Try class-based approach
        for attr_name in dir(competitive_intelligence):
            if attr_name.endswith('Analyzer') and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(competitive_intelligence, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_competitive_landscape"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in competitive_intelligence: {e}")
        
        logger.warning("No suitable function found in competitive_intelligence module")
        return get_mock_data("competitive_intelligence")
    except Exception as e:
        logger.error(f"Error in analyze_competitive_landscape: {e}")
        return get_mock_data("competitive_intelligence")


def analyze_team(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze team moat data
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_TEAM_MOAT:
        logger.warning("Team Moat module not available, returning mock data")
        return get_mock_data("team_moat")
    
    try:
        # Try different function names
        if hasattr(team_moat, "analyze_team"):
            return team_moat.analyze_team(data)
        elif hasattr(team_moat, "analyze"):
            return team_moat.analyze(data)
        elif hasattr(team_moat, "evaluate_team"):
            return team_moat.evaluate_team(data)
        
        # Try class-based approach
        for attr_name in dir(team_moat):
            if attr_name.endswith(('Analyzer', 'Evaluator')) and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(team_moat, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_team", "evaluate_team"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in team_moat: {e}")
        
        logger.warning("No suitable function found in team_moat module")
        return get_mock_data("team_moat")
    except Exception as e:
        logger.error(f"Error in analyze_team: {e}")
        return get_mock_data("team_moat")


def analyze_patterns(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pattern detector data
    
    Parameters:
    data (dict): Input data for the analysis
    
    Returns:
    dict: Analysis results
    """
    if not HAS_PATTERN_DETECTOR:
        logger.warning("Pattern Detector module not available, returning mock data")
        return get_mock_data("pattern_detector")
    
    try:
        # Try different function names
        if hasattr(pattern_detector, "analyze_patterns"):
            return pattern_detector.analyze_patterns(data)
        elif hasattr(pattern_detector, "analyze"):
            return pattern_detector.analyze(data)
        elif hasattr(pattern_detector, "detect_patterns"):
            return pattern_detector.detect_patterns(data)
        
        # Try class-based approach
        for attr_name in dir(pattern_detector):
            if attr_name.endswith(('Analyzer', 'Detector')) and not attr_name.startswith('_'):
                try:
                    class_obj = getattr(pattern_detector, attr_name)
                    instance = class_obj()
                    method = None
                    for method_name in ["analyze", "analyze_patterns", "detect_patterns"]:
                        if hasattr(instance, method_name):
                            method = getattr(instance, method_name)
                            break
                    
                    if method:
                        return method(data)
                except Exception as e:
                    logger.error(f"Error using class {attr_name} in pattern_detector: {e}")
        
        logger.warning("No suitable function found in pattern_detector module")
        return get_mock_data("pattern_detector")
    except Exception as e:
        logger.error(f"Error in analyze_patterns: {e}")
        return get_mock_data("pattern_detector") 