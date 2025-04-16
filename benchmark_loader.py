"""
Loader module to ensure benchmark tab implementation is properly initialized
This file should be imported early in the application startup process
"""
import logging
import importlib.util
import sys

# Configure logger
logger = logging.getLogger("benchmark_loader")

def load_benchmark_tab():
    """
    Ensures the benchmark tab implementation is loaded and properly patched
    into the main application flow. This is designed to be imported and run
    early in the application startup process.
    """
    try:
        # First import our implementation module
        if 'benchmark_tab_implementation' not in sys.modules:
            logger.info("Loading benchmark tab implementation module")
            import benchmark_tab_implementation
        
        # Apply the patch to analysis_flow.py
        logger.info("Applying benchmark tab patch")
        import benchmark_tab_patch
        benchmark_tab_patch.apply_patch()
        
        logger.info("Benchmark tab successfully loaded and patched")
        return True
    except Exception as e:
        logger.error(f"Error loading benchmark tab: {str(e)}")
        return False

# Load when imported
load_successful = load_benchmark_tab()
