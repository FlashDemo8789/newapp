"""
Color utilities for FlashDNA application.
Provides consistent color access with fallbacks to prevent errors.
"""

import logging

logger = logging.getLogger("color_utils")

# Complete color palette with all needed colors
COMPLETE_COLOR_PALETTE = {
    # Core colors
    "primary": "#3A86FF",     # Blue
    "secondary": "#8338EC",   # Purple
    "success": "#06D6A0",     # Green
    "warning": "#FFD166",     # Yellow
    "danger": "#FF006E",      # Magenta
    "info": "#00B4D8",        # Light blue
    "dark": "#1E293B",        # Dark blue-gray
    "light": "#F8FAFC",       # Light gray
    "background": "#F1F5F9",  # Light gray-blue
    "text": "#334155",        # Dark gray
    "border": "#E2E8F0",      # Light border color
    
    # Text colors
    "text_primary": "#334155",      # Dark text
    "text_secondary": "#64748B",    # Medium gray text
    
    # Primary variations
    "primary_dark": "#0043CE",      # Darker blue
    "primary_light": "#60A5FA",     # Lighter blue
    
    # Secondary variations
    "secondary_dark": "#6D28D9",    # Darker purple
    "secondary_light": "#A78BFA",   # Light purple
    
    # CAMP framework colors
    "camp_capital": "#3A86FF",      # Blue for Capital
    "camp_advantage": "#FF006E",    # Magenta for Advantage
    "camp_market": "#FFD166",       # Yellow for Market
    "camp_people": "#06D6A0",       # Green for People
}

def get_color(color_name, default="#333333"):
    """
    Get a color from the palette with a fallback to prevent errors.
    
    Args:
        color_name: Name of the color in the palette
        default: Default color to use if not found
        
    Returns:
        The color value as a string
    """
    if color_name in COMPLETE_COLOR_PALETTE:
        return COMPLETE_COLOR_PALETTE[color_name]
    
    logger.warning(f"Color '{color_name}' not found in palette, using default color")
    return default

def get_color_palette(names=None):
    """
    Get a list of colors from the palette for charts.
    
    Args:
        names: List of color names to include, or None for default palette
        
    Returns:
        List of color values
    """
    if names is None:
        # Default color sequence for charts
        names = ["primary", "secondary", "success", "warning", "danger", 
                "info", "primary_light", "secondary_light"]
    
    return [get_color(name) for name in names]

def patch_color_palette():
    """
    Patch all COLOR_PALETTE dictionaries in loaded modules with our complete palette.
    """
    import sys
    
    # Modules that might have a COLOR_PALETTE
    target_modules = [
        "analysis_flow", 
        "visualization", 
        "report_generator",
        "chart_utils"
    ]
    
    for module_name in target_modules:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "COLOR_PALETTE"):
                # Update the module's color palette with our complete one
                module.COLOR_PALETTE.update(COMPLETE_COLOR_PALETTE)
                logger.info(f"Patched COLOR_PALETTE in {module_name}")

# Patch color palettes when this module is imported
patch_color_palette() 