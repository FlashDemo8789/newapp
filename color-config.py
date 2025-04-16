"""
Color configuration for FlashDNA Analysis System.
This module provides a direct replacement for the COLOR_PALETTE dictionary
with all the necessary color keys to prevent 'text_primary' and similar errors.
"""

import logging
import importlib
import sys

logger = logging.getLogger("color_config")

# Enhanced full color palette with all required keys
FULL_COLOR_PALETTE = {
    # Base colors
    "primary": "#3A86FF",     # Blue
    "primary_dark": "#0043CE", # Darker blue for gradients
    "primary_light": "#60A5FA", # Lighter blue for gradients
    "secondary": "#8338EC",   # Purple
    "secondary_dark": "#6025C9", # Darker purple for gradients
    "secondary_light": "#A875F5", # Lighter purple for gradients
    "success": "#06D6A0",     # Green
    "success_dark": "#05A57E", # Darker green for gradients
    "success_light": "#12F1B8", # Lighter green for gradients
    "warning": "#FFD166",     # Yellow
    "warning_dark": "#FFC233", # Darker yellow for gradients
    "warning_light": "#FFDF99", # Lighter yellow for gradients
    "danger": "#FF006E",      # Magenta
    "danger_dark": "#C8005C", # Darker magenta for gradients
    "danger_light": "#FF4D9B", # Lighter magenta for gradients
    "info": "#00B4D8",        # Light blue
    "info_dark": "#0090B0",   # Darker light blue for gradients
    "info_light": "#5FDFFA",  # Lighter light blue for gradients
    "dark": "#1E293B",        # Dark blue-gray
    "light": "#F8FAFC",       # Light gray
    "background": "#F1F5F9",  # Light gray-blue
    
    # Text colors
    "text": "#334155",        # Dark gray - general text
    "text_primary": "#334155", # Dark gray - primary text (same as text)
    "text_secondary": "#64748B", # Medium gray - secondary text
    "text_muted": "#94A3B8",   # Light gray - muted text
    
    # Border colors
    "border": "#E2E8F0",      # Light border color
    "border_light": "#F1F5F9", # Lighter border color
    "border_dark": "#CBD5E1",  # Darker border color
    
    # CAMP-specific colors
    "camp_capital": "#3A86FF", # Blue for Capital
    "camp_advantage": "#FF006E", # Magenta for Advantage
    "camp_market": "#FFD166",  # Yellow for Market
    "camp_people": "#06D6A0"   # Green for People
}

def configure_colors():
    """
    Configure colors by patching the global COLOR_PALETTE in analysis_flow.py.
    This function should be imported and called at the start of the application.
    """
    try:
        # Try to import analysis_flow
        try:
            import analysis_flow
            
            # Replace the COLOR_PALETTE with our enhanced version
            analysis_flow.COLOR_PALETTE = FULL_COLOR_PALETTE.copy()
            
            # If the module is already imported elsewhere, reload it
            if 'analysis_flow' in sys.modules:
                importlib.reload(sys.modules['analysis_flow'])
                
            logger.info("Successfully configured colors by patching analysis_flow.COLOR_PALETTE")
            return True
            
        except ImportError:
            logger.warning("Could not import analysis_flow module")
            return False
            
    except Exception as e:
        logger.error(f"Error configuring colors: {e}")
        return False

# Monkey patch functions for color management
def get_color(key, default=None):
    """
    Get a color from the palette by key, with fallback to default.
    
    Args:
        key (str): The color key to look up
        default (str, optional): Default color if key not found
        
    Returns:
        str: The color value
    """
    if key in FULL_COLOR_PALETTE:
        return FULL_COLOR_PALETTE[key]
    
    # Try to get a base color if the key has a suffix
    base_key = key.split('_')[0]
    if f"{base_key}_dark" in FULL_COLOR_PALETTE and key.endswith('_dark'):
        return FULL_COLOR_PALETTE[f"{base_key}_dark"]
    if f"{base_key}_light" in FULL_COLOR_PALETTE and key.endswith('_light'):
        return FULL_COLOR_PALETTE[f"{base_key}_light"]
    
    # Return default or primary color
    if default is not None:
        return default
    return FULL_COLOR_PALETTE.get('primary', "#3A86FF")

# Automatic configuration on import
configure_colors()
