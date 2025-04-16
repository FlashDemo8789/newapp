"""
Color utilities and management for FlashDNA Analysis System.
This module provides centralized color management, utilities, and patching functions
to ensure consistent color usage throughout the application.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger("color_utils")

# Enhanced default color palette
DEFAULT_COLOR_PALETTE = {
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

# Global variable to store the current color palette
_current_palette = DEFAULT_COLOR_PALETTE.copy()

def get_color_palette() -> Dict[str, str]:
    """
    Get the current color palette.
    
    Returns:
        Dict[str, str]: The current color palette.
    """
    return _current_palette.copy()

def set_color_palette(palette: Dict[str, str]) -> None:
    """
    Set the current color palette.
    
    Args:
        palette (Dict[str, str]): The new color palette to use.
    """
    global _current_palette
    _current_palette = palette.copy()
    logger.info(f"Color palette updated with {len(palette)} colors")

def get_color(key: str, default: str = None) -> str:
    """
    Get a color from the current palette by key.
    
    Args:
        key (str): The color key to look up.
        default (str, optional): The default color to return if key not found.
            If None and key not found, returns the 'primary' color.
            
    Returns:
        str: The color value.
    """
    if key in _current_palette:
        return _current_palette[key]
    
    # Handle case when key has a suffix (_dark, _light, etc.)
    base_key = key.split('_')[0]
    if f"{base_key}_dark" in _current_palette and key.endswith('_dark'):
        return _current_palette[f"{base_key}_dark"]
    if f"{base_key}_light" in _current_palette and key.endswith('_light'):
        return _current_palette[f"{base_key}_light"]
        
    # Log warning for missing color key
    logger.warning(f"Color key '{key}' not found in palette")
    
    # Return default value if provided, otherwise return primary color
    if default is not None:
        return default
    return _current_palette.get('primary', "#3A86FF")

def get_gradient(start_color: str, end_color: str, steps: int = 10) -> List[str]:
    """
    Create a gradient between two colors.
    
    Args:
        start_color (str): Starting color in hex format.
        end_color (str): Ending color in hex format.
        steps (int): Number of color steps in the gradient.
        
    Returns:
        List[str]: List of hex colors forming a gradient.
    """
    try:
        import numpy as np
        
        # Convert hex to RGB
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        
        # Create gradient
        rgb_gradient = []
        for i in range(steps):
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (steps - 1))
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (steps - 1))
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (steps - 1))
            rgb_gradient.append((r, g, b))
        
        # Convert back to hex
        return [rgb_to_hex(rgb) for rgb in rgb_gradient]
    except Exception as e:
        logger.error(f"Error creating gradient: {e}")
        return [start_color, end_color]  # Fallback

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color (str): Color in hex format (e.g., "#3A86FF").
        
    Returns:
        Tuple[int, int, int]: RGB color tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color.
    
    Args:
        rgb (Tuple[int, int, int]): RGB color tuple.
        
    Returns:
        str: Color in hex format.
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def get_contrasting_text_color(bg_color: str) -> str:
    """
    Get a contrasting text color (black or white) for a given background color.
    
    Args:
        bg_color (str): Background color in hex format.
        
    Returns:
        str: '#FFFFFF' for dark backgrounds or '#000000' for light backgrounds.
    """
    try:
        r, g, b = hex_to_rgb(bg_color)
        # Calculate brightness using a common formula
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 125 else "#FFFFFF"
    except Exception as e:
        logger.error(f"Error calculating contrasting color: {e}")
        return "#000000"  # Default to black text

def enhance_color_palette(palette: Dict[str, str]) -> Dict[str, str]:
    """
    Enhance a color palette by adding derived colors like dark and light variants.
    
    Args:
        palette (Dict[str, str]): The base color palette.
        
    Returns:
        Dict[str, str]: Enhanced color palette with derived colors.
    """
    enhanced = palette.copy()
    
    # Base colors to generate variants for
    base_colors = ["primary", "secondary", "success", "warning", "danger", "info"]
    
    for color in base_colors:
        if color in palette:
            # Add dark variant if not already present
            if f"{color}_dark" not in enhanced:
                try:
                    # Darken the base color
                    r, g, b = hex_to_rgb(palette[color])
                    r = max(0, int(r * 0.8))
                    g = max(0, int(g * 0.8))
                    b = max(0, int(b * 0.8))
                    enhanced[f"{color}_dark"] = rgb_to_hex((r, g, b))
                except Exception:
                    # Fallback if calculation fails
                    enhanced[f"{color}_dark"] = palette[color]
            
            # Add light variant if not already present
            if f"{color}_light" not in enhanced:
                try:
                    # Lighten the base color
                    r, g, b = hex_to_rgb(palette[color])
                    r = min(255, int(r + (255 - r) * 0.4))
                    g = min(255, int(g + (255 - g) * 0.4))
                    b = min(255, int(b + (255 - b) * 0.4))
                    enhanced[f"{color}_light"] = rgb_to_hex((r, g, b))
                except Exception:
                    # Fallback if calculation fails
                    enhanced[f"{color}_light"] = palette[color]
    
    # Ensure text colors exist
    if "text" in palette and "text_primary" not in enhanced:
        enhanced["text_primary"] = palette["text"]
    
    if "text" in palette and "text_secondary" not in enhanced:
        try:
            r, g, b = hex_to_rgb(palette["text"])
            r = min(255, int(r + (255 - r) * 0.3))
            g = min(255, int(g + (255 - g) * 0.3))
            b = min(255, int(b + (255 - b) * 0.3))
            enhanced["text_secondary"] = rgb_to_hex((r, g, b))
        except Exception:
            enhanced["text_secondary"] = "#64748B"  # Default medium gray
    
    if "text_muted" not in enhanced:
        enhanced["text_muted"] = "#94A3B8"  # Default light gray
    
    return enhanced

def patch_global_color_palette():
    """
    Patch the global COLOR_PALETTE with missing color keys to ensure compatibility.
    This function should be called early in the application initialization.
    """
    try:
        # Try to import the global COLOR_PALETTE
        try:
            from analysis_flow import COLOR_PALETTE as global_palette
            
            # Create enhanced palette with all necessary colors
            enhanced_palette = enhance_color_palette(global_palette)
            
            # Update the global palette in the analysis_flow module
            import analysis_flow
            analysis_flow.COLOR_PALETTE = enhanced_palette
            
            # Also update our local reference
            set_color_palette(enhanced_palette)
            
            logger.info(f"Successfully patched global COLOR_PALETTE with {len(enhanced_palette)} colors")
            
        except ImportError:
            logger.warning("Could not import global COLOR_PALETTE, using default palette")
            set_color_palette(DEFAULT_COLOR_PALETTE)
            
    except Exception as e:
        logger.error(f"Error patching global COLOR_PALETTE: {e}")
        # Ensure we at least have a valid local palette
        set_color_palette(DEFAULT_COLOR_PALETTE)

# Patch the global COLOR_PALETTE on module import
patch_global_color_palette()
