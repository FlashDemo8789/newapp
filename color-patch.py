"""
Patch script to update the COLOR_PALETTE in analysis_flow.py
Run this once to fix missing color keys in the application.
"""

import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("patch_colors")

# The updated color palette to inject
UPDATED_COLOR_PALETTE = """
# Enhanced color palette with all required keys
COLOR_PALETTE = {
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
"""

def patch_analysis_flow():
    """Patch the COLOR_PALETTE in analysis_flow.py"""
    
    filename = "analysis_flow.py"
    logger.info(f"Preparing to patch {filename}")
    
    # Check if file exists
    if not os.path.exists(filename):
        logger.error(f"File {filename} not found")
        return False
    
    try:
        # Read the file content
        with open(filename, 'r') as f:
            content = f.read()
        
        # Back up the original file
        backup_file = f"{filename}.bak"
        with open(backup_file, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_file}")
        
        # Look for the COLOR_PALETTE definition
        color_palette_pattern = r'COLOR_PALETTE\s*=\s*\{[^}]*\}'
        match = re.search(color_palette_pattern, content, re.DOTALL)
        
        if not match:
            logger.error("Could not find COLOR_PALETTE definition in file")
            return False
        
        # Replace the COLOR_PALETTE definition
        new_content = content.replace(match.group(0), UPDATED_COLOR_PALETTE.strip())
        
        # Write the updated content back to the file
        with open(filename, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Successfully patched {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error patching file: {e}")
        return False

if __name__ == "__main__":
    if patch_analysis_flow():
        print("Patch applied successfully!")
    else:
        print("Failed to apply patch. Check logs for details.")
