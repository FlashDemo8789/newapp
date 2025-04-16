import pandas as pd
import matplotlib
# Force use of non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF
import logging
import os
from datetime import datetime
import numpy as np
import tempfile
import traceback
import copy
import streamlit as st
import json
from io import StringIO, BytesIO
import csv
import math
from pdf_generator import generate_enhanced_pdf

logger = logging.getLogger("report_generator")

class ReportPDF(FPDF):
    """Extended FPDF class with header and footer for investor reports."""
    
    def __init__(self, title="Investor Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        # Set up margins
        self.set_margins(15, 15, 15)
        # Set auto page break
        self.set_auto_page_break(True, margin=25)
        
    def header(self):
        # Logo (if available)
        try:
            if os.path.exists(os.path.join("static", "logo.png")):
                self.image(os.path.join("static", "logo.png"), 10, 8, 30)
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
        
        # Report title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, self.title, 0, 1, 'C')
        
        # Line break
        self.ln(10)
    
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
        
    def chapter_body(self, text):
        """Add chapter body text"""
        self.set_font('Arial', '', 11)
        # Split text into lines and output
        self.multi_cell(0, 7, text)
        self.ln(7)
        
    def section_title(self, title):
        """Add a section title (smaller than chapter)"""
        self.set_font('Arial', 'B', 11)
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(3)
        
    def info_cell(self, label, value, width1=60, width2=0):
        """Add an information cell with label and value"""
        self.set_font('Arial', 'B', 10)
        self.cell(width1, 7, label, 0, 0, 'L')
        self.set_font('Arial', '', 10)
        if width2 == 0:
            width2 = self.w - width1 - self.r_margin - self.l_margin
        self.cell(width2, 7, str(value), 0, 1, 'L')
        
    def add_image(self, img_data, caption="", w=0, h=0, x=None, y=None, center=False):
        """Add an image to the PDF."""
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
            
        # Adjust x for centering if requested
        if center and w > 0:
            x = (self.w - w) / 2
        
        # Check if image data exists
        if img_data is None:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, "Image data not available", 0, 1, 'C')
            return
        
        try:
            # Check if it's a file path string
            if isinstance(img_data, str) and os.path.exists(img_data):
                # It's a file path
                self.image(img_data, x=x, y=y, w=w, h=h)
            else:
                # It's binary data, save to temp file first
                fd, temp_path = tempfile.mkstemp(suffix='.png')
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(img_data)
                    
                    # Add the image to the PDF
                    self.image(temp_path, x=x, y=y, w=w, h=h)
                finally:
                    try:
                        os.close(fd)
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up temp file: {str(e)}")
            
            # Move below the image and add caption if provided
            if h > 0:
                self.set_y(y + h + 5)
            else:
                self.ln(5)
            
            if caption:
                self.set_font('Arial', 'I', 9)
                self.cell(0, 5, caption, 0, 1, 'C')
                self.ln(5)
                
        except Exception as e:
            logger.error(f"Error adding image to PDF: {str(e)}")
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, "Error displaying image", 0, 1, 'C')
            
    def add_cover_page(self, startup_name, startup_stage, startup_sector):
        """Add a cover page to the report."""
        self.add_page()
        
        # Title
        self.set_font('Arial', 'B', 24)
        self.cell(0, 20, "Investor Report", 0, 1, 'C')
        
        # Startup name
        self.set_font('Arial', 'B', 20)
        self.cell(0, 30, startup_name, 0, 1, 'C')
        
        # Stage and sector
        self.set_font('Arial', '', 14)
        if startup_stage:
            self.cell(0, 15, f"Stage: {startup_stage}", 0, 1, 'C')
        if startup_sector:
            self.cell(0, 15, f"Sector: {startup_sector}", 0, 1, 'C')
        
        # Date
        self.set_font('Arial', 'I', 12)
        
        # Confidentiality notice
        self.set_y(-60)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, "CONFIDENTIAL", 0, 1, 'C')
        self.multi_cell(0, 5, "This document contains confidential information about the company and is intended only for the named recipient. If you are not the intended recipient, please notify the sender immediately and destroy this document.")
    
    def add_section(self, title):
        """Add a new section with a title."""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def add_paragraph(self, text, bold=False, italics=False, left_indent=0):
        """Add a paragraph of text."""
        style = ''
        if bold:
            style += 'B'
        if italics:
            style += 'I'
        
        self.set_font('Arial', style, 11)
        self.set_x(self.get_x() + left_indent)
        self.multi_cell(0, 6, text)
        self.ln(3)
    
    def add_table(self, data, header=True):
        """Add a table to the PDF."""
        # Calculate column widths
        num_cols = len(data[0])
        col_width = (self.w - self.l_margin - self.r_margin) / num_cols
        
        # Set font for header if there is one
        if header:
            self.set_font('Arial', 'B', 11)
            for i, cell in enumerate(data[0]):
                self.cell(col_width, 7, str(cell), 1, 0, 'C')
            self.ln()
            self.set_font('Arial', '', 11)
            start_row = 1
        else:
            start_row = 0
        
        # Add data rows
        for row in data[start_row:]:
            for cell in row:
                self.cell(col_width, 7, str(cell), 1, 0, 'L')
            self.ln()
        
        self.ln(5)

def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for embedding in PDF."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Close the figure to free memory
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error converting figure to bytes: {str(e)}")
        # Create a simple error image
        try:
            # Create a simple error figure
            error_fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Chart Generation Error: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_axis_off()
            
            # Save to buffer
            buf = io.BytesIO()
            error_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(error_fig)
            return buf.getvalue()
        except:
            logger.error("Failed to create error chart", exc_info=True)
            return None

def generate_camp_radar_chart(doc):
    """Generate CAMP framework radar chart."""
    # Extract CAMP scores
    capital_score = doc.get("capital_score", 0)
    advantage_score = doc.get("advantage_score", 0)
    market_score = doc.get("market_score", 0)
    people_score = doc.get("people_score", 0)
    
    # Create radar chart
    categories = ['Capital\nEfficiency', 'Market\nDynamics', 'Advantage\nMoat', 'People &\nPerformance']
    values = [capital_score, market_score, advantage_score, people_score]
    
    # Convert to numpy array and close the loop
    values = np.concatenate((values, [values[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # Calculate angle for each category
    N = len(categories) - 1  # -1 because we added one to close the loop
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
    
    # Add title
    plt.title("CAMP Framework Analysis", size=14, color='blue', y=1.1)
    
    return fig_to_bytes(fig)

def create_bar_chart(labels, values, title="Metrics"):
    """Generate a simple bar chart with the given labels and values."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(labels, values, width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(title)
    
    # Rotate x-axis labels if there are many or if they're long
    if len(labels) > 5 or any(len(str(label)) > 10 for label in labels):
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig_to_bytes(fig)

def generate_user_growth_chart(doc):
    """Generate user growth projection chart."""
    sys_dynamics = doc.get("system_dynamics", {})
    
    if not isinstance(sys_dynamics, dict) or "users" not in sys_dynamics:
        return None
    
    users = sys_dynamics.get("users", [])
    months = list(range(len(users)))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, users, marker='o', linestyle='-', color='blue')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Users')
    ax.set_title('User Growth Projection')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(users) > 1:
        z = np.polyfit(months, users, 1)
        p = np.poly1d(z)
        ax.plot(months, p(months), "r--", alpha=0.8)
    
    fig.tight_layout()
    return fig_to_bytes(fig)

def generate_financial_chart(doc):
    """Generate financial projection chart."""
    forecast = doc.get("financial_forecast", {})
    
    if not isinstance(forecast, dict) or "months" not in forecast or "revenue" not in forecast:
        return None
    
    months = forecast.get("months", [])
    revenue = forecast.get("revenue", [])
    profit = forecast.get("profit", []) if "profit" in forecast else None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot revenue
    ax.plot(months, revenue, marker='o', linestyle='-', color='green', label='Revenue')
    
    # Plot profit if available
    if profit and len(profit) == len(months):
        ax.plot(months, profit, marker='s', linestyle='-', color='blue', label='Profit')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount ($)')
    ax.set_title('Financial Projections')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    return fig_to_bytes(fig)

def generate_competitive_chart(doc):
    """Generate competitive positioning chart."""
    positioning = doc.get("competitive_positioning", {})
    
    if not isinstance(positioning, dict):
        return None
    
    # Extract positioning data
    dimensions = positioning.get("dimensions", [])
    company_position = positioning.get("company_position", {})
    competitor_positions = positioning.get("competitor_positions", {})
    
    if not dimensions or not company_position or not competitor_positions or len(dimensions) < 2:
        return None
    
    # Pick first two dimensions
    x_dim = dimensions[0]
    y_dim = dimensions[1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot company position
    company_x = company_position.get(x_dim, 50)
    company_y = company_position.get(y_dim, 50)
    ax.scatter(company_x, company_y, color='blue', s=100, marker='o', label='Your Company')
    
    # Plot competitor positions
    for comp_name, comp_pos in competitor_positions.items():
        comp_x = comp_pos.get(x_dim, 50)
        comp_y = comp_pos.get(y_dim, 50)
        ax.scatter(comp_x, comp_y, color='red', s=80, alpha=0.7)
        ax.annotate(comp_name, (comp_x, comp_y), xytext=(5, 5), textcoords='offset points')
    
    # Add quadrant lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title(f'Competitive Positioning: {x_dim} vs {y_dim}')
    
    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add legend
    ax.legend()
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    return fig_to_bytes(fig)

def format_currency(value):
    """Format a numeric value as currency."""
    try:
        # Convert to float if not already
        value = float(value)
        
        # Format based on size
        if value >= 1e9:  # Billions
            return f"${value/1e9:.1f}B"
        elif value >= 1e6:  # Millions
            return f"${value/1e6:.1f}M"
        elif value >= 1e3:  # Thousands
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    except (ValueError, TypeError):
        return f"${0:.2f}"  # Return $0.00 for invalid values

def sanitize_text_for_pdf(text):
    """
    Sanitize text for PDF output with latin1 encoding.
    Replace common Unicode characters with ASCII equivalents.
    
    Args:
        text (str): Text to sanitize
        
    Returns:
        str: Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2022': '-',  # bullet point -> hyphen
        '\u2023': '>',  # triangular bullet -> greater than
        '\u2043': '-',  # hyphen bullet -> hyphen
        '\u2024': '.',  # one dot leader -> period
        '\u2025': '..',  # two dot leader -> two periods
        '\u2026': '...',  # ellipsis -> three periods
        '\u2019': "'",  # right single quotation mark -> apostrophe
        '\u2018': "'",  # left single quotation mark -> apostrophe
        '\u201C': '"',  # left double quotation mark -> quotation mark
        '\u201D': '"',  # right double quotation mark -> quotation mark
        '\u2014': '--',  # em dash -> two hyphens
        '\u2013': '-',  # en dash -> hyphen
        '\u00A9': '(c)',  # copyright sign -> (c)
        '\u00AE': '(R)',  # registered sign -> (R)
        '\u2122': '(TM)',  # trademark sign -> (TM)
        '\u20AC': 'EUR',  # euro sign -> EUR
        '\u00A3': 'GBP',  # pound sign -> GBP
        '\u00A5': 'JPY',  # yen sign -> JPY
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Replace any other non-latin1 characters with '?'
    return ''.join(c if ord(c) < 256 else '?' for c in text)

def emergency_sanitize_and_generate_pdf(pdf):
    """
    Emergency fallback to sanitize all text in the PDF and regenerate.
    
    Args:
        pdf (ReportPDF): PDF object with content
        
    Returns:
        bytes: PDF report as bytes with sanitized content
    """
    try:
        # Force all text in the PDF to be ASCII-compatible
        if hasattr(pdf, '_out') and callable(pdf._out):
            original_out = pdf._out
            
            def sanitized_out(s):
                if isinstance(s, str):
                    s = sanitize_text_for_pdf(s)
                return original_out(s)
            
            pdf._out = sanitized_out
        
        # Output the PDF
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        logger.error(f"Emergency PDF sanitization failed: {e}")
        # Create a simple error PDF as last resort
        error_pdf = ReportPDF()
        error_pdf.add_page()
        error_pdf.set_font('Arial', 'B', 16)
        error_pdf.cell(0, 10, "Error Generating Full Report", 0, 1, 'C')
        error_pdf.set_font('Arial', '', 12)
        error_pdf.multi_cell(0, 10, "A technical error occurred while generating the report. Please try again or contact support.")
        return error_pdf.output(dest='S').encode('latin1')

def generate_investor_report(doc, report_type="full", sections=None):
    """
    Generate a PDF investor report for a startup.
    
    Args:
        doc (dict): Document containing the startup data
        report_type (str): Type of report to generate 
                          ("executive", "full", or "custom")
        sections (dict): Dictionary of sections to include (for custom reports)
        
    Returns:
        bytes: PDF report as bytes
    """
    # Create a deep copy of the document to avoid modifying the original
    try:
        import copy
        import tempfile
        import os
        import logging
        logger = logging.getLogger("report_generator")
        
        doc_copy = copy.deepcopy(doc)
        logger.info("Created deep copy of document for report generation")
        
        # Pre-process known problematic data types 
        if "cohort_data" in doc_copy and isinstance(doc_copy["cohort_data"], dict):
            for key, value in list(doc_copy["cohort_data"].items()):
                if hasattr(pd, 'Period') and hasattr(value, 'index') and hasattr(value.index, 'dtype'):
                    if 'period' in str(value.index.dtype).lower():
                        doc_copy["cohort_data"][key] = value.copy()
                        doc_copy["cohort_data"][key].index = [str(idx) for idx in value.index]
                        logger.info(f"Converted Period index to strings in cohort_data[{key}]")
    except Exception as e:
        logger.warning(f"Error preprocessing document data: {e}")
        doc_copy = doc  # Fall back to original if any errors
        
    # Set up default sections based on report type
    default_sections = {
        "executive": {
            "Executive Summary": True,
            "Business Model": True,
            "Market Analysis": True,
            "Team Assessment": True
        },
        "full": {
            "Executive Summary": True,
            "Business Model": True,
            "Market Analysis": True,
            "Financial Projections": True,
            "Team Assessment": True,
            "Competitive Analysis": True,
            "Growth Metrics": True,
            "Risk Assessment": True,
            "Exit Strategy": True,
            "Technical Assessment": True
        }
    }
    
    # Use provided sections or defaults
    if report_type == "custom" and sections:
        active_sections = sections
    else:
        active_sections = default_sections.get(report_type, default_sections["executive"])
    
    # Initialize PDF report object
    pdf = ReportPDF(title=f"{doc_copy.get('name', 'Startup')} Investment Analysis")
    
    # Set fonts and colors for consistent styling
    pdf.set_text_color(44, 62, 80)  # Dark blue-gray
    pdf.set_font('Arial', '', 11)
    
    # Add cover page
    startup_name = doc_copy.get("name", "Startup")
    startup_stage = doc_copy.get("stage", "")
    startup_sector = doc_copy.get("sector", "")
    pdf.add_cover_page(startup_name, startup_stage, startup_sector)
    
    # Add table of contents page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 16, "Table of Contents", 0, 1, 'C')
    pdf.ln(5)
    
    # Store current position for TOC content (to be filled later)
    toc_y_pos = pdf.get_y()
    toc_page = pdf.page_no()  # Store the page number for TOC
    
    # Add CAMP Framework analysis section
    pdf.add_section("CAMP Framework Analysis")
    
    # Create a better description of CAMP framework
    pdf.add_paragraph("The CAMP Framework evaluates startups across four critical dimensions that have been shown to correlate with startup success. Each dimension is scored on a scale of 0-100, with higher scores indicating stronger performance.", False)
    
    # Add CAMP definition
    camp_definitions = [
        ("Capital Efficiency (C)", "Measures financial health, sustainability, and how efficiently capital is deployed."),
        ("Advantage Moat (A)", "Evaluates competitive advantages, defensibility, and barriers to entry for competitors."),
        ("Market Dynamics (M)", "Assesses market opportunity, growth potential, and the startup's positioning."),
        ("People & Performance (P)", "Evaluates team strength, experience, and execution capability.")
    ]
    
    for title, desc in camp_definitions:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, title, 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)
    
    # Extract CAMP scores and ensure reasonable values
    def normalize_score(score):
        """Ensure score is within 0-100 range and not unrealistically high."""
        if score > 95:  # Cap perfect scores to make them more realistic
            return 95.0 + (score - 95) * 0.5  # Scale values above 95 to max out at 97.5
        return score
    
    capital_score = normalize_score(doc_copy.get("capital_score", 0))
    market_score = normalize_score(doc_copy.get("market_score", 0))
    advantage_score = normalize_score(doc_copy.get("advantage_score", 0))
    people_score = normalize_score(doc_copy.get("people_score", 0))
    
    # Calculate weighted CAMP score
    camp_score = (
        capital_score * 0.30 +
        market_score * 0.25 +
        advantage_score * 0.20 +
        people_score * 0.25
    )
    
    try:
        # Create a CAMP radar chart
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Set up figure with better styling
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot categories
        categories = ['Capital\nEfficiency', 'Market\nDynamics', 'Advantage\nMoat', 'People &\nPerformance']
        values = [capital_score, market_score, advantage_score, people_score]
        
        # Convert to numpy array and close the loop
        values = np.concatenate((values, [values[0]]))
        categories = np.concatenate((categories, [categories[0]]))
        
        # Calculate angle for each category
        N = len(categories) - 1
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Set up the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], color='#7f8c8d')
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add title
        plt.title("CAMP Framework Analysis", size=16, color='#2c3e50', y=1.1)
        
        # Set background color
        fig.patch.set_facecolor('#f9f9f9')
        ax.set_facecolor('#f9f9f9')
        
        # Create temp file and save
        temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_chart.name, dpi=200, bbox_inches='tight', facecolor='#f9f9f9')
        plt.close(fig)
        temp_chart.close()
        
        # Add the chart to the PDF
        pdf.add_image(temp_chart.name, w=180, h=140, center=True, 
                     caption="CAMP Framework Radar Analysis")
        
        # Clean up temp file
        try:
            os.unlink(temp_chart.name)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")
            
    except Exception as e:
        logger.error(f"Error creating CAMP radar chart: {e}")
    
    # Add CAMP scores table
    camp_table_data = [
        ["Dimension", "Score", "Weight", "Description"],
    ]
    
    # Set up a more stylish table with better spacing
    pdf.set_font('Arial', 'B', 10)
    pdf.ln(10)  # Add some space
    
    # Table column widths
    col_widths = [50, 25, 25, 80]
    row_height = 8  # Row height
    
    # Draw table header with background
    pdf.set_fill_color(44, 62, 80)  # Dark header background
    pdf.set_text_color(255, 255, 255)  # White text for header
    pdf.set_font('Arial', 'B', 10)
    
    for i, width in enumerate(col_widths):
        pdf.cell(width, row_height, camp_table_data[0][i], 1, 0, 'C', 1)
    pdf.ln()
    
    # Draw table data
    pdf.set_text_color(44, 62, 80)  # Dark text
    fill = False
    for row in camp_table_data[1:]:
        # Alternate row colors
        fill = not fill
        if row[0] == "Overall CAMP Score":
            pdf.set_fill_color(217, 240, 255)  # Light blue for overall score
            pdf.set_font('Arial', 'B', 10)
        else:
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_font('Arial', '', 10)
            
        for i, width in enumerate(col_widths):
            pdf.cell(width, row_height, row[i], 1, 0, 'C', 1)
        pdf.ln()
    
    pdf.set_font('Arial', '', 10)  # Reset font
    
    # Add key insights from CAMP analysis
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "Key Insights from CAMP Analysis", 0, 1, 'L')
    
    # Generate insights based on CAMP scores
    insights = []
    
    # Identify the strongest dimension
    scores = [
        ("Capital Efficiency", capital_score),
        ("Market Dynamics", market_score),
        ("Advantage Moat", advantage_score),
        ("People & Performance", people_score)
    ]
    strongest = max(scores, key=lambda x: x[1])
    weakest = min(scores, key=lambda x: x[1])
    
    # Add insight for strongest dimension
    insights.append(f"Strongest Dimension: {strongest[0]} ({strongest[1]:.1f}/100)")
    
    # Add insight for weakest dimension
    insights.append(f"Area for Improvement: {weakest[0]} ({weakest[1]:.1f}/100)")
    
    # Add general assessment based on overall score
    if camp_score >= 85:
        insights.append(f"Outstanding overall CAMP score ({camp_score:.1f}/100) indicates strong potential for success.")
    elif camp_score >= 70:
        insights.append(f"Strong overall CAMP score ({camp_score:.1f}/100) shows good potential with some areas to strengthen.")
    elif camp_score >= 50:
        insights.append(f"Average overall CAMP score ({camp_score:.1f}/100) indicates moderate potential with clear improvement areas.")
    else:
        insights.append(f"Below average CAMP score ({camp_score:.1f}/100) suggests significant challenges that need to be addressed.")
    
    # Add dimension-specific insights
    if capital_score < 50:
        insights.append("Capital efficiency requires significant improvement for sustainability.")
    
    if market_score > 80:
        insights.append("Strong market positioning provides excellent growth potential.")
    
    if advantage_score < 60:
        insights.append("Competitive moat needs strengthening to defend market position.")
    
    if people_score > 80:
        insights.append("Strong team and execution capability is a key asset.")
    
    # Add the insights to the PDF
    for insight in insights:
        pdf.set_font('Arial', '', 10)
        pdf.cell(10, 6, "-", 0, 0, 'R')  # Replace Unicode bullet with hyphen
        pdf.cell(0, 6, insight, 0, 1, 'L')
    
    # Add remaining sections based on active_sections
    # ... (implementation of all the other sections)
    
    # NOW FOR THE TABLE OF CONTENTS
    # Remember current page number
    current_page = pdf.page_no()
    
    # Go back to TOC page
    pdf.page = toc_page  # Go back to the TOC page we stored earlier
    pdf.set_y(toc_y_pos)
    
    # Set TOC styling
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(44, 62, 80)
    
    # Add TOC entries
    sections_added = []
    
    if active_sections.get("Executive Summary", False):
        sections_added.append("Executive Summary")
    
    # Add other sections based on active_sections
    for section_name, is_active in active_sections.items():
        if is_active and section_name not in sections_added and section_name != "CAMP Framework Analysis":  # Skip CAMP since it's already added
            sections_added.append(section_name)
    
    # Add CAMP Framework first
    pdf.cell(0, 10, "CAMP Framework Analysis", 0, 0, 'L')
    pdf.cell(0, 10, "3", 0, 1, 'R')
    
    # Add page numbers for all sections in order
    section_page = 3  # CAMP Framework starts on page 3
    
    for section in sections_added:
        # Each section typically takes at least 2 pages
        section_page += 2
        
        if section_page > current_page:
            section_page = current_page
            
        pdf.cell(0, 10, section, 0, 0, 'L')
        pdf.cell(0, 10, f"{section_page}", 0, 1, 'R')
        
        # Add dotted lines between text and page number
        pdf.ln(2)
    
    # Return to the last page where we were working
    pdf.page = current_page
    
    # Continue with the rest of the report sections
    # (Executive Summary, Business Model, Market Analysis, etc.)
    # Code for additional sections would go here...
    
    # Return the PDF as bytes
    try:
        # Replace bullet points (•) with a hyphen (-) to avoid encoding errors
        def safe_output():
            """Generate PDF output with safe character replacements"""
            # Store original state
            original_buffer = pdf.buffer
            pdf.buffer = bytearray()
            
            # Process each page to replace problematic characters
            for page in original_buffer.split(b'BT'):
                if page:
                    # Replace bullet points and other problematic characters
                    safe_page = page.replace(b'\xe2\x80\xa2', b'-')  # Replace bullet with hyphen
                    pdf.buffer += safe_page
                    if page != original_buffer.split(b'BT')[-1]:  # Not the last segment
                        pdf.buffer += b'BT'
            
            # Get the final output
            output = bytes(pdf.buffer)
            return output
            
        # Try the standard output method first
        try:
            output = pdf.output(dest='S')
            return output
        except UnicodeEncodeError:
            # Fall back to safe output with character replacement
            logger.warning("Handling Unicode characters in PDF generation")
            return safe_output()
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        # Create a simple error PDF instead
        error_pdf = ReportPDF()
        error_pdf.add_page()
        error_pdf.set_font('Arial', 'B', 16)
        error_pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
        error_pdf.set_font('Arial', '', 12)
        error_pdf.multi_cell(0, 10, f"An error occurred while generating the report: {str(e)}")
        return error_pdf.output(dest='S')

# Custom JSON encoder to handle special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle pandas objects
        if hasattr(pd, 'Period') and isinstance(obj, pd.Period):
            return str(obj)
        if hasattr(pd, 'Timestamp') and isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(pd, 'DataFrame') and isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if hasattr(pd, 'Series') and isinstance(obj, pd.Series):
            return obj.to_dict()
        
        # Handle numpy types
        if hasattr(obj, 'dtype'):
            return obj.item()
        
        # Handle any other non-serializable objects
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# Helper function to flatten the nested metrics into a flat structure for CSV
def flatten_metrics(doc):
    """Convert nested document structure into a flat dictionary for CSV export."""
    flat_dict = {}
    
    # Add company info
    flat_dict["Company Name"] = doc.get("name", "")
    flat_dict["Sector"] = doc.get("sector", "")
    flat_dict["Stage"] = doc.get("stage", "")
    
    # Add key metrics
    flat_dict["CAMP Score"] = doc.get("camp_score", 0)
    flat_dict["Capital Score"] = doc.get("capital_score", 0)
    flat_dict["Advantage Score"] = doc.get("advantage_score", 0)
    flat_dict["Market Score"] = doc.get("market_score", 0)
    flat_dict["People Score"] = doc.get("people_score", 0)
    flat_dict["Success Probability"] = doc.get("success_prob", 0)
    flat_dict["Runway (months)"] = doc.get("runway_months", 0)
    
    # Add financial metrics if available
    flat_dict["Monthly Revenue"] = doc.get("monthly_revenue", 0)
    flat_dict["Monthly Burn"] = doc.get("burn_rate", 0)
    
    # Add unit economics if available
    unit_econ = doc.get("unit_economics", {})
    if unit_econ:
        flat_dict["LTV"] = unit_econ.get("ltv", 0)
        flat_dict["CAC"] = unit_econ.get("cac", 0)
        flat_dict["LTV:CAC Ratio"] = unit_econ.get("ltv_cac_ratio", 0)
        flat_dict["Gross Margin"] = unit_econ.get("gross_margin", 0)
    
    # Add market metrics
    flat_dict["Market Size"] = doc.get("market_size", 0)
    flat_dict["Market Growth Rate"] = doc.get("market_growth_rate", 0)
    flat_dict["Market Share"] = doc.get("market_share", 0)
    
    # Add team metrics
    flat_dict["Team Size"] = doc.get("employee_count", 0)
    flat_dict["Tech Talent Ratio"] = doc.get("tech_talent_ratio", 0)
    
    return flat_dict

def render_report_tab(doc: dict):
    """Render the report tab with download options for enhanced visual reports."""
    st.header("Investor Report")
    
    try:
        # Create a deep copy of the document to avoid modifying the original
        logger = logging.getLogger("pdf_generator")
        
        doc_copy = copy.deepcopy(doc)
        
        # Pre-process pandas Period indices to ensure proper serialization
        if "cohort_data" in doc_copy and isinstance(doc_copy["cohort_data"], dict):
            for key, value in list(doc_copy["cohort_data"].items()):
                if hasattr(value, 'index') and hasattr(pd, 'Period'):
                    # Safe check for dtype attribute and type
                    is_period_index = isinstance(value.index, pd.PeriodIndex)
                    if (hasattr(value.index, 'dtype') and 
                        isinstance(value.index.dtype, type) and 
                        'period' in str(value.index.dtype).lower()) or is_period_index:
                        
                        doc_copy["cohort_data"][key] = value.copy()
                        doc_copy["cohort_data"][key].index = [str(idx) for idx in value.index]
                        logging.info(f"Converted Period index to strings in cohort_data[{key}]")
        
        # Create download buttons
        col1, col2, col3 = st.columns(3)
        
        # Create JSON string with custom encoder
        json_str = json.dumps(doc_copy, indent=2, cls=CustomJSONEncoder)
        
        with col1:
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="startup_report.json",
                mime="application/json"
            )
        
        with col2:
            # Convert to CSV format
            try:
                # Extract key metrics into a flat dictionary
                flat_metrics = flatten_metrics(doc)
                
                # Convert to CSV
                csv_buffer = StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow(flat_metrics.keys())
                writer.writerow(flat_metrics.values())
                
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="startup_metrics.csv",
                    mime="text/csv"
                )
            except Exception as e:
                logging.error(f"Error creating CSV: {str(e)}")
                st.error("Could not generate CSV format. Please use JSON download.")
        
        with col3:
            # Enhanced PDF generation button
            if st.button("Generate PDF Report"):
                with st.spinner("Generating enhanced PDF report..."):
                    try:
                        # Generate the enhanced PDF
                        pdf_bytes = generate_enhanced_pdf(doc_copy)
                        
                        if pdf_bytes:
                            st.download_button(
                                label="Download Enhanced PDF Report",
                                data=pdf_bytes,
                                file_name=f"{doc_copy.get('name', 'startup')}_report.pdf",
                                mime="application/pdf"
                            )
                            st.success("Enhanced PDF report generated successfully!")
                        else:
                            st.error("PDF generation failed. Please try JSON or CSV format instead.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("Please use JSON or CSV download formats instead.")
        
        # Report sections selection
        st.subheader("Customize Report Sections")
        st.write("Select sections to include in your next PDF report:")
        
        # Create columns for section selection
        section_cols = st.columns(3)
        section_options = {
            "Executive Summary": True,
            "Business Model": True,
            "Market Analysis": True,
            "Financial Projections": True,
            "Team Assessment": True,
            "Competitive Analysis": True,
            "Growth Metrics": True,
            "Risk Assessment": True,
            "Exit Strategy": True,
            "Technical Assessment": True
        }
        
        selected_sections = {}
        i = 0
        for section, default in section_options.items():
                selected_sections[section] = st.checkbox(section, value=default)
                i += 1
        
        # Generate custom report button
        if st.button("Generate Custom PDF Report"):
            with st.spinner("Generating custom PDF report..."):
                try:
                    # Generate the enhanced custom PDF
                    pdf_bytes = generate_enhanced_pdf(doc_copy, "custom", selected_sections)
                    
                    if pdf_bytes:
                        st.download_button(
                            label="Download Custom PDF Report",
                            data=pdf_bytes,
                            file_name=f"{doc_copy.get('name', 'startup')}_custom_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("Custom PDF report generated successfully!")
                    else:
                        st.error("Custom PDF generation failed. Please try JSON or CSV format instead.")
                except Exception as e:
                    st.error(f"Custom PDF generation failed: {str(e)}")
                    st.info("Please use JSON or CSV download formats instead.")
        
        # Display preview
        with st.expander("Preview Report Data"):
            try:
                st.json(json.loads(json_str))
            except Exception as e:
                logging.error(f"Error displaying JSON preview: {str(e)}")
                st.error("Could not display report preview. Please download to view.")
    
    except Exception as e:
        logging.error(f"Error in report generation: {str(e)}")
        st.error("An error occurred while preparing the report data. Please try again.")

# Add a wrapper function to connect with enhanced_pdf_generator
def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """
    Generate an enhanced PDF report with charts, graphs, and better formatting.
    This is a wrapper function that calls generate_investor_report for compatibility.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data or None if generation failed
    """
    try:
        # First, try to use the enhanced PDF generator
        from enhanced_pdf_generator import generate_enhanced_pdf as enhanced_pdf_gen
        logger = logging.getLogger("report_generator")
        
        logger.info("Using enhanced PDF generator for report generation")
        return enhanced_pdf_gen(doc, report_type, sections)
    except Exception as e:
        # If enhanced generator fails, fall back to standard implementation
        logger.warning(f"Enhanced PDF generator failed, using fallback: {e}")
        return generate_investor_report(doc, report_type, sections)
