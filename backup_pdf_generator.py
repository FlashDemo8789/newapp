"""
Investor-Grade Report Generator for FlashDNA

Built with ReportLab to produce investment banking quality reports for fundraising.
"""

import os
import io
import logging
import tempfile
import copy
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
import traceback
from pydantic import BaseModel, confloat, conint, field_validator
from typing import Optional

# Install ReportLab if not present
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, KeepTogether, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.charts.textlabels import Label
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, KeepTogether, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.charts.textlabels import Label

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("investor_report")

# Brand colors for consistent styling
BRAND_PRIMARY = colors.HexColor('#1F77B4')  # Blue
BRAND_SECONDARY = colors.HexColor('#FF7F0E')  # Orange
BRAND_SUCCESS = colors.HexColor('#2CA02C')  # Green
BRAND_WARNING = colors.HexColor('#D62728')  # Red
BRAND_NEUTRAL = colors.HexColor('#7F7F7F')  # Gray
BRAND_ACCENT = colors.HexColor('#17BECF')  # Light blue

class DocumentSchema(BaseModel):
    name: str
    camp_score: confloat(ge=0, le=100)
    success_prob: confloat(ge=0, le=100)
    runway_months: confloat(ge=0)
    monthly_revenue: float
    burn_rate: float
    ltv_cac_ratio: confloat(ge=0)
    capital_score: confloat(ge=0, le=100)
    market_score: confloat(ge=0, le=100)
    advantage_score: confloat(ge=0, le=100)
    people_score: confloat(ge=0, le=100)

    @field_validator('*', mode='before')
    def set_defaults(cls, value, info):
        if value is None:
            return info.field_info.default if info.field_info.default is not None else 0.0
        return value

class BasePDFGenerator:
    """Abstract base class for PDF generation"""
    
    def __init__(self, doc_data):
        self.doc_data = doc_data
        self._setup_core_styles()
        self.story = []
        
    def _setup_core_styles(self):
        # Consolidated style initialization
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.black,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            'CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            'CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            'CustomBodyText',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.black,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            'CustomItalic',
            parent=self.styles['Italic'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.black,
            fontName='Helvetica-Oblique'
        ))
        
        # Add metric styles
        self.styles.add(ParagraphStyle(
            'MetricValue',
            parent=self.styles['BodyText'],
            fontSize=14,
            spaceAfter=2,
            textColor=colors.black,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            'MetricLabel',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=12,
            textColor=colors.grey,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Add table styles
        self.styles.add(ParagraphStyle(
            'TableHeader',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.white,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#2C3E50')
        ))
        
        self.styles.add(ParagraphStyle(
            'TableCell',
            parent=self.styles['BodyText'],
            fontSize=9,
            textColor=colors.black,
            alignment=TA_LEFT,
            fontName='Helvetica'
        ))

class ValidatedPDFGenerator(BasePDFGenerator):
    """Adds schema validation to generation process"""
    
    def __init__(self, doc_data):
        self.validated_data = DocumentSchema(**doc_data).dict()
        super().__init__(self.validated_data)

class InvestorReportGenerator(ValidatedPDFGenerator):
    """Handles investor-specific report generation"""
    
    def __init__(self, doc_data, report_type="full", sections=None):
        super().__init__(doc_data)
        self.report_type = report_type
        self.sections = sections
        self.active_sections = self._determine_active_sections()
        self.colors = {
            'primary': colors.HexColor('#2C3E50'),  # Dark blue
            'secondary': colors.HexColor('#3498DB'),  # Light blue
            'accent': colors.HexColor('#E74C3C'),  # Red
            'success': colors.HexColor('#2ECC71'),  # Green
            'warning': colors.HexColor('#F1C40F'),  # Yellow
            'background': colors.HexColor('#F8F9FA'),  # Light grey
            'text': colors.HexColor('#2C3E50'),  # Dark text
            'light_text': colors.HexColor('#7F8C8D'),  # Light text
            'border': colors.HexColor('#BDC3C7')  # Border color
        }
        
        # Define table styles
        self.table_style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['text']),
        ])
        
        # Define alternating row style
        self.alternating_row_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['text']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['background']])
        ])
        
        # Define metric table style
        self.metric_table_style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.colors['text']),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
        
        logger.info(f"Initialized InvestorReport for {doc_data.get('name', 'Unknown Company')}")
    
    def _determine_active_sections(self):
        """Determine which sections to include in the report."""
        if self.report_type == "custom" and self.sections is not None:
            return self.sections
        else:
            # Default sections for full report
            return {
                "Executive Summary": True,
                "Business Model": True,
                "Market Analysis": True,
                "Financial Projections": True,
                "Team Assessment": True,
                "Competitive Analysis": True,
                "Growth Metrics": True,
                "Risk Assessment": True,
                "Exit Strategy": True,
                "Technical Assessment": True,
                "CAMP Details": True,
                "PMF Analysis": True,
                "Clustering Analysis": True,
                "Summary Dashboard": True
            }
            
            # For executive report, limit sections
            if self.report_type == "executive":
                for section in ["Growth Metrics", "Risk Assessment", "Technical Assessment", "PMF Analysis"]:
                    self.active_sections[section] = False
    
    def format_currency(self, value):
        """Format value as currency with proper notation for large numbers."""
        try:
            value = float(value)
            if value >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif value >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:.2f}"
        except:
            return str(value)
    
    def create_cover_page(self):
        """Create an attractive cover page."""
        try:
            # Add startup logo if available
            logo_paths = [
                "static/logo.png",
                "static/img/logo.png",
                os.path.join(os.path.dirname(__file__), "logo.png"),
                "/app/static/logo.png"
            ]
            
            logo_added = False
            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    try:
                        img = Image(logo_path, width=2*inch, height=2*inch)
                        img.hAlign = 'CENTER'
                        self.story.append(img)
                        self.story.append(Spacer(1, 0.5*inch))
                        logo_added = True
                        logger.info(f"Added logo from {logo_path}")
                        break
                    except Exception as logo_err:
                        logger.warning(f"Error adding logo from {logo_path}: {str(logo_err)}")
            
            # Add report title
            self.story.append(Paragraph("Investor Report", self.styles['CustomTitle']))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add company name in large font
            company_style = ParagraphStyle(
                'CompanyName',
                parent=self.styles['CustomTitle'],
                fontSize=36,
                leading=40,
                alignment=TA_CENTER,
                spaceAfter=30
            )
            self.story.append(Paragraph(self.doc_data.get('name', 'Startup'), company_style))
            self.story.append(Spacer(1, 0.25*inch))
            
            # Add company info
            info_style = ParagraphStyle(
                'CompanyInfo',
                parent=self.styles['Normal'],
                fontSize=14,
                leading=18,
                alignment=TA_CENTER
            )
            
            self.story.append(Paragraph(f"Sector: {self.doc_data.get('sector', 'Technology')}", info_style))
            self.story.append(Paragraph(f"Stage: {self.doc_data.get('stage', 'Growth')}", info_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add CAMP score
            camp_style = ParagraphStyle(
                'CampScore',
                parent=self.styles['CustomHeading1'],
                fontSize=20,
                leading=24,
                alignment=TA_CENTER,
                textColor=BRAND_PRIMARY
            )
            
            camp_score = self.doc_data.get('camp_score', 0)
            self.story.append(Paragraph(f"CAMP Score: {camp_score:.1f}/100", camp_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            # Add key metrics in a table
            data = [
                ["Success Probability", "Runway", "Monthly Revenue"],
                [
                    f"{self.doc_data.get('success_prob', 0):.1f}%", 
                    f"{self.doc_data.get('runway_months', 0):.1f} months",
                    self.format_currency(self.doc_data.get('monthly_revenue', 0))
                ]
            ]
            
            metrics_table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (-1, 0), BRAND_PRIMARY),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, 1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, 1), BRAND_PRIMARY),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 14),
                ('TOPPADDING', (0, 1), (-1, 1), 12),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
                ('BOX', (0, 0), (-1, 1), 1, colors.lightgrey),
                ('GRID', (0, 0), (-1, 1), 0.5, colors.lightgrey),
            ]))
            
            self.story.append(metrics_table)
            self.story.append(Spacer(1, inch))
            
            # Add date and confidentiality notice
            date_style = ParagraphStyle(
                'Date',
                parent=self.styles['Normal'],
                fontSize=12,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.darkgrey
            )
            
            self.story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", date_style))
            self.story.append(Spacer(1, 0.5*inch))
            
            confidential_style = ParagraphStyle(
                'Confidential',
                parent=self.styles['Italic'],
                fontSize=9,
                leading=11,
                alignment=TA_CENTER,
                textColor=colors.darkgrey
            )
            
            self.story.append(Paragraph("CONFIDENTIAL", confidential_style))
            self.story.append(Paragraph("This report contains confidential information about the company and is intended only for the named recipient.", confidential_style))
            
            # Add page break
            self.story.append(PageBreak())
            logger.info("Added cover page")
            
        except Exception as e:
            logger.error(f"Error creating cover page: {str(e)}\n{traceback.format_exc()}")
            # Add minimal cover page if error occurs
            self.story.append(Paragraph("Investor Report", self.styles['CustomTitle']))
            self.story.append(Spacer(1, 0.25*inch))
            self.story.append(Paragraph(self.doc_data.get('name', 'Startup'), self.styles['CustomHeading1']))
            self.story.append(PageBreak())

    def create_table_of_contents(self):
        """Create a table of contents for the report."""
        try:
            self.story.append(Paragraph("Table of Contents", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 0.2*inch))
            
            toc_sections = []
            
            if self.active_sections.get("Executive Summary", True):
                toc_sections.append("Executive Summary")
                
            if self.active_sections.get("CAMP Details", True):
                toc_sections.append("CAMP Framework Details")
                
            for section, include in self.active_sections.items():
                if include and section not in ["Executive Summary", "CAMP Details"]:
                    toc_sections.append(section)
            
            toc_style = ParagraphStyle(
                'TOC',
                parent=self.styles['Normal'],
                fontSize=12,
                leading=24  # Increased spacing between lines
            )
            
            for section in toc_sections:
                self.story.append(Paragraph(section, toc_style))
            
            self.story.append(PageBreak())
            logger.info("Added table of contents")
            
        except Exception as e:
            logger.error(f"Error creating table of contents: {str(e)}")
            # Skip TOC if error occurs
    
    def _safe_table(self, data, colWidths=None, style=None):
        """Create a table safely, ensuring data is valid."""
        try:
            # Validate data
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning("Empty or invalid table data")
                return Paragraph("Table data unavailable", self.styles['CustomBodyText'])
                
            # Ensure all data is string type
            data = [[str(cell) if cell is not None else '' for cell in row] for row in data]
                
            # Check that all rows have the same number of columns
            col_count = len(data[0])
            for row in data:
                if len(row) != col_count:
                    logger.warning(f"Inconsistent table column count: expected {col_count}, got {len(row)}")
                    # Pad or truncate to make consistent
                    if len(row) < col_count:
                        row.extend([''] * (col_count - len(row)))
                    else:
                        row = row[:col_count]
                        
            # Create the table with default style if none provided
            table = Table(data, colWidths=colWidths)
            
            # Apply style if provided
            if style:
                try:
                    table.setStyle(style)
                except Exception as style_error:
                    logger.warning(f"Error applying table style: {style_error}")
                    # Apply basic style as fallback
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('TOPPADDING', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                    ]))
                
            return table
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            return Paragraph("Error displaying table data", self.styles['CustomBodyText'])
            
    def create_executive_summary(self):
        """Create the executive summary section with detailed analysis."""
        try:
            logger.info("Adding executive summary")
            
            # Add section title
            self.story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 12))
            
            # Company Overview
            self.story.append(Paragraph("Company Overview", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            overview_text = f"""
            {self.doc_data.get('name', 'The Company')} is a {self.doc_data.get('industry', 'technology')} company 
            focused on {self.doc_data.get('value_proposition', 'delivering innovative solutions')}. 
            The company has demonstrated strong growth with a {self.doc_data.get('revenue_growth_rate', 0)}% 
            year-over-year revenue increase and has established a significant market presence.
            """
            self.story.append(Paragraph(overview_text, self.styles['CustomBodyText']))
            self.story.append(Spacer(1, 12))
            
            # Key Metrics
            self.story.append(Paragraph("Key Performance Metrics", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            # Create metrics table
            metrics_data = [
                ['Metric', 'Value', 'Industry Benchmark'],
                ['Monthly Revenue', f"${self.doc_data.get('monthly_revenue', 0):,.2f}", 'Industry Average'],
                ['Growth Rate', f"{self.doc_data.get('revenue_growth_rate', 0)}%", 'Industry Average'],
                ['LTV/CAC Ratio', f"{self.doc_data.get('ltv_cac_ratio', 0):.2f}", '>3.0'],
                ['Market Share', f"{self.doc_data.get('market_share', 0)}%", 'Industry Average'],
                ['Runway', f"{self.doc_data.get('runway_months', 0)} months", '12-18 months']
            ]
            
            metrics_table = self._safe_table(metrics_data, colWidths=[2*inch, inch, 1.5*inch], style=self.metric_table_style)
            self.story.append(metrics_table)
            self.story.append(Spacer(1, 12))
            
            # CAMP Framework Analysis
            self.story.append(Paragraph("CAMP Framework Analysis", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            camp_data = [
                ['Component', 'Score', 'Analysis'],
                ['Capital', f"{self.doc_data.get('capital_score', 0)}/100", self._get_capital_analysis()],
                ['Advantage', f"{self.doc_data.get('advantage_score', 0)}/100", self._get_advantage_analysis()],
                ['Market', f"{self.doc_data.get('market_score', 0)}/100", self._get_market_analysis()],
                ['People', f"{self.doc_data.get('people_score', 0)}/100", self._get_people_analysis()]
            ]
            
            camp_table = self._safe_table(camp_data, colWidths=[1.5*inch, inch, 3*inch], style=self.alternating_row_style)
            self.story.append(camp_table)
            self.story.append(Spacer(1, 12))
            
            # Investment Highlights
            self.story.append(Paragraph("Investment Highlights", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            highlights = [
                "Strong market position with significant growth potential",
                "Proven business model with scalable operations",
                "Experienced management team with track record of success",
                "Clear path to profitability and sustainable growth",
                "Competitive advantages in technology and market positioning"
            ]
            
            for highlight in highlights:
                self.story.append(Paragraph(f"• {highlight}", self.styles['CustomBodyText']))
            
            self.story.append(PageBreak())
            logger.info("Added executive summary")
            
        except Exception as e:
            logger.error(f"Error creating executive summary: {str(e)}")
            self.story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating executive summary content.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def _get_capital_analysis(self):
        """Generate analysis for capital component."""
        score = self.doc_data.get('capital_score', 0)
        if score >= 80:
            return "Strong financial position with healthy cash reserves and efficient capital allocation."
        elif score >= 60:
            return "Adequate capital structure with room for optimization in resource allocation."
        else:
            return "Capital position needs improvement; focus on optimizing resource allocation and cash management."
    
    def _get_advantage_analysis(self):
        """Generate analysis for advantage component."""
        score = self.doc_data.get('advantage_score', 0)
        if score >= 80:
            return "Strong competitive advantages with significant barriers to entry and unique value proposition."
        elif score >= 60:
            return "Moderate competitive advantages with some differentiation in the market."
        else:
            return "Competitive advantages need strengthening; focus on differentiation and value proposition."
    
    def _get_market_analysis(self):
        """Generate analysis for market component."""
        score = self.doc_data.get('market_score', 0)
        if score >= 80:
            return "Strong market position with significant growth potential and favorable market dynamics."
        elif score >= 60:
            return "Good market position with opportunities for growth and market expansion."
        else:
            return "Market position needs improvement; focus on market penetration and growth strategies."
    
    def _get_people_analysis(self):
        """Generate analysis for people component."""
        score = self.doc_data.get('people_score', 0)
        if score >= 80:
            return "Strong team with experienced leadership and skilled workforce."
        elif score >= 60:
            return "Competent team with potential for growth and development."
        else:
            return "Team capabilities need strengthening; focus on talent acquisition and development."
    
    def create_camp_details(self):
        """Create the CAMP framework details section."""
        try:
            if not self.active_sections.get("CAMP Details", True):
                return
                
            self.story.append(Paragraph("CAMP Framework Details", self.styles['CustomHeading1']))
            
            # Capital efficiency
            self.story.append(Paragraph("Capital Efficiency", self.styles['CustomHeading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('capital_score', 0):.1f}/100", self.styles['CustomHeading3']))
            
            capital_metrics = [
                ("Monthly Revenue", self.format_currency(self.doc_data.get('monthly_revenue', 0))),
                ("Burn Rate", self.format_currency(self.doc_data.get('burn_rate', 0))),
                ("Runway", f"{self.doc_data.get('runway_months', 0):.1f} months")
            ]
            
            self.add_metric_row(capital_metrics)
            
            capital_metrics2 = [
                ("Gross Margin", f"{self.doc_data.get('gross_margin_percent', 0):.1f}%"),
                ("LTV:CAC Ratio", f"{self.doc_data.get('ltv_cac_ratio', 0):.2f}"),
                ("CAC", self.format_currency(self.doc_data.get('customer_acquisition_cost', 0)))
            ]
            
            self.add_metric_row(capital_metrics2)
            
            capital_metrics3 = [
                ("Annual Recurring Revenue", self.format_currency(self.doc_data.get('annual_recurring_revenue', 0))),
                ("Operating Margin", f"{self.doc_data.get('operating_margin_percent', 0):.1f}%"),
                ("Debt Ratio", f"{self.doc_data.get('debt_ratio', 0):.2f}")
            ]
            
            self.add_metric_row(capital_metrics3)
            
            capital_metrics4 = [
                ("Avg Revenue Per User", self.format_currency(self.doc_data.get('avg_revenue_per_user', 0))),
                ("LTV", self.format_currency(self.doc_data.get('ltv', 0))),
                ("Cash on Hand", self.format_currency(self.doc_data.get('current_cash', 0)))
            ]
            
            self.add_metric_row(capital_metrics4)
            
            # Add cash flow chart if available
            cash_flow = self.doc_data.get("cash_flow", [])
            if cash_flow:
                self.add_line_chart(
                    list(range(len(cash_flow))),
                    cash_flow,
                    "Cash Flow Projection",
                    "Month",
                    "Cash ($)"
                )
            
            # Market dynamics
            self.story.append(Paragraph("Market Dynamics", self.styles['CustomHeading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('market_score', 0):.1f}/100", self.styles['CustomHeading3']))
            
            market_metrics = [
                ("Market Size", self.format_currency(self.doc_data.get('market_size', 0))),
                ("Market Growth Rate", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/yr"),
                ("Market Share", f"{self.doc_data.get('market_share', 0):.2f}%")
            ]
            
            self.add_metric_row(market_metrics)
            
            market_metrics2 = [
                ("User Growth", f"{self.doc_data.get('user_growth_rate', 0):.1f}%/mo"),
                ("Revenue Growth", f"{self.doc_data.get('revenue_growth_rate', 0):.1f}%/mo"),
                ("Churn Rate", f"{self.doc_data.get('churn_rate', 0):.1f}%/mo")
            ]
            
            self.add_metric_row(market_metrics2)
            
            market_metrics3 = [
                ("Net Retention Rate", f"{self.doc_data.get('net_retention_rate', 0):.1f}%"),
                ("Community Growth", f"{self.doc_data.get('community_growth_rate', 0):.1f}%"),
                ("Viral Coefficient", f"{self.doc_data.get('viral_coefficient', 0):.2f}")
            ]
            
            self.add_metric_row(market_metrics3)
            
            market_metrics4 = [
                ("Category Leadership", f"{self.doc_data.get('category_leadership_score', 0):.1f}/100"),
                ("Current Users", f"{self.doc_data.get('current_users', 0):,}"),
                ("Conversion Rate", f"{self.doc_data.get('conversion_rate', 0):.1f}%")
            ]
            
            self.add_metric_row(market_metrics4)
            
            # Advantage moat
            self.story.append(Paragraph("Advantage Moat", self.styles['CustomHeading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('advantage_score', 0):.1f}/100", self.styles['CustomHeading3']))
            
            advantage_metrics = [
                ("Technical Innovation", f"{self.doc_data.get('technical_innovation_score', 0):.1f}/100"),
                ("Product Maturity", f"{self.doc_data.get('product_maturity_score', 0):.1f}/100"),
                ("Moat Score", f"{self.doc_data.get('moat_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(advantage_metrics)
            
            advantage_metrics2 = [
                ("Patent Count", f"{self.doc_data.get('patent_count', 0)}"),
                ("API Integrations", f"{self.doc_data.get('api_integrations_count', 0)}"),
                ("Scalability", f"{self.doc_data.get('scalability_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(advantage_metrics2)
            
            advantage_metrics3 = [
                ("Security Score", f"{self.doc_data.get('product_security_score', 0):.1f}/100"),
                ("Technical Debt", f"{self.doc_data.get('technical_debt_score', 0):.1f}/100"),
                ("Business Model Strength", f"{self.doc_data.get('business_model_strength', 0):.1f}/100")
            ]
            
            self.add_metric_row(advantage_metrics3)
            
            advantage_metrics4 = [
                ("Data Moat Strength", f"{self.doc_data.get('data_moat_strength', 0):.1f}/100"),
                ("Algorithm Uniqueness", f"{self.doc_data.get('algorithm_uniqueness', 0):.1f}/100"),
                ("Category Leadership", f"{self.doc_data.get('category_leadership_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(advantage_metrics4)
            
            # People & performance
            self.story.append(Paragraph("People & Performance", self.styles['CustomHeading2']))
            self.story.append(Paragraph(f"Score: {self.doc_data.get('people_score', 0):.1f}/100", self.styles['CustomHeading3']))
            
            people_metrics = [
                ("Team Score", f"{self.doc_data.get('team_score', 0):.1f}/100"),
                ("Founder Experience", f"{self.doc_data.get('founder_domain_exp_yrs', 0)} years"),
                ("Previous Exits", f"{self.doc_data.get('founder_exits', 0)}")
            ]
            
            self.add_metric_row(people_metrics)
            
            people_metrics2 = [
                ("Team Size", f"{self.doc_data.get('employee_count', 0)} employees"),
                ("Tech Talent Ratio", f"{self.doc_data.get('tech_talent_ratio', 0)*100:.1f}%"),
                ("Team Diversity", f"{self.doc_data.get('founder_diversity_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(people_metrics2)
            
            people_metrics3 = [
                ("Employee Turnover", f"{self.doc_data.get('employee_turnover_rate', 0):.1f}%"),
                ("NPS Score", f"{self.doc_data.get('nps_score', 0)}"),
                ("Support SLA Hit %", f"{self.doc_data.get('support_ticket_sla_percent', 0):.1f}%")
            ]
            
            self.add_metric_row(people_metrics3)
            
            people_metrics4 = [
                ("Management Satisfaction", f"{self.doc_data.get('management_satisfaction_score', 0):.1f}/100"),
                ("Support Ticket Volume", f"{self.doc_data.get('support_ticket_volume', 0):,}/mo"),
                ("Leadership", f"CTO: {'Yes' if self.doc_data.get('has_cto', False) else 'No'}, CMO: {'Yes' if self.doc_data.get('has_cmo', False) else 'No'}, CFO: {'Yes' if self.doc_data.get('has_cfo', False) else 'No'}")
            ]
            
            self.add_metric_row(people_metrics4)
            
            self.story.append(PageBreak())
            logger.info("Added CAMP details")
            
        except Exception as e:
            logger.error(f"Error creating CAMP details: {str(e)}\n{traceback.format_exc()}")
            # Add basic CAMP details if error occurs
            self.story.append(Paragraph("CAMP Framework Details", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating full CAMP details.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def create_business_model_section(self):
        """Create the business model section with detailed analysis."""
        try:
            logger.info("Adding business model section")
            
            # Add section title
            self.story.append(Paragraph("Business Model Analysis", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 12))
            
            # Revenue Model
            self.story.append(Paragraph("Revenue Model", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            revenue_text = f"""
            The company operates on a {self.doc_data.get('revenue_model', 'subscription-based')} revenue model 
            with an average monthly revenue of ${self.doc_data.get('monthly_revenue', 0):,.2f}. The business 
            demonstrates strong unit economics with a customer lifetime value (LTV) to customer acquisition 
            cost (CAC) ratio of {self.doc_data.get('ltv_cac_ratio', 0):.2f}, indicating efficient customer 
            acquisition and retention strategies.
            """
            self.story.append(Paragraph(revenue_text, self.styles['CustomBodyText']))
            self.story.append(Spacer(1, 12))
            
            # Key Metrics
            self.story.append(Paragraph("Key Business Metrics", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            metrics_data = [
                ['Metric', 'Value', 'Industry Benchmark'],
                ['Monthly Revenue', f"${self.doc_data.get('monthly_revenue', 0):,.2f}", 'Industry Average'],
                ['Growth Rate', f"{self.doc_data.get('revenue_growth_rate', 0)}%", 'Industry Average'],
                ['LTV/CAC Ratio', f"{self.doc_data.get('ltv_cac_ratio', 0):.2f}", '>3.0'],
                ['Gross Margin', f"{self.doc_data.get('gross_margin_percent', 0)}%", 'Industry Average'],
                ['Operating Margin', f"{self.doc_data.get('operating_margin_percent', 0)}%", 'Industry Average']
            ]
            
            metrics_table = self._safe_table(metrics_data, colWidths=[2*inch, inch, 1.5*inch], style=self.metric_table_style)
            self.story.append(metrics_table)
            self.story.append(Spacer(1, 12))
            
            # LTV vs CAC Analysis
            self.story.append(Paragraph("LTV vs CAC Analysis", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            # Add bar chart for LTV vs CAC
            ltv = self.doc_data.get('ltv', 0)
            cac = self.doc_data.get('cac', 0)
            
            if ltv > 0 and cac > 0:
                self.add_bar_chart(
                    ['LTV', 'CAC'],
                    [ltv, cac],
                    'LTV vs CAC Comparison',
                    colors=[self.colors['success'], self.colors['warning']]
                )
            
            # Cost Structure
            self.story.append(Paragraph("Cost Structure", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            cost_data = [
                ['Cost Category', 'Monthly Cost', 'Percentage of Revenue'],
                ['Customer Acquisition', f"${self.doc_data.get('cac', 0):,.2f}", f"{self.doc_data.get('cac_percentage', 0)}%"],
                ['Operations', f"${self.doc_data.get('operating_costs', 0):,.2f}", f"{self.doc_data.get('operating_costs_percentage', 0)}%"],
                ['Research & Development', f"${self.doc_data.get('rd_costs', 0):,.2f}", f"{self.doc_data.get('rd_costs_percentage', 0)}%"],
                ['General & Administrative', f"${self.doc_data.get('ga_costs', 0):,.2f}", f"{self.doc_data.get('ga_costs_percentage', 0)}%"]
            ]
            
            cost_table = self._safe_table(cost_data, colWidths=[2*inch, inch, 1.5*inch], style=self.alternating_row_style)
            self.story.append(cost_table)
            self.story.append(Spacer(1, 12))
            
            # Growth Strategy
            self.story.append(Paragraph("Growth Strategy", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            growth_text = f"""
            The company's growth strategy focuses on {self.doc_data.get('growth_strategy', 'market expansion and customer acquisition')}. 
            With a current market share of {self.doc_data.get('market_share', 0)}% in a ${self.doc_data.get('market_size', 0):,.2f} 
            market, there is significant opportunity for growth. The company plans to achieve this through:
            
            • {self.doc_data.get('growth_initiative1', 'Expanding into new market segments')}
            • {self.doc_data.get('growth_initiative2', 'Enhancing product offerings')}
            • {self.doc_data.get('growth_initiative3', 'Strengthening customer relationships')}
            • {self.doc_data.get('growth_initiative4', 'Optimizing operational efficiency')}
            """
            self.story.append(Paragraph(growth_text, self.styles['CustomBodyText']))
            
            self.story.append(PageBreak())
            logger.info("Added business model section")
            
        except Exception as e:
            logger.error(f"Error creating business model section: {str(e)}")
            self.story.append(Paragraph("Business Model Analysis", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating business model content.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def create_market_analysis_section(self):
        """Create the market analysis section."""
        try:
            if not self.active_sections.get("Market Analysis", True):
                return
                
            self.story.append(Paragraph("Market Analysis", self.styles['CustomHeading1']))
            
            # Market metrics
            market_metrics = [
                ("Market Size", self.format_currency(self.doc_data.get('market_size', 0))),
                ("Market Growth Rate", f"{self.doc_data.get('market_growth_rate', 0):.1f}%/yr"),
                ("Market Share", f"{self.doc_data.get('market_share', 0):.2f}%")
            ]
            
            self.add_metric_row(market_metrics)
            
            # Market breakdown
            market_trends = self.doc_data.get("market_trends", {})
            if isinstance(market_trends, dict) and "trends" in market_trends:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Market Trends", self.styles['CustomHeading2']))
                
                trends = market_trends.get("trends", [])
                trend_items = []
                
                for trend in trends:
                    if isinstance(trend, dict):
                        trend_text = f"{trend.get('name', '')}: {trend.get('description', '')}"
                        trend_items.append(ListItem(Paragraph(trend_text, self.styles['CustomBodyText'])))
                    elif isinstance(trend, str):
                        trend_items.append(ListItem(Paragraph(trend, self.styles['CustomBodyText'])))
                
                if trend_items:
                    self.story.append(ListFlowable(trend_items, bulletType='bullet', start=''))
            
            # Competitive position
            competitive_pos = self.doc_data.get("competitive_positioning", {})
            if competitive_pos:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Competitive Position", self.styles['CustomHeading2']))
                
                position = competitive_pos.get("position", "challenger")
                self.story.append(Paragraph(f"Current Position: {position.capitalize()}", self.styles['CustomBodyText']))
                
                # Add competitive advantages
                advantages = competitive_pos.get("advantages", [])
                if advantages:
                    self.story.append(Spacer(1, 0.1*inch))
                    self.story.append(Paragraph("Competitive Advantages", self.styles['CustomHeading3']))
                    
                    advantages_names = []
                    advantages_scores = []
                    
                    for adv in advantages:
                        if isinstance(adv, dict):
                            advantages_names.append(adv.get("name", ""))
                            advantages_scores.append(adv.get("score", 0))
                    
                    if advantages_names and advantages_scores:
                        self.add_bar_chart(
                            advantages_names,
                            advantages_scores,
                            "Competitive Advantages",
                            "Score"
                        )
            
            # PMF Analysis if available
            pmf = self.doc_data.get("pmf_analysis", {})
            if pmf:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Product-Market Fit", self.styles['CustomHeading2']))
                
                pmf_score = pmf.get("pmf_score", 0)
                pmf_stage = pmf.get("stage", "")
                
                pmf_metrics = [
                    ("PMF Score", f"{pmf_score:.1f}/100"),
                    ("Stage", f"{pmf_stage}"),
                    ("Retention Rate", f"{pmf.get('retention_rate', 0):.1f}%")
                ]
                
                self.add_metric_row(pmf_metrics)
            
            # Market penetration chart
            market_penetration = self.doc_data.get("market_penetration", {})
            if isinstance(market_penetration, dict) and "timeline" in market_penetration and "penetration" in market_penetration:
                timeline = market_penetration.get("timeline", [])
                penetration = [p * 100 for p in market_penetration.get("penetration", [])]  # Convert to percentage
                
                if timeline and penetration and len(timeline) == len(penetration):
                    self.story.append(Spacer(1, 0.2*inch))
                    self.add_line_chart(
                        timeline,
                        penetration,
                        "Market Penetration Projection",
                        "Month",
                        "Penetration (%)"
                    )
            
            self.story.append(PageBreak())
            logger.info("Added market analysis section")
            
        except Exception as e:
            logger.error(f"Error creating market analysis section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Market Analysis", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating full market analysis section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def create_team_assessment_section(self):
        """Create the team assessment section."""
        try:
            if not self.active_sections.get("Team Assessment", True):
                return
                
            self.story.append(Paragraph("Team Assessment", self.styles['CustomHeading1']))
            
            # Team metrics
            team_metrics = [
                ("Team Score", f"{self.doc_data.get('team_score', 0):.1f}/100"),
                ("Founder Experience", f"{self.doc_data.get('founder_domain_exp_yrs', 0)} years"),
                ("Previous Exits", f"{self.doc_data.get('founder_exits', 0)}")
            ]
            
            self.add_metric_row(team_metrics)
            
            team_metrics2 = [
                ("Team Size", f"{self.doc_data.get('employee_count', 0)} employees"),
                ("Tech Talent Ratio", f"{self.doc_data.get('tech_talent_ratio', 0)*100:.1f}%"),
                ("Team Diversity", f"{self.doc_data.get('founder_diversity_score', 0):.1f}/100")
            ]
            
            self.add_metric_row(team_metrics2)
            
            # Leadership team
            self.story.append(Spacer(1, 0.2*inch))
            self.story.append(Paragraph("Leadership Team", self.styles['CustomHeading2']))
            
            leadership = {
                "CEO": True,  # Assumed always present
                "CTO": self.doc_data.get("has_cto", False),
                "CMO": self.doc_data.get("has_cmo", False),
                "CFO": self.doc_data.get("has_cfo", False)
            }
            
            # Create a visual representation of the leadership team
            leaders = list(leadership.keys())
            status = [1 if v else 0 for v in leadership.values()]
            
            if leaders and status and len(leaders) == len(status):
                self.add_bar_chart(
                    leaders,
                    status,
                    "Leadership Positions",
                    "Present (1) / Absent (0)",
                    ['#2CA02C' if s else '#D62728' for s in status]
                )
            
            # Execution risk
            execution_risk = self.doc_data.get("execution_risk", {})
            if isinstance(execution_risk, dict) and "risk_factors" in execution_risk:
                risk_factors = execution_risk.get("risk_factors", {})
                
                if risk_factors:
                    self.story.append(Spacer(1, 0.2*inch))
                    self.story.append(Paragraph("Execution Risk Factors", self.styles['CustomHeading2']))
                    
                    factors = list(risk_factors.keys())
                    scores = list(risk_factors.values())
                    
                    if factors and scores and len(factors) == len(scores):
                        self.add_bar_chart(
                            factors,
                            scores,
                            "Risk Factors",
                            "Risk Level"
                        )
            
            self.story.append(PageBreak())
            logger.info("Added team assessment section")
        
        except Exception as e:
            logger.error(f"Error creating team assessment section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Team Assessment", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating full team assessment section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def create_competitive_analysis(self):
        """Create the competitive analysis section."""
        try:
            if not self.active_sections.get("Competitive Analysis", True):
                return
                
            self.story.append(Paragraph("Competitive Analysis", self.styles['CustomHeading1']))
            
            # Competitors list
            competitors = self.doc_data.get("competitors", [])
            if competitors and all(isinstance(comp, dict) for comp in competitors):
                self.story.append(Paragraph("Key Competitors", self.styles['CustomHeading2']))
                
                # Prepare header and data for the table
                headers = ["Competitor", "Funding", "Founded", "Threat Level"]
                data = [headers]
                
                for comp in competitors:
                    row = [
                        comp.get("name", ""),
                        self.format_currency(comp.get("funding", 0)),
                        comp.get("founded", ""),
                        comp.get("threat_level", "Medium")
                    ]
                    data.append(row)
                
                # Create the table
                competitors_table = self._safe_table(data, colWidths=[2*inch, inch, inch, 1.5*inch], style=self.table_style)
                
                # Add alternating row style
                if isinstance(competitors_table, Table):
                    competitors_table.setStyle(self.alternating_row_style)
                
                self.story.append(competitors_table)
                self.story.append(Spacer(1, 0.2*inch))
            
            # Competitive positioning
            positioning = self.doc_data.get("competitive_positioning", {})
            if positioning:
                self.story.append(Paragraph("Competitive Positioning", self.styles['CustomHeading2']))
                
                # Extract positioning data
                dimensions = positioning.get("dimensions", [])
                company_position = positioning.get("company_position", {})
                competitor_positions = positioning.get("competitor_positions", {})
                
                if dimensions and company_position and competitor_positions and len(dimensions) >= 2:
                    # Add description of positioning
                    position_text = f"Current competitive position: {positioning.get('position', 'Challenger').capitalize()}"
                    self.story.append(Paragraph(position_text, self.styles['CustomBodyText']))
                    self.story.append(Spacer(1, 0.1*inch))
                    
                    # Create a competitor comparison table
                    if len(dimensions) >= 2:
                        x_dim = dimensions[0]
                        y_dim = dimensions[1]
                        
                        headers = ["Company", x_dim, y_dim]
                        data = [headers]
                        
                        # Add company data
                        company_x = company_position.get(x_dim, 50)
                        company_y = company_position.get(y_dim, 50)
                        data.append([self.doc_data.get('name', 'Startup'), f"{company_x:.1f}", f"{company_y:.1f}"])
                        
                        # Add competitor data
                        for comp_name, comp_pos in competitor_positions.items():
                            comp_x = comp_pos.get(x_dim, 50)
                            comp_y = comp_pos.get(y_dim, 50)
                            data.append([comp_name, f"{comp_x:.1f}", f"{comp_y:.1f}"])
                        
                        # Create the table
                        position_table = self._safe_table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=self.table_style)
                        
                        # Add alternating row style
                        if isinstance(position_table, Table):
                            position_table.setStyle(self.alternating_row_style)
                        
                        self.story.append(position_table)
            
            # Network effects
            network = self.doc_data.get("network_analysis", {})
            if network:
                self.story.append(Spacer(1, 0.2*inch))
                self.story.append(Paragraph("Network Effects Analysis", self.styles['CustomHeading2']))
                
                # Network effect score
                ne_score = network.get("network_effect_score", 0)
                self.story.append(Paragraph(f"Network Effect Score: {ne_score:.1f}/100", self.styles['CustomBodyText']))
                
                # Network effect types
                ne_types = network.get("network_effect_types", {})
                
                if ne_types:
                    types = list(ne_types.keys())
                    scores = list(ne_types.values())
                    
                    if types and scores and len(types) == len(scores):
                        self.add_bar_chart(
                            types,
                            scores,
                            "Network Effect Strength by Type",
                            "Strength"
                        )
            
            self.story.append(PageBreak())
            logger.info("Added competitive analysis section")
            
        except Exception as e:
            logger.error(f"Error creating competitive analysis section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Competitive Analysis", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating full competitive analysis section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())

    def add_remaining_sections(self):
        """Add any remaining sections based on active_sections configuration."""
        try:
            # Forecast section
            if self.active_sections.get("Growth Metrics", True):
                self.story.append(Paragraph("Growth Metrics & Forecasts", self.styles['CustomHeading1']))
                
                # User growth projection
                sys_dynamics = self.doc_data.get("system_dynamics", {})
                if isinstance(sys_dynamics, dict) and "users" in sys_dynamics:
                    users = sys_dynamics.get("users", [])
                    months = list(range(len(users)))
                    
                    if users and months:
                        self.story.append(Paragraph("User Growth Projection", self.styles['CustomHeading2']))
                        self.add_line_chart(
                            months, 
                            users, 
                            "Projected User Growth", 
                            "Months", 
                            "Users"
                        )
                
                # Monte Carlo simulation
                monte_carlo = self.doc_data.get("monte_carlo", {})
                if monte_carlo and "user_projections" in monte_carlo:
                    self.story.append(Paragraph("Monte Carlo Simulation", self.styles['CustomHeading2']))
                    
                    # Extract key metrics
                    success_prob = monte_carlo.get("success_probability", 0)
                    median_runway = monte_carlo.get("median_runway_months", 0)
                    
                    # Add metrics
                    mc_metrics = [
                        ("Success Probability", f"{success_prob:.1f}%"),
                        ("Median Runway", f"{median_runway:.1f} months"),
                        ("Simulations", f"{monte_carlo.get('simulation_count', 0)}")
                    ]
                    
                    self.add_metric_row(mc_metrics)
                
                self.story.append(PageBreak())
                logger.info("Added growth metrics section")
            
            # Risk assessment
            if self.active_sections.get("Risk Assessment", True):
                self.create_risk_assessment()
                
            # Exit strategy
            if self.active_sections.get("Exit Strategy", True):
                self.story.append(Paragraph("Exit Strategy", self.styles['CustomHeading1']))
                
                exit_analysis = self.doc_data.get("exit_path_analysis", {})
                exit_recs = self.doc_data.get("exit_recommendations", {})
                
                if exit_analysis or exit_recs:
                    # Top metrics and optimal path
                    optimal_path = exit_recs.get("optimal_path", "")
                    readiness = exit_recs.get("readiness", 0)
                    
                    self.story.append(Paragraph(f"Exit Readiness: {readiness:.1f}/100", self.styles['CustomBodyText']))
                    
                    if optimal_path:
                        self.story.append(Paragraph(f"Optimal Exit Path: {exit_recs.get('path_details', {}).get('description', optimal_path)}", self.styles['CustomBodyText']))
                    
                    # Exit timeline
                    timeline = exit_recs.get("timeline", {})
                    if timeline:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Exit Timeline", self.styles['CustomHeading2']))
                        
                        timeline_metrics = [
                            ("Years to Exit", f"{timeline.get('years_to_exit', 0):.1f}"),
                            ("Exit Year", f"{timeline.get('exit_year', 0)}"),
                            ("Exit Valuation", self.format_currency(timeline.get('exit_valuation', 0)))
                        ]
                        
                        self.add_metric_row(timeline_metrics)
                    
                    # Exit scenarios
                    scenarios = exit_analysis.get("scenarios", [])
                    if scenarios and all(isinstance(s, dict) for s in scenarios):
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Exit Path Scenarios", self.styles['CustomHeading2']))
                        
                        # Create scenarios table
                        headers = ["Exit Path", "Valuation", "Probability", "Time to Exit"]
                        data = [headers]
                        
                        for scenario in scenarios:
                            row = [
                                scenario.get("path_name", ""),
                                self.format_currency(scenario.get("exit_valuation", 0)),
                                f"{scenario.get('probability', 0)*100:.1f}%",
                                f"{scenario.get('time_to_exit', 0):.1f} years"
                            ]
                            data.append(row)
                        
                        # Create the table
                        scenario_table = self._safe_table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch], style=self.table_style)
                        
                        # Add alternating row style
                        if isinstance(scenario_table, Table):
                            scenario_table.setStyle(self.alternating_row_style)
                        
                        self.story.append(scenario_table)
                
                self.story.append(PageBreak())
                logger.info("Added exit strategy section")
            
            # Technical assessment
            if self.active_sections.get("Technical Assessment", True):
                self.story.append(Paragraph("Technical Assessment", self.styles['CustomHeading1']))
                
                tech_assessment = self.doc_data.get("tech_assessment", {})
                if tech_assessment:
                    # Overall tech score
                    tech_score = tech_assessment.get("overall_score", 0)
                    self.story.append(Paragraph(f"Technical Assessment Score: {tech_score:.1f}/100", self.styles['CustomBodyText']))
                    
                    # Component scores
                    scores = tech_assessment.get("scores", {})
                    if scores:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Component Scores", self.styles['CustomHeading2']))
                        
                        categories = list(scores.keys())
                        values = list(scores.values())
                        
                        if categories and values:
                            self.add_bar_chart(
                                categories,
                                values,
                                "Technical Component Scores",
                                "Score"
                            )
                    
                    # Tech stack
                    tech_stack = tech_assessment.get("tech_stack", {})
                    if tech_stack:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Technology Stack", self.styles['CustomHeading2']))
                        
                        stack_categories = list(tech_stack.keys())
                        
                        for category in stack_categories:
                            self.story.append(Paragraph(category, self.styles['CustomHeading3']))
                            
                            technologies = tech_stack[category]
                            
                            if isinstance(technologies, list):
                                tech_text = ", ".join(technologies)
                                self.story.append(Paragraph(tech_text, self.styles['CustomBodyText']))
                            elif isinstance(technologies, dict):
                                tech_items = []
                                for tech, details in technologies.items():
                                    tech_items.append(ListItem(Paragraph(f"{tech}: {details}", self.styles['CustomBodyText'])))
                                
                                if tech_items:
                                    self.story.append(ListFlowable(tech_items, bulletType='bullet', start=''))
                    
                    # Recommendations
                    recommendations = tech_assessment.get("recommendations", [])
                    if recommendations:
                        self.story.append(Spacer(1, 0.2*inch))
                        self.story.append(Paragraph("Technical Recommendations", self.styles['CustomHeading2']))
                        
                        rec_items = []
                        for rec in recommendations:
                            rec_items.append(ListItem(Paragraph(rec, self.styles['CustomBodyText'])))
                        
                        if rec_items:
                            self.story.append(ListFlowable(rec_items, bulletType='bullet', start=''))
                
                self.story.append(PageBreak())
                logger.info("Added technical assessment section")
            
            # Clustering Analysis Section
            if self.active_sections.get("Clustering Analysis", True):
                self.create_clustering_analysis_section()
                
            # Summary Dashboard Section
            if self.active_sections.get("Summary Dashboard", True):
                self.create_summary_dashboard_section()
                
        except Exception as e:
            logger.error(f"Error adding remaining sections: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Additional Sections", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating additional sections.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def add_metric_row(self, metrics):
        """Add a row of metrics with equal styling."""
        try:
            # Calculate column widths
            col_width = 6.5 * inch / len(metrics)
            
            # Create data for the table
            labels = [m[0] for m in metrics]
            values = [m[1] for m in metrics]
            
            data = [labels, values]
            
            # Create the table
            table = Table(data, colWidths=[col_width] * len(metrics))
            
            # Style the table
            table.setStyle(TableStyle([
                # Headers (labels)
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                
                # Values
                ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
                ('TEXTCOLOR', (0, 1), (-1, 1), BRAND_PRIMARY),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 14),
                ('TOPPADDING', (0, 1), (-1, 1), 6),
                
                # Table styling
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            self.story.append(table)
            self.story.append(Spacer(1, 0.3*inch))
            
        except Exception as e:
            logger.error(f"Error adding metric row: {str(e)}")
            # Fallback to simple format
            for label, value in metrics:
                self.story.append(Paragraph(f"{label}: {value}", self.styles['CustomBodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_camp_radar_chart(self):
        """Add a CAMP framework radar chart using matplotlib."""
        try:
            # Get CAMP scores
            camp_scores = [
                self.doc_data.get('capital_score', 0),
                self.doc_data.get('market_score', 0),
                self.doc_data.get('advantage_score', 0),
                self.doc_data.get('people_score', 0)
            ]
            
            categories = ['Capital', 'Market', 'Advantage', 'People']
            
            # Create radar chart with matplotlib
            plt.clf()
            plt.close('all')
            
            # Increase figure size and DPI for better quality
            fig = plt.figure(figsize=(8, 8), dpi=300)
            ax = fig.add_subplot(111, polar=True)
            
            # Ensure the plot forms a complete circle by appending the first value at the end
            values = np.array(camp_scores + [camp_scores[0]])
            cat_labels = categories + [categories[0]]
            
            # Compute angle for each category (in radians)
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            angles = np.append(angles, angles[0])
            
            # Draw the chart with enhanced styling
            ax.plot(angles, values, 'o-', linewidth=3, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # Set category labels with larger font
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            
            # Set radial limits
            ax.set_ylim(0, 100)
            
            # Add gridlines
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set title with larger font
            plt.title("CAMP Framework Scores", size=16, color='#1f77b4', y=1.1)
            
            # Save to BytesIO with high quality settings
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.1)
            plt.close()
            
            # Create image and add to story with larger dimensions
            img = Image(buf)
            img.drawHeight = 4*inch
            img.drawWidth = 4*inch
            img.hAlign = 'CENTER'
            
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
            logger.info("Added CAMP radar chart")
            
        except Exception as e:
            logger.error(f"Error creating CAMP radar chart: {str(e)}\n{traceback.format_exc()}")
            # Skip chart if error occurs
            self.story.append(Paragraph("CAMP Framework Visualization", self.styles['CustomHeading3']))
            self.story.append(Paragraph("Error generating CAMP radar chart.", self.styles['CustomBodyText']))
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_bar_chart(self, categories, values, title="", ylabel="Value", colors=None):
        """Add a bar chart to the report."""
        try:
            # Create figure with increased size and DPI
            plt.figure(figsize=(8, 5), dpi=300)
            
            # Generate colors if not provided
            if colors is None:
                colors = ['#2C3E50', '#3498DB', '#E74C3C']  # Default colors in hex format
            elif isinstance(colors, (list, tuple)):
                # Convert reportlab colors to hex if needed
                colors = [color.hexval() if hasattr(color, 'hexval') else color for color in colors]
            
            # Create bar chart
            bars = plt.bar(categories, values, color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10)
            
            # Customize chart
            plt.title(title, fontsize=14, pad=20)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add image to story
            buf.seek(0)
            self.story.append(Image(buf, width=6.5*inch, height=3.5*inch))
            self.story.append(Spacer(1, 0.5*inch))
            
            logger.info(f"Added bar chart: {title}")
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            self.story.append(Paragraph(f"Error displaying chart: {title}", self.styles['CustomBodyText']))

    def add_line_chart(self, x_data, y_data, title="", xlabel="", ylabel="", color=None):
        """Add a line chart to the report."""
        try:
            plt.figure(figsize=(8, 5), dpi=300)
            
            # Convert color to hex if it's a reportlab color
            if color and hasattr(color, 'hexval'):
                color = color.hexval()
            elif color is None:
                color = '#2C3E50'  # Default color in hex format
                
            ax = plt.gca()
            ax.plot(x_data, y_data, marker='o', color=color, linewidth=2)
            
            # Customize chart
            plt.title(title, fontsize=14, pad=20)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis for financial metrics
            if "Revenue" in ylabel or "Cash" in ylabel:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', quality=100)
            plt.close()
            
            # Add image to story
            buf.seek(0)
            self.story.append(Image(buf, width=6.5*inch, height=3.5*inch))
            self.story.append(Spacer(1, 0.5*inch))
            
            logger.info(f"Added line chart: {title}")
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            self.story.append(Paragraph(f"Error displaying chart: {title}", self.styles['CustomBodyText']))
    
    def generate_report(self):
        """Generate the complete report."""
        try:
            # Create a buffer for the PDF
            buffer = io.BytesIO()
            
            # Create the document with enhanced settings
            doc = SimpleDocTemplate(
                buffer, 
                pagesize=letter,
                leftMargin=0.5*inch,
                rightMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch,
                allowSplitting=0,
                title=self.doc_data.get('name', 'Investor Report'),
                author='FlashDNA',
                subject='Investor Report',
                creator='FlashDNA Report Generator'
            )
            
            # Add cover page
            self.create_cover_page()
            
            # Add table of contents
            self.create_table_of_contents()
            
            # Add executive summary
            self.create_executive_summary()
            
            # Add CAMP details
            self.create_camp_details()
            
            # Add business model section
            self.create_business_model_section()
            
            # Add market analysis section
            self.create_market_analysis_section()
            
            # Add team assessment section
            self.create_team_assessment_section()
            
            # Add competitive analysis
            self.create_competitive_analysis()
            
            # Add remaining sections
            self.add_remaining_sections()
            
            # Build the document with enhanced settings
            doc.build(self.story, onFirstPage=self._on_first_page, onLaterPages=self._on_later_pages)
            
            # Get the PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            logger.info("PDF report generation completed successfully")
            return pdf_data
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}\n{traceback.format_exc()}")
            return generate_emergency_pdf(self.doc_data)

    def _on_first_page(self, canvas, doc):
        """Custom first page handler."""
        canvas.saveState()
        canvas.setTitle(self.doc_data.get('name', 'Investor Report'))
        canvas.restoreState()

    def _on_later_pages(self, canvas, doc):
        """Custom later pages handler."""
        canvas.saveState()
        canvas.setTitle(self.doc_data.get('name', 'Investor Report'))
        canvas.restoreState()

    def create_risk_assessment(self):
        """Create the risk assessment section."""
        try:
            if not self.active_sections.get("Risk Assessment", True):
                return
                
            self.story.append(Paragraph("Risk Assessment", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 12))
            
            # Extract risk factors
            risk_factors = self.doc_data.get("risk_factors", {})
            if not risk_factors:
                # Generate comprehensive default risk factors
                risk_factors = {
                    "Market Risk": {
                        "severity": 7.5,
                        "mitigation": "Diversify revenue streams and expand market reach",
                        "description": "Risk of market saturation or changing market conditions",
                        "impact": "High impact on revenue growth and market share"
                    },
                    "Competitive Risk": {
                        "severity": 6.0,
                        "mitigation": "Strengthen competitive advantages and IP protection",
                        "description": "Threat from existing and potential competitors",
                        "impact": "Medium to high impact on market position"
                    },
                    "Operational Risk": {
                        "severity": 5.5,
                        "mitigation": "Implement robust operational processes and controls",
                        "description": "Risk of operational inefficiencies or failures",
                        "impact": "Medium impact on business continuity"
                    },
                    "Financial Risk": {
                        "severity": 6.5,
                        "mitigation": "Maintain healthy cash reserves and monitor burn rate",
                        "description": "Risk of financial instability or cash flow issues",
                        "impact": "High impact on business sustainability"
                    },
                    "Regulatory Risk": {
                        "severity": 4.0,
                        "mitigation": "Stay compliant with industry regulations and standards",
                        "description": "Risk of regulatory changes or non-compliance",
                        "impact": "Medium impact on operations"
                    }
                }
            
            # Create detailed risk table
            headers = ["Risk Factor", "Severity", "Description", "Impact", "Mitigation Strategy"]
            data = [headers]
            
            for factor, details in risk_factors.items():
                if isinstance(details, dict):
                    severity = details.get('severity', 0)
                    description = details.get('description', 'No description available')
                    impact = details.get('impact', 'Impact not specified')
                    mitigation = details.get('mitigation', 'No mitigation strategy defined')
                    data.append([factor, f"{severity:.1f}/10", description, impact, mitigation])
                else:
                    data.append([factor, f"{details:.1f}/10", "No description available", 
                               "Impact not specified", "No mitigation strategy defined"])
            
            # Create the table with adjusted column widths
            risk_table = self._safe_table(data, 
                                        colWidths=[1.5*inch, 0.8*inch, 2*inch, 1.5*inch, 2*inch],
                                        style=self.table_style)
            
            # Add alternating row style
            if isinstance(risk_table, Table):
                risk_table.setStyle(self.alternating_row_style)
            
            self.story.append(risk_table)
            self.story.append(Spacer(1, 12))
            
            # Add risk severity visualization
            factors = [row[0] for row in data[1:]]
            severities = [float(row[1].split('/')[0]) for row in data[1:]]
            
            # Create color gradient based on severity
            colors = ['#E74C3C' if s >= 7 else '#F39C12' if s >= 5 else '#2ECC71' for s in severities]
            
            self.add_bar_chart(
                factors,
                severities,
                "Risk Severity by Factor",
                "Severity (out of 10)",
                colors
            )
            
            # Add risk summary
            self.story.append(Paragraph("Risk Summary", self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 6))
            
            # Calculate overall risk metrics
            avg_severity = sum(severities) / len(severities)
            high_risk_count = sum(1 for s in severities if s >= 7)
            medium_risk_count = sum(1 for s in severities if 5 <= s < 7)
            
            summary_text = f"""
            Overall Risk Assessment:
            • Average Risk Severity: {avg_severity:.1f}/10
            • High Risk Factors: {high_risk_count}
            • Medium Risk Factors: {medium_risk_count}
            • Total Risk Factors: {len(severities)}
            
            Key Risk Areas:
            • Market and Competitive Risks: {sum(severities[:2])/2:.1f}/10
            • Operational and Financial Risks: {sum(severities[2:4])/2:.1f}/10
            • Regulatory Risk: {severities[-1]:.1f}/10
            """
            
            self.story.append(Paragraph(summary_text, self.styles['CustomBodyText']))
            self.story.append(PageBreak())
            logger.info("Added detailed risk assessment section")
            
        except Exception as e:
            logger.error(f"Error creating risk assessment section: {str(e)}")
            self.story.append(Paragraph("Risk Assessment", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating full risk assessment section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())

    def create_clustering_analysis_section(self):
        """Create the clustering analysis section."""
        try:
            self.story.append(Paragraph("Startup Clustering Analysis", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 0.2*inch))
            
            # Get clustering data
            clustering_data = self.doc_data.get("clustering_analysis", {})
            
            # Cluster identification
            cluster_name = clustering_data.get("cluster_name", "High-Growth SaaS")
            cluster_description = clustering_data.get("cluster_description", 
                "B2B SaaS companies with strong growth metrics and product-led acquisition")
            
            self.story.append(Paragraph("Cluster Identification", self.styles['CustomHeading2']))
            self.story.append(Paragraph(f"<b>Your Cluster:</b> {cluster_name}", self.styles['CustomBodyText']))
            self.story.append(Paragraph(cluster_description, self.styles['CustomBodyText']))
            
            # Add key cluster metrics
            cluster_metrics = clustering_data.get("cluster_metrics", {})
            if cluster_metrics:
                self.story.append(Paragraph("Cluster Metrics Comparison", self.styles['CustomHeading2']))
                
                # Create table data
                data = [["Metric", "Your Value", "Cluster Mean", "Cluster Median"]]
                
                # Format metrics for display
                for metric_name, values in cluster_metrics.items():
                    # Format metric name
                    display_name = metric_name.replace("_", " ").title()
                    
                    # Format values based on metric type
                    if metric_name == "arr":
                        mean = self.format_currency(values.get('mean', 0))
                        median = self.format_currency(values.get('median', 0))
                        your_value = self.format_currency(values.get('your_value', 0))
                    elif metric_name in ["growth_rate", "gross_margin"]:
                        mean = f"{values.get('mean', 0)*100:.0f}%"
                        median = f"{values.get('median', 0)*100:.0f}%"
                        your_value = f"{values.get('your_value', 0)*100:.0f}%"
                    elif metric_name == "cac_payback":
                        mean = f"{values.get('mean', 0):.1f} months"
                        median = f"{values.get('median', 0):.1f} months"
                        your_value = f"{values.get('your_value', 0):.1f} months"
                    else:
                        mean = f"{values.get('mean', 0):.1f}"
                        median = f"{values.get('median', 0):.1f}"
                        your_value = f"{values.get('your_value', 0):.1f}"
                    
                    data.append([display_name, your_value, mean, median])
                
                # Create the table
                table = self._safe_table(data, colWidths=[1.6*inch, 1.6*inch, 1.6*inch, 1.6*inch])
                self.story.append(table)
                self.story.append(Spacer(1, 0.3*inch))
            
            # Similar companies
            similar_companies = clustering_data.get("closest_companies", [])
            if similar_companies:
                self.story.append(Paragraph("Most Similar Companies", self.styles['CustomHeading2']))
                
                # Create table data
                data = [["Company", "Similarity", "Description"]]
                
                for company in similar_companies:
                    data.append([
                        company.get("name", ""),
                        f"{company.get('similarity', 0)*100:.0f}%",
                        company.get("description", "")
                    ])
                
                # Create the table
                table = self._safe_table(data, colWidths=[1.5*inch, 1*inch, 4*inch])
                self.story.append(table)
                self.story.append(Spacer(1, 0.3*inch))
            
            # Cluster transitions
            transitions = self.doc_data.get("cluster_transitions", [])
            if transitions:
                self.story.append(Paragraph("Cluster Evolution Pathways", self.styles['CustomHeading2']))
                
                # Create table data
                data = [["Target Cluster", "Probability", "Timeline", "Key Changes"]]
                
                for transition in transitions:
                    key_changes = transition.get("key_changes", [])
                    changes_text = ", ".join(key_changes[:2])
                    if len(key_changes) > 2:
                        changes_text += "..."
                        
                    data.append([
                        transition.get("to_cluster", ""),
                        f"{transition.get('probability', 0)*100:.0f}%",
                        transition.get("typical_timeline", ""),
                        changes_text
                    ])
                
                # Create the table
                table = self._safe_table(data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 2.5*inch])
                self.story.append(table)
            
            # Add recommendations
            recommendations = self.doc_data.get("cluster_recommendations", [])
            if recommendations:
                self.story.append(Paragraph("Cluster-Based Recommendations", self.styles['CustomHeading2']))
                
                for i, rec in enumerate(recommendations):
                    self.story.append(Paragraph(f"<b>{i+1}. {rec.get('title', '')}</b> (Impact: {rec.get('impact', 'Medium')})", 
                                              self.styles['CustomBodyText']))
                    self.story.append(Paragraph(f"<i>{rec.get('description', '')}</i>", self.styles['CustomBodyTextSmall']))
                    self.story.append(Spacer(1, 0.1*inch))
            
            self.story.append(PageBreak())
            logger.info("Added clustering analysis section")
            
        except Exception as e:
            logger.error(f"Error creating clustering analysis section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Clustering Analysis", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating clustering analysis section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
    
    def create_summary_dashboard_section(self):
        """Create the summary dashboard section."""
        try:
            self.story.append(Paragraph("Executive Summary Dashboard", self.styles['CustomHeading1']))
            self.story.append(Spacer(1, 0.2*inch))
            
            # Overall score and recommendation
            overall_score = self.doc_data.get("overall_score", 78)
            
            # Add top-level metrics
            metrics = [
                ("Overall Score", f"{overall_score}/100"),
                ("Success Probability", f"{self.doc_data.get('success_prob', 70):.0f}%"),
                ("CAMP Score", f"{self.doc_data.get('camp_score', 75):.0f}/100")
            ]
            
            self.add_metric_row(metrics)
            
            # Add top recommendation
            top_recommendation = self.doc_data.get("top_recommendation", 
                "Focus on strengthening market positioning and optimizing customer acquisition channels to accelerate growth trajectory.")
            
            self.story.append(Paragraph("Key Recommendation:", self.styles['CustomHeading3']))
            self.story.append(Paragraph(top_recommendation, self.styles['CustomBodyText']))
            self.story.append(Spacer(1, 0.2*inch))
            
            # CAMP Framework Summary
            self.story.append(Paragraph("CAMP Framework Assessment", self.styles['CustomHeading2']))
            
            # Add CAMP radar chart
            self.add_camp_radar_chart()
            
            # Key metrics dashboard
            self.story.append(Paragraph("Key Performance Metrics", self.styles['CustomHeading2']))
            
            # Top metrics from across all analyses
            key_metrics = self.doc_data.get("key_metrics", {})
            
            # Metrics table with 3 rows of 3 columns
            if key_metrics:
                data = []
                current_row = []
                
                # Convert dict to list of tuples
                metrics_list = [(k.replace("_", " ").title(), v) for k, v in key_metrics.items()]
                
                # Create rows of 3 columns
                for i, (name, details) in enumerate(metrics_list):
                    if i > 0 and i % 3 == 0:
                        data.append(current_row)
                        current_row = []
                    
                    value = details.get("value", "N/A")
                    change = details.get("change", "")
                    
                    # Format the cell content
                    cell_content = f"<b>{name}</b>\n{value}"
                    if change:
                        cell_content += f"\n{change}"
                    
                    current_row.append(cell_content)
                
                # Add any remaining metrics
                if current_row:
                    while len(current_row) < 3:
                        current_row.append("")
                    data.append(current_row)
                
                # Create the table
                if data:
                    table = Table(data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
                    
                    # Style the table
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                        ('TOPPADDING', (0, 0), (-1, -1), 12),
                        ('GRID', (0, 0), (-1, -1), 1, self.colors['border'])
                    ]))
                    
                    self.story.append(table)
                    self.story.append(Spacer(1, 0.3*inch))
            
            # Top Recommendations
            self.story.append(Paragraph("Top Recommendations", self.styles['CustomHeading2']))
            
            # Get recommendations from various analyses
            recommendations = self.doc_data.get("top_recommendations", [])
            
            if recommendations:
                data = [["Category", "Recommendation", "Impact"]]
                
                for rec in recommendations[:5]:  # Limit to top 5
                    data.append([
                        rec.get("category", "General"),
                        rec.get("title", "") + "\n" + rec.get("description", ""),
                        rec.get("impact", "Medium")
                    ])
                
                # Create the table
                table = self._safe_table(data, colWidths=[1*inch, 4.5*inch, 1*inch])
                self.story.append(table)
            else:
                self.story.append(Paragraph("No recommendations available.", self.styles['CustomBodyText']))
            
            # Next Steps
            self.story.append(Paragraph("Next Steps", self.styles['CustomHeading2']))
            
            next_steps = self.doc_data.get("next_steps", [
                "Implement the recommended pricing strategy to improve unit economics",
                "Accelerate go-to-market efforts targeting the identified customer segments",
                "Strengthen the executive team with key hires in sales and marketing",
                "Develop the product roadmap to enhance defensibility and stickiness",
                "Prepare for the next funding round with a focus on key metrics"
            ])
            
            if next_steps:
                for i, step in enumerate(next_steps):
                    self.story.append(Paragraph(f"{i+1}. {step}", self.styles['CustomBodyText']))
                    self.story.append(Spacer(1, 0.1*inch))
            
            self.story.append(PageBreak())
            logger.info("Added summary dashboard section")
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard section: {str(e)}\n{traceback.format_exc()}")
            # Add basic section if error occurs
            self.story.append(Paragraph("Executive Summary Dashboard", self.styles['CustomHeading1']))
            self.story.append(Paragraph("Error generating summary dashboard section.", self.styles['CustomBodyText']))
            self.story.append(PageBreak())
```

```python
def generate_enhanced_pdf(doc, report_type="full", sections=None):
    """
    Generate an investment banking quality PDF report with ReportLab.
    
    Args:
        doc: The document data dictionary
        report_type: The type of report ("full", "executive", "custom")
        sections: Dictionary of sections to include if report_type is "custom"
        
    Returns:
        bytes: The PDF data
    """
    logger.info(f"Starting enhanced PDF generation with report type: {report_type}")
    
    try:
        # Create the report generator
        report = InvestorReportGenerator(doc, report_type, sections)
        
        # Generate the report
        pdf_data = report.generate_report()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error in generate_enhanced_pdf: {e}")
        try:
            # First fallback: try to just create a simple report with cover and executive summary
            logger.info("Trying simplified report generation")
            buffer = io.BytesIO()
            
            # Create the document
            doc_template = SimpleDocTemplate(
                buffer, 
                pagesize=letter,
                leftMargin=0.5*inch,
                rightMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create the story
            story = []
            
            # Add title
            story.append(Paragraph(f"{doc.get('name', 'Startup')} - Investor Report", styles['Title']))
            story.append(Spacer(1, 0.5*inch))
            
            # Add CAMP score
            story.append(Paragraph(f"CAMP Score: {doc.get('camp_score', 0):.1f}/100", styles['Heading1']))
            story.append(Spacer(1, 0.25*inch))
            
            # Add CAMP breakdown
            story.append(Paragraph("CAMP Framework Scores", styles['Heading2']))
            
            story.append(Paragraph(f"Capital Efficiency: {doc.get('capital_score', 0):.1f}/100", styles['Normal']))
            story.append(Paragraph(f"Market Dynamics: {doc.get('market_score', 0):.1f}/100", styles['Normal']))
            story.append(Paragraph(f"Advantage Moat: {doc.get('advantage_score', 0):.1f}/100", styles['Normal']))
            story.append(Paragraph(f"People & Performance: {doc.get('people_score', 0):.1f}/100", styles['Normal']))
            
            # Try to build the document
            doc_template.build(story)
            
            # Get the PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            logger.info("Successfully generated simplified report")
            
            return pdf_data
            
        except Exception as e2:
            logger.error(f"Simplified report generation also failed: {e2}")
            # Fall back to the emergency PDF
            return generate_emergency_pdf(doc)

def generate_emergency_pdf(doc):
    """Generate a minimal emergency PDF when the enhanced version fails."""
    logger.info("Generating emergency PDF")
    
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the document
        doc_template = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create the story
        story = []
        
        # Add title
        story.append(Paragraph(f"{doc.get('name', 'Startup')} - Investor Report", styles['Title']))
        story.append(Spacer(1, 0.25*inch))
        
        # Add basic info
        story.append(Paragraph(f"CAMP Score: {doc.get('camp_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Success Probability: {doc.get('success_prob', 0):.1f}%", styles['Normal']))
        story.append(Paragraph(f"Runway: {doc.get('runway_months', 0):.1f} months", styles['Normal']))
        
        # CAMP breakdown
        story.append(Spacer(1, 0.25*inch))
        story.append(Paragraph("CAMP Framework Scores", styles['Heading2']))
        
        story.append(Paragraph(f"Capital Efficiency: {doc.get('capital_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Market Dynamics: {doc.get('market_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"Advantage Moat: {doc.get('advantage_score', 0):.1f}/100", styles['Normal']))
        story.append(Paragraph(f"People & Performance: {doc.get('people_score', 0):.1f}/100", styles['Normal']))
        
        # Add core metrics
        story.append(Spacer(1, 0.25*inch))
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        
        def format_currency(value):
            value = float(value)
            if value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif value >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:.2f}"
        
        story.append(Paragraph(f"Monthly Revenue: {format_currency(doc.get('monthly_revenue', 0))}", styles['Normal']))
        story.append(Paragraph(f"Burn Rate: {format_currency(doc.get('burn_rate', 0))}", styles['Normal']))
        story.append(Paragraph(f"LTV:CAC Ratio: {doc.get('ltv_cac_ratio', 0):.2f}", styles['Normal']))
        
        # Emergency message
        story.append(Spacer(1, 0.25*inch))
        note_style = ParagraphStyle('Note', parent=styles['Italic'], textColor=colors.red)
        story.append(Paragraph("Note: This is a simplified emergency report. A fully detailed report with visualizations will be available soon.", note_style))
        
        # Build the document
        doc_template.build(story)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Emergency PDF generation also failed: {e}")
        # Return an empty PDF if all else fails
        buffer = io.BytesIO()
        doc_template = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Error Generating Report", styles['Title'])]
        doc_template.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

# Alias for backward compatibility
generate_investor_report = generate_enhanced_pdf 

def generate_pdf(document, strategy=InvestorReportGenerator):
    try:
        validated_doc = DocumentSchema(**document).dict()
        return strategy(validated_doc).generate_report()
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise ValueError("Invalid document structure") from e