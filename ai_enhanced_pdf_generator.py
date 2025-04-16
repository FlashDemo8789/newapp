"""
AI-Enhanced PDF Generator for FlashDNA

This module extends the unified PDF generator with AI-powered narrative content
to generate more insightful investor reports.
"""

import logging
import copy
from io import BytesIO
from datetime import datetime
import tempfile
import os
from typing import Dict, List, Any, Optional, Union

# Import the base PDF generator
from unified_pdf_generator import InvestorReport, generate_emergency_pdf

# Import our AI Narrative Engine
from ai_narrative_engine import AINarrativeEngine, initialize_narrative_engine

# Configure logging
logger = logging.getLogger("ai_enhanced_pdf")
logger.setLevel(logging.INFO)

class AIEnhancedReportPDF(InvestorReport):
    """
    Enhanced version of InvestorReport that incorporates AI-generated narrative content.
    """
    
    def __init__(self, doc_data: Dict, report_type: str = "full", sections: Optional[List[str]] = None):
        """Initialize the AI-enhanced report generator."""
        super().__init__(doc_data, report_type, sections)
        self.ai_engine = AINarrativeEngine()
        logger.info(f"Initialized AI-enhanced report for {doc_data.get('company_name', 'Unknown Company')}")
    
    def add_ai_executive_summary(self):
        """Add AI-enhanced executive summary section."""
        try:
            # Generate AI-enhanced summary
            summary = self.ai_engine.generate_executive_summary(self.doc_data)
            
            # Add to story
            self.story.extend([
                Paragraph("Executive Summary", self.styles['Heading1']),
                Spacer(1, 0.25*inch),
                Paragraph(summary, self.styles['Normal']),
                Spacer(1, 0.5*inch)
            ])
            logger.info("Added AI-enhanced executive summary")
        except Exception as e:
            logger.error(f"Error adding AI executive summary: {e}")
            self.story.append(Paragraph("Executive summary generation failed. Using standard summary.", self.styles['Normal']))
    
    def add_ai_competitive_analysis(self):
        """Add AI-enhanced competitive analysis section."""
        try:
            # Generate AI-enhanced analysis
            analysis = self.ai_engine.generate_competitive_analysis(self.doc_data)
            
            # Add to story
            self.story.extend([
                Paragraph("Competitive Analysis", self.styles['Heading1']),
                Spacer(1, 0.25*inch),
                Paragraph(analysis, self.styles['Normal']),
                Spacer(1, 0.5*inch)
            ])
            logger.info("Added AI-enhanced competitive analysis")
        except Exception as e:
            logger.error(f"Error adding AI competitive analysis: {e}")
            self.story.append(Paragraph("Competitive analysis generation failed. Using standard analysis.", self.styles['Normal']))
    
    def add_ai_recommendations(self):
        """Add AI-generated strategic recommendations section."""
        try:
            # Generate recommendations
            recommendations = self.ai_engine.generate_strategic_recommendations(self.doc_data)
            
            # Add to story
            self.story.extend([
                Paragraph("Strategic Recommendations", self.styles['Heading1']),
                Spacer(1, 0.25*inch),
                Paragraph(recommendations, self.styles['Normal']),
                Spacer(1, 0.5*inch)
            ])
            logger.info("Added AI-generated strategic recommendations")
        except Exception as e:
            logger.error(f"Error adding AI recommendations: {e}")
            self.story.append(Paragraph("Recommendations generation failed. Using standard recommendations.", self.styles['Normal']))
    
    def add_ai_market_trends(self):
        """Add AI-generated market trends analysis section."""
        try:
            # Generate market trends analysis
            trends = self.ai_engine.generate_market_trends_analysis(self.doc_data)
            
            # Add to story
            self.story.extend([
                Paragraph("Market Trends Analysis", self.styles['Heading1']),
                Spacer(1, 0.25*inch),
                Paragraph(trends, self.styles['Normal']),
                Spacer(1, 0.5*inch)
            ])
            logger.info("Added AI-generated market trends analysis")
        except Exception as e:
            logger.error(f"Error adding AI market trends: {e}")
            self.story.append(Paragraph("Market trends analysis generation failed. Using standard analysis.", self.styles['Normal']))
    
    def add_intangible_factors_analysis(self):
        """Add AI-generated intangible factors analysis section."""
        try:
            # Generate intangible factors analysis
            analysis = self.ai_engine.analyze_intangible_factors(self.doc_data)
            
            # Add to story
            self.story.extend([
                Paragraph("Intangible Factors Analysis", self.styles['Heading1']),
                Spacer(1, 0.25*inch),
                Paragraph(analysis, self.styles['Normal']),
                Spacer(1, 0.5*inch)
            ])
            logger.info("Added AI-generated intangible factors analysis")
        except Exception as e:
            logger.error(f"Error adding AI intangible factors analysis: {e}")
            self.story.append(Paragraph("Intangible factors analysis generation failed. Using standard analysis.", self.styles['Normal']))

def generate_ai_enhanced_pdf(
    doc_data: Dict,
    output_path: str,
    report_type: str = "full",
    sections: Optional[List[str]] = None
) -> bool:
    """
    Generate an AI-enhanced PDF report.
    
    Args:
        doc_data: Dictionary containing document data
        output_path: Path where the PDF will be saved
        report_type: Type of report to generate
        sections: List of sections to include
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting AI-enhanced PDF generation for {doc_data.get('company_name', 'Unknown Company')}")
        
        # Create AI-enhanced report
        report = AIEnhancedReportPDF(doc_data, report_type, sections)
        
        # Generate the report
        pdf_data = report.generate_report()
        
        # Save the PDF data to the output path
        with open(output_path, 'wb') as f:
            f.write(pdf_data)
        
        logger.info(f"Successfully generated AI-enhanced PDF: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating AI-enhanced PDF: {e}")
        # Fallback to emergency PDF
        try:
            emergency_pdf = generate_emergency_pdf(doc_data)
            with open(output_path, 'wb') as f:
                f.write(emergency_pdf)
            logger.info(f"Generated emergency PDF: {output_path}")
            return True
        except Exception as e2:
            logger.error(f"Error generating emergency PDF: {e2}")
            return False


# Replace the standard PDF generation functions with our AI-enhanced versions
generate_enhanced_pdf = generate_ai_enhanced_pdf
generate_investor_report = generate_ai_enhanced_pdf 