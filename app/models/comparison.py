from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from uuid import UUID

class ComparisonRequest(BaseModel):
    """Request model for comparing multiple analyses"""
    analysis_ids: List[str] = Field(..., min_items=2, max_items=5)

class AnalysisSummary(BaseModel):
    """Summary of a startup analysis for comparison purposes"""
    id: str
    name: str
    industry: str
    stage: str
    founding_date: Optional[date] = None
    created_at: datetime
    overall_score: float = Field(..., ge=0, le=1)
    capital_score: float = Field(..., ge=0, le=1)
    advantage_score: float = Field(..., ge=0, le=1)
    market_score: float = Field(..., ge=0, le=1)
    people_score: float = Field(..., ge=0, le=1)
    
class ComparisonInsight(BaseModel):
    """Insight generated from comparing startup analyses"""
    category: str  # Can be 'capital', 'advantage', 'market', 'people', or 'overall'
    insight: str
    severity: str = Field(..., pattern='^(critical|important|moderate|info)$')
    
class DimensionComparison(BaseModel):
    """Comparison of a specific dimension across multiple startups"""
    dimension: str
    metrics: Dict[str, List[float]]  # Metric name to list of values (one per startup)
    insights: List[str]
    
class AnalysisComparison(BaseModel):
    """Full comparison result of multiple startup analyses"""
    analyses: List[AnalysisSummary]
    insights: List[ComparisonInsight]
    dimension_comparisons: List[DimensionComparison]
    comparison_date: datetime = Field(default_factory=datetime.now)
    strongest_startup: Optional[str] = None  # ID of the startup with highest overall score

class ExportRequest(BaseModel):
    """Request model for exporting an analysis or comparison to PDF"""
    analyses: List[AnalysisSummary]
    insights: List[ComparisonInsight]
    dimension_comparisons: List[DimensionComparison]
    title: Optional[str] = "FlashDNA Startup Comparison"
    description: Optional[str] = None
    include_recommendations: bool = True
