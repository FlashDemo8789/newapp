from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import date
from app.core.constants import BUSINESS_MODELS, STARTUP_STAGES
from app.core.industries import INDUSTRIES

class StartupBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    industry: str = Field(..., description="Industry sector")
    stage: str = Field(..., description="Funding stage")
    founded_date: date = Field(...)
    description: Optional[str] = Field(None, max_length=2000)
    business_model: str = Field(..., description="Primary business model")
    
    @validator('industry')
    def validate_industry(cls, v):
        if v not in INDUSTRIES:
            raise ValueError(f"Industry must be one of: {', '.join(INDUSTRIES)}")
        return v
    
    @validator('stage')
    def validate_stage(cls, v):
        if v not in STARTUP_STAGES:
            raise ValueError(f"Stage must be one of: {', '.join(STARTUP_STAGES)}")
        return v
    
    @validator('business_model')
    def validate_business_model(cls, v):
        if v not in BUSINESS_MODELS:
            raise ValueError(f"Business model must be one of: {', '.join(BUSINESS_MODELS)}")
        return v

class CapitalMetrics(BaseModel):
    monthly_revenue: float = Field(0.0, ge=0)
    annual_recurring_revenue: float = Field(0.0, ge=0)
    burn_rate: float = Field(0.0, ge=0)
    runway_months: Optional[float] = Field(None, ge=0)
    cash_balance: Optional[float] = Field(None, ge=0)
    gross_margin: Optional[float] = Field(None, ge=0, le=1)
    ltv_cac_ratio: Optional[float] = Field(None, ge=0)
    customer_acquisition_cost: Optional[float] = Field(None, ge=0)
    unit_economics_score: Optional[float] = Field(None, ge=0, le=1)
    
class AdvantageMetrics(BaseModel):
    competition_level: Optional[int] = Field(None, ge=1, le=10)
    patents_count: Optional[int] = Field(None, ge=0)
    tech_innovation_score: Optional[float] = Field(None, ge=0, le=1)
    technical_debt_ratio: Optional[float] = Field(None, ge=0, le=1)
    network_effects_score: Optional[float] = Field(None, ge=0, le=1)
    ip_protection_level: Optional[float] = Field(None, ge=0, le=1)
    data_moat_score: Optional[float] = Field(None, ge=0, le=1)
    business_model_strength: Optional[float] = Field(None, ge=0, le=1)
    api_integrations_count: Optional[int] = Field(None, ge=0)
    product_security_score: Optional[float] = Field(None, ge=0, le=1)
    
class MarketMetrics(BaseModel):
    tam_size: Optional[float] = Field(None, ge=0)
    sam_size: Optional[float] = Field(None, ge=0)
    market_growth_rate: Optional[float] = Field(None, ge=0, le=1)
    user_growth_rate: Optional[float] = Field(None, ge=0, le=10)
    active_users: Optional[int] = Field(None, ge=0)
    retention_rate: Optional[float] = Field(None, ge=0, le=1)
    market_penetration: Optional[float] = Field(None, ge=0, le=1)
    industry_trends_score: Optional[float] = Field(None, ge=0, le=1)
    category_leadership_score: Optional[float] = Field(None, ge=0, le=1)
    churn_rate: Optional[float] = Field(None, ge=0, le=1)
    viral_coefficient: Optional[float] = Field(None, ge=0)

class PeopleMetrics(BaseModel):
    team_size: Optional[int] = Field(None, ge=0)
    founder_experience: Optional[float] = Field(None, ge=0, le=1)
    technical_skill_score: Optional[float] = Field(None, ge=0, le=1)
    leadership_score: Optional[float] = Field(None, ge=0, le=1)
    diversity_score: Optional[float] = Field(None, ge=0, le=1)
    employee_turnover_rate: Optional[float] = Field(None, ge=0, le=1)
    nps_score: Optional[int] = Field(None, ge=-100, le=100)
    support_ticket_sla_percent: Optional[float] = Field(None, ge=0, le=1)
    support_ticket_volume: Optional[int] = Field(None, ge=0)
    team_score: Optional[float] = Field(None, ge=0, le=1)
    founder_domain_exp_yrs: Optional[int] = Field(None, ge=0)
    
class PitchMetrics(BaseModel):
    text: Optional[str] = Field(None)
    clarity: Optional[float] = Field(None, ge=0, le=1)
    impact: Optional[float] = Field(None, ge=0, le=1)

class StartupAnalysisInput(StartupBase):
    capital_metrics: Optional[CapitalMetrics] = None
    advantage_metrics: Optional[AdvantageMetrics] = None
    market_metrics: Optional[MarketMetrics] = None
    people_metrics: Optional[PeopleMetrics] = None
    pitch_metrics: Optional[PitchMetrics] = None

class CAMPScore(BaseModel):
    score: float = Field(..., ge=0, le=10)
    description: str
    color: str
    metrics_breakdown: Dict[str, float]

class CAMPAnalysisResult(BaseModel):
    id: Optional[str] = None
    overall_score: float = Field(..., ge=0, le=10)
    capital_score: CAMPScore
    advantage_score: CAMPScore
    market_score: CAMPScore
    people_score: CAMPScore
    success_probability: float = Field(..., ge=0, le=100)
    recommendations: List[str]
    top_strengths: List[str]
    improvement_areas: List[str]
