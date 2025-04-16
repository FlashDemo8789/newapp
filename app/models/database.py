from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Date
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime
import uuid

class Analysis(Base):
    """Main analysis model that stores the results of a startup assessment"""
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    stage = Column(String, nullable=False)
    business_model = Column(String, nullable=True)
    founding_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    # CAMP framework scores
    overall_score = Column(Float, nullable=False)
    capital_score = Column(Float, nullable=False)
    advantage_score = Column(Float, nullable=False)
    market_score = Column(Float, nullable=False)
    people_score = Column(Float, nullable=False)
    
    # Success probability and other metrics
    success_probability = Column(Float, nullable=True)
    
    # Relationship with dimension metrics
    capital_metrics = relationship("CapitalMetricsDB", back_populates="analysis", 
                                   uselist=False, cascade="all, delete-orphan")
    advantage_metrics = relationship("AdvantageMetricsDB", back_populates="analysis", 
                                     uselist=False, cascade="all, delete-orphan")
    market_metrics = relationship("MarketMetricsDB", back_populates="analysis", 
                                  uselist=False, cascade="all, delete-orphan")
    people_metrics = relationship("PeopleMetricsDB", back_populates="analysis", 
                                  uselist=False, cascade="all, delete-orphan")
    
    # Insights and recommendations
    recommendations = relationship("Recommendation", back_populates="analysis", 
                                   cascade="all, delete-orphan")
    strengths = relationship("Strength", back_populates="analysis", 
                            cascade="all, delete-orphan")
    improvements = relationship("Improvement", back_populates="analysis", 
                               cascade="all, delete-orphan")

class CapitalMetricsDB(Base):
    """Capital dimension metrics for a startup analysis"""
    __tablename__ = "capital_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="capital_metrics")
    
    # Capital metrics
    monthly_revenue = Column(Float, nullable=True)
    annual_recurring_revenue = Column(Float, nullable=True)
    burn_rate = Column(Float, nullable=True)
    runway_months = Column(Float, nullable=True)
    cash_balance = Column(Float, nullable=True)
    gross_margin = Column(Float, nullable=True)
    ltv_cac_ratio = Column(Float, nullable=True)
    customer_acquisition_cost = Column(Float, nullable=True)
    unit_economics_score = Column(Float, nullable=True)
    
    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = Column(JSON, nullable=True)

class AdvantageMetricsDB(Base):
    """Advantage dimension metrics for a startup analysis"""
    __tablename__ = "advantage_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="advantage_metrics")
    
    # Advantage metrics
    competition_level = Column(Integer, nullable=True)
    patents_count = Column(Integer, nullable=True)
    tech_innovation_score = Column(Float, nullable=True)
    technical_debt_ratio = Column(Float, nullable=True)
    network_effects_score = Column(Float, nullable=True)
    ip_protection_level = Column(Float, nullable=True)
    data_moat_score = Column(Float, nullable=True)
    business_model_strength = Column(Float, nullable=True)
    api_integrations_count = Column(Integer, nullable=True)
    product_security_score = Column(Float, nullable=True)
    
    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = Column(JSON, nullable=True)

class MarketMetricsDB(Base):
    """Market dimension metrics for a startup analysis"""
    __tablename__ = "market_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="market_metrics")
    
    # Market metrics
    tam_size = Column(Float, nullable=True)
    sam_size = Column(Float, nullable=True)
    market_growth_rate = Column(Float, nullable=True)
    user_growth_rate = Column(Float, nullable=True)
    active_users = Column(Integer, nullable=True)
    retention_rate = Column(Float, nullable=True)
    market_penetration = Column(Float, nullable=True)
    industry_trends_score = Column(Float, nullable=True)
    category_leadership_score = Column(Float, nullable=True)
    churn_rate = Column(Float, nullable=True)
    viral_coefficient = Column(Float, nullable=True)
    
    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = Column(JSON, nullable=True)

class PeopleMetricsDB(Base):
    """People dimension metrics for a startup analysis"""
    __tablename__ = "people_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="people_metrics")
    
    # People metrics
    team_size = Column(Integer, nullable=True)
    founder_experience = Column(Float, nullable=True)
    technical_skill_score = Column(Float, nullable=True)
    leadership_score = Column(Float, nullable=True)
    diversity_score = Column(Float, nullable=True)
    employee_turnover_rate = Column(Float, nullable=True)
    nps_score = Column(Integer, nullable=True)
    support_ticket_sla_percent = Column(Float, nullable=True)
    support_ticket_volume = Column(Integer, nullable=True)
    team_score = Column(Float, nullable=True)
    founder_domain_exp_yrs = Column(Integer, nullable=True)
    
    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = Column(JSON, nullable=True)

class Recommendation(Base):
    """Strategic recommendations for a startup analysis"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="recommendations")
    
    text = Column(Text, nullable=False)
    category = Column(String, nullable=True)  # Can be capital, advantage, market, people, or overall
    priority = Column(Integer, nullable=True)  # Higher number = higher priority

class Strength(Base):
    """Identified strengths in a startup analysis"""
    __tablename__ = "strengths"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="strengths")
    
    text = Column(Text, nullable=False)
    category = Column(String, nullable=True)  # Can be capital, advantage, market, people, or overall
    score = Column(Float, nullable=True)  # Score related to this strength

class Improvement(Base):
    """Identified areas for improvement in a startup analysis"""
    __tablename__ = "improvements"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    analysis = relationship("Analysis", back_populates="improvements")
    
    text = Column(Text, nullable=False)
    category = Column(String, nullable=True)  # Can be capital, advantage, market, people, or overall
    score = Column(Float, nullable=True)  # Score related to this improvement area
