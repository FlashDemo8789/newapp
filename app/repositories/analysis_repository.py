from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models.database import (
    Analysis, CapitalMetricsDB, AdvantageMetricsDB, 
    MarketMetricsDB, PeopleMetricsDB, Recommendation,
    Strength, Improvement
)
from app.models.startup import StartupAnalysisInput, CAMPAnalysisResult
import logging
import json
from datetime import datetime

logger = logging.getLogger("flashdna.repository")

class AnalysisRepository:
    """Repository for managing startup analysis data in the database"""
    
    @staticmethod
    async def create_analysis(db: Session, analysis_data: CAMPAnalysisResult, startup_data: StartupAnalysisInput) -> Analysis:
        """
        Create a new analysis record in the database
        
        Args:
            db: Database session
            analysis_data: Analysis results from the CAMP framework assessment
            startup_data: Original startup data input
            
        Returns:
            The created analysis database record
        """
        try:
            # Create main analysis record
            db_analysis = Analysis(
                name=startup_data.name,
                industry=startup_data.industry,
                stage=startup_data.stage,
                business_model=getattr(startup_data, 'business_model', None),
                founding_date=startup_data.founded_date,
                overall_score=analysis_data.overall_score / 10.0,  # Scale from 0-10 to 0-1
                capital_score=analysis_data.capital_score.score / 10.0,
                advantage_score=analysis_data.advantage_score.score / 10.0,
                market_score=analysis_data.market_score.score / 10.0,
                people_score=analysis_data.people_score.score / 10.0,
                success_probability=analysis_data.success_probability / 100.0  # Scale from 0-100 to 0-1
            )
            
            db.add(db_analysis)
            db.flush()  # Get the ID without committing
            
            # Add capital metrics
            if startup_data.capital_metrics:
                capital_dict = startup_data.capital_metrics.dict()
                db_capital = CapitalMetricsDB(
                    analysis_id=db_analysis.id,
                    **{k: v for k, v in capital_dict.items() if k in CapitalMetricsDB.__table__.columns.keys()},
                    additional_metrics=json.dumps({k: v for k, v in capital_dict.items() 
                                                 if k not in CapitalMetricsDB.__table__.columns.keys() 
                                                 and k != "additional_metrics"})
                )
                db.add(db_capital)
            
            # Add advantage metrics
            if startup_data.advantage_metrics:
                advantage_dict = startup_data.advantage_metrics.dict()
                db_advantage = AdvantageMetricsDB(
                    analysis_id=db_analysis.id,
                    **{k: v for k, v in advantage_dict.items() if k in AdvantageMetricsDB.__table__.columns.keys()},
                    additional_metrics=json.dumps({k: v for k, v in advantage_dict.items() 
                                                  if k not in AdvantageMetricsDB.__table__.columns.keys()
                                                  and k != "additional_metrics"})
                )
                db.add(db_advantage)
            
            # Add market metrics
            if startup_data.market_metrics:
                market_dict = startup_data.market_metrics.dict()
                db_market = MarketMetricsDB(
                    analysis_id=db_analysis.id,
                    **{k: v for k, v in market_dict.items() if k in MarketMetricsDB.__table__.columns.keys()},
                    additional_metrics=json.dumps({k: v for k, v in market_dict.items() 
                                                 if k not in MarketMetricsDB.__table__.columns.keys()
                                                 and k != "additional_metrics"})
                )
                db.add(db_market)
            
            # Add people metrics
            if startup_data.people_metrics:
                people_dict = startup_data.people_metrics.dict()
                db_people = PeopleMetricsDB(
                    analysis_id=db_analysis.id,
                    **{k: v for k, v in people_dict.items() if k in PeopleMetricsDB.__table__.columns.keys()},
                    additional_metrics=json.dumps({k: v for k, v in people_dict.items() 
                                                 if k not in PeopleMetricsDB.__table__.columns.keys()
                                                 and k != "additional_metrics"})
                )
                db.add(db_people)
            
            # Add recommendations
            for i, rec_text in enumerate(analysis_data.recommendations):
                db_rec = Recommendation(
                    analysis_id=db_analysis.id,
                    text=rec_text,
                    priority=len(analysis_data.recommendations) - i  # Higher index = lower priority
                )
                db.add(db_rec)
            
            # Add strengths
            for i, strength_text in enumerate(analysis_data.top_strengths):
                db_strength = Strength(
                    analysis_id=db_analysis.id,
                    text=strength_text,
                    score=1.0 - (i * 0.1)  # Simple scoring: first = 1.0, second = 0.9, etc.
                )
                db.add(db_strength)
            
            # Add improvement areas
            for i, improvement_text in enumerate(analysis_data.improvement_areas):
                db_improvement = Improvement(
                    analysis_id=db_analysis.id,
                    text=improvement_text,
                    score=1.0 - (i * 0.1)  # Simple scoring: first = 1.0, second = 0.9, etc.
                )
                db.add(db_improvement)
            
            # Commit the transaction
            db.commit()
            db.refresh(db_analysis)
            
            return db_analysis
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating analysis: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def get_analysis_by_id(db: Session, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a complete analysis by ID
        
        Args:
            db: Database session
            analysis_id: ID of the analysis to retrieve
            
        Returns:
            Complete analysis data or None if not found
        """
        try:
            # Query the main analysis record with all related data
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return None
            
            # Convert to dictionary format for API response
            result = {
                "id": analysis.id,
                "company_info": {
                    "name": analysis.name,
                    "industry": analysis.industry,
                    "stage": analysis.stage,
                    "business_model": analysis.business_model,
                    "founding_date": analysis.founding_date.isoformat() if analysis.founding_date else None
                },
                "created_at": analysis.created_at.isoformat(),
                "updated_at": analysis.updated_at.isoformat(),
                "overall_score": analysis.overall_score,
                "capital_score": analysis.capital_score,
                "advantage_score": analysis.advantage_score,
                "market_score": analysis.market_score,
                "people_score": analysis.people_score,
                "success_probability": analysis.success_probability
            }
            
            # Add metrics if available
            if analysis.capital_metrics:
                # Start with columns from the table
                capital_metrics = {c.name: getattr(analysis.capital_metrics, c.name) 
                                 for c in analysis.capital_metrics.__table__.columns
                                 if c.name not in ('id', 'analysis_id')}
                
                # Add any additional metrics stored in JSON
                if analysis.capital_metrics.additional_metrics:
                    try:
                        additional = json.loads(analysis.capital_metrics.additional_metrics)
                        capital_metrics.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                result["capital_metrics"] = capital_metrics
            
            if analysis.advantage_metrics:
                advantage_metrics = {c.name: getattr(analysis.advantage_metrics, c.name) 
                                    for c in analysis.advantage_metrics.__table__.columns
                                    if c.name not in ('id', 'analysis_id')}
                
                if analysis.advantage_metrics.additional_metrics:
                    try:
                        additional = json.loads(analysis.advantage_metrics.additional_metrics)
                        advantage_metrics.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                result["advantage_metrics"] = advantage_metrics
            
            if analysis.market_metrics:
                market_metrics = {c.name: getattr(analysis.market_metrics, c.name) 
                                 for c in analysis.market_metrics.__table__.columns
                                 if c.name not in ('id', 'analysis_id')}
                
                if analysis.market_metrics.additional_metrics:
                    try:
                        additional = json.loads(analysis.market_metrics.additional_metrics)
                        market_metrics.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                result["market_metrics"] = market_metrics
            
            if analysis.people_metrics:
                people_metrics = {c.name: getattr(analysis.people_metrics, c.name) 
                                 for c in analysis.people_metrics.__table__.columns
                                 if c.name not in ('id', 'analysis_id')}
                
                if analysis.people_metrics.additional_metrics:
                    try:
                        additional = json.loads(analysis.people_metrics.additional_metrics)
                        people_metrics.update(additional)
                    except json.JSONDecodeError:
                        pass
                
                result["people_metrics"] = people_metrics
            
            # Add recommendations, strengths, and improvements
            result["recommendations"] = [rec.text for rec in analysis.recommendations]
            result["top_strengths"] = [s.text for s in analysis.strengths]
            result["improvement_areas"] = [imp.text for imp in analysis.improvements]
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving analysis: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def get_all_analyses(db: Session, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of all analyses with basic information
        
        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            
        Returns:
            List of analysis summaries
        """
        try:
            analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
            
            results = []
            for analysis in analyses:
                results.append({
                    "id": analysis.id,
                    "name": analysis.name,
                    "industry": analysis.industry,
                    "stage": analysis.stage,
                    "created_at": analysis.created_at.isoformat(),
                    "overall_score": analysis.overall_score,
                    "capital_score": analysis.capital_score,
                    "advantage_score": analysis.advantage_score,
                    "market_score": analysis.market_score,
                    "people_score": analysis.people_score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving analyses: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def get_analyses_by_ids(db: Session, analysis_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple analyses by their IDs
        
        Args:
            db: Database session
            analysis_ids: List of analysis IDs to retrieve
            
        Returns:
            List of complete analysis data for the requested IDs
        """
        try:
            results = []
            
            for analysis_id in analysis_ids:
                analysis = await AnalysisRepository.get_analysis_by_id(db, analysis_id)
                if analysis:
                    results.append(analysis)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving multiple analyses: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def delete_analysis(db: Session, analysis_id: str) -> bool:
        """
        Delete an analysis by ID
        
        Args:
            db: Database session
            analysis_id: ID of the analysis to delete
            
        Returns:
            True if the analysis was deleted, False if not found
        """
        try:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return False
            
            db.delete(analysis)
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting analysis: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def update_analysis_name(db: Session, analysis_id: str, new_name: str) -> bool:
        """
        Update the name of an analysis
        
        Args:
            db: Database session
            analysis_id: ID of the analysis to update
            new_name: New name for the analysis
            
        Returns:
            True if the analysis was updated, False if not found
        """
        try:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return False
            
            analysis.name = new_name
            analysis.updated_at = datetime.now()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating analysis name: {str(e)}", exc_info=True)
            raise
