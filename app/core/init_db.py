import logging
from sqlalchemy.exc import SQLAlchemyError
from app.core.database import engine, Base, SessionLocal
from app.models.database import Analysis  # This imports all models
import os

logger = logging.getLogger("flashdna.init_db")

def init_db():
    """Initialize the database by creating all tables"""
    try:
        # Create tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize with seed data for development
        if os.environ.get("ENVIRONMENT", "development") == "development":
            initialize_seed_data()
        
        return True
    except SQLAlchemyError as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        return False

def initialize_seed_data():
    """Initialize the database with seed data for development"""
    db = SessionLocal()
    try:
        # Check if we already have data
        if db.query(Analysis).count() > 0:
            logger.info("Database already contains data, skipping seed data")
            return
        
        # Add seed data
        logger.info("Adding seed data...")
        from app.core.seed_data import generate_seed_data
        
        # Generate 10 sample analyses for testing
        analysis_ids = generate_seed_data(db, 10)
        
        logger.info(f"Seed data added successfully: {len(analysis_ids)} analyses")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error adding seed data: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error adding seed data: {str(e)}", exc_info=True)
    finally:
        db.close()
