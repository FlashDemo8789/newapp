import logging
import os
import sys
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.services.repository_factory import initialize_repository, get_repository
from app.routes import specialized_analysis
from app.routes import metadata
import logging.handlers

# Add root directory to path to ensure MongoRepository can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=5)
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug("[Startup] Logging system initialized and set to DEBUG level.")

# Define API description
API_DESCRIPTION = """
# FlashDNA Startup Assessment Platform API

This API provides specialized ML-powered analysis of startup potential using the CAMP framework:
- **Capital**: Financial metrics and runway
- **Advantage**: Product, technology, and competitive differentiation
- **Market**: Market size, growth, and dynamics
- **People**: Team, expertise, and execution

The API offers both core analysis endpoints and specialized analysis modules.
"""

# Initialize FastAPI app
app = FastAPI(
    title="FlashDNA Startup Assessment Platform",
    description=API_DESCRIPTION,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up dependency for repository access
def get_repo_dependency():
    repo = get_repository()
    if repo is None:
        raise HTTPException(status_code=503, detail="Repository not available")
    return repo

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(specialized_analysis.router)
app.include_router(metadata.router, prefix="/api")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing repository on startup")
    try:
        await initialize_repository()
        logger.info("Repository initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize repository: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information and navigation links
    """
    return {
        "name": "FlashDNA Startup Assessment Platform API",
        "version": "2.0",
        "description": "Advanced ML-powered startup analysis with the CAMP framework",
        "links": {
            "documentation": "/docs",
            "specialized_analysis": "/specialized",
            "core_analysis": "/api/analyze",
            "metadata": "/api/industries"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
