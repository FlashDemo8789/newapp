class Config:
    SECRET_KEY = 'dev-secret-key-replace-in-production'
    DEBUG = True
    TESTING = False
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    
    # Timeout settings for long-running analyses
    ANALYSIS_TIMEOUT = 300  # seconds
