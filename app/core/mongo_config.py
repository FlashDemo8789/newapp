import os
import logging
from functools import lru_cache
from app.services.repository_factory import get_repository, get_db

logger = logging.getLogger("flashdna.mongo_config")

# MongoDB URI from environment or default
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/flash_dna")

# Re-export the repository getter and dependency for backward compatibility
get_mongo_repository = get_repository
get_mongo_db = get_db
