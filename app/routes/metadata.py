"""
API routes for startup metadata (industries, business models, stages, etc.)
"""
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter(tags=["metadata"])

# Industry categories
INDUSTRIES = [
    "SaaS",
    "Fintech",
    "Healthtech",
    "Edtech",
    "E-commerce",
    "Marketplace",
    "AI/ML",
    "Blockchain",
    "IoT",
    "Consumer",
    "Gaming",
    "Enterprise",
    "Hardware",
    "Biotech",
    "Cleantech",
    "Agtech",
    "Real Estate",
    "Travel",
    "Media",
    "Other"
]

# Business models
BUSINESS_MODELS = [
    "B2B",
    "B2C",
    "B2B2C",
    "Marketplace",
    "SaaS",
    "Subscription",
    "Freemium",
    "Transaction Fee",
    "Advertising",
    "Enterprise",
    "Consumer",
    "Hardware",
    "Open Source",
    "Licensing",
    "Data Monetization",
    "API",
    "Other"
]

# Startup stages
STARTUP_STAGES = [
    "Pre-seed",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Series D+",
    "Growth",
    "Pre-IPO",
    "Public",
    "Other"
]

@router.get("/industries", response_model=List[str])
async def get_industries():
    """
    Get list of industry categories
    """
    return INDUSTRIES

@router.get("/business-models", response_model=List[str])
async def get_business_models():
    """
    Get list of business models
    """
    return BUSINESS_MODELS

@router.get("/startup-stages", response_model=List[str])
async def get_startup_stages():
    """
    Get list of startup stages
    """
    return STARTUP_STAGES
