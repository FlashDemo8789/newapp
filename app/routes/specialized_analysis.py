"""
API routes for specialized startup analysis modules
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.services.product_market_fit_service import pmf_service
from app.services.cohort_analysis_service import cohort_service
from app.services.acquisition_fit_service import acquisition_fit_service
from app.services.monte_carlo_service import monte_carlo_service
from app.services.competitive_intelligence_service import competitive_intelligence_service
from app.services.benchmarking_service import benchmarking_service
from app.services.exit_path_service import exit_path_service
from app.services.ecosystem_map_service import ecosystem_map_service
from app.services.funding_trajectory_service import funding_trajectory_service

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/specialized", tags=["specialized"])

# Helper function to extract data from various request formats
def extract_data_from_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalized startup data from various request formats
    handles both flattened and nested formats
    """
    try:
        logger.info(f"Received data structure: {list(data.keys())}")
        
        # If the data is already in the expected format with direct metrics
        if "capital_metrics" in data and "advantage_metrics" in data and "market_metrics" in data:
            logger.info("Data is already in properly structured format")
            return data
            
        # If data is in the nested format with company, capital, advantage, etc.
        if "company" in data and isinstance(data["company"], dict):
            logger.info("Data is in nested format with 'company' key")
            
            # Transform to expected format
            transformed_data = {
                "name": data["company"].get("name", ""),
                "industry": data["company"].get("industry", ""),
                "business_model": data["company"].get("business_model", ""),
                "founded_date": data["company"].get("founding_date", ""),
                "stage": data["company"].get("stage", ""),
                "description": data["company"].get("description", ""),
                
                # Get the nested metrics
                "capital_metrics": data.get("capital", {}),
                "advantage_metrics": data.get("advantage", {}),
                "market_metrics": data.get("market", {}),
                "people_metrics": data.get("people", {})
            }
            
            logger.info(f"Transformed data: {transformed_data['name']}, {transformed_data['industry']}")
            return transformed_data
        
        # If the data is flattened
        logger.info("Using data as-is")
        return data
        
    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}")
        # Return the original data if extraction fails
        return data

@router.post("/pmf")
async def analyze_product_market_fit(startup_data: Dict[str, Any] = Body(...)):
    """
    Analyze product-market fit for a startup using the specialized PMF analyzer
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        results = await pmf_service.analyze_pmf(structured_data)
        return results
    except Exception as e:
        logger.error(f"Error in PMF analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/cohorts")
async def analyze_cohorts(startup_data: Dict[str, Any] = Body(...), cohort_data: Optional[Dict[str, Any]] = None):
    """
    Analyze user cohorts and retention patterns
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        results = await cohort_service.analyze_cohorts(structured_data, cohort_data)
        return results
    except Exception as e:
        logger.error(f"Error in cohort analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/acquisition-fit")
async def analyze_acquisition_fit(startup_data: Dict[str, Any] = Body(...), acquirer_data: Optional[Dict[str, Any]] = None):
    """
    Analyze acquisition fit and potential for a startup
    """
    logger.info("[API] /specialized/acquisition-fit endpoint called")
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        logger.debug(f"[API] Structured data for acquisition fit: {structured_data}")
        results = await acquisition_fit_service.analyze_acquisition_fit(structured_data, acquirer_data)
        logger.debug(f"[API] Acquisition fit results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error in acquisition fit analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/monte-carlo")
async def run_monte_carlo_simulation(
    startup_data: Dict[str, Any] = Body(...), 
    simulation_params: Optional[Dict[str, Any]] = None,
    num_iterations: int = Query(1000, description="Number of simulation iterations")
):
    """
    Run Monte Carlo simulations for startup forecasting
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        results = await monte_carlo_service.run_monte_carlo_simulation(
            structured_data, 
            simulation_params, 
            num_iterations
        )
        return results
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/competitive-intelligence")
async def analyze_competitive_landscape(
    startup_data: Dict[str, Any] = Body(...), 
    competitor_data: Optional[Dict[str, Any]] = None
):
    """
    Analyze competitive landscape and positioning
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        results = await competitive_intelligence_service.analyze_competitive_landscape(
            structured_data, 
            competitor_data
        )
        return results
    except Exception as e:
        logger.error(f"Error in competitive landscape analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/benchmarking")
async def analyze_benchmarks(
    startup_data: Dict[str, Any] = Body(...),
    industry: Optional[str] = None,
    stage: Optional[str] = None
):
    """
    Analyze startup performance against industry benchmarks
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        # Extract industry and stage from the data if not provided separately
        if industry is None and "industry" in structured_data:
            industry = structured_data["industry"]
            
        if stage is None and "stage" in structured_data:
            stage = structured_data["stage"]
            
        results = await benchmarking_service.analyze_benchmarks(
            structured_data, 
            industry,
            stage
        )
        return results
    except Exception as e:
        logger.error(f"Error in benchmark analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

@router.post("/exit-path")
async def analyze_exit_paths(
    startup_data: Dict[str, Any] = Body(...),
    target_timeframe_years: int = Query(5, description="Target timeframe for exit analysis in years")
):
    """
    Analyze potential exit paths for the startup
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        logger.info(f"Analyzing exit paths with timeframe: {target_timeframe_years} years")
        results = await exit_path_service.analyze_exit_paths(
            structured_data, 
            target_timeframe_years
        )
        return results
    except Exception as e:
        logger.error(f"Error in exit path analysis: {str(e)}")
        return {"error": str(e), "status": "error"}

# Adding explicit endpoint mappings for frontend compatibility
@router.post("/funding-trajectory")
async def analyze_funding_trajectory(
    startup_data: Dict[str, Any] = Body(...)
):
    """
    Analyze funding trajectory and future fundraising potential
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        # Use the funding trajectory service
        trajectory_analysis = await funding_trajectory_service.analyze_funding_trajectory(structured_data)
        
        logger.info(f"Generated funding trajectory analysis for: {structured_data.get('name', 'Unknown Company')}")
        return trajectory_analysis
    except Exception as e:
        logger.error(f"Error analyzing funding trajectory: {str(e)}")
        # Generate fallback funding trajectory data
        stage = structured_data.get("stage", "seed")
        
        return {
            "funding_trajectory": {
                "current_stage": stage,
                "current_valuation": 5000000 if stage.lower() == "seed" else 20000000,
                "funding_score": 0.65,
                "projected_stages": [
                    {
                        "name": "SERIES A" if stage.lower() == "seed" else "SERIES B",
                        "probability": 0.7,
                        "timeline_months": 12,
                        "typical_raise": "$5M-$15M" if stage.lower() == "seed" else "$15M-$30M"
                    },
                    {
                        "name": "SERIES B" if stage.lower() == "seed" else "SERIES C",
                        "probability": 0.4,
                        "timeline_months": 30,
                        "typical_raise": "$15M-$30M" if stage.lower() == "seed" else "$30M-$100M"
                    }
                ],
                "valuation_projections": [
                    {
                        "timeline_months": 12,
                        "valuation_range": {
                            "low": 8000000,
                            "mid": 12000000,
                            "high": 18000000
                        }
                    },
                    {
                        "timeline_months": 24,
                        "valuation_range": {
                            "low": 15000000,
                            "mid": 25000000,
                            "high": 40000000
                        }
                    }
                ],
                "funding_recommendations": [
                    "Focus on key growth metrics to prepare for next funding round",
                    "Develop relationships with venture capital firms",
                    "Consider strategic partnerships to accelerate growth",
                    "Improve unit economics to strengthen funding position"
                ]
            },
            "success_factors": [
                {
                    "name": "Team Strength",
                    "current_score": 0.7,
                    "impact": "High"
                },
                {
                    "name": "Product Quality",
                    "current_score": 0.8,
                    "impact": "High"
                },
                {
                    "name": "Market Size Potential",
                    "current_score": 0.6,
                    "impact": "Medium"
                },
                {
                    "name": "Revenue Growth",
                    "current_score": 0.5,
                    "impact": "High"
                }
            ],
            "is_fallback": True
        }

@router.post("/ecosystem-map")
async def generate_ecosystem_map(
    startup_data: Dict[str, Any] = Body(...)
):
    """
    Generate a map of the startup's ecosystem including competitors, partners, and customers
    """
    try:
        # Extract properly structured data
        structured_data = extract_data_from_request(startup_data)
        
        # Use the ecosystem map service
        ecosystem_map = await ecosystem_map_service.analyze_ecosystem(structured_data)
        
        logger.info(f"Generated ecosystem map for: {structured_data.get('name', 'Unknown Company')}")
        return ecosystem_map
    except Exception as e:
        logger.error(f"Error generating ecosystem map: {str(e)}")
        # Return a fallback response with basic data
        industry = structured_data.get("industry", "software")
        business_model = structured_data.get("business_model", "saas")
        
        return {
            "ecosystem_map": {
                "direct_competitors": _get_competitors_for_industry(industry, business_model),
                "indirect_competitors": _get_indirect_competitors(industry, business_model),
                "potential_partners": _get_partners_for_industry(industry, business_model),
                "customer_segments": _get_customer_segments(industry, business_model),
                "market_trends": [
                    {"name": "Increasing AI adoption", "impact": "High", "opportunity_level": 0.8},
                    {"name": "Remote work transformation", "impact": "Medium", "opportunity_level": 0.7},
                    {"name": "Data privacy regulations", "impact": "Medium", "opportunity_level": 0.5}
                ]
            },
            "positioning": {
                "competitive_differentiators": [
                    "Advanced AI capabilities",
                    "Superior user experience",
                    "Enterprise-grade security"
                ],
                "whitespace_opportunities": [
                    "Integration with legacy systems",
                    "Vertical-specific solutions",
                    "Mid-market expansion"
                ]
            },
            "is_fallback": True
        }

# Helper functions for ecosystem map generation
def _get_competitors_for_industry(industry, business_model):
    competitors = {
        "software": ["Microsoft", "Oracle", "SAP", "Salesforce", "Adobe"],
        "fintech": ["PayPal", "Stripe", "Square", "Plaid", "Adyen"],
        "healthcare": ["UnitedHealth", "Teladoc", "Cerner", "Epic", "Doximity"],
        "ecommerce": ["Amazon", "Shopify", "eBay", "Etsy", "Walmart"]
    }
    return competitors.get(industry, ["Competitor 1", "Competitor 2", "Competitor 3"])

def _get_indirect_competitors(industry, business_model):
    indirect = {
        "software": ["Custom in-house solutions", "Legacy systems", "Open-source alternatives"],
        "fintech": ["Traditional banks", "Credit card networks", "Cash-based solutions"],
        "healthcare": ["Traditional healthcare providers", "Paper-based systems", "Mail-order services"],
        "ecommerce": ["Brick-and-mortar retail", "Direct-to-consumer brands", "Phone orders"]
    }
    return indirect.get(industry, ["Indirect Competitor 1", "Indirect Competitor 2"])

def _get_partners_for_industry(industry, business_model):
    partners = {
        "software": ["AWS", "Google Cloud", "Microsoft Azure", "Systems integrators", "Consultancies"],
        "fintech": ["Banks", "Credit card providers", "Accounting software providers", "KYC services"],
        "healthcare": ["Insurance providers", "Medical device manufacturers", "Pharmacies", "Labs"],
        "ecommerce": ["Payment processors", "Logistics providers", "Marketing agencies", "Review platforms"]
    }
    return partners.get(industry, ["Partner 1", "Partner 2", "Partner 3"])

def _get_customer_segments(industry, business_model):
    segments = {
        "software": ["Enterprises", "SMBs", "Individuals"],
        "fintech": ["Consumers", "Small businesses", "Enterprises"],
        "healthcare": ["Patients", "Providers", "Payers"],
        "ecommerce": ["Consumers", "Small businesses", "Enterprises"]
    }
    return segments.get(industry, ["Segment 1", "Segment 2", "Segment 3"])
