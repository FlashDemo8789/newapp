import logging
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.database import (
    Analysis, CapitalMetricsDB, AdvantageMetricsDB, 
    MarketMetricsDB, PeopleMetricsDB, Recommendation,
    Strength, Improvement
)
import uuid

logger = logging.getLogger("flashdna.seed_data")

def generate_seed_data(db: Session, count: int = 10) -> List[str]:
    """
    Generate seed data for development and testing
    
    Args:
        db: Database session
        count: Number of analysis records to generate
        
    Returns:
        List of generated analysis IDs
    """
    logger.info(f"Generating {count} seed analyses")
    
    # Check if we already have data
    existing_count = db.query(Analysis).count()
    if existing_count > 0:
        logger.info(f"Database already contains {existing_count} analyses, skipping seed data generation")
        # Return IDs of existing analyses
        return [a.id for a in db.query(Analysis.id).limit(count).all()]
    
    # Sample company names
    company_names = [
        "TechNova", "DataSphere", "QuantumLeap", "NexusAI", "CloudPulse", 
        "InnovateX", "CyberSync", "BioGenetics", "EcoSustain", "FinEdge",
        "MediTrack", "EdTechPro", "RetailFlow", "AgriTech", "SpaceVentures",
        "VirtualReality", "BlockchainWave", "AudioStream", "SocialConnect", "FoodDelivery"
    ]
    
    # Sample industries
    industries = ["SaaS", "FinTech", "HealthTech", "EdTech", "E-commerce", 
                 "AI/ML", "Blockchain", "IoT", "Clean Energy", "Biotech"]
    
    # Sample business models
    business_models = ["B2B", "B2C", "B2B2C", "Marketplace", "SaaS", 
                      "Subscription", "Freemium", "On-demand", "Ad-supported"]
    
    # Sample stages
    stages = ["Pre-seed", "Seed", "Series A", "Series B", "Series C", "Growth", "Late Stage"]
    
    # Sample recommendations
    recommendation_templates = [
        "Focus on improving {dimension} metrics to strengthen overall performance",
        "Consider optimizing {metric} to enhance {dimension} score",
        "Invest in {dimension} development to gain competitive advantage",
        "Strengthen {metric} to improve {dimension} positioning in the market",
        "Address challenges in {dimension} to improve investor attractiveness",
        "Leverage strong {dimension} performance to accelerate growth",
        "Develop a strategic plan to improve {metric} within 6 months",
        "Maintain current strength in {dimension} while focusing on {other_dimension}",
        "Conduct regular assessment of {metric} to ensure continued success",
        "Benchmark {dimension} performance against industry leaders"
    ]
    
    # Sample strengths
    strength_templates = [
        "Strong {dimension} fundamentals provide a solid foundation for growth",
        "Exceptional {metric} performance compared to industry benchmarks",
        "Robust {dimension} strategy demonstrating market leadership",
        "Innovative approach to {metric} creates significant competitive advantage",
        "Well-executed {dimension} positioning for long-term success",
        "Industry-leading {metric} metrics indicating sustainable advantage",
        "Strategic excellence in {dimension} management",
        "Compelling {metric} results attracting investor interest",
        "Advanced {dimension} infrastructure supporting rapid scaling",
        "Optimized {metric} driving overall performance"
    ]
    
    # Sample improvement areas
    improvement_templates = [
        "{dimension} metrics showing room for optimization",
        "{metric} performance below industry benchmarks",
        "{dimension} strategy requiring refinement for maximum impact",
        "{metric} approach needing modernization to meet market standards",
        "{dimension} execution showing inconsistency",
        "{metric} efficiency presenting opportunity for enhancement",
        "{dimension} scaling challenges requiring strategic attention",
        "{metric} optimization needed for competitive positioning",
        "{dimension} resources requiring better allocation",
        "Untapped potential in {metric} development"
    ]
    
    # Generate analyses
    generated_ids = []
    
    for i in range(count):
        # Generate base scores for dimensions (0.3 to 0.95)
        capital_score = random.uniform(0.3, 0.95)
        advantage_score = random.uniform(0.3, 0.95)
        market_score = random.uniform(0.3, 0.95)
        people_score = random.uniform(0.3, 0.95)
        
        # Calculate overall score (weighted average)
        overall_score = (capital_score * 0.25 + advantage_score * 0.25 + 
                         market_score * 0.25 + people_score * 0.25)
        
        # Success probability
        success_probability = min(0.9, max(0.1, overall_score + random.uniform(-0.1, 0.1)))
        
        # Choose company profile
        company_name = random.choice(company_names)
        industry = random.choice(industries)
        business_model = random.choice(business_models)
        stage = random.choice(stages)
        
        # Generate founding date (1-5 years ago)
        days_ago = random.randint(365, 365 * 5)
        founding_date = (datetime.now() - timedelta(days=days_ago)).date()
        
        # Create analysis record
        analysis = Analysis(
            id=str(uuid.uuid4()),
            name=f"{company_name} ({i+1})",
            industry=industry,
            stage=stage,
            business_model=business_model,
            founding_date=founding_date,
            overall_score=overall_score,
            capital_score=capital_score,
            advantage_score=advantage_score,
            market_score=market_score,
            people_score=people_score,
            success_probability=success_probability
        )
        
        db.add(analysis)
        db.flush()  # Get the ID without committing
        
        generated_ids.append(analysis.id)
        
        # Generate Capital metrics
        capital_metrics = CapitalMetricsDB(
            analysis_id=analysis.id,
            monthly_revenue=random.uniform(10000, 500000),
            annual_recurring_revenue=random.uniform(100000, 5000000),
            burn_rate=random.uniform(20000, 300000),
            runway_months=random.uniform(6, 36),
            cash_balance=random.uniform(50000, 2000000),
            gross_margin=random.uniform(0.2, 0.8),
            ltv_cac_ratio=random.uniform(1, 5),
            customer_acquisition_cost=random.uniform(500, 5000),
            unit_economics_score=random.uniform(0.3, 0.9)
        )
        db.add(capital_metrics)
        
        # Generate Advantage metrics
        advantage_metrics = AdvantageMetricsDB(
            analysis_id=analysis.id,
            competition_level=random.randint(3, 9),
            patents_count=random.randint(0, 10),
            tech_innovation_score=random.uniform(0.3, 0.9),
            technical_debt_ratio=random.uniform(0.1, 0.7),
            network_effects_score=random.uniform(0.2, 0.9),
            ip_protection_level=random.uniform(0.2, 0.8),
            data_moat_score=random.uniform(0.1, 0.9),
            business_model_strength=random.uniform(0.3, 0.9),
            api_integrations_count=random.randint(1, 20),
            product_security_score=random.uniform(0.5, 0.95)
        )
        db.add(advantage_metrics)
        
        # Generate Market metrics
        market_metrics = MarketMetricsDB(
            analysis_id=analysis.id,
            tam_size=random.uniform(1e9, 1e11),  # $1B to $100B
            sam_size=random.uniform(1e8, 1e10),  # $100M to $10B
            market_growth_rate=random.uniform(0.05, 0.4),
            user_growth_rate=random.uniform(0.1, 1.0),
            active_users=random.randint(1000, 1000000),
            retention_rate=random.uniform(0.6, 0.95),
            market_penetration=random.uniform(0.01, 0.3),
            industry_trends_score=random.uniform(0.3, 0.9),
            category_leadership_score=random.uniform(0.2, 0.8),
            churn_rate=random.uniform(0.05, 0.3),
            viral_coefficient=random.uniform(0.5, 1.5)
        )
        db.add(market_metrics)
        
        # Generate People metrics
        people_metrics = PeopleMetricsDB(
            analysis_id=analysis.id,
            team_size=random.randint(5, 200),
            founder_experience=random.uniform(0.3, 0.9),
            technical_skill_score=random.uniform(0.4, 0.9),
            leadership_score=random.uniform(0.3, 0.9),
            diversity_score=random.uniform(0.2, 0.9),
            employee_turnover_rate=random.uniform(0.05, 0.4),
            nps_score=random.randint(-20, 80),
            support_ticket_sla_percent=random.uniform(0.7, 0.98),
            support_ticket_volume=random.randint(50, 5000),
            team_score=random.uniform(0.4, 0.9),
            founder_domain_exp_yrs=random.randint(1, 20)
        )
        db.add(people_metrics)
        
        # Generate recommendations (3-5)
        dimensions = ["capital", "advantage", "market", "people"]
        metrics = {
            "capital": ["revenue", "cash flow", "unit economics", "margins", "runway"],
            "advantage": ["technology", "patents", "innovation", "IP", "product"],
            "market": ["growth", "retention", "user acquisition", "market fit", "positioning"],
            "people": ["talent", "leadership", "team", "culture", "skills"]
        }
        
        rec_count = random.randint(3, 5)
        for j in range(rec_count):
            dimension = random.choice(dimensions)
            other_dimension = random.choice([d for d in dimensions if d != dimension])
            metric = random.choice(metrics[dimension])
            
            template = random.choice(recommendation_templates)
            recommendation = template.format(
                dimension=dimension.title(),
                other_dimension=other_dimension.title(),
                metric=metric
            )
            
            db_rec = Recommendation(
                analysis_id=analysis.id,
                text=recommendation,
                category=dimension,
                priority=rec_count - j
            )
            db.add(db_rec)
        
        # Generate strengths (2-4)
        strength_count = random.randint(2, 4)
        best_dimensions = sorted(
            [("capital", capital_score), ("advantage", advantage_score), 
             ("market", market_score), ("people", people_score)],
            key=lambda x: x[1], reverse=True
        )
        
        for j in range(strength_count):
            dimension = best_dimensions[j % 4][0]
            metric = random.choice(metrics[dimension])
            
            template = random.choice(strength_templates)
            strength = template.format(
                dimension=dimension.title(),
                metric=metric
            )
            
            db_strength = Strength(
                analysis_id=analysis.id,
                text=strength,
                category=dimension,
                score=1.0 - (j * 0.1)  
            )
            db.add(db_strength)
        
        # Generate improvement areas (2-4)
        improvement_count = random.randint(2, 4)
        worst_dimensions = sorted(
            [("capital", capital_score), ("advantage", advantage_score), 
             ("market", market_score), ("people", people_score)],
            key=lambda x: x[1]
        )
        
        for j in range(improvement_count):
            dimension = worst_dimensions[j % 4][0]
            metric = random.choice(metrics[dimension])
            
            template = random.choice(improvement_templates)
            improvement = template.format(
                dimension=dimension.title(),
                metric=metric
            )
            
            db_improvement = Improvement(
                analysis_id=analysis.id,
                text=improvement,
                category=dimension,
                score=1.0 - (j * 0.1)
            )
            db.add(db_improvement)
    
    # Commit all changes
    db.commit()
    logger.info(f"Successfully generated {count} seed analyses")
    
    return generated_ids

def seed_database(count: int = 10) -> List[str]:
    """
    Seed the database with sample data
    
    Args:
        count: Number of analysis records to generate
        
    Returns:
        List of generated analysis IDs
    """
    db = SessionLocal()
    try:
        return generate_seed_data(db, count)
    finally:
        db.close()

if __name__ == "__main__":
    # Set up logging for CLI execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Generate seed data
    generated_ids = seed_database(10)
    print(f"Generated {len(generated_ids)} seed analyses")
    print("Analysis IDs:")
    for id in generated_ids:
        print(f"  - {id}")
