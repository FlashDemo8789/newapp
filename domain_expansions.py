def apply_domain_expansions(doc: dict) -> dict:
    """
    Flash DNA analysis approach:
    expansions for fintech, biotech, saas, marketplace, crypto, etc.
    We unify expansions to a single dict so every doc has same dimension.
    """
    expansions = {
        "compliance_index": 0.0,
        "regulatory_risk": 0.0,
        "fraud_risk_factor": 0.0,

        "development_complexity": 0.0,
        "time_to_market_years": 0.0,

        "retention_factor": 0.0,
        "expansion_revenue_index": 0.0,

        "liquidity_score": 0.0,
        "cross_side_network_effect": 0.0,

        "token_utility_score": 0.0,
        "decentralization_factor": 0.0,
        "regulatory_uncertainty": 0.0,

        "data_moat_strength": 0.0,
        "algorithm_uniqueness": 0.0,

        "net_growth_factor": 0.0
    }

    sector = doc.get("sector","other").lower()

    if sector == "fintech":
        lic = doc.get("licenses_count",0)
        dfr = doc.get("default_rate",0.02)
        expansions["compliance_index"] = 10* lic - (dfr*100)
        expansions["regulatory_risk"] = min(1.0, max(0.0, dfr * 5))
        expansions["fraud_risk_factor"] = min(1.0, max(0.0, 0.5 - (lic * 0.05)))

    elif sector in ["biotech","healthtech"]:
        phase = doc.get("clinical_phase", 0)
        expansions["regulatory_risk"] = 1.0 if phase<=0 else (1.0/ phase)
        expansions["development_complexity"] = min(1.0, max(0.0,1.0 - (phase * 0.2)))
        expansions["time_to_market_years"] = max(1,8 - phase*1.5)

    elif sector == "saas":
        nr = doc.get("net_retention_rate",1.0)
        expansions["retention_factor"] = (nr -1)* 100
        expansions["expansion_revenue_index"] = max(0, (nr -1)* 200)

    elif sector == "marketplace":
        expansions["liquidity_score"] = min(100, doc.get("monthly_active_users",0)/ 1000)
        expansions["cross_side_network_effect"] = min(1.0, doc.get("viral_coefficient",0)*2)

    elif sector in ["crypto","blockchain"]:
        expansions["token_utility_score"]   = doc.get("token_utility_score",50)
        expansions["decentralization_factor"]= doc.get("decentralization_factor",0.5)
        expansions["regulatory_uncertainty"]= 0.7

    elif sector == "ai":
        expansions["data_moat_strength"]   = min(100, doc.get("data_volume_tb",0)*0.5)
        expansions["algorithm_uniqueness"] = doc.get("patent_count",0)*10

    # always net_growth_factor
    churn = doc.get("churn_rate",0.05)
    growth= doc.get("user_growth_rate",0.1)
    expansions["net_growth_factor"] = max(-1.0, growth - churn)

    return expansions

def get_sector_recommendations(doc: dict) -> list:
    """
    Flash DNA sector-based suggestions
    + NEW(UI) lines (no omissions).
    """
    sector = doc.get("sector","other").lower()
    stage  = doc.get("stage","seed").lower()
    recs   = []

    if sector == "fintech":
        recs.append("Focus on regulatory compliance and security certifications")
        if doc.get("licenses_count",0)<2:
            recs.append("Secure additional financial licenses to reduce regulatory risk")
        if doc.get("default_rate",0)>0.05:
            recs.append("Implement stronger risk assessment models to reduce default rates")

    elif sector in ["biotech","healthtech"]:
        recs.append("Accelerate clinical progress while strengthening IP portfolio")
        if doc.get("patent_count",0)<3:
            recs.append("Prioritize patent applications for core technology")
        if doc.get("clinical_phase",0)<2:
            recs.append("Focus resources on advancing to Phase 2 trials")

    elif sector == "saas":
        recs.append("Focus on reducing churn and increasing expansion revenue")
        if doc.get("net_retention_rate",1.0)<1.1:
            recs.append("Implement upsell/cross-sell strategy to boost net retention above 110%")
        if doc.get("churn_rate",0.05)>0.03:
            recs.append("Develop customer success program to reduce monthly churn below 3%")

    elif sector == "marketplace":
        recs.append("Prioritize liquidity in core segments before expanding")
        if doc.get("session_frequency",0)<3:
            recs.append("Increase engagement through gamification and retention hooks")

    elif sector in ["crypto","blockchain"]:
        recs.append("Clarify token utility and regulatory compliance approach")
        recs.append("Develop cross-chain compatibility to maximize market reach")

    elif sector == "ai":
        recs.append("Secure proprietary data sources to strengthen competitive moat")
        recs.append("Demonstrate clear ROI metrics for enterprise customers")

    # stage-specific Flash DNA expansions
    if stage in ["pre-seed","seed"]:
        recs.append("Focus on product-market fit before scaling go-to-market")
    elif stage == "series-a":
        recs.append("Develop scalable acquisition channels with predictable CAC")
    elif stage in ["series-b","series-c","growth"]:
        recs.append("Optimize unit economics to demonstrate path to profitability")

    return recs[:5]
