from typing import Dict, List, Union, Any

# CAMP Framework categories for metrics
CAMP_CATEGORIES: Dict[str, Dict[str, Union[str, float, List[str]]]] = {
    "capital": {
        "name": "Capital Efficiency",
        "weight": 0.30,
        "description": "Financial health, sustainability, and capital deployment efficiency",
        "metrics": [
            "monthly_revenue",
            "annual_recurring_revenue",
            "lifetime_value_ltv",
            "gross_margin_percent",
            "operating_margin_percent",
            "burn_rate",
            "runway_months",
            "cash_on_hand_million",
            "debt_ratio",
            "ltv_cac_ratio",
            "customer_acquisition_cost",
            "avg_revenue_per_user"
        ]
    },
    "market": {
        "name": "Market Dynamics",
        "weight": 0.25,
        "description": "Market opportunity, growth potential, and competitive positioning",
        "metrics": [
            "market_size",
            "market_growth_rate",
            "market_share",
            "user_growth_rate",
            "viral_coefficient",
            "category_leadership_score",
            "revenue_growth_rate",
            "net_retention_rate",
            "churn_rate",
            "community_growth_rate",
            "referral_rate",
            "current_users",
            "conversion_rate"
        ]
    },
    "advantage": {
        "name": "Advantage Moat",
        "weight": 0.20,
        "description": "Competitive advantages and defensibility",
        "metrics": [
            "patent_count",
            "technical_innovation_score",
            "product_maturity_score",
            "api_integrations_count",
            "scalability_score",
            "technical_debt_score",
            "product_security_score",
            "business_model_strength",
            "category_leadership_score",
            "data_moat_strength",
            "algorithm_uniqueness",
            "moat_score"
        ]
    },
    "people": {
        "name": "People & Performance",
        "weight": 0.25,
        "description": "Team strength, experience, and execution ability",
        "metrics": [
            "founder_exits",
            "founder_domain_exp_yrs",
            "employee_count",
            "founder_diversity_score",
            "management_satisfaction_score",
            "tech_talent_ratio",
            "has_cto",
            "has_cmo",
            "has_cfo",
            "employee_turnover_rate",
            "nps_score",
            "support_ticket_sla_percent",
            "support_ticket_volume",
            "team_score"
        ]
    },
    "intangible": {
        "name": "Intangible Factor",
        "weight": 0.20,
        "description": "AI-derived assessment of qualitative factors from pitch materials",
        "metrics": [
            "intangible",
            "pitch_sentiment"
        ]
    }
}

# Business models list
BUSINESS_MODELS: List[str] = [
    "SaaS",
    "Marketplace",
    "E-commerce",
    "Consumer",
    "Enterprise",
    "Hardware",
    "Biotech",
    "Fintech",
    "AI/ML",
    "Crypto/Blockchain",
    "Media",
    "EdTech"
]

# Startup stages
STARTUP_STAGES: List[str] = [
    "Pre-seed",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Series D+",
    "Growth",
    "Pre-IPO"
]

# CAMP scoring thresholds
CAMP_SCORE_BANDS: Dict[str, Dict[str, Any]] = {
    "outstanding": {"min": 80, "max": 100, "color": "#4caf50", "label": "Outstanding"},
    "strong": {"min": 65, "max": 80, "color": "#8bc34a", "label": "Strong"},
    "average": {"min": 50, "max": 65, "color": "#ff9800", "label": "Average"},
    "below-average": {"min": 0, "max": 50, "color": "#f44336", "label": "Below Average"}
}

# Brand Colors
BRAND_COLORS: Dict[str, str] = {
    "primary": "#4169E1",       # Royal Blue
    "secondary": "#38B6FF",     # Light Blue
    "accent": "#FF5C77",        # Coral
    "neutral": "#354259",       # Dark Blue-Gray
    "success": "#4CAF50",       # Green
    "warning": "#FF9800",       # Orange
    "danger": "#F44336",        # Red
    "error": "#F44336",         # Red (alias for danger)
    "info": "#2196F3",          # Info Blue
    "background": "#F8FAFC",    # Light Gray
    
    # Text colors
    "text_primary": "#1A202C",   # Dark text
    "text_secondary": "#4A5568", # Medium text
    "text_muted": "#A0AEC0",     # Light text
    
    # Border colors
    "border_color": "#E2E8F0",   # Border color
    "divider": "#E2E8F0",        # Divider color
}

# Chart color palettes
CHART_PALETTE: List[str] = [
    "#4169E1",  # Primary blue
    "#FF5C77",  # Coral
    "#38B6FF",  # Light blue
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#607D8B",  # Blue Gray
    "#E91E63",  # Pink
]

# Metric tooltips - comprehensive glossary of metrics with detailed explanations
METRIC_TOOLTIPS: Dict[str, str] = {
    "monthly_revenue": "Average monthly revenue over the past 3 months. A key indicator of current business performance.",
    "annual_recurring_revenue": "Predictable revenue component normalized on an annual basis. Critical for subscription businesses.",
    "lifetime_value_ltv": "Total revenue expected from a single customer throughout their relationship with your company.",
    "gross_margin_percent": "Percentage of revenue retained after accounting for direct costs of goods sold. Higher margins provide more flexibility for growth.",
    "operating_margin_percent": "Percentage of revenue remaining after both COGS and operating expenses. Indicates operational efficiency.",
    "burn_rate": "Rate at which the company is spending its cash reserves each month. Critical for runway calculations.",
    "runway_months": "Months of operation remaining at current burn rate before depleting cash reserves. Key survival metric.",
    "cash_on_hand_million": "Current cash reserves in millions of dollars. Important for assessing financial stability.",
    "debt_ratio": "Ratio of total debt to total assets. Lower values indicate stronger financial health.",
    "financing_round_count": "Number of funding rounds completed. Indicator of fundraising history and investor interest.",
    "monthly_active_users": "Number of unique users who performed an action in the last month. Key engagement metric.",
    "daily_active_users": "Number of unique users who performed an action in the last day. Indicates product stickiness.",
    "user_growth_rate": "Month-over-month percentage growth in active users. Strong indicator of product-market fit.",
    "churn_rate": "Percentage of customers who stop using your product in a given period. Lower is better.",
    "churn_cohort_6mo": "Percentage of users who stop using the product within 6 months. Indicator of long-term retention.",
    "activation_rate": "Percentage of users who complete the desired onboarding actions. Indicates initial product value.",
    "conversion_rate": "Percentage of users who convert from free to paid. Critical for assessing monetization efficiency.",
    "repeat_purchase_rate": "Percentage of customers who make more than one purchase. Indicates product satisfaction.",
    "referral_rate": "Percentage of new users who come from existing user referrals. Indicator of product virality.",
    "session_frequency": "Average number of sessions per user per month. Indicator of user engagement.",
    "product_maturity_score": "Rating of product completeness and stability on a scale of 0-10.",
    "product_security_score": "Evaluation of the product's security measures and resistance to threats.",
    "technical_debt_score": "Assessment of code quality, documentation, and maintainability. Lower technical debt enhances future development.",
    "release_frequency": "Number of product releases per month. Indicator of development velocity and agility.",
    "api_integrations_count": "Number of external APIs integrated with the product. Indicates ecosystem connectivity.",
    "uptime_percent": "Percentage of time the service is available and operational. Critical for SaaS businesses.",
    "scalability_score": "Assessment of the system's ability to handle growth in users, transactions, or data.",
    "esg_sustainability_score": "Rating of environmental, social, and governance practices.",
    "patent_count": "Number of patents or pending patent applications.",
    "technical_innovation_score": "Assessment of technical differentiation and innovation level of the product.",
    "employee_count": "Total number of full-time employees in the company.",
    "employee_turnover_rate": "Percentage of employees who leave the company annually.",
    "founder_diversity_score": "Measure of diversity within the founding team.",
    "management_satisfaction_score": "Employee satisfaction with management, typically measured through surveys.",
    "tech_talent_ratio": "Percentage of technical roles within the company. Indicator of technical focus.",
    "founder_exits": "Number of previous successful exits by the founding team. Indicator of entrepreneurial experience.",
    "founder_domain_exp_yrs": "Average years of domain experience among the founding team.",
    "founder_network_reach": "Size and quality of founders' professional networks. Can impact hiring, partnerships, and funding.",
    "board_experience_score": "Assessment of board members' relevant experience and track record.",
    "hiring_velocity_score": "Speed and efficiency of the hiring process. Indicator of organizational growth capability.",
    "nps_score": "Net Promoter Score, measuring customer loyalty and satisfaction.",
    "customer_acquisition_cost": "Total cost to acquire a new customer, including marketing and sales expenses.",
    "roi_on_ad_spend": "Return on investment for advertising expenditures.",
    "lead_conversion_percent": "Percentage of leads that convert to customers.",
    "organic_traffic_share": "Percentage of web traffic from organic (non-paid) sources.",
    "channel_partner_count": "Number of channel partners distributing or selling the product.",
    "support_ticket_volume": "Average monthly volume of customer support tickets.",
    "support_ticket_sla_percent": "Percentage of support tickets resolved within SLA targets.",
    "investor_interest_score": "Level of interest from investors based on meetings, follow-ups, and term sheets.",
    "category_leadership_score": "Company's leadership position within its category or market segment.",
    "business_model_strength": "Robustness and sustainability of the company's business model.",
    "market_size": "Total addressable market size in dollars.",
    "market_growth_rate": "Annual percentage growth rate of the target market.",
    "market_share": "Percentage of the total addressable market captured by the company.",
    "viral_coefficient": "Number of new users that each existing user generates. Values >1 indicate viral growth.",
    "revenue_growth_rate": "Year-over-year percentage increase in revenue.",
    "net_retention_rate": "Revenue retained from existing customers after accounting for expansion, downgrades, and churn.",
    "community_growth_rate": "Growth rate of the company's community (forums, social media, etc.).",
    "upsell_rate": "Percentage of customers who purchase additional products or upgrades.",
    "ltv_cac_ratio": "Ratio of customer lifetime value to acquisition cost. Ideal ratio is typically 3:1 or higher.",
    "default_rate": "Percentage of loans or payments that enter default status. Relevant for fintech startups.",
    "licenses_count": "Number of licenses granted or acquired. Relevant for IP-heavy businesses.",
    "clinical_phase": "Current phase of clinical trials. Relevant for biotech and healthcare startups.",
    "avg_revenue_per_user": "Average revenue generated by each user or customer, typically monthly.",
    "current_users": "Total number of current active users of the product or service.",
    "moat_score": "Overall assessment of competitive advantages and barriers to entry for competitors.",
    "data_moat_strength": "Competitive advantage derived from proprietary data and its strategic value.",
    "algorithm_uniqueness": "Uniqueness and effectiveness of core algorithms compared to alternatives.",
    "team_score": "Overall assessment of team quality, cohesion, and capability.",
    "has_cto": "Whether the company has a dedicated Chief Technology Officer.",
    "has_cmo": "Whether the company has a dedicated Chief Marketing Officer.",
    "has_cfo": "Whether the company has a dedicated Chief Financial Officer.",
    "intangible": "AI-derived assessment of qualitative aspects not captured by specific metrics.",
    "pitch_sentiment": "AI analysis of pitch materials, assessing clarity, confidence, and persuasiveness."
}

# Named metrics list - comprehensive list of all potential metrics used in the system
NAMED_METRICS_50: List[str] = [
    "monthly_revenue",
    "annual_recurring_revenue",
    "lifetime_value_ltv",
    "gross_margin_percent",
    "operating_margin_percent",
    "burn_rate",
    "runway_months",
    "cash_on_hand_million",
    "debt_ratio",
    "financing_round_count",
    "monthly_active_users",
    "daily_active_users",
    "user_growth_rate",
    "churn_rate",
    "churn_cohort_6mo",
    "activation_rate",
    "conversion_rate",
    "repeat_purchase_rate",
    "referral_rate",
    "session_frequency",
    "product_maturity_score",
    "product_security_score",
    "technical_debt_score",
    "release_frequency",
    "api_integrations_count",
    "uptime_percent",
    "scalability_score",
    "esg_sustainability_score",
    "patent_count",
    "technical_innovation_score",
    "employee_count",
    "employee_turnover_rate",
    "founder_diversity_score",
    "management_satisfaction_score",
    "tech_talent_ratio",
    "founder_exits",
    "founder_domain_exp_yrs",
    "founder_network_reach",
    "board_experience_score",
    "hiring_velocity_score",
    "nps_score",
    "customer_acquisition_cost",
    "roi_on_ad_spend",
    "lead_conversion_percent",
    "organic_traffic_share",
    "channel_partner_count",
    "support_ticket_volume",
    "support_ticket_sla_percent",
    "investor_interest_score",
    "category_leadership_score",
    "business_model_strength",
    "market_size",
    "market_growth_rate",
    "market_share",
    "viral_coefficient",
    "revenue_growth_rate",
    "net_retention_rate",
    "community_growth_rate",
    "upsell_rate",
    "ltv_cac_ratio",
    "default_rate",
    "licenses_count",
    "clinical_phase",
    "avg_revenue_per_user",
    "current_users",
    "moat_score",
    "data_moat_strength",
    "algorithm_uniqueness",
    "team_score",
    "has_cto",
    "has_cmo",
    "has_cfo",
    "intangible",
    "pitch_sentiment"
]
