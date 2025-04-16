from scenario_runway_fix import safe_scenario_runway
import numpy as np
from typing import Tuple, Dict, List, Any, Optional, Union

def calculate_unit_economics(doc: dict) -> dict:
    """
    Calculate key unit economics metrics for a startup.
    
    This function computes metrics such as ARPU, CAC, LTV, 
    gross margin, churn rate, LTV:CAC ratio, and CAC payback period.
    
    Formula references:
    - LTV = Monthly Contribution Margin / Churn Rate
    - LTV:CAC Ratio = LTV / CAC
    - CAC Payback = CAC / Monthly Contribution Margin
    
    Args:
        doc (dict): Dictionary containing startup metrics including:
            - avg_revenue_per_user: Average revenue per user (monthly)
            - customer_acquisition_cost: Cost to acquire a customer
            - gross_margin_percent: Gross margin as a percentage
            - gross_margin: Gross margin as a decimal (alternative to gross_margin_percent)
            - churn_rate: Monthly customer churn rate
    
    Returns:
        dict: Dictionary containing calculated unit economics metrics:
            - arpu: Average revenue per user
            - cac: Customer acquisition cost
            - ltv: Lifetime value
            - gross_margin: Gross margin as a decimal
            - churn_rate: Monthly churn rate
            - ltv_cac_ratio: LTV to CAC ratio
            - cac_payback_months: Months to recoup CAC
    """
    # Extract values with defaults
    arpu = doc.get("avg_revenue_per_user", 0)
    cac = doc.get("customer_acquisition_cost", 0)
    gm = doc.get("gross_margin_percent", doc.get("gross_margin", 0.7))
    
    # Convert percentage to decimal if needed
    if gm > 1:
        gm = gm / 100
    
    # Ensure churn rate is valid (non-zero, non-negative)
    churn = doc.get("churn_rate", 0.05)
    if churn <= 0:
        churn = 0.01  # Set minimum churn to avoid division by zero
    
    # Calculate monthly contribution margin
    monthly_contribution = arpu * gm
    
    # Calculate lifetime value (LTV)
    ltv = monthly_contribution / churn if churn > 0 else 999999
    
    # Calculate LTV to CAC ratio
    ratio = ltv / cac if cac > 0 else 999999
    
    # Calculate CAC payback period in months
    payback = cac / monthly_contribution if monthly_contribution > 0 else 999999

    return {
        "arpu": arpu,
        "cac": cac,
        "ltv": ltv,
        "gross_margin": gm,
        "churn_rate": churn,
        "ltv_cac_ratio": ratio,
        "cac_payback_months": payback
    }
def scenario_runway(
    burn_rate: float, 
    current_cash: float,
    monthly_revenue: float = 0.0,
    rev_growth: float = 0.0,
    cost_growth: float = 0.0,
    months: int = 24
) -> Tuple[int, float, List[float]]:
    """Calculate cash runway under different growth assumptions."""
    return safe_scenario_runway(
        burn_rate=burn_rate,
        current_cash=current_cash,
        monthly_revenue=monthly_revenue,
        rev_growth=rev_growth,
        cost_growth=cost_growth,
        months=months
    )

    return (runway, cash, flow)

def forecast_financials(doc: dict, years: int = 5) -> Dict[str, Any]:
    """
    Generate detailed financial forecasts for a startup.
    
    This function creates monthly and annual financial projections including
    revenue, costs, gross profit, net profit, and cash flow based on
    the startup's current metrics and growth assumptions.
    
    Args:
        doc (dict): Dictionary containing startup metrics including:
            - monthly_revenue: Current monthly revenue
            - revenue_growth_rate: Monthly revenue growth rate
            - gross_margin_percent: Gross margin as percentage
            - gross_margin: Gross margin as decimal (alternative to gross_margin_percent)
            - burn_rate: Monthly burn rate (expenses)
            - burn_growth_rate: Monthly growth rate for expenses
            - current_cash: Current cash balance
            - stage: Startup stage (pre-seed, seed, series-a, etc.)
        years (int, optional): Number of years to forecast. Defaults to 5.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - monthly: Monthly projections for revenue, costs, gross profit, net profit, cash flow
            - annual: Annual totals for revenue, costs, profit
            - metrics: Key metrics including month/year of profitability, ending cash, CAGR
    """
    # Extract input parameters with defaults
    monthly_revenue = doc.get("monthly_revenue", 0)
    monthly_growth = doc.get("revenue_growth_rate", 0)
    if monthly_growth > 1:  # Convert percentage to decimal if needed
        monthly_growth /= 100
        
    gm = doc.get("gross_margin_percent", doc.get("gross_margin", 0.7))
    if gm > 1:  # Convert percentage to decimal if needed
        gm = gm / 100
        
    burn_rate = doc.get("burn_rate", 0)
    burn_growth = doc.get("burn_growth_rate", 0.05)
    
    # Initialize arrays for projections
    months = years * 12
    rev_arr = np.zeros(months)
    cost_arr = np.zeros(months)
    gp_arr = np.zeros(months)  # Gross profit
    np_arr = np.zeros(months)  # Net profit
    cash_flow = np.zeros(months)
    
    # Set initial values
    rev_arr[0] = monthly_revenue
    cost_arr[0] = burn_rate
    gp_arr[0] = rev_arr[0] * gm
    np_arr[0] = gp_arr[0] - cost_arr[0]
    
    # Get starting cash balance
    current_cash = doc.get("current_cash", 0)
    cash_balance = current_cash
    cash_flow[0] = cash_balance
    
    # Set growth decay rate based on startup stage
    # Growth typically slows more gradually for later-stage companies
    stage = doc.get("stage", "seed").lower()
    if stage in ["series-a", "series-b", "growth"]:
        growth_decay = 0.95  # 5% reduction in growth rate per year
    else:
        growth_decay = 0.93  # 7% reduction in growth rate per year
    
    # Calculate projections for each month
    for i in range(1, months):
        # Apply growth decay over time (growth slows as company matures)
        adj_growth = max(0.01, monthly_growth * (growth_decay ** (i/12)))
        
        # Calculate metrics for this month
        rev_arr[i] = rev_arr[i-1] * (1 + adj_growth)
        # Costs grow more slowly than revenue as company scales
        cost_arr[i] = cost_arr[i-1] * (1 + min(burn_growth, adj_growth * 0.5))
        gp_arr[i] = rev_arr[i] * gm
        np_arr[i] = gp_arr[i] - cost_arr[i]
        
        # Update cash balance
        cash_balance += np_arr[i]
        cash_flow[i] = cash_balance
    
    # Calculate annual totals
    annual_rev = []
    annual_cost = []
    annual_profit = []
    for y in range(years):
        start_idx = y * 12
        end_idx = start_idx + 12
        annual_rev.append(np.sum(rev_arr[start_idx:end_idx]))
        annual_cost.append(np.sum(cost_arr[start_idx:end_idx]))
        annual_profit.append(np.sum(np_arr[start_idx:end_idx]))
    
    # Find when the company becomes profitable
    profitable_month = -1
    for i in range(months):
        if np_arr[i] > 0 and profitable_month < 0:
            profitable_month = i + 1
            break
    profitable_year = profitable_month // 12 + 1 if profitable_month > 0 else -1
    
    # Calculate compound annual growth rate (CAGR)
    cagr = 0
    if rev_arr[0] > 0:
        cagr = ((rev_arr[-1] * 12) / (rev_arr[0] * 12)) ** (1 / years) - 1
    
    # Return detailed financial projections and metrics
    return {
        "monthly": {
            "revenue": rev_arr.tolist(),
            "costs": cost_arr.tolist(),
            "gross_profit": gp_arr.tolist(),
            "net_profit": np_arr.tolist(),
            "cash_flow": cash_flow.tolist()
        },
        "annual": {
            "revenue": annual_rev,
            "costs": annual_cost,
            "profit": annual_profit
        },
        "metrics": {
            "profitable_month": profitable_month,
            "profitable_year": profitable_year,
            "ending_cash": cash_flow[-1],
            "cagr": cagr
        }
    }

def calculate_valuation_metrics(doc: dict) -> dict:
    """
    Calculate valuation metrics for a startup using various methodologies.
    
    This function computes valuations based on:
    - Revenue multiple approach (adjusted for sector, growth, and stage)
    - Rule of 40 score
    - Berkhus method (5x forward ARR)
    
    Args:
        doc (dict): Dictionary containing startup metrics including:
            - monthly_revenue: Current monthly revenue
            - revenue_growth_rate: Monthly revenue growth rate
            - sector: Industry sector (saas, marketplace, ecommerce, etc.)
            - stage: Startup stage (pre-seed, seed, series-a, etc.)
            - operating_margin_percent: Operating margin as percentage
    
    Returns:
        dict: Dictionary containing valuation metrics:
            - revenue_multiple: Applied revenue multiple
            - arr_valuation: Valuation based on ARR multiple
            - rule_of_40_score: Score based on growth + profitability
            - berkhus_valuation: Valuation based on Berkhus method
            - forward_arr: Projected ARR for next year
            - annual_growth_rate: Annualized growth rate
            - justification: Explanation of multiple calculation
    """
    # Extract input parameters with defaults
    monthly_rev = doc.get("monthly_revenue", 0)
    annual_rev = monthly_rev * 12  # Calculate Annual Recurring Revenue (ARR)
    
    growth_rate = doc.get("revenue_growth_rate", 0)
    if growth_rate > 1:  # Convert percentage to decimal if needed
        growth_rate /= 100
        
    sector = doc.get("sector", "saas").lower()
    stage = doc.get("stage", "seed").lower()
    
    # Set base multiple based on sector
    # Different sectors command different revenue multiples
    if sector in ["saas", "software", "ai", "ml"]:
        base_multiple = 8  # SaaS companies typically have higher multiples
    elif sector in ["marketplace", "platform"]:
        base_multiple = 6
    elif sector in ["ecommerce", "d2c", "retail"]:
        base_multiple = 3  # Lower margins = lower multiples
    elif sector in ["fintech", "financial"]:
        base_multiple = 5
    elif sector in ["biotech", "healthcare", "medtech"]:
        base_multiple = 4
    else:
        base_multiple = 4  # Default multiple
    
    # Adjust multiple based on growth rate
    growth_adjustment = 0
    if growth_rate > 0.2:  # >20% monthly growth
        growth_adjustment = 4  # Significant premium for high growth
    elif growth_rate > 0.1:  # >10% monthly growth
        growth_adjustment = 2
    elif growth_rate > 0.05:  # >5% monthly growth
        growth_adjustment = 1
    
    # Adjust multiple based on startup stage
    stage_adjustment = 0
    if stage in ["pre-seed", "seed"]:
        stage_adjustment = 1  # Premium for early stage (higher risk but more growth potential)
    elif stage in ["series-c", "growth", "pre-ipo"]:
        stage_adjustment = -1  # Discount for later stage (lower risk but less growth potential)
    
    # Calculate final revenue multiple and ARR-based valuation
    revenue_multiple = base_multiple + growth_adjustment + stage_adjustment
    arr_valuation = annual_rev * revenue_multiple
    
    # Calculate Rule of 40 score (growth rate + profit margin)
    op_margin = doc.get("operating_margin_percent", 0)
    if op_margin > 1:
        op_margin = op_margin / 100
    
    # Convert monthly growth to annual growth
    annual_growth = ((1 + growth_rate) ** 12) - 1 if growth_rate > 0 else 0
    
    # Rule of 40: Sum of growth rate and profit margin should exceed 40%
    rule_of_40 = (annual_growth * 100) + (op_margin * 100)
    
    # Calculate forward ARR and Berkhus valuation (5x forward ARR)
    forward_arr = annual_rev * (1 + annual_growth)
    berkhus_val = forward_arr * 5
    
    # Return valuation metrics with explanation
    return {
        "revenue_multiple": revenue_multiple,
        "arr_valuation": arr_valuation,
        "rule_of_40_score": rule_of_40,
        "berkhus_valuation": berkhus_val,
        "forward_arr": forward_arr,
        "annual_growth_rate": annual_growth,
        "justification": f"Base multiple={base_multiple}, growth adj={growth_adjustment}, stage adj={stage_adjustment}"
    }

def calculate_discounted_cash_flow(
    doc: dict, 
    years: int = 5, 
    terminal_multiple: float = 10,
    discount_rate: float = 0.25
) -> dict:
    """
    Calculate company valuation using Discounted Cash Flow (DCF) method.
    
    This function projects future cash flows and discounts them back to present value.
    It also calculates a terminal value based on a multiple of the final year's revenue.
    
    Args:
        doc (dict): Dictionary containing startup metrics
        years (int, optional): Forecast period in years. Defaults to 5.
        terminal_multiple (float, optional): Exit multiple for terminal value. Defaults to 10.
        discount_rate (float, optional): Annual discount rate. Defaults to 0.25 (25%).
    
    Returns:
        dict: Dictionary containing DCF valuation metrics:
            - dcf_valuation: Total DCF valuation
            - present_value_cash_flows: PV of projected cash flows
            - terminal_value: Calculated terminal value
            - present_value_terminal: PV of terminal value
            - assumptions: Discount rate and terminal multiple used
    """
    # Generate financial forecast
    forecast = forecast_financials(doc, years)
    
    # Extract annual cash flows (using profit as proxy if no separate cash flow)
    annual_profits = forecast["annual"]["profit"]
    
    # Calculate present value of each year's cash flow
    present_values = []
    for i, cf in enumerate(annual_profits):
        year = i + 1
        present_values.append(cf / ((1 + discount_rate) ** year))
    
    # Calculate terminal value based on final year revenue with multiple
    final_year_revenue = forecast["annual"]["revenue"][-1]
    terminal_value = final_year_revenue * terminal_multiple
    
    # Discount terminal value to present
    present_value_terminal = terminal_value / ((1 + discount_rate) ** years)
    
    # Calculate total DCF valuation
    present_value_cash_flows = sum(present_values)
    dcf_valuation = present_value_cash_flows + present_value_terminal
    
    return {
        "dcf_valuation": dcf_valuation,
        "present_value_cash_flows": present_value_cash_flows,
        "terminal_value": terminal_value,
        "present_value_terminal": present_value_terminal,
        "annual_cash_flows": annual_profits,
        "discounted_cash_flows": present_values,
        "assumptions": {
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "forecast_years": years
        }
    }

def calculate_capital_requirements(
    doc: dict, 
    months_to_raise: int = 18,
    buffer_multiple: float = 1.5
) -> dict:
    """
    Calculate startup capital requirements based on burn rate and growth plans.
    
    This function estimates how much capital a startup should raise based on:
    - Current burn rate
    - Growth projections
    - Desired runway
    - Safety buffer
    
    Args:
        doc (dict): Dictionary containing startup metrics
        months_to_raise (int, optional): Desired runway in months. Defaults to 18.
        buffer_multiple (float, optional): Safety buffer multiplier. Defaults to 1.5.
    
    Returns:
        dict: Dictionary containing capital requirements:
            - base_capital_required: Capital needed for specified runway
            - recommended_raise: Capital needed with safety buffer
            - burn_rate_monthly: Current monthly burn rate
            - projected_monthly_burn: Projected average monthly burn
            - time_to_profitability: Estimated months until profitable
    """
    # Extract current burn rate
    burn_rate = doc.get("burn_rate", 0)
    monthly_revenue = doc.get("monthly_revenue", 0)
    
    # Calculate net burn (burn minus revenue)
    net_burn = max(0, burn_rate - monthly_revenue)
    
    # Generate financial forecast
    forecast = forecast_financials(doc, years=3)
    
    # Calculate average projected burn over the runway period
    costs = forecast["monthly"]["costs"][:months_to_raise]
    revenues = forecast["monthly"]["revenue"][:months_to_raise]
    
    # Calculate net burns for each month
    net_burns = [max(0, cost - rev) for cost, rev in zip(costs, revenues)]
    avg_projected_burn = sum(net_burns) / len(net_burns) if net_burns else 0
    
    # Calculate base capital required
    base_capital_required = sum(net_burns)
    
    # Add safety buffer
    recommended_raise = base_capital_required * buffer_multiple
    
    # Estimate time to profitability
    time_to_profitability = forecast["metrics"]["profitable_month"]
    
    return {
        "base_capital_required": base_capital_required,
        "recommended_raise": recommended_raise,
        "burn_rate_monthly": burn_rate,
        "net_burn_monthly": net_burn,
        "projected_monthly_burn": avg_projected_burn,
        "time_to_profitability": time_to_profitability,
        "assumptions": {
            "months_to_raise": months_to_raise,
            "buffer_multiple": buffer_multiple
        }
    }
