from flask import Blueprint, request, jsonify
from services.analysis_service import AnalysisService
from datetime import datetime

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/v1/analysis')
status_bp = Blueprint('status', __name__, url_prefix='/api/v1')
root_bp = Blueprint('root', __name__, url_prefix='')

analysis_service = AnalysisService()

def init_routes(app):
    app.register_blueprint(analysis_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(root_bp)

# General analysis endpoints
@analysis_bp.route('/compute', methods=['POST'])
def compute_analysis():
    data = request.json
    result = analysis_service.compute_full_analysis(data)
    return jsonify(result)

@analysis_bp.route('/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    result = analysis_service.get_analysis_by_id(analysis_id)
    return jsonify(result)

# Frontend compatibility routes
@status_bp.route('/analyses', methods=['GET'])
def get_analyses():
    """Return list of available analyses for frontend compatibility"""
    return jsonify([
        {
            "id": "mock-analysis-001",
            "name": "Sample Startup Analysis",
            "date": datetime.now().isoformat(),
            "stage": "Series A",
            "sector": "Software",
            "score": 0.78
        }
    ])

@status_bp.route('/analysis-details/<analysis_id>', methods=['GET'])
def get_analysis_details(analysis_id):
    """Route for frontend compatibility - maps to the main get_analysis function"""
    result = analysis_service.get_analysis_by_id(analysis_id)
    return jsonify(result)

@analysis_bp.route('/<analysis_id>/tabs/<tab_name>', methods=['GET'])
def get_tab_data(analysis_id, tab_name):
    result = analysis_service.get_tab_data(analysis_id, tab_name)
    return jsonify(result)

# Specific analysis type endpoints
@analysis_bp.route('/monteCarlo', methods=['POST'])
def monte_carlo_analysis():
    data = request.json
    result = analysis_service.run_monte_carlo_analysis(data)
    return jsonify(result)

@analysis_bp.route('/teamMoat', methods=['POST'])
def team_moat_analysis():
    data = request.json
    result = analysis_service.run_team_moat_analysis(data)
    return jsonify(result)

@analysis_bp.route('/acquisition', methods=['POST'])
def acquisition_analysis():
    data = request.json
    result = analysis_service.run_acquisition_analysis(data)
    return jsonify(result)

@analysis_bp.route('/pmf', methods=['POST'])
def pmf_analysis():
    data = request.json
    result = analysis_service.run_pmf_analysis(data)
    return jsonify(result)

@analysis_bp.route('/technical', methods=['POST'])
def technical_analysis():
    data = request.json
    result = analysis_service.run_technical_analysis(data)
    return jsonify(result)

@analysis_bp.route('/competitive', methods=['POST'])
def competitive_analysis():
    data = request.json
    result = analysis_service.run_competitive_analysis(data)
    return jsonify(result)

@analysis_bp.route('/exitPath', methods=['POST'])
def exit_path_analysis():
    data = request.json
    result = analysis_service.run_exit_path_analysis(data)
    return jsonify(result)

@analysis_bp.route('/pattern', methods=['POST'])
def pattern_analysis():
    data = request.json
    result = analysis_service.run_pattern_analysis(data)
    return jsonify(result)

@analysis_bp.route('/clustering', methods=['POST'])
def clustering_analysis():
    data = request.json
    result = analysis_service.run_clustering_analysis(data)
    return jsonify(result)

@analysis_bp.route('/benchmarks', methods=['POST'])
def benchmarks_analysis():
    data = request.json
    result = analysis_service.run_benchmarks_analysis(data)
    return jsonify(result)

@analysis_bp.route('/campDetails', methods=['POST'])
def camp_details_analysis():
    data = request.json
    result = analysis_service.run_camp_details_analysis(data)
    return jsonify(result)

@analysis_bp.route('/cohort', methods=['POST'])
def cohort_analysis():
    data = request.json
    result = analysis_service.run_cohort_analysis(data)
    return jsonify(result)

@analysis_bp.route('/dna', methods=['POST'])
def dna_analysis():
    data = request.json
    result = analysis_service.run_dna_analysis(data)
    return jsonify(result)

# Report generation endpoint
@analysis_bp.route('/<analysis_id>/report', methods=['POST'])
def generate_report():
    analysis_id = request.view_args['analysis_id']
    options = request.json
    result = analysis_service.generate_report(analysis_id, options)
    return jsonify(result)

# Add a simple status endpoint route
@status_bp.route('/status', methods=['GET'])
def api_status():
    """Simple endpoint to check API status"""
    return jsonify({
        'status': 'ok',
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'message': 'API is running'
    })

# Add a startup stages endpoint
@status_bp.route('/startup-stages', methods=['GET'])
def get_startup_stages():
    """Return list of startup stages"""
    return jsonify([
        "Pre-seed", "Seed", "Series A", "Series B", "Series C", "Series D+", "Growth"
    ])

# Debug endpoint to log received data
@analysis_bp.route('/debug', methods=['POST'])
def debug_analysis():
    """Debug endpoint that logs and returns the received data"""
    data = request.json
    print(f"DEBUG received data: {data}")
    return jsonify({
        'status': 'debug',
        'received_data': data,
        'timestamp': datetime.now().isoformat()
    })

# Add routes for the missing endpoints that are being requested by the frontend
@status_bp.route('/industries', methods=['GET'])
def get_industries():
    """Return list of supported industries"""
    return jsonify([
        "SaaS", "E-commerce", "FinTech", "HealthTech", "EdTech", 
        "AI/ML", "Cybersecurity", "MarTech", "Enterprise Software", "Consumer Apps",
        "Hardware", "IoT", "Clean Energy", "Biotech", "Gaming"
    ])

@status_bp.route('/business-models', methods=['GET'])
def get_business_models():
    """Return list of supported business models"""
    return jsonify([
        "Subscription", "Freemium", "Marketplace", "E-commerce", 
        "SaaS", "PaaS", "IaaS", "Transaction Fees", "Advertising", 
        "Licensing", "Usage-Based", "Hardware + Services"
    ])

@status_bp.route('/metrics-tooltips', methods=['GET'])
def get_metrics_tooltips():
    """Return tooltips for metrics"""
    return jsonify({
        "arr": "Annual Recurring Revenue - Total value of recurring revenue normalized to a one-year period",
        "cac": "Customer Acquisition Cost - Total cost to acquire a new customer",
        "ltv": "Lifetime Value - Predicted revenue from a customer over their entire relationship",
        "nrr": "Net Revenue Retention - Measures expansion and contraction of existing customers",
        "runway": "Cash Runway - How long your current cash will last at current burn rate",
        "growth_rate": "Year-over-Year revenue growth rate",
        "gross_margin": "Revenue minus cost of goods sold, divided by revenue",
        "burn_rate": "Rate at which a company uses up its cash reserves",
        "conversion_rate": "Percentage of leads that convert to paying customers",
        "churn_rate": "Percentage of customers who cancel/don't renew subscriptions"
    })
