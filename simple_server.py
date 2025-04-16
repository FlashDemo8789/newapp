#!/usr/bin/env python3

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the analysis data once at startup
mock_analysis_data = {}
sample_path = os.path.join('data', 'analyses', 'sample_analysis.json')
if os.path.exists(sample_path):
    try:
        with open(sample_path, 'r') as f:
            mock_analysis_data = json.load(f)
    except Exception as e:
        print(f"Error loading sample analysis: {e}")

# Try to import module_exports
try:
    import module_exports
    HAS_MODULE_EXPORTS = True
    print("Successfully imported module_exports - analysis functions available")
except ImportError as e:
    HAS_MODULE_EXPORTS = False
    print(f"Warning: module_exports not available: {e}")

@app.route('/api/v1/status')
def status():
    return jsonify({
        'status': 'ok',
        'version': '1.0',
        'timestamp': '2025-04-16T12:00:00Z',
        'message': 'API is running',
        'module_exports_available': HAS_MODULE_EXPORTS
    })

@app.route('/api/v1/analyses')
def analyses():
    return jsonify([
        {
            "id": "mock-analysis-001",
            "name": "Sample Startup Analysis",
            "date": "2025-04-16T12:00:00Z",
            "stage": "Series A",
            "sector": "Software",
            "score": 0.78
        }
    ])

@app.route('/api/v1/analysis/<analysis_id>')
def get_analysis(analysis_id):
    # Return our mock data
    return jsonify(mock_analysis_data)

@app.route('/api/v1/analysis-details/<analysis_id>')
def get_analysis_details(analysis_id):
    # Same as get_analysis - return mock data
    return jsonify(mock_analysis_data)

@app.route('/api/v1/analysis/team', methods=['POST'])
def team_analysis():
    # Get the request data
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'No data provided',
            'status': 'error'
        }), 400
    
    # Use module_exports if available, otherwise return mock data
    if HAS_MODULE_EXPORTS:
        try:
            # Call the analyze_team function from module_exports
            result = module_exports.analyze_team(data)
            
            # Convert to standard frontend format
            formatted_result = {
                'id': result.get('analysis_id', str(uuid.uuid4())),
                'timestamp': result.get('analysis_timestamp', datetime.now().isoformat()),
                'scores': {
                    'overall': result.get('combined_score', 0) / 10,  # Normalize to 0-1 scale
                    'team_depth': result.get('team_depth_score', 0) / 10,
                    'moat': result.get('competitive_moat_score', 0) / 10,
                    'risk': 1 - result.get('execution_risk', 0)  # Invert risk score
                },
                'metrics': {
                    'experience_diversity': result.get('team_components', {}).get('team_diversity', 'Low'),
                    'domain_expertise': result.get('team_components', {}).get('domain_expertise', 'Low'),
                    'execution_track_record': result.get('risk_breakdown', {}).get('execution_history', 'Moderate'),
                    'leadership_quality': result.get('team_components', {}).get('management_satisfaction', 'Moderate')
                },
                'summary': "Team analysis shows " + (
                    "strong potential" if result.get('combined_score', 0) > 6 else 
                    "moderate strengths" if result.get('combined_score', 0) > 3 else 
                    "significant areas for improvement"
                ),
                'recommendations': [
                    {'title': rec.split(':')[0], 'description': rec.split(':', 1)[1].strip()} 
                    if ':' in rec else {'title': 'Recommendation', 'description': rec}
                    for rec in result.get('recommendations', [])
                ],
                'composition': {
                    'description': f"Team of {data.get('team_size', 0)} with {data.get('founders', 0)} founders, "
                                 f"{data.get('engineers', 0)} engineers, and other key roles.",
                    'roles': [
                        {
                            'title': 'Founders',
                            'count': data.get('founders', 0),
                            'description': 'Leadership and vision'
                        },
                        {
                            'title': 'Engineering',
                            'count': data.get('engineers', 0),
                            'description': 'Technical implementation'
                        },
                        {
                            'title': 'Sales',
                            'count': data.get('sales', 0),
                            'description': 'Revenue generation'
                        },
                        {
                            'title': 'Marketing',
                            'count': data.get('marketing', 0),
                            'description': 'Market positioning'
                        },
                        {
                            'title': 'Operations',
                            'count': data.get('operations', 0),
                            'description': 'Business operations'
                        }
                    ]
                },
                'moat': {
                    'description': "Team moat analysis evaluates how the team composition contributes to competitive advantage.",
                    'factors': [
                        {
                            'name': 'Domain Expertise', 
                            'description': 'Specialized knowledge in the industry'
                        },
                        {
                            'name': 'Prior Exits',
                            'description': 'Experience with successful exits'
                        },
                        {
                            'name': 'Team Completeness',
                            'description': 'Coverage of all critical business functions'
                        },
                        {
                            'name': 'Technical Talent',
                            'description': 'Engineering and product development capabilities'
                        }
                    ]
                }
            }
            
            return jsonify(formatted_result)
        except Exception as e:
            # Log the error and return a helpful message
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'error': str(e),
                'status': 'error',
                'trace': traceback.format_exc(),
                'message': 'Error running team analysis'
            }), 500
    else:
        # Return mock data if module_exports is not available
        return jsonify({
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'scores': {
                'overall': 0.68,
                'team_depth': 0.72,
                'moat': 0.55,
                'risk': 0.65
            },
            'metrics': {
                'experience_diversity': 'Moderate',
                'domain_expertise': 'High',
                'execution_track_record': 'Moderate',
                'leadership_quality': 'High'
            },
            'summary': "Mock team analysis for demonstration purposes. This startup has a moderately strong founding team with good domain expertise but could benefit from more diverse experience and additional key hires.",
            'recommendations': [
                {
                    'title': 'Hire Key Executives',
                    'description': 'Add experienced CTO and CMO to round out the executive team'
                },
                {
                    'title': 'Improve Domain Diversity',
                    'description': 'Bring in team members with experience in adjacent sectors'
                },
                {
                    'title': 'Advisory Board',
                    'description': 'Create an advisory board with industry veterans to complement the founding team'
                }
            ],
            'composition': {
                'description': f"Team of {data.get('team_size', 10)} with {data.get('founders', 2)} founders and key roles across product, engineering, and business functions.",
                'roles': [
                    {
                        'title': 'Founders',
                        'count': data.get('founders', 2),
                        'description': 'Leadership and vision'
                    },
                    {
                        'title': 'Engineering',
                        'count': data.get('engineers', 4),
                        'description': 'Technical implementation'
                    },
                    {
                        'title': 'Sales',
                        'count': data.get('sales', 2),
                        'description': 'Revenue generation'
                    },
                    {
                        'title': 'Marketing',
                        'count': data.get('marketing', 1),
                        'description': 'Market positioning'
                    },
                    {
                        'title': 'Operations',
                        'count': data.get('operations', 1),
                        'description': 'Business operations'
                    }
                ]
            },
            'moat': {
                'description': "The team provides a moderate competitive moat with strong domain expertise but lacks diversity in prior company-building experience.",
                'factors': [
                    {
                        'name': 'Domain Expertise', 
                        'description': 'Founders have 8+ years in the industry'
                    },
                    {
                        'name': 'Prior Exits',
                        'description': 'One founder has previous successful exit'
                    },
                    {
                        'name': 'Team Completeness',
                        'description': 'Missing key executive roles'
                    },
                    {
                        'name': 'Technical Talent',
                        'description': 'Strong engineering team with relevant skills'
                    }
                ]
            }
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'message': 'Simple Flash DNA API is running'
    })

# Industry and business model data
@app.route('/api/v1/industries')
def get_industries():
    return jsonify([
        "SaaS", "E-commerce", "FinTech", "HealthTech", "EdTech", 
        "AI/ML", "Cybersecurity", "MarTech", "Enterprise Software", "Consumer Apps",
        "Hardware", "IoT", "Clean Energy", "Biotech", "Gaming"
    ])

@app.route('/api/v1/business-models')
def get_business_models():
    return jsonify([
        "Subscription", "Freemium", "Marketplace", "E-commerce", 
        "SaaS", "PaaS", "IaaS", "Transaction Fees", "Advertising", 
        "Licensing", "Usage-Based", "Hardware + Services"
    ])

@app.route('/api/v1/metrics-tooltips')
def get_metrics_tooltips():
    return jsonify({
        "arr": "Annual Recurring Revenue",
        "cac": "Customer Acquisition Cost",
        "ltv": "Lifetime Value",
        "runway": "Cash Runway",
        "growth_rate": "Growth Rate"
    })

# Serve any other static files from the data directory
@app.route('/data/<path:path>')
def serve_static(path):
    return send_from_directory('data', path)

if __name__ == '__main__':
    # Make sure we can find the data directory
    if not os.path.exists('data/analyses'):
        os.makedirs('data/analyses', exist_ok=True)
        
    # Create a sample analysis file if it doesn't exist
    if not os.path.exists(sample_path):
        sample_data = {
            "id": "mock-analysis-001",
            "name": "Sample Startup Analysis",
            "description": "This is a sample startup analysis for demo purposes",
            "industry": "Software",
            "business_model": "SaaS",
            "stage": "Series A",
            "timestamp": "2025-04-16T12:00:00Z",
            "analysis_timestamp": "2025-04-16T12:00:00Z",
            "scores": {
                "capital": 0.75,
                "advantage": 0.82,
                "market": 0.68,
                "people": 0.77,
                "composite": 0.76
            },
            "summary": {
                "overview": "Sample startup with strong product-market fit",
                "key_strengths": [
                    "Strong team with domain expertise",
                    "Growing market with low competition",
                    "Robust SaaS business model"
                ]
            },
            "dna": {
                "attributes": {
                    "innovation": 0.85,
                    "execution": 0.72,
                    "scalability": 0.78,
                    "defensibility": 0.65
                }
            },
            "pmf": {
                "score": 0.82,
                "feedback": {
                    "positive": 65,
                    "neutral": 25,
                    "negative": 10
                }
            }
        }
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        # Load the data we just created    
        mock_analysis_data = sample_data
    
    print(f"Starting simple server with mock data for analysis ID: {mock_analysis_data.get('id', 'unknown')}")
    app.run(debug=True, host='0.0.0.0', port=5001) 