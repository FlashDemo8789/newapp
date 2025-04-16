"""
Adapter module for technical_due_diligence.py
This provides a standardized interface to the Technical Due Diligence analysis module
"""
from backend.adapters.base_adapter import BaseAnalysisAdapter
import json
import os
from datetime import datetime

class TechnicalDueDiligenceAdapter(BaseAnalysisAdapter):
    """Adapter for technical_due_diligence analysis module"""
    
    def __init__(self):
        """Initialize the adapter"""
        super().__init__('technical_due_diligence')
        self.main_function = "run_technical_analysis"
        self.fallback_functions = ["analyze_technical_stack", "assess_technical_architecture"]
    
    def get_mock_data(self):
        """Get mock data for Technical Due Diligence analysis"""
        # Try to load mock data from the React frontend
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            frontend_mock_path = os.path.join(root_dir, 'frontend/src/services/analysis/mockAnalysisData.js')
            if os.path.exists(frontend_mock_path):
                # This is a simple extraction approach - in production you'd use a proper parser
                with open(frontend_mock_path, 'r') as f:
                    content = f.read()
                    technical_section = self._extract_section(content, 'technicalMockData')
                    if technical_section:
                        return json.loads(technical_section)
        except Exception as e:
            print(f"Error loading mock data: {e}")
        
        # Fallback basic mock data
        return {
            "overallScore": 73,
            "riskLevel": "Medium",
            "architecture": {
                "rating": "B+",
                "score": 78,
                "description": "Evaluation of technical architecture quality",
                "components": [
                    {
                        "name": "System Design",
                        "score": 80,
                        "description": "Overall system architecture design"
                    },
                    {
                        "name": "Microservices Implementation",
                        "score": 75,
                        "description": "Quality of microservices structure"
                    },
                    {
                        "name": "API Design",
                        "score": 82,
                        "description": "Quality and consistency of API design"
                    }
                ]
            },
            "codeQuality": {
                "rating": "B",
                "score": 76,
                "description": "Evaluation of source code quality",
                "metrics": [
                    {
                        "name": "Test Coverage",
                        "value": "68%",
                        "benchmark": "75%",
                        "description": "Percentage of code covered by automated tests"
                    },
                    {
                        "name": "Code Complexity",
                        "value": "Medium",
                        "benchmark": "Low-Medium",
                        "description": "Cyclomatic complexity of codebase"
                    },
                    {
                        "name": "Documentation",
                        "value": "Adequate",
                        "benchmark": "Good",
                        "description": "Quality and completeness of documentation"
                    }
                ]
            },
            "scalability": {
                "rating": "B-",
                "score": 72,
                "description": "Ability to scale under increased load",
                "assessment": "The system should handle up to 10x current load with moderate infrastructure upgrades",
                "bottlenecks": [
                    {
                        "component": "Database",
                        "issue": "Potential connection pool limitations",
                        "severity": "Medium",
                        "remediation": "Implement connection pooling optimizations"
                    },
                    {
                        "component": "File Processing Service",
                        "issue": "Single-threaded implementation",
                        "severity": "High",
                        "remediation": "Refactor to asynchronous processing"
                    }
                ]
            },
            "security": {
                "rating": "B",
                "score": 75,
                "description": "Security posture assessment",
                "vulnerabilities": [
                    {
                        "type": "Authentication",
                        "severity": "Medium",
                        "description": "Token expiration policy too generous",
                        "remediation": "Reduce token lifetime and implement refresh tokens"
                    },
                    {
                        "type": "Data Encryption",
                        "severity": "Low",
                        "description": "Some PII not encrypted at rest",
                        "remediation": "Implement field-level encryption for all PII"
                    }
                ]
            },
            "technicalDebt": {
                "rating": "C+",
                "score": 65,
                "description": "Assessment of accumulated technical debt",
                "areas": [
                    {
                        "name": "Legacy Components",
                        "severity": "High",
                        "description": "Several core components using outdated frameworks",
                        "remediationEffort": "3-4 months",
                        "impact": "Medium"
                    },
                    {
                        "name": "Testing Gaps",
                        "severity": "Medium",
                        "description": "Integration test coverage inadequate",
                        "remediationEffort": "1-2 months",
                        "impact": "High"
                    }
                ]
            },
            "recommendations": [
                {
                    "area": "Architecture",
                    "recommendation": "Refactor file processing service to asynchronous model",
                    "priority": "High",
                    "effort": "Medium",
                    "impact": "High"
                },
                {
                    "area": "Security",
                    "recommendation": "Implement field-level encryption for PII",
                    "priority": "Medium",
                    "effort": "Low",
                    "impact": "Medium"
                },
                {
                    "area": "Technical Debt",
                    "recommendation": "Upgrade legacy components to current frameworks",
                    "priority": "Medium",
                    "effort": "High",
                    "impact": "Medium"
                }
            ]
        }
    
    def _extract_section(self, content, section_name):
        """Extract a JavaScript object literal from a JavaScript file"""
        try:
            # This is a simplistic approach - in production, use a proper JS parser
            start_marker = f"export const {section_name} = "
            start = content.find(start_marker)
            if start == -1:
                return None
            
            start = content.find('{', start)
            if start == -1:
                return None
            
            # Track nested braces to find the end of the object
            brace_count = 1
            end = start + 1
            while brace_count > 0 and end < len(content):
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            if brace_count != 0:
                return None
            
            # Extract the object literal
            js_object = content[start:end]
            
            # Convert JavaScript to JSON
            # This is a very simplistic approach - in production use a proper converter
            js_object = js_object.replace('true', 'true')
            js_object = js_object.replace('false', 'false')
            js_object = js_object.replace('null', 'null')
            
            # Handle trailing commas (valid in JS, invalid in JSON)
            js_object = js_object.replace(',}', '}')
            js_object = js_object.replace(',]', ']')
            
            return js_object
        except Exception as e:
            print(f"Error extracting section: {e}")
            return None

# Create a singleton instance
technical_due_diligence_adapter = TechnicalDueDiligenceAdapter()

# Export the main function
def run_technical_analysis(data):
    """
    Run technical due diligence analysis
    
    Parameters:
    data (dict): Input data about the technical aspects
    
    Returns:
    dict: Formatted technical analysis results matching the React frontend structure
    """
    return technical_due_diligence_adapter.run_analysis(data)
