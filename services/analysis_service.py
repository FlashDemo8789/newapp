import sys
import os
import json
import time
from datetime import datetime

# Add the root directory to Python path to import the analysis files
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

# Import adapter modules
from backend.adapters import monte_carlo_adapter
from backend.adapters import team_moat_adapter
from backend.adapters import acquisition_fit_adapter
from backend.adapters import pmf_adapter
from backend.adapters import technical_due_diligence_adapter
from backend.adapters import competitive_intelligence_adapter
from backend.adapters import exit_path_adapter
from backend.adapters import pattern_adapter
from backend.adapters import clustering_adapter
from backend.adapters import benchmarks_adapter
from backend.adapters import camp_details_adapter
from backend.adapters import cohort_adapter
from backend.adapters import dna_adapter
from backend.utils.path_utils import ensure_path_setup

# Ensure all paths are set up correctly
ensure_path_setup()

class AnalysisService:
    def __init__(self):
        # Initialize analysis modules via adapters
        self.monte_carlo = monte_carlo_adapter
        self.team_moat = team_moat_adapter
        self.acquisition_fit = acquisition_fit_adapter
        self.product_market_fit = pmf_adapter
        self.technical_due_diligence = technical_due_diligence_adapter
        self.competitive_intelligence = competitive_intelligence_adapter
        self.exit_path = exit_path_adapter
        self.pattern = pattern_adapter
        self.clustering = clustering_adapter
        self.benchmarks = benchmarks_adapter
        self.camp_details = camp_details_adapter
        self.cohort = cohort_adapter
        self.dna = dna_adapter
        
        # For demo/testing, store analyses in memory
        # In production, this would be a database
        self.analyses = {}

    def compute_full_analysis(self, data):
        """Run a full analysis on all dimensions"""
        analysis_id = data.get('id', f"analysis-{int(time.time())}")
        
        try:
            # Run individual analyses
            result = {
                'id': analysis_id,
                'name': data.get('name', 'Flash DNA Analysis'),
                'description': data.get('description', 'Comprehensive startup analysis'),
                'createdAt': datetime.now().isoformat(),
                'updatedAt': datetime.now().isoformat(),
                'status': 'completed',
                
                # Analysis tab data
                'monteCarlo': self.run_monte_carlo_analysis(data),
                'teamMoat': self.run_team_moat_analysis(data),
                'acquisition': self.run_acquisition_analysis(data),
                'pmf': self.run_pmf_analysis(data),
                'technical': self.run_technical_analysis(data),
                'competitive': self.run_competitive_analysis(data),
                'exitPath': self.run_exit_path_analysis(data),
                'pattern': self.run_pattern_analysis(data),
                'clustering': self.run_clustering_analysis(data),
                'benchmarks': self.run_benchmarks_analysis(data),
                'campDetails': self.run_camp_details_analysis(data),
                'cohort': self.run_cohort_analysis(data),
                'dna': self.run_dna_analysis(data)
            }
            
            # Store the analysis
            self.analyses[analysis_id] = result
            
            return result
        except Exception as e:
            print(f"Error in full analysis computation: {e}")
            return {
                'id': analysis_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_analysis_by_id(self, analysis_id):
        """Get a previously computed analysis by ID"""
        # Check if we have this analysis stored
        if analysis_id in self.analyses:
            return self.analyses[analysis_id]
        
        # Special handling for mock-analysis-001
        if analysis_id == 'mock-analysis-001':
            # Load from the sample file
            import os
            import json
            
            sample_path = os.path.join('backend', 'data', 'analyses', 'sample_analysis.json')
            if os.path.exists(sample_path):
                try:
                    with open(sample_path, 'r') as f:
                        sample_data = json.load(f)
                    
                    # Store it for future access
                    self.analyses[analysis_id] = sample_data
                    return sample_data
                except Exception as e:
                    print(f"Error loading sample analysis: {e}")
                    # Fall through to mock generation if file read fails
        
        # Check if we have a sample file with this ID
        import os
        import json
        
        sample_path = os.path.join('backend', 'data', 'analyses', 'sample_analysis.json')
        if os.path.exists(sample_path):
            try:
                with open(sample_path, 'r') as f:
                    sample_data = json.load(f)
                
                if sample_data.get('id') == analysis_id or analysis_id == 'sample-123':
                    # Store it for future access
                    self.analyses[analysis_id] = sample_data
                    return sample_data
            except Exception as e:
                print(f"Error loading sample analysis: {e}")
        
        # If not found, generate a mock analysis
        # In production, you'd return a 404 error
        mock_result = self.compute_full_analysis({
            'id': analysis_id,
            'name': f"Analysis {analysis_id}"
        })
        return mock_result
    
    def get_tab_data(self, analysis_id, tab_name):
        """Get specific tab data for an analysis"""
        # Get the full analysis
        analysis = self.get_analysis_by_id(analysis_id)
        
        # Return the specific tab data if it exists
        if tab_name in analysis:
            return analysis[tab_name]
        
        # If tab doesn't exist, run a specific analysis
        analysis_functions = {
            'monteCarlo': self.run_monte_carlo_analysis,
            'teamMoat': self.run_team_moat_analysis,
            'acquisition': self.run_acquisition_analysis,
            'pmf': self.run_pmf_analysis,
            'technical': self.run_technical_analysis,
            'competitive': self.run_competitive_analysis,
            'exitPath': self.run_exit_path_analysis,
            'pattern': self.run_pattern_analysis,
            'clustering': self.run_clustering_analysis,
            'benchmarks': self.run_benchmarks_analysis,
            'campDetails': self.run_camp_details_analysis,
            'cohort': self.run_cohort_analysis,
            'dna': self.run_dna_analysis
        }
        
        if tab_name in analysis_functions:
            # Use the analysis ID as input data
            tab_result = analysis_functions[tab_name]({'id': analysis_id})
            
            # Update the stored analysis
            analysis[tab_name] = tab_result
            self.analyses[analysis_id] = analysis
            
            return tab_result
        
        # If tab doesn't exist and no matching function, return empty result
        return {}
    
    def generate_report(self, analysis_id, options):
        """Generate a report for an analysis"""
        # Get the analysis
        analysis = self.get_analysis_by_id(analysis_id)
        
        # In a real implementation, this would generate a PDF or other report format
        # For now, just return a mock URL
        report_url = f"/reports/{analysis_id}.pdf"
        
        return {
            'reportUrl': report_url,
            'format': options.get('format', 'pdf'),
            'sections': options.get('sections', []),
            'generatedAt': datetime.now().isoformat()
        }
    
    def run_monte_carlo_analysis(self, data):
        """Run Monte Carlo simulation analysis"""
        try:
            # Call the monte_carlo adapter module
            return self.monte_carlo.run_analysis(data)
        except Exception as e:
            print(f"Error in Monte Carlo analysis: {e}")
            return {}
    
    def run_team_moat_analysis(self, data):
        """Run team moat analysis"""
        try:
            # Call the team_moat adapter module
            return self.team_moat.analyze_team(data)
        except Exception as e:
            print(f"Error in Team Moat analysis: {e}")
            return {}
    
    def run_acquisition_analysis(self, data):
        """Run acquisition fit analysis"""
        try:
            # Use the acquisition_fit adapter
            return self.acquisition_fit.analyze_acquisition_fit(data)
        except Exception as e:
            print(f"Error in Acquisition analysis: {e}")
            return {}
    
    def run_pmf_analysis(self, data):
        """Run product-market fit analysis"""
        try:
            # Use the pmf adapter
            return self.product_market_fit.analyze_pmf(data)
        except Exception as e:
            print(f"Error in PMF analysis: {e}")
            return {}
    
    def run_technical_analysis(self, data):
        """Run technical due diligence analysis"""
        try:
            # Use the technical due diligence adapter
            return self.technical_due_diligence.run_technical_analysis(data)
        except Exception as e:
            print(f"Error in Technical analysis: {e}")
            return {}
    
    def run_competitive_analysis(self, data):
        """Run competitive intelligence analysis"""
        try:
            # Use the competitive intelligence adapter
            return self.competitive_intelligence.analyze_competition(data)
        except Exception as e:
            print(f"Error in Competitive analysis: {e}")
            return {}
    
    def run_exit_path_analysis(self, data):
        """Run exit path analysis"""
        try:
            # Use the exit path adapter
            return self.exit_path.generate_exit_path_analysis(data)
        except Exception as e:
            print(f"Error in Exit Path analysis: {e}")
            return {}
    
    def run_pattern_analysis(self, data):
        """Run pattern detection analysis"""
        try:
            # Use the pattern adapter
            return self.pattern.analyze_patterns(data)
        except Exception as e:
            print(f"Error in Pattern analysis: {e}")
            return {}
    
    def run_clustering_analysis(self, data):
        """Run clustering analysis"""
        try:
            # Use the clustering adapter
            return self.clustering.analyze_clusters(data)
        except Exception as e:
            print(f"Error in Clustering analysis: {e}")
            return {}
    
    def run_benchmarks_analysis(self, data):
        """Run benchmarking analysis"""
        try:
            # Use the benchmarks adapter
            return self.benchmarks.analyze_benchmarks(data)
        except Exception as e:
            print(f"Error in Benchmarks analysis: {e}")
            return {}
    
    def run_camp_details_analysis(self, data):
        """Run CAMP details analysis"""
        try:
            # Use the camp details adapter
            return self.camp_details.analyze_camp_details(data)
        except Exception as e:
            print(f"Error in CAMP Details analysis: {e}")
            return {}
    
    def run_cohort_analysis(self, data):
        """Run cohort analysis"""
        try:
            # Use the cohort adapter
            return self.cohort.analyze_cohorts(data)
        except Exception as e:
            print(f"Error in Cohort analysis: {e}")
            return {}
    
    def run_dna_analysis(self, data):
        """Run DNA analysis"""
        try:
            # Use the dna adapter
            return self.dna.analyze_dna(data)
        except Exception as e:
            print(f"Error in DNA analysis: {e}")
            return {}
