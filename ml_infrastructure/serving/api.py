"""
ML Infrastructure API

This module provides a REST API for accessing the various ML infrastructure components,
including model prediction, analysis, and data processing capabilities.

Features:
- Model prediction endpoints
- Time series forecasting endpoints 
- Bayesian analysis endpoints
- Clustering and similarity endpoints
- Model ensemble endpoints
- Feature store access
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from flask import Flask, request, jsonify
from waitress import serve
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import infrastructure components
try:
    from ml_infrastructure.model_registry.registry import ModelRegistry
    from ml_infrastructure.feature_store.feature_store import FeatureStore, FeatureTransformer
    from ml_infrastructure.pipeline.pipeline import Pipeline, PipelineBuilder
    from ml_infrastructure.model_ensemble import EnsembleModel, LightGBMModel, PyTorchModel
    ML_INFRA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML infrastructure import error: {e}")
    ML_INFRA_AVAILABLE = False

# Import statistical and time series modules
try:
    from ml_infrastructure.statistics.bayesian_analysis import (
        BayesianModel, LinearGrowthModel, ExponentialGrowthModel, 
        fit_bayesian_growth_model
    )
    from ml_infrastructure.time_series.forecasting import (
        ProphetForecaster, ScenarioForecaster, 
        forecast_startup_growth, create_scenario_analysis
    )
    from ml_infrastructure.clustering.clustering import (
        HDBSCANClusteringAnalyzer, SimilarityAnalyzer, 
        cluster_and_visualize, find_similar_companies
    )
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced analytics import error: {e}")
    ADVANCED_ANALYTICS_AVAILABLE = False

# Import adapter for backend integration
try:
    from backend.adapters.ml_infrastructure_adapter import MLInfrastructureAdapter
    ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML infrastructure adapter import error: {e}")
    ADAPTER_AVAILABLE = False

class MLInfrastructureAPI:
    """
    API for accessing ML infrastructure components
    
    This class provides a Flask API for using various ML infrastructure
    components, including model prediction, analysis, and data processing.
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5500,
        model_registry_dir: str = './model_store',
        feature_store_dir: str = './feature_store'
    ):
        """
        Initialize ML infrastructure API
        
        Args:
            host: Host to bind the server
            port: Port to bind the server
            model_registry_dir: Model registry directory
            feature_store_dir: Feature store directory
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        
        # Initialize infrastructure components if available
        self.model_registry = None
        self.feature_store = None
        self.adapter = None
        
        if ML_INFRA_AVAILABLE:
            try:
                self.model_registry = ModelRegistry(model_registry_dir)
                self.feature_store = FeatureStore(feature_store_dir)
                logger.info("Initialized ML infrastructure components")
            except Exception as e:
                logger.error(f"Error initializing ML infrastructure: {e}")
        
        if ADAPTER_AVAILABLE:
            try:
                self.adapter = MLInfrastructureAdapter()
                logger.info("Initialized ML infrastructure adapter")
            except Exception as e:
                logger.error(f"Error initializing ML infrastructure adapter: {e}")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            components_status = {
                'model_registry': self.model_registry is not None,
                'feature_store': self.feature_store is not None,
                'adapter': self.adapter is not None,
                'advanced_analytics': ADVANCED_ANALYTICS_AVAILABLE
            }
            
            return jsonify({
                'status': 'healthy',
                'components': components_status,
                'timestamp': datetime.now().isoformat()
            })
        
        # Model endpoints
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available models"""
            if not self.model_registry:
                return jsonify({'error': 'Model registry not available'}), 503
            
            try:
                models = self.model_registry.list_models()
                return jsonify({'models': models})
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/models/<model_name>/predict', methods=['POST'])
        def predict_model(model_name):
            """Make predictions with a model"""
            if not self.model_registry:
                return jsonify({'error': 'Model registry not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get features if provided in a standard format
                if 'features' in input_data:
                    features = input_data['features']
                else:
                    features = input_data
                
                # Load model
                model = self.model_registry.get_model(model_name)
                
                # Make prediction
                if hasattr(model, 'predict'):
                    # Convert inputs to numpy array if needed
                    if isinstance(features, list):
                        features = np.array(features)
                    
                    # Handle both single instance and batch predictions
                    if len(np.array(features).shape) == 1:
                        features = np.array([features])
                    
                    # Make prediction
                    predictions = model.predict(features)
                    
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(predictions, np.ndarray):
                        predictions = predictions.tolist()
                    
                    return jsonify({
                        'model': model_name,
                        'predictions': predictions
                    })
                else:
                    return jsonify({'error': 'Model does not support predictions'}), 400
            
            except Exception as e:
                logger.error(f"Error making prediction with model {model_name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/ensemble/<ensemble_name>/predict', methods=['POST'])
        def predict_ensemble(ensemble_name):
            """Make predictions with an ensemble model"""
            if not ML_INFRA_AVAILABLE:
                return jsonify({'error': 'ML infrastructure not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get features if provided in a standard format
                if 'features' in input_data:
                    features = input_data['features']
                else:
                    features = input_data
                
                # Try to load the ensemble
                ensemble_path = os.path.join(
                    self.model_registry.registry_dir,
                    'ensembles',
                    f"{ensemble_name}.pkl"
                )
                
                if not os.path.exists(ensemble_path):
                    return jsonify({'error': f'Ensemble model {ensemble_name} not found'}), 404
                
                # Load ensemble
                ensemble = EnsembleModel.load(ensemble_path)
                
                # Convert inputs to appropriate format
                if isinstance(features, list):
                    features = np.array(features)
                
                # Make prediction
                predictions = ensemble.predict(features)
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(predictions, np.ndarray):
                    predictions = predictions.tolist()
                
                # Get feature importance if available
                feature_importance = ensemble.get_feature_importance()
                
                return jsonify({
                    'ensemble': ensemble_name,
                    'predictions': predictions,
                    'feature_importance': feature_importance
                })
            
            except Exception as e:
                logger.error(f"Error making prediction with ensemble {ensemble_name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Time series forecasting endpoints
        @self.app.route('/forecast', methods=['POST'])
        def forecast():
            """Generate time series forecast"""
            if not ADVANCED_ANALYTICS_AVAILABLE:
                return jsonify({'error': 'Advanced analytics not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get required parameters
                if 'dates' not in input_data or 'values' not in input_data:
                    return jsonify({'error': 'Both dates and values are required'}), 400
                
                dates = input_data['dates']
                values = input_data['values']
                
                if len(dates) != len(values):
                    return jsonify({'error': 'Dates and values must have the same length'}), 400
                
                # Get optional parameters
                periods = input_data.get('periods', 12)
                model_type = input_data.get('model_type', 'prophet')
                include_holidays = input_data.get('include_holidays', True)
                include_seasonality = input_data.get('include_seasonality', True)
                
                # Create DataFrame
                data = pd.DataFrame({
                    'ds': pd.to_datetime(dates),
                    'y': values
                })
                
                # Generate forecast
                forecast_df = forecast_startup_growth(
                    data=data,
                    periods=periods,
                    model_type=model_type,
                    include_holidays=include_holidays,
                    include_seasonality=include_seasonality
                )
                
                # Convert to JSON-serializable format
                result = {
                    'dates': forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'forecast': forecast_df['yhat'].tolist(),
                    'lower_bound': forecast_df['yhat_lower'].tolist(),
                    'upper_bound': forecast_df['yhat_upper'].tolist()
                }
                
                # Include components if available
                if 'trend' in forecast_df.columns:
                    result['trend'] = forecast_df['trend'].tolist()
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error generating forecast: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/forecast/scenarios', methods=['POST'])
        def forecast_scenarios():
            """Generate scenario-based forecasts"""
            if not ADVANCED_ANALYTICS_AVAILABLE:
                return jsonify({'error': 'Advanced analytics not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get required parameters
                if 'dates' not in input_data or 'values' not in input_data:
                    return jsonify({'error': 'Both dates and values are required'}), 400
                
                dates = input_data['dates']
                values = input_data['values']
                
                if len(dates) != len(values):
                    return jsonify({'error': 'Dates and values must have the same length'}), 400
                
                # Get optional parameters
                periods = input_data.get('periods', 12)
                scenarios = input_data.get('scenarios', {
                    'optimistic': {'growth_multiplier': 1.5},
                    'pessimistic': {'growth_multiplier': 0.7}
                })
                
                # Create DataFrame
                data = pd.DataFrame({
                    'ds': pd.to_datetime(dates),
                    'y': values
                })
                
                # Generate scenario forecasts
                scenario_dfs = create_scenario_analysis(
                    data=data,
                    periods=periods,
                    scenarios=scenarios
                )
                
                # Convert to JSON-serializable format
                result = {}
                
                for scenario_name, scenario_df in scenario_dfs.items():
                    result[scenario_name] = {
                        'dates': scenario_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                        'forecast': scenario_df['yhat'].tolist(),
                        'lower_bound': scenario_df['yhat_lower'].tolist(),
                        'upper_bound': scenario_df['yhat_upper'].tolist()
                    }
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error generating scenario forecasts: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Bayesian analysis endpoints
        @self.app.route('/bayesian/growth', methods=['POST'])
        def bayesian_growth():
            """Fit Bayesian growth model to data"""
            if not ADVANCED_ANALYTICS_AVAILABLE:
                return jsonify({'error': 'Advanced analytics not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get time points and values
                if 'x' not in input_data or 'y' not in input_data:
                    return jsonify({'error': 'Both x and y data are required'}), 400
                
                time_points = np.array(input_data['x'])
                values = np.array(input_data['y'])
                
                if len(time_points) != len(values):
                    return jsonify({'error': 'X and Y must have the same length'}), 400
                
                # Get optional parameters
                model_type = input_data.get('model_type', 'linear')
                samples = input_data.get('samples', 1000)
                future_points = input_data.get('future_points', 6)
                
                # Fit Bayesian growth model
                if model_type == 'linear':
                    model = LinearGrowthModel('BayesianGrowth')
                elif model_type == 'exponential':
                    model = ExponentialGrowthModel('BayesianGrowth')
                else:
                    return jsonify({'error': f'Unsupported model type: {model_type}'}), 400
                
                # Fit model
                model.fit({'x': time_points, 'y': values}, samples=samples)
                
                # Make predictions for future points
                future_x = np.arange(time_points.min(), time_points.max() + future_points + 1)
                mean_pred, lower_pred, upper_pred = model.predict(future_x)
                
                # Get growth rate parameters
                if model_type == 'linear':
                    alpha_samples = model.get_posterior_samples('alpha')
                    beta_samples = model.get_posterior_samples('beta')
                    
                    growth_params = {
                        'intercept': float(np.mean(alpha_samples)),
                        'slope': float(np.mean(beta_samples)),
                        'slope_ci': [
                            float(np.percentile(beta_samples, 2.5)),
                            float(np.percentile(beta_samples, 97.5))
                        ]
                    }
                else:  # exponential
                    alpha_samples = model.get_posterior_samples('alpha')
                    beta_samples = model.get_posterior_samples('beta')
                    
                    growth_params = {
                        'initial_value': float(np.mean(alpha_samples)),
                        'growth_rate': float(np.mean(beta_samples)),
                        'growth_rate_ci': [
                            float(np.percentile(beta_samples, 2.5)),
                            float(np.percentile(beta_samples, 97.5))
                        ]
                    }
                
                # Format results
                result = {
                    'model_type': model_type,
                    'x': future_x.tolist(),
                    'mean_prediction': mean_pred.tolist(),
                    'lower_bound': lower_pred.tolist(),
                    'upper_bound': upper_pred.tolist(),
                    'growth_parameters': growth_params
                }
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error in Bayesian growth analysis: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Clustering and similarity endpoints
        @self.app.route('/clustering', methods=['POST'])
        def cluster_data():
            """Cluster data points"""
            if not ADVANCED_ANALYTICS_AVAILABLE:
                return jsonify({'error': 'Advanced analytics not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get required parameters
                if 'data' not in input_data:
                    return jsonify({'error': 'Data is required'}), 400
                
                data = input_data['data']
                
                # Get optional parameters
                method = input_data.get('method', 'hdbscan')
                n_clusters = input_data.get('n_clusters', 3)
                labels = input_data.get('labels', None)
                
                # Convert data to numpy array
                X = np.array(data)
                
                # Perform clustering
                result = cluster_and_visualize(
                    X=X,
                    method=method,
                    n_clusters=n_clusters,
                    labels=labels
                )
                
                # Convert numpy arrays to lists for JSON serialization
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error in clustering: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/similarity', methods=['POST'])
        def find_similar():
            """Find similar items"""
            if not ADVANCED_ANALYTICS_AVAILABLE:
                return jsonify({'error': 'Advanced analytics not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Get required parameters
                if 'data' not in input_data or 'query_index' not in input_data:
                    return jsonify({'error': 'Both data and query_index are required'}), 400
                
                data = np.array(input_data['data'])
                query_idx = input_data['query_index']
                
                # Get optional parameters
                n_similar = input_data.get('n_similar', 5)
                ids = input_data.get('ids', None)
                metric = input_data.get('metric', 'cosine')
                
                # Find similar items
                similar_items = find_similar_companies(
                    X=data,
                    company_ids=ids or list(range(len(data))),
                    query_idx=query_idx,
                    n_similar=n_similar,
                    metric=metric
                )
                
                # Format results
                result = [
                    {'id': item_id, 'similarity': float(score)}
                    for item_id, score in similar_items
                ]
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error finding similar items: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Feature store endpoints
        @self.app.route('/features', methods=['GET'])
        def list_features():
            """List available features"""
            if not self.feature_store:
                return jsonify({'error': 'Feature store not available'}), 503
            
            try:
                # Get optional tag filter
                tags = request.args.get('tags', None)
                if tags:
                    tags = tags.split(',')
                
                # List features
                features = self.feature_store.list_features(tags)
                return jsonify({'features': features})
            
            except Exception as e:
                logger.error(f"Error listing features: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Adapter endpoint for comprehensive analysis
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            """Run comprehensive analysis using the adapter"""
            if not self.adapter:
                return jsonify({'error': 'ML infrastructure adapter not available'}), 503
            
            try:
                # Get input data
                input_data = request.json
                if not input_data:
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Run analysis through adapter
                results = self.adapter.run_advanced_analysis(input_data)
                return jsonify(results)
            
            except Exception as e:
                logger.error(f"Error in comprehensive analysis: {e}")
                return jsonify({'error': str(e)}), 500
    
    def start(self, threaded: bool = False):
        """
        Start the API server
        
        Args:
            threaded: Whether to run in a separate thread
        """
        if threaded:
            # Start in a separate thread
            thread = threading.Thread(target=self._run_server)
            thread.daemon = True
            thread.start()
            return thread
        else:
            # Run in the current thread (blocking)
            self._run_server()
    
    def _run_server(self):
        """Internal method to run the server"""
        logger.info(f"Starting ML infrastructure API on {self.host}:{self.port}")
        serve(self.app, host=self.host, port=self.port, threads=10)

# Create API instance when module is imported
api = MLInfrastructureAPI()

def start_api(host: str = '0.0.0.0', port: int = 5500):
    """
    Convenience function to start the API server
    
    Args:
        host: Host to bind the server
        port: Port to bind the server
    """
    global api
    api = MLInfrastructureAPI(host=host, port=port)
    api.start()

if __name__ == '__main__':
    # Start the API server when script is run directly
    start_api() 