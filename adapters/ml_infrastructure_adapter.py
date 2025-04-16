"""
ML Infrastructure Adapter

This module provides an adapter that connects the existing API
with the new ML infrastructure components for enhanced analytical capabilities.
It enables seamless integration of advanced ML features into the current system.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional

# Import base adapter class
from backend.adapters.base_adapter import BaseAnalysisAdapter

# Add ML infrastructure to path to enable imports
ml_infra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml_infrastructure'))
if ml_infra_path not in sys.path:
    sys.path.append(ml_infra_path)

# Import ML infrastructure components with error handling
try:
    from ml_infrastructure.model_registry.registry import ModelRegistry
    from ml_infrastructure.feature_store.feature_store import FeatureStore, FeatureTransformer
    from ml_infrastructure.pipeline.pipeline import Pipeline, PipelineBuilder
    from ml_infrastructure.model_ensemble import EnsembleModel, LightGBMModel, PyTorchModel
    ML_INFRA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML infrastructure import error: {e}")
    ML_INFRA_AVAILABLE = False

# Import statistical and time series modules with error handling
try:
    from ml_infrastructure.statistics.bayesian_analysis import BayesianModel, LinearGrowthModel, ExponentialGrowthModel
    from ml_infrastructure.time_series.forecasting import ProphetForecaster, ScenarioForecaster
    from ml_infrastructure.clustering.clustering import HDBSCANClusteringAnalyzer, SimilarityAnalyzer
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced analytics import error: {e}")
    ADVANCED_ANALYTICS_AVAILABLE = False

class MLInfrastructureAdapter(BaseAnalysisAdapter):
    """
    Adapter for ML infrastructure components
    
    This adapter provides an interface between the existing API
    and the new ML infrastructure for enhanced analytical capabilities.
    """
    
    def __init__(self):
        """Initialize ML infrastructure adapter"""
        super().__init__("ml_infrastructure")
        
        # Check if ML infrastructure is available
        if not ML_INFRA_AVAILABLE:
            logging.warning("ML infrastructure not available. Using fallback mechanisms.")
            self.use_mock = True
        
        # Initialize infrastructure components
        self.model_registry = None
        self.feature_store = None
        self.registry_path = os.path.abspath(os.path.join(ml_infra_path, 'model_store'))
        self.feature_store_path = os.path.abspath(os.path.join(ml_infra_path, 'feature_store'))
        
        # Create model registry and feature store if available
        if ML_INFRA_AVAILABLE:
            try:
                self.model_registry = ModelRegistry(self.registry_path)
                self.feature_store = FeatureStore(self.feature_store_path)
                logging.info("Initialized ML infrastructure components")
            except Exception as e:
                logging.error(f"Error initializing ML infrastructure: {e}")
                self.use_mock = True
    
    def run_advanced_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run advanced analysis using ML infrastructure
        
        Args:
            data: Input data for analysis
            
        Returns:
            Analysis results
        """
        if self.use_mock:
            return self.get_mock_data()
        
        results = {}
        
        # Extract startup data
        startup_data = data.get('startup_data', {})
        
        # Run relevant analyses based on data content
        if 'financial_metrics' in startup_data:
            results.update(self._analyze_financial_metrics(startup_data['financial_metrics']))
        
        if 'time_series' in startup_data:
            results.update(self._forecast_metrics(startup_data['time_series']))
        
        if 'competitors' in startup_data:
            results.update(self._analyze_competitive_landscape(
                startup_data.get('competitors', []), 
                startup_data.get('company_name', 'Target')
            ))
        
        # Add model-based predictions if available
        model_predictions = self._get_model_predictions(startup_data)
        if model_predictions:
            results['model_predictions'] = model_predictions
        
        return results
    
    def _analyze_financial_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial metrics using statistical models
        
        Args:
            financial_data: Financial metrics data
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Skip if advanced analytics not available
        if not ADVANCED_ANALYTICS_AVAILABLE:
            return {"error": "Advanced analytics modules not available"}
        
        # Extract time series data if present
        if 'monthly_revenue' in financial_data and len(financial_data['monthly_revenue']) > 0:
            # Prepare data for Bayesian analysis
            revenue_data = financial_data['monthly_revenue']
            time_points = np.arange(len(revenue_data))
            values = np.array(revenue_data)
            
            # Try different growth models
            try:
                # Create linear growth model
                linear_model = LinearGrowthModel("RevenueGrowthLinear")
                linear_model.fit({"x": time_points, "y": values})
                
                # Predict with uncertainty
                future_months = 6
                forecast_points = np.arange(len(time_points) + future_months)
                mean_pred, lower_pred, upper_pred = linear_model.predict(forecast_points)
                
                # Add to results
                results['revenue_forecast'] = {
                    'mean': mean_pred[-future_months:].tolist(),
                    'lower_bound': lower_pred[-future_months:].tolist(),
                    'upper_bound': upper_pred[-future_months:].tolist(),
                    'model_type': 'bayesian_linear'
                }
                
                # Get growth rate from model
                alpha_samples = linear_model.get_posterior_samples('alpha')
                beta_samples = linear_model.get_posterior_samples('beta')
                
                results['growth_metrics'] = {
                    'monthly_growth_rate': float(np.mean(beta_samples)),
                    'growth_rate_uncertainty': float(np.std(beta_samples)),
                    'growth_rate_ci': [
                        float(np.percentile(beta_samples, 2.5)),
                        float(np.percentile(beta_samples, 97.5))
                    ]
                }
                
            except Exception as e:
                logging.error(f"Error in Bayesian modeling: {e}")
        
        # Analyze unit economics if available
        if all(k in financial_data for k in ['cac', 'ltv', 'gross_margin']):
            try:
                cac = financial_data['cac']
                ltv = financial_data['ltv']
                gross_margin = financial_data['gross_margin']
                
                # Calculate key metrics
                ltv_to_cac = ltv / cac if cac > 0 else float('inf')
                payback_period = cac / (ltv * gross_margin / 100) if gross_margin > 0 else float('inf')
                
                results['unit_economics'] = {
                    'ltv_to_cac': ltv_to_cac,
                    'payback_period_months': payback_period,
                    'profitable_unit': ltv_to_cac > 3,
                    'assessment': self._assess_unit_economics(ltv_to_cac, payback_period)
                }
            except Exception as e:
                logging.error(f"Error in unit economics analysis: {e}")
        
        return results
    
    def _forecast_metrics(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast time series metrics
        
        Args:
            time_series_data: Time series data
            
        Returns:
            Forecast results
        """
        results = {}
        
        # Skip if advanced analytics not available
        if not ADVANCED_ANALYTICS_AVAILABLE:
            return {"error": "Time series forecasting not available"}
        
        # Check for available metrics
        available_metrics = [k for k, v in time_series_data.items() if isinstance(v, list) and len(v) >= 6]
        
        for metric in available_metrics:
            try:
                # Extract values and dates
                values = time_series_data[metric]
                
                # Create date range (assume monthly data if not specified)
                if 'dates' in time_series_data:
                    dates = time_series_data['dates']
                else:
                    # Generate monthly dates for last N months
                    from datetime import datetime, timedelta
                    end_date = datetime.now()
                    dates = [(end_date - timedelta(days=30 * i)).strftime('%Y-%m-%d') 
                             for i in range(len(values)-1, -1, -1)]
                
                # Create DataFrame for Prophet
                df = pd.DataFrame({
                    'ds': pd.to_datetime(dates),
                    'y': values
                })
                
                # Create forecaster
                forecaster = ProphetForecaster(
                    name=f"{metric}_forecast",
                    frequency='M'  # Monthly frequency
                )
                
                # Fit model
                forecaster.fit(df)
                
                # Generate forecast (6 months)
                forecast = forecaster.predict(6)
                
                # Add to results
                results[f'{metric}_forecast'] = {
                    'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': forecast['yhat'].tolist(),
                    'lower_bound': forecast['yhat_lower'].tolist(),
                    'upper_bound': forecast['yhat_upper'].tolist()
                }
                
                # Calculate growth metrics
                last_value = values[-1]
                forecasted_values = forecast['yhat'].tolist()
                future_growth = (forecasted_values[-1] - last_value) / last_value if last_value > 0 else 0
                
                results[f'{metric}_growth'] = {
                    'future_growth_rate': future_growth,
                    'growth_trajectory': 'increasing' if future_growth > 0.1 else 'stable' if -0.1 <= future_growth <= 0.1 else 'decreasing'
                }
                
            except Exception as e:
                logging.error(f"Error forecasting {metric}: {e}")
        
        return results
    
    def _analyze_competitive_landscape(
        self,
        competitors: List[Dict[str, Any]],
        company_name: str
    ) -> Dict[str, Any]:
        """
        Analyze competitive landscape using clustering
        
        Args:
            competitors: List of competitor data
            company_name: Company name
            
        Returns:
            Competitive analysis results
        """
        results = {}
        
        # Skip if not enough competitors or missing advanced analytics
        if len(competitors) < 3 or not ADVANCED_ANALYTICS_AVAILABLE:
            return {"error": "Insufficient data for competitive analysis"}
        
        try:
            # Extract features for clustering
            feature_names = ['funding', 'employees', 'growth_rate', 'market_share']
            available_features = [f for f in feature_names if all(f in comp for comp in competitors)]
            
            if len(available_features) < 2:
                return {"error": "Insufficient features for competitive analysis"}
            
            # Create DataFrame with competitor data
            df = pd.DataFrame(competitors)
            
            # Add target company if not in competitors
            if not any(comp.get('name') == company_name for comp in competitors):
                # Assume average values if target company not in data
                target_company = {feature: df[feature].mean() for feature in available_features}
                target_company['name'] = company_name
                df = pd.concat([df, pd.DataFrame([target_company])], ignore_index=True)
            
            # Extract features for clustering
            X = df[available_features].values
            company_names = df['name'].tolist()
            
            # Try HDBSCAN clustering if available
            cluster_analyzer = HDBSCANClusteringAnalyzer(
                name="CompetitorClustering",
                min_cluster_size=2,
                min_samples=1,
                scale_data=True
            )
            
            # Fit clustering model
            cluster_analyzer.fit(X)
            
            # Get cluster assignments
            df['cluster'] = cluster_analyzer.labels_
            
            # Find target company's cluster
            target_cluster = df.loc[df['name'] == company_name, 'cluster'].iloc[0]
            
            # Get similar companies (same cluster)
            similar_companies = df[df['cluster'] == target_cluster]['name'].tolist()
            similar_companies.remove(company_name)  # Remove target company itself
            
            # Get cluster profiles
            profiles = cluster_analyzer.get_cluster_profiles(df[available_features])
            
            # Format results
            results['competitive_analysis'] = {
                'similar_companies': similar_companies,
                'num_clusters': len(profiles),
                'cluster_profiles': profiles.to_dict('records'),
                'target_cluster': int(target_cluster),
                'target_position': self._assess_competitive_position(df, company_name, available_features)
            }
            
            # Add similarity analysis
            similarity_analyzer = SimilarityAnalyzer(
                name="CompetitorSimilarity",
                n_neighbors=min(5, len(df)),
                metric='cosine'
            )
            
            # Fit similarity model
            similarity_analyzer.fit(df[available_features], ids=company_names)
            
            # Find company index
            company_idx = company_names.index(company_name)
            
            # Get similar companies with similarity scores
            query = df.iloc[company_idx][available_features]
            similar_with_scores = similarity_analyzer.get_recommendation_ranking(query, k=3)
            
            results['similar_competitors'] = [
                {'name': name, 'similarity_score': float(score)}
                for name, score in similar_with_scores
            ]
            
        except Exception as e:
            logging.error(f"Error in competitive analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _get_model_predictions(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from registered models
        
        Args:
            startup_data: Startup data
            
        Returns:
            Model predictions
        """
        if not ML_INFRA_AVAILABLE or self.model_registry is None:
            return {}
        
        predictions = {}
        
        try:
            # Check if we have relevant models
            available_models = self.model_registry.list_models()
            
            # Get growth potential prediction if model exists
            if 'growth_potential_predictor' in available_models:
                # Prepare features for prediction
                features = self._extract_prediction_features(startup_data)
                
                # Get model
                model = self.model_registry.get_model('growth_potential_predictor')
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    # Classification model with probabilities
                    probs = model.predict_proba([features])[0]
                    predictions['growth_potential'] = {
                        'score': float(probs[1]),  # Assuming binary classification with positive class at index 1
                        'high_growth_probability': float(probs[1]),
                        'classification': 'high_growth' if probs[1] > 0.7 else 'medium_growth' if probs[1] > 0.3 else 'low_growth'
                    }
                else:
                    # Regression model
                    score = float(model.predict([features])[0])
                    predictions['growth_potential'] = {
                        'score': score,
                        'percentile': self._score_to_percentile(score),
                        'classification': 'high_growth' if score > 0.7 else 'medium_growth' if score > 0.3 else 'low_growth'
                    }
            
            # Get exit valuation prediction if model exists
            if 'exit_valuation_predictor' in available_models:
                # Prepare features for prediction
                features = self._extract_prediction_features(startup_data)
                
                # Get model
                model = self.model_registry.get_model('exit_valuation_predictor')
                
                # Make prediction
                valuation = float(model.predict([features])[0])
                
                predictions['exit_valuation'] = {
                    'predicted_valuation': valuation,
                    'valuation_range': [valuation * 0.7, valuation * 1.3],  # Simple uncertainty range
                    'expected_multiple': self._calculate_multiple(valuation, startup_data)
                }
        
        except Exception as e:
            logging.error(f"Error getting model predictions: {e}")
        
        return predictions
    
    def _extract_prediction_features(self, startup_data: Dict[str, Any]) -> List[float]:
        """
        Extract features for model prediction
        
        Args:
            startup_data: Startup data
            
        Returns:
            List of features
        """
        # Define default values for missing data
        defaults = {
            'funding_amount': 0,
            'monthly_revenue': 0,
            'employee_count': 1,
            'customer_count': 0,
            'churn_rate': 0.05,
            'growth_rate': 0,
            'gross_margin': 0.5,
            'runway_months': 12,
            'cac': 1000,
            'ltv': 0
        }
        
        # Try to extract values from startup data
        features = []
        
        # Funding amount
        features.append(startup_data.get('funding_amount', defaults['funding_amount']))
        
        # Monthly revenue (use latest if it's a list)
        monthly_revenue = startup_data.get('monthly_revenue', defaults['monthly_revenue'])
        if isinstance(monthly_revenue, list) and monthly_revenue:
            features.append(monthly_revenue[-1])
        else:
            features.append(defaults['monthly_revenue'])
        
        # Employee count
        features.append(startup_data.get('employee_count', defaults['employee_count']))
        
        # Customer count
        features.append(startup_data.get('customer_count', defaults['customer_count']))
        
        # Churn rate
        features.append(startup_data.get('churn_rate', defaults['churn_rate']))
        
        # Growth rate
        features.append(startup_data.get('growth_rate', defaults['growth_rate']))
        
        # Gross margin
        features.append(startup_data.get('gross_margin', defaults['gross_margin']))
        
        # Runway months
        features.append(startup_data.get('runway_months', defaults['runway_months']))
        
        # Customer acquisition cost
        features.append(startup_data.get('cac', defaults['cac']))
        
        # Lifetime value
        features.append(startup_data.get('ltv', defaults['ltv']))
        
        return features
    
    def _score_to_percentile(self, score: float) -> float:
        """
        Convert a score to a percentile (simple implementation)
        
        Args:
            score: Raw score
            
        Returns:
            Percentile (0-100)
        """
        # Simple linear mapping from 0-1 to 0-100
        return min(100, max(0, score * 100))
    
    def _calculate_multiple(self, valuation: float, startup_data: Dict[str, Any]) -> float:
        """
        Calculate exit multiple based on valuation and current metrics
        
        Args:
            valuation: Predicted exit valuation
            startup_data: Startup data
            
        Returns:
            Exit multiple
        """
        # Get total funding if available
        funding = startup_data.get('funding_amount', 0)
        
        # Get current valuation if available
        current_valuation = startup_data.get('current_valuation', None)
        
        # If current valuation not available, estimate it
        if current_valuation is None:
            # Simple heuristic: current valuation is typically 3-5x funding
            current_valuation = funding * 4 if funding > 0 else valuation / 5
        
        # Calculate multiple
        return valuation / current_valuation if current_valuation > 0 else 10.0  # Default to 10x if can't calculate
    
    def _assess_unit_economics(self, ltv_to_cac: float, payback_period: float) -> str:
        """
        Assess unit economics health
        
        Args:
            ltv_to_cac: LTV to CAC ratio
            payback_period: Payback period in months
            
        Returns:
            Assessment text
        """
        if ltv_to_cac > 3 and payback_period < 12:
            return "excellent"
        elif ltv_to_cac > 3 or payback_period < 12:
            return "good"
        elif ltv_to_cac > 1:
            return "fair"
        else:
            return "poor"
    
    def _assess_competitive_position(
        self,
        df: pd.DataFrame,
        company_name: str,
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Assess competitive position relative to peers
        
        Args:
            df: DataFrame with competitor data
            company_name: Company name
            features: List of features to assess
            
        Returns:
            Competitive position assessment
        """
        # Get company row
        company_row = df[df['name'] == company_name]
        
        # Calculate percentiles for each feature
        position = {}
        
        for feature in features:
            # Get company value
            company_value = company_row[feature].iloc[0]
            
            # Calculate percentile
            percentile = (df[feature] <= company_value).mean() * 100
            
            # Determine if higher is better (default assumption)
            higher_is_better = feature not in ['churn_rate', 'cac']
            
            # Adjust percentile if lower is better
            if not higher_is_better:
                percentile = 100 - percentile
            
            # Add to position dict
            position[feature] = {
                'percentile': percentile,
                'ranking': int((df[feature].rank(ascending=not higher_is_better)).loc[company_row.index[0]]),
                'total_companies': len(df)
            }
        
        # Calculate overall position
        avg_percentile = sum(pos['percentile'] for pos in position.values()) / len(position)
        
        position['overall'] = {
            'percentile': avg_percentile,
            'assessment': 'leader' if avg_percentile > 75 else 'strong' if avg_percentile > 50 else 'challenger' if avg_percentile > 25 else 'laggard'
        }
        
        return position
    
    def get_mock_data(self) -> Dict[str, Any]:
        """
        Get mock data for the analysis
        
        Returns:
            Mock analysis results
        """
        return {
            "notice": "Using mock data because ML infrastructure is not available",
            "revenue_forecast": {
                "mean": [105000, 110000, 115000, 120000, 125000, 130000],
                "lower_bound": [95000, 97000, 99000, 101000, 103000, 105000],
                "upper_bound": [115000, 123000, 131000, 139000, 147000, 155000],
                "model_type": "mock"
            },
            "growth_metrics": {
                "monthly_growth_rate": 0.05,
                "growth_rate_uncertainty": 0.02,
                "growth_rate_ci": [0.01, 0.09]
            },
            "competitive_analysis": {
                "similar_companies": ["CompetitorA", "CompetitorB"],
                "num_clusters": 3,
                "cluster_profiles": [
                    {"funding": 1000000, "employees": 15, "growth_rate": 0.1, "market_share": 0.05}
                ],
                "target_cluster": 1,
                "target_position": {
                    "funding": {"percentile": 60, "ranking": 3, "total_companies": 5},
                    "overall": {"percentile": 55, "assessment": "strong"}
                }
            },
            "model_predictions": {
                "growth_potential": {
                    "score": 0.75,
                    "high_growth_probability": 0.75,
                    "classification": "high_growth"
                },
                "exit_valuation": {
                    "predicted_valuation": 50000000,
                    "valuation_range": [35000000, 65000000],
                    "expected_multiple": 5.0
                }
            }
        }

# Create an instance of the adapter
ml_infrastructure_adapter = MLInfrastructureAdapter() 