"""
ML Infrastructure API Client

This module provides a client for interacting with the ML infrastructure API,
making it easy to access prediction, forecasting, and analysis functionality
from Python applications.
"""

import requests
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLInfrastructureClient:
    """
    Client for accessing ML infrastructure API
    
    This class provides methods for interacting with the ML infrastructure API,
    abstracting away HTTP requests and response handling.
    """
    
    def __init__(self, base_url: str = "http://localhost:5500"):
        """
        Initialize ML infrastructure client
        
        Args:
            base_url: Base URL of the ML infrastructure API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the ML infrastructure API
        
        Returns:
            Health status information
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models
        
        Returns:
            List of available models
        """
        url = f"{self.base_url}/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['models']
    
    def predict(
        self,
        model_name: str,
        features: Union[List[float], List[List[float]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Make predictions with a model
        
        Args:
            model_name: Name of the model to use
            features: Input features for prediction
            
        Returns:
            Prediction results
        """
        url = f"{self.base_url}/models/{model_name}/predict"
        payload = {'features': features}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_ensemble(
        self,
        ensemble_name: str,
        features: Union[List[float], List[List[float]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Make predictions with an ensemble model
        
        Args:
            ensemble_name: Name of the ensemble model to use
            features: Input features for prediction
            
        Returns:
            Prediction results
        """
        url = f"{self.base_url}/ensemble/{ensemble_name}/predict"
        payload = {'features': features}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def forecast(
        self,
        dates: List[str],
        values: List[float],
        periods: int = 12,
        model_type: str = 'prophet',
        include_holidays: bool = True,
        include_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Generate time series forecast
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
            values: List of values corresponding to dates
            periods: Number of periods to forecast
            model_type: Type of forecasting model to use
            include_holidays: Whether to include holidays in the model
            include_seasonality: Whether to include seasonality in the model
            
        Returns:
            Forecast results
        """
        url = f"{self.base_url}/forecast"
        payload = {
            'dates': dates,
            'values': values,
            'periods': periods,
            'model_type': model_type,
            'include_holidays': include_holidays,
            'include_seasonality': include_seasonality
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def forecast_scenarios(
        self,
        dates: List[str],
        values: List[float],
        periods: int = 12,
        scenarios: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate scenario-based forecasts
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
            values: List of values corresponding to dates
            periods: Number of periods to forecast
            scenarios: Dictionary of scenario configurations
            
        Returns:
            Scenario forecast results
        """
        url = f"{self.base_url}/forecast/scenarios"
        
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = {
                'optimistic': {'growth_multiplier': 1.5},
                'pessimistic': {'growth_multiplier': 0.7}
            }
        
        payload = {
            'dates': dates,
            'values': values,
            'periods': periods,
            'scenarios': scenarios
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def bayesian_growth_analysis(
        self,
        x: List[float],
        y: List[float],
        model_type: str = 'linear',
        samples: int = 1000,
        future_points: int = 6
    ) -> Dict[str, Any]:
        """
        Perform Bayesian growth analysis
        
        Args:
            x: List of time points
            y: List of values
            model_type: Type of growth model ('linear' or 'exponential')
            samples: Number of MCMC samples
            future_points: Number of future points to predict
            
        Returns:
            Growth analysis results
        """
        url = f"{self.base_url}/bayesian/growth"
        payload = {
            'x': x,
            'y': y,
            'model_type': model_type,
            'samples': samples,
            'future_points': future_points
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def cluster_data(
        self,
        data: List[List[float]],
        method: str = 'hdbscan',
        n_clusters: int = 3,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Cluster data points
        
        Args:
            data: List of data points (each point is a list of features)
            method: Clustering method
            n_clusters: Number of clusters (for methods that require it)
            labels: Optional labels for data points
            
        Returns:
            Clustering results
        """
        url = f"{self.base_url}/clustering"
        payload = {
            'data': data,
            'method': method,
            'n_clusters': n_clusters
        }
        
        if labels:
            payload['labels'] = labels
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def find_similar(
        self,
        data: List[List[float]],
        query_index: int,
        n_similar: int = 5,
        ids: List[Any] = None,
        metric: str = 'cosine'
    ) -> List[Dict[str, Any]]:
        """
        Find similar items
        
        Args:
            data: List of data points (each point is a list of features)
            query_index: Index of the query point
            n_similar: Number of similar items to return
            ids: Optional IDs for data points
            metric: Similarity metric
            
        Returns:
            List of similar items with similarity scores
        """
        url = f"{self.base_url}/similarity"
        payload = {
            'data': data,
            'query_index': query_index,
            'n_similar': n_similar,
            'metric': metric
        }
        
        if ids:
            payload['ids'] = ids
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_features(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List available features
        
        Args:
            tags: Optional filter by tags
            
        Returns:
            List of available features
        """
        url = f"{self.base_url}/features"
        
        params = {}
        if tags:
            params['tags'] = ','.join(tags)
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()['features']
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive analysis using the ML infrastructure adapter
        
        Args:
            data: Input data for analysis
            
        Returns:
            Analysis results
        """
        url = f"{self.base_url}/analyze"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

def create_client(base_url: str = "http://localhost:5500") -> MLInfrastructureClient:
    """
    Create a new ML infrastructure client
    
    Args:
        base_url: Base URL of the ML infrastructure API
        
    Returns:
        ML infrastructure client
    """
    return MLInfrastructureClient(base_url)

# Example usage
if __name__ == "__main__":
    # Create client
    client = create_client()
    
    # Check health
    try:
        health = client.health_check()
        print(f"API health: {health['status']}")
        print(f"Components: {health['components']}")
    except Exception as e:
        print(f"Error checking health: {e}")
    
    # Example forecast
    try:
        # Simple example with monthly revenue data
        dates = [
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", 
            "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01"
        ]
        values = [10000, 12000, 15000, 14000, 16000, 18000, 19000, 22000]
        
        forecast = client.forecast(dates, values, periods=6)
        print("\nForecast results:")
        print(f"Future dates: {forecast['dates'][-6:]}")
        print(f"Forecasted values: {forecast['forecast'][-6:]}")
    except Exception as e:
        print(f"Error generating forecast: {e}")
    
    # Example Bayesian analysis
    try:
        # Monthly growth data
        months = [1, 2, 3, 4, 5, 6, 7, 8]
        growth = [5, 7, 10, 15, 19, 26, 32, 40]
        
        growth_analysis = client.bayesian_growth_analysis(
            x=months, 
            y=growth, 
            model_type='exponential'
        )
        
        print("\nBayesian growth analysis:")
        print(f"Growth rate: {growth_analysis['growth_parameters']['growth_rate']:.2f}")
        print(f"Growth rate 95% CI: {growth_analysis['growth_parameters']['growth_rate_ci']}")
    except Exception as e:
        print(f"Error performing Bayesian analysis: {e}")
    
    # Example clustering
    try:
        # Simple 2D data for visualization
        data_points = [
            [1, 2], [1.5, 1.8], [2, 2.1], [2.2, 1.7],  # Cluster 1
            [5, 6], [5.5, 5.5], [6, 5.8], [5.8, 6.2],  # Cluster 2
            [9, 1], [9.5, 1.5], [10, 1.2], [9.8, 0.8]   # Cluster 3
        ]
        
        clustering = client.cluster_data(
            data=data_points,
            method='kmeans',
            n_clusters=3
        )
        
        print("\nClustering results:")
        print(f"Clusters: {clustering['labels']}")
        print(f"Silhouette score: {clustering.get('silhouette_score', 'N/A')}")
    except Exception as e:
        print(f"Error performing clustering: {e}") 