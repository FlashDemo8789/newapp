"""
Bayesian Statistical Analysis

This module provides Bayesian modeling tools for statistical analysis
with rigorous uncertainty quantification. It enables inference on
startup metrics, probabilistic forecasting, and risk assessment.

Features:
- Bayesian parameter estimation
- Credible intervals for predictions
- Prior specification for domain knowledge
- Posterior predictive checks
- Model comparison
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Optional, Tuple, Callable

# Try to import PyMC3
try:
    import pymc3 as pm
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    logging.warning("PyMC3 not available. Bayesian modeling will be limited.")

# Try to import ArviZ for diagnostics and visualization
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    logging.warning("ArviZ not available. Bayesian diagnostics will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BayesianModel:
    """Base class for Bayesian models"""
    
    def __init__(self, name: str):
        """
        Initialize Bayesian model
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.trace = None
        self.is_fitted = False
    
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build PyMC3 model
        
        Args:
            data: Input data
            
        Returns:
            PyMC3 model
        """
        raise NotImplementedError("Subclasses must implement build_model")
    
    def fit(
        self,
        data: Dict[str, Any],
        samples: int = 1000,
        tune: int = 1000,
        cores: int = 2,
        chains: int = 2
    ) -> None:
        """
        Fit the model using MCMC
        
        Args:
            data: Input data
            samples: Number of samples to draw
            tune: Number of tuning steps
            cores: Number of cores to use
            chains: Number of chains to run
        """
        if not PYMC3_AVAILABLE:
            raise ImportError("PyMC3 is required for Bayesian modeling")
        
        # Build model
        self.model = self.build_model(data)
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=samples,
                tune=tune,
                cores=cores,
                chains=chains,
                return_inferencedata=ARVIZ_AVAILABLE
            )
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} with {samples} samples")
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_ci: bool = True,
        ci_width: float = 0.95
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty
        
        Args:
            X: Input features
            return_ci: Whether to return credible intervals
            ci_width: Width of credible interval (0-1)
            
        Returns:
            If return_ci is True: (mean, lower, upper)
            If return_ci is False: mean
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict")
    
    def get_posterior_samples(self, param_name: str) -> np.ndarray:
        """
        Get posterior samples for a parameter
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Array of posterior samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting posterior samples")
        
        if ARVIZ_AVAILABLE and hasattr(self.trace, "posterior"):
            # ArviZ InferenceData format
            return self.trace.posterior[param_name].values.flatten()
        else:
            # Direct PyMC3 trace
            return self.trace[param_name]
    
    def get_credible_interval(
        self,
        param_name: str,
        ci_width: float = 0.95
    ) -> Tuple[float, float]:
        """
        Get credible interval for a parameter
        
        Args:
            param_name: Name of the parameter
            ci_width: Width of credible interval (0-1)
            
        Returns:
            (lower, upper) bounds of the credible interval
        """
        samples = self.get_posterior_samples(param_name)
        
        # Calculate quantiles for credible interval
        alpha = (1 - ci_width) / 2
        lower = np.quantile(samples, alpha)
        upper = np.quantile(samples, 1 - alpha)
        
        return lower, upper
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics
        
        Returns:
            Dictionary of diagnostic metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting diagnostics")
        
        if not ARVIZ_AVAILABLE:
            return {"error": "ArviZ required for diagnostics"}
        
        # Basic convergence diagnostics
        try:
            summary = az.summary(self.trace)
            return {
                "summary": summary.to_dict(),
                "r_hat": summary["r_hat"].to_dict()
            }
        except Exception as e:
            logger.error(f"Error in diagnostics: {e}")
            return {"error": str(e)}

class LinearGrowthModel(BayesianModel):
    """Bayesian linear growth model with uncertainty"""
    
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build PyMC3 model for linear growth
        
        Args:
            data: Dictionary with 'x' (time) and 'y' (metric) arrays
            
        Returns:
            PyMC3 model
        """
        x = data['x']
        y = data['y']
        
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.HalfNormal('beta', sigma=5)  # Growth rate is positive
            sigma = pm.HalfNormal('sigma', sigma=1)  # Observation noise
            
            # Linear model
            mu = alpha + beta * x
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        return model
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_ci: bool = True,
        ci_width: float = 0.95
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty
        
        Args:
            X: Input features (time points)
            return_ci: Whether to return credible intervals
            ci_width: Width of credible interval (0-1)
            
        Returns:
            If return_ci is True: (mean, lower, upper)
            If return_ci is False: mean
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get posterior samples
        alpha_samples = self.get_posterior_samples('alpha')
        beta_samples = self.get_posterior_samples('beta')
        
        # Reshape X if needed
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) > 1 and X.shape[1] == 1:
            X = X.flatten()
        
        # Calculate predictions for each posterior sample
        n_samples = len(alpha_samples)
        n_points = len(X)
        
        predictions = np.zeros((n_samples, n_points))
        for i in range(n_samples):
            predictions[i] = alpha_samples[i] + beta_samples[i] * X
        
        # Calculate mean and credible intervals
        mean_pred = predictions.mean(axis=0)
        
        if return_ci:
            alpha = (1 - ci_width) / 2
            lower = np.quantile(predictions, alpha, axis=0)
            upper = np.quantile(predictions, 1 - alpha, axis=0)
            return mean_pred, lower, upper
        else:
            return mean_pred

class ExponentialGrowthModel(BayesianModel):
    """Bayesian exponential growth model with uncertainty"""
    
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build PyMC3 model for exponential growth
        
        Args:
            data: Dictionary with 'x' (time) and 'y' (metric) arrays
            
        Returns:
            PyMC3 model
        """
        x = data['x']
        y = data['y']
        
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)  # Initial value
            growth_rate = pm.Normal('growth_rate', mu=0.1, sigma=0.5)  # Growth rate
            sigma = pm.HalfNormal('sigma', sigma=1)  # Observation noise
            
            # Exponential model
            mu = alpha * pm.math.exp(growth_rate * x)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        return model
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_ci: bool = True,
        ci_width: float = 0.95
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty
        
        Args:
            X: Input features (time points)
            return_ci: Whether to return credible intervals
            ci_width: Width of credible interval (0-1)
            
        Returns:
            If return_ci is True: (mean, lower, upper)
            If return_ci is False: mean
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get posterior samples
        alpha_samples = self.get_posterior_samples('alpha')
        growth_rate_samples = self.get_posterior_samples('growth_rate')
        
        # Reshape X if needed
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) > 1 and X.shape[1] == 1:
            X = X.flatten()
        
        # Calculate predictions for each posterior sample
        n_samples = len(alpha_samples)
        n_points = len(X)
        
        predictions = np.zeros((n_samples, n_points))
        for i in range(n_samples):
            predictions[i] = alpha_samples[i] * np.exp(growth_rate_samples[i] * X)
        
        # Calculate mean and credible intervals
        mean_pred = predictions.mean(axis=0)
        
        if return_ci:
            alpha = (1 - ci_width) / 2
            lower = np.quantile(predictions, alpha, axis=0)
            upper = np.quantile(predictions, 1 - alpha, axis=0)
            return mean_pred, lower, upper
        else:
            return mean_pred

class BayesianMonteCarlo:
    """Monte Carlo simulation with Bayesian parameters"""
    
    def __init__(self, name: str, param_distributions: Dict[str, Any] = None):
        """
        Initialize Bayesian Monte Carlo simulation
        
        Args:
            name: Simulation name
            param_distributions: Dictionary of parameter distributions
        """
        self.name = name
        self.param_distributions = param_distributions or {}
        self.traces: Dict[str, Any] = {}
        self.sample_cache = {}
    
    def add_parameter_model(
        self,
        param_name: str,
        model: BayesianModel,
        transform_func: Callable = None
    ) -> None:
        """
        Add a fitted Bayesian model for a parameter
        
        Args:
            param_name: Parameter name
            model: Fitted Bayesian model
            transform_func: Function to transform samples if needed
        """
        if not model.is_fitted:
            raise ValueError(f"Model for {param_name} must be fitted")
        
        self.traces[param_name] = {
            'model': model,
            'transform': transform_func
        }
    
    def add_parameter_distribution(
        self,
        param_name: str,
        distribution: Any,
        **params
    ) -> None:
        """
        Add a parameter distribution
        
        Args:
            param_name: Parameter name
            distribution: NumPy/SciPy distribution function
            **params: Parameters for the distribution
        """
        self.param_distributions[param_name] = {
            'distribution': distribution,
            'params': params
        }
    
    def _sample_parameters(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Sample parameters from distributions and models
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dictionary of parameter samples
        """
        samples = {}
        
        # Sample from parameter distributions
        for param_name, param_info in self.param_distributions.items():
            distribution = param_info['distribution']
            params = param_info['params']
            
            # Sample from distribution
            samples[param_name] = distribution(**params, size=n_samples)
        
        # Sample from Bayesian model traces
        for param_name, trace_info in self.traces.items():
            model = trace_info['model']
            transform = trace_info['transform']
            
            # Get samples from posterior
            posterior_samples = model.get_posterior_samples(param_name)
            
            # Randomly select n_samples
            indices = np.random.choice(len(posterior_samples), size=n_samples, replace=True)
            param_samples = posterior_samples[indices]
            
            # Apply transformation if needed
            if transform:
                param_samples = transform(param_samples)
            
            samples[param_name] = param_samples
        
        return samples
    
    def run_simulation(
        self,
        simulation_func: Callable,
        n_samples: int = 1000,
        seed: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation
        
        Args:
            simulation_func: Function that takes parameter samples and returns outputs
            n_samples: Number of simulation samples
            seed: Random seed
            
        Returns:
            Dictionary of simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample parameters
        param_samples = self._sample_parameters(n_samples)
        
        # Run simulation
        results = simulation_func(param_samples)
        
        # Cache samples for later analysis
        self.sample_cache = {
            'parameters': param_samples,
            'results': results
        }
        
        return results
    
    def get_simulation_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of simulation results
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.sample_cache or 'results' not in self.sample_cache:
            raise ValueError("No simulation results to summarize. Run simulation first.")
        
        results = self.sample_cache['results']
        
        summary = {}
        for key, values in results.items():
            summary[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        return summary
    
    def get_credible_intervals(
        self,
        ci_width: float = 0.95
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Get credible intervals for simulation results
        
        Args:
            ci_width: Width of credible interval (0-1)
            
        Returns:
            Dictionary of credible intervals
        """
        if not self.sample_cache or 'results' not in self.sample_cache:
            raise ValueError("No simulation results to analyze. Run simulation first.")
        
        results = self.sample_cache['results']
        
        alpha = (1 - ci_width) / 2
        intervals = {}
        
        for key, values in results.items():
            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))
            intervals[key] = (lower, upper)
        
        return intervals
    
    def compute_var(
        self,
        metric_name: str,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute Value at Risk (VaR) for a metric
        
        Args:
            metric_name: Name of the metric
            confidence_level: Confidence level (0-1)
            
        Returns:
            VaR value
        """
        if not self.sample_cache or 'results' not in self.sample_cache:
            raise ValueError("No simulation results to analyze. Run simulation first.")
        
        if metric_name not in self.sample_cache['results']:
            raise ValueError(f"Metric {metric_name} not found in simulation results")
        
        values = self.sample_cache['results'][metric_name]
        
        # Compute VaR
        return np.percentile(values, 100 * (1 - confidence_level))
    
    def compute_conditional_var(
        self,
        metric_name: str,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute Conditional Value at Risk (CVaR) for a metric
        
        Args:
            metric_name: Name of the metric
            confidence_level: Confidence level (0-1)
            
        Returns:
            CVaR value
        """
        if not self.sample_cache or 'results' not in self.sample_cache:
            raise ValueError("No simulation results to analyze. Run simulation first.")
        
        if metric_name not in self.sample_cache['results']:
            raise ValueError(f"Metric {metric_name} not found in simulation results")
        
        values = self.sample_cache['results'][metric_name]
        
        # Compute VaR
        var = np.percentile(values, 100 * (1 - confidence_level))
        
        # Compute CVaR (mean of values beyond VaR)
        cvar_values = values[values <= var] if metric_name.startswith('loss') else values[values >= var]
        
        return np.mean(cvar_values) if len(cvar_values) > 0 else var

def fit_bayesian_growth_model(
    time_points: np.ndarray,
    values: np.ndarray,
    model_type: str = 'linear',
    samples: int = 1000,
    return_model: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], BayesianModel]:
    """
    Fit a Bayesian growth model and predict with uncertainty
    
    Args:
        time_points: Array of time points
        values: Array of values
        model_type: 'linear' or 'exponential'
        samples: Number of MCMC samples
        return_model: Whether to return the fitted model object
        
    Returns:
        If return_model is True: Fitted model
        If return_model is False: (mean, lower, upper) for predictions
    """
    if not PYMC3_AVAILABLE:
        raise ImportError("PyMC3 is required for Bayesian modeling")
    
    # Select model class
    if model_type == 'linear':
        model = LinearGrowthModel(name=f"Bayesian{model_type.capitalize()}Growth")
    elif model_type == 'exponential':
        model = ExponentialGrowthModel(name=f"Bayesian{model_type.capitalize()}Growth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare data
    data = {
        'x': time_points,
        'y': values
    }
    
    # Fit model
    model.fit(data, samples=samples)
    
    if return_model:
        return model
    else:
        # Predict with credible intervals
        return model.predict(time_points)

def run_bayesian_simulation(
    param_models: Dict[str, BayesianModel],
    simulation_func: Callable,
    n_samples: int = 1000,
    return_full_results: bool = False
) -> Union[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Run a simulation with Bayesian parameter estimates
    
    Args:
        param_models: Dictionary mapping parameter names to fitted Bayesian models
        simulation_func: Function that takes parameter samples and returns outputs
        n_samples: Number of simulation samples
        return_full_results: Whether to return full simulation results
        
    Returns:
        If return_full_results is True: Full simulation results
        If return_full_results is False: Summary statistics
    """
    # Create Monte Carlo simulator
    simulator = BayesianMonteCarlo(name="BayesianSimulation")
    
    # Add parameter models
    for param_name, model in param_models.items():
        simulator.add_parameter_model(param_name, model)
    
    # Run simulation
    results = simulator.run_simulation(simulation_func, n_samples=n_samples)
    
    if return_full_results:
        return {
            'parameters': simulator.sample_cache['parameters'],
            'results': results,
            'summary': simulator.get_simulation_summary(),
            'intervals': simulator.get_credible_intervals()
        }
    else:
        return simulator.get_simulation_summary() 