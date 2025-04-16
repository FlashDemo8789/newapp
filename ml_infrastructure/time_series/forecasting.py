"""
Time Series Forecasting

This module provides tools for time series forecasting using Prophet
and other models. It enables startup growth projection, scenario analysis,
and uncertainty quantification in time-based predictions.

Features:
- Prophet-based forecasting with custom seasonality
- Scenario analysis with different assumptions
- Monte Carlo simulations for risk assessment
- Uncertainty visualization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Prophet
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Time series forecasting will be limited.")

class TimeSeriesForecaster:
    """Base class for time series forecasting models"""
    
    def __init__(self, name: str, frequency: str = 'D'):
        """
        Initialize forecaster
        
        Args:
            name: Model name
            frequency: Time series frequency (D=daily, W=weekly, M=monthly, etc.)
        """
        self.name = name
        self.frequency = frequency
        self.model = None
        self.is_fitted = False
        self.future = None
        self.forecast = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model
        
        Args:
            data: Time series data with 'ds' (dates) and 'y' (values) columns
        """
        raise NotImplementedError("Subclasses must implement fit")
    
    def predict(
        self,
        periods: int,
        include_history: bool = True,
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Generate forecast
        
        Args:
            periods: Number of periods to forecast
            include_history: Whether to include historical data in the forecast
            return_components: Whether to return individual forecast components
            
        Returns:
            DataFrame with forecast
        """
        raise NotImplementedError("Subclasses must implement predict")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        raise NotImplementedError("Subclasses must implement get_metrics")
    
    def plot_forecast(
        self,
        uncertainty: bool = True,
        xlabel: str = 'Date',
        ylabel: str = 'Value',
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot the forecast
        
        Args:
            uncertainty: Whether to show uncertainty intervals
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.forecast is None:
            raise ValueError("Model must be fitted and predict called before plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        return fig

class ProphetForecaster(TimeSeriesForecaster):
    """Prophet-based time series forecaster"""
    
    def __init__(
        self,
        name: str,
        frequency: str = 'D',
        yearly_seasonality: Union[bool, int] = 'auto',
        weekly_seasonality: Union[bool, int] = 'auto',
        daily_seasonality: Union[bool, int] = 'auto'
    ):
        """
        Initialize Prophet forecaster
        
        Args:
            name: Model name
            frequency: Time series frequency (D=daily, W=weekly, M=monthly, etc.)
            yearly_seasonality: Yearly seasonality setting
            weekly_seasonality: Weekly seasonality setting
            daily_seasonality: Daily seasonality setting
        """
        super().__init__(name, frequency)
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for ProphetForecaster")
        
        # Default Prophet model
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        **kwargs
    ) -> None:
        """
        Add custom seasonality
        
        Args:
            name: Name of the seasonality
            period: Period of seasonality in days
            fourier_order: Number of Fourier components
            **kwargs: Additional arguments for Prophet's add_seasonality
        """
        if self.is_fitted:
            logger.warning("Adding seasonality after fitting has no effect")
            return
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            **kwargs
        )
    
    def add_country_holidays(self, country_name: str) -> None:
        """
        Add country holidays as a regressor
        
        Args:
            country_name: Country code (e.g., 'US', 'UK', etc.)
        """
        if self.is_fitted:
            logger.warning("Adding holidays after fitting has no effect")
            return
        
        self.model.add_country_holidays(country_name=country_name)
    
    def add_regressor(
        self,
        name: str,
        prior_scale: float = 10.0,
        standardize: bool = True,
        mode: str = 'additive'
    ) -> None:
        """
        Add an additional regressor
        
        Args:
            name: Name of the regressor
            prior_scale: Prior scale for the regressor
            standardize: Whether to standardize the regressor
            mode: 'additive' or 'multiplicative'
        """
        if self.is_fitted:
            logger.warning("Adding regressor after fitting has no effect")
            return
        
        self.model.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize,
            mode=mode
        )
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model
        
        Args:
            data: Time series data with 'ds' (dates) and 'y' (values) columns
        """
        if not all(col in data.columns for col in ['ds', 'y']):
            raise ValueError("Data must contain 'ds' (dates) and 'y' (values) columns")
        
        # Fit the model
        self.model.fit(data)
        self.is_fitted = True
        
        logger.info(f"Fitted Prophet model {self.name}")
    
    def predict(
        self,
        periods: int,
        include_history: bool = True,
        return_components: bool = False,
        freq: str = None
    ) -> pd.DataFrame:
        """
        Generate forecast
        
        Args:
            periods: Number of periods to forecast
            include_history: Whether to include historical data in the forecast
            return_components: Whether to return individual forecast components
            freq: Frequency of the forecast (if None, uses self.frequency)
            
        Returns:
            DataFrame with forecast
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        if freq is None:
            freq = self.frequency
        
        self.future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        # Generate forecast
        self.forecast = self.model.predict(self.future)
        
        if return_components:
            return self.forecast
        else:
            return self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting metrics")
        
        # Simple metrics based on training data
        train_data = self.model.history
        train_pred = self.model.predict(train_data[['ds']])
        
        y_true = train_data['y'].values
        y_pred = train_pred['yhat'].values
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def cross_validate(
        self,
        horizon: str = '30 days',
        period: str = '180 days',
        initial: str = '365 days'
    ) -> pd.DataFrame:
        """
        Perform cross-validation
        
        Args:
            horizon: Forecast horizon for cross-validation
            period: Period between cutoff dates
            initial: Initial training period
            
        Returns:
            DataFrame with cross-validation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        # Run cross-validation
        cv_results = cross_validation(
            model=self.model,
            horizon=horizon,
            period=period,
            initial=initial
        )
        
        # Get performance metrics
        cv_metrics = performance_metrics(cv_results)
        
        return cv_metrics
    
    def plot_forecast(
        self,
        uncertainty: bool = True,
        xlabel: str = 'Date',
        ylabel: str = 'Value',
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot the forecast
        
        Args:
            uncertainty: Whether to show uncertainty intervals
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.forecast is None:
            raise ValueError("Model must be fitted and predict called before plotting")
        
        # Create figure
        fig = self.model.plot(self.forecast, figsize=figsize)
        
        # Customize plot
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{self.name} Forecast")
        
        # Show uncertainty
        if uncertainty:
            ax = fig.get_axes()[0]
            ax.fill_between(
                self.forecast['ds'],
                self.forecast['yhat_lower'],
                self.forecast['yhat_upper'],
                color='#0072B2',
                alpha=0.2,
                label='Uncertainty Interval'
            )
        
        return fig
    
    def plot_components(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the forecast components
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.forecast is None:
            raise ValueError("Model must be fitted and predict called before plotting components")
        
        return self.model.plot_components(self.forecast, figsize=figsize)

class ScenarioForecaster:
    """
    Scenario-based forecaster for what-if analysis
    
    This class enables scenario analysis by running multiple forecasts
    with different assumptions and comparing results.
    """
    
    def __init__(self, base_model: TimeSeriesForecaster):
        """
        Initialize scenario forecaster
        
        Args:
            base_model: Base forecasting model
        """
        self.base_model = base_model
        self.scenarios = {}
        self.forecasts = {}
    
    def add_scenario(
        self,
        name: str,
        description: str,
        adjustments: Dict[str, Any]
    ) -> None:
        """
        Add a scenario
        
        Args:
            name: Scenario name
            description: Scenario description
            adjustments: Dict of parameter adjustments for this scenario
        """
        self.scenarios[name] = {
            'description': description,
            'adjustments': adjustments
        }
    
    def run_scenarios(
        self,
        periods: int,
        data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Run forecasts for all scenarios
        
        Args:
            periods: Number of periods to forecast
            data: Historical data
            
        Returns:
            Dictionary mapping scenario names to forecast DataFrames
        """
        # Run base scenario
        base_model = self.base_model
        base_model.fit(data)
        base_forecast = base_model.predict(periods)
        
        self.forecasts['base'] = base_forecast
        
        # Run other scenarios
        for name, scenario_info in self.scenarios.items():
            adjustments = scenario_info['adjustments']
            
            # Create a modified copy of the data
            scenario_data = data.copy()
            
            # Apply adjustments to the data
            if 'growth_multiplier' in adjustments:
                # Apply growth rate adjustment (simplified)
                multiplier = adjustments['growth_multiplier']
                dates = scenario_data['ds']
                values = scenario_data['y']
                
                # Calculate percentage changes
                pct_changes = values.pct_change().fillna(0)
                
                # Apply multiplier to percentage changes
                adjusted_pct_changes = pct_changes * multiplier
                
                # Recalculate values
                new_values = [values.iloc[0]]
                for i in range(1, len(values)):
                    new_value = new_values[-1] * (1 + adjusted_pct_changes.iloc[i])
                    new_values.append(new_value)
                
                scenario_data['y'] = new_values
            
            # Create a new model instance
            if hasattr(self.base_model, '__class__'):
                model_class = self.base_model.__class__
                scenario_model = model_class(f"{name}_model")
            else:
                # Fallback to Prophet
                scenario_model = ProphetForecaster(f"{name}_model")
            
            # Set model parameters based on adjustments
            if 'seasonality_mode' in adjustments and hasattr(scenario_model, 'model'):
                scenario_model.model.seasonality_mode = adjustments['seasonality_mode']
            
            if 'seasonality_prior_scale' in adjustments and hasattr(scenario_model, 'model'):
                scenario_model.model.seasonality_prior_scale = adjustments['seasonality_prior_scale']
            
            # Fit and predict
            scenario_model.fit(scenario_data)
            forecast = scenario_model.predict(periods)
            
            self.forecasts[name] = forecast
        
        return self.forecasts
    
    def plot_scenarios(
        self,
        metric: str = 'yhat',
        xlabel: str = 'Date',
        ylabel: str = 'Value',
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot all scenarios for comparison
        
        Args:
            metric: Which metric to plot ('yhat', 'trend', etc.)
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Run scenarios first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each scenario
        for name, forecast in self.forecasts.items():
            label = name
            if name in self.scenarios:
                label = f"{name}: {self.scenarios[name]['description']}"
            elif name == 'base':
                label = 'Base Scenario'
            
            ax.plot(forecast['ds'], forecast[metric], label=label)
        
        # Add labels and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Scenario Comparison')
        ax.legend()
        
        return fig
    
    def get_scenario_metrics(self) -> pd.DataFrame:
        """
        Get comparison metrics for all scenarios
        
        Returns:
            DataFrame with scenario metrics
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Run scenarios first.")
        
        metrics = {}
        
        # Calculate metrics for each scenario
        for name, forecast in self.forecasts.items():
            # Get final value
            final_value = forecast['yhat'].iloc[-1]
            
            # Get max value
            max_value = forecast['yhat'].max()
            
            # Get growth rate (last value / first value - 1)
            growth_rate = (forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1
            
            # Get average
            avg_value = forecast['yhat'].mean()
            
            metrics[name] = {
                'final_value': final_value,
                'max_value': max_value,
                'growth_rate': growth_rate,
                'avg_value': avg_value
            }
        
        return pd.DataFrame.from_dict(metrics, orient='index')

def forecast_startup_growth(
    data: pd.DataFrame,
    periods: int = 12,
    model_type: str = 'prophet',
    include_holidays: bool = True,
    include_seasonality: bool = True,
    return_model: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TimeSeriesForecaster]]:
    """
    Forecast startup growth metrics
    
    Args:
        data: DataFrame with 'ds' (date) and 'y' (metric) columns
        periods: Number of periods to forecast
        model_type: Type of forecasting model ('prophet')
        include_holidays: Whether to include holidays
        include_seasonality: Whether to include seasonality
        return_model: Whether to return the fitted model
        
    Returns:
        If return_model is True: (forecast, model)
        If return_model is False: forecast
    """
    if model_type == 'prophet':
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for forecasting")
        
        # Create Prophet model
        model = ProphetForecaster(
            name="StartupGrowth",
            frequency='M'  # Monthly data is common for startups
        )
        
        # Add seasonality and holidays if requested
        if include_seasonality:
            # No changes needed - Prophet includes seasonality by default
            pass
        else:
            # Disable seasonality
            model.model.seasonality_mode = 'additive'
            model.model.yearly_seasonality = False
            model.model.weekly_seasonality = False
            model.model.daily_seasonality = False
        
        if include_holidays:
            model.add_country_holidays('US')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit model
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(periods)
    
    if return_model:
        return forecast, model
    else:
        return forecast

def create_scenario_analysis(
    data: pd.DataFrame,
    periods: int = 12,
    scenarios: Dict[str, Dict[str, Any]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create scenario analysis for startup growth
    
    Args:
        data: DataFrame with 'ds' (date) and 'y' (metric) columns
        periods: Number of periods to forecast
        scenarios: Dictionary mapping scenario names to adjustment dictionaries
        
    Returns:
        Dictionary mapping scenario names to forecast DataFrames
    """
    # Create base model
    base_model = ProphetForecaster(
        name="BaseScenario",
        frequency='M'  # Monthly data is common for startups
    )
    
    # Create scenario forecaster
    scenario_forecaster = ScenarioForecaster(base_model)
    
    # Add default scenarios if none provided
    if scenarios is None:
        scenarios = {
            'optimistic': {
                'description': 'Optimistic Growth',
                'adjustments': {'growth_multiplier': 1.5}
            },
            'pessimistic': {
                'description': 'Pessimistic Growth',
                'adjustments': {'growth_multiplier': 0.5}
            },
            'seasonal': {
                'description': 'High Seasonality',
                'adjustments': {'seasonality_prior_scale': 15.0}
            }
        }
    
    # Add scenarios
    for name, scenario_info in scenarios.items():
        scenario_forecaster.add_scenario(
            name=name,
            description=scenario_info.get('description', name),
            adjustments=scenario_info.get('adjustments', {})
        )
    
    # Run scenarios
    return scenario_forecaster.run_scenarios(periods, data) 