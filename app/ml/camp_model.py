import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("camp_model")

class CAMPModel:
    """
    XGBoost-based machine learning model for CAMP framework startup analysis.
    This model provides evaluation across Capital, Advantage, Market, and People dimensions.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the CAMP model with optional model directory for saved models."""
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model components
        self.models = {
            'overall': None,
            'capital': None,
            'advantage': None, 
            'market': None,
            'people': None
        }
        self.scaler = None
        self.feature_columns = []
        self.is_trained = False
        
        # Try to load pre-trained models
        try:
            self.load_models()
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {str(e)}")
    
    def load_models(self) -> bool:
        """Load pre-trained models from the model directory."""
        try:
            # Check if all required model files exist
            model_files = {
                'overall': os.path.join(self.model_dir, 'overall_model.pkl'),
                'capital': os.path.join(self.model_dir, 'capital_model.pkl'),
                'advantage': os.path.join(self.model_dir, 'advantage_model.pkl'),
                'market': os.path.join(self.model_dir, 'market_model.pkl'),
                'people': os.path.join(self.model_dir, 'people_model.pkl')
            }
            scaler_file = os.path.join(self.model_dir, 'scaler.pkl')
            metadata_file = os.path.join(self.model_dir, 'metadata.pkl')
            
            # Check if all files exist
            all_files_exist = all(os.path.exists(f) for f in list(model_files.values()) + [scaler_file, metadata_file])
            
            if not all_files_exist:
                logger.info("Not all model files found, will need to train model first")
                return False
            
            # Load each model
            for model_name, model_path in model_files.items():
                self.models[model_name] = joblib.load(model_path)
            
            # Load scaler and metadata
            self.scaler = joblib.load(scaler_file)
            metadata = joblib.load(metadata_file)
            self.feature_columns = metadata.get('feature_columns', [])
            
            self.is_trained = True
            logger.info(f"Successfully loaded pre-trained models with {len(self.feature_columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_trained = False
            return False
    
    def save_models(self) -> bool:
        """Save trained models to the model directory."""
        try:
            # Ensure all models are trained
            if not all(model is not None for model in self.models.values()):
                logger.error("Cannot save models - not all models are trained")
                return False
            
            # Save each model
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
                joblib.dump(model, model_path)
            
            # Save scaler and metadata
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'training_date': datetime.now().isoformat(),
                'model_version': '1.0'
            }
            joblib.dump(metadata, os.path.join(self.model_dir, 'metadata.pkl'))
            
            logger.info("Successfully saved all models")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def prepare_features(self, startup_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract and prepare features from startup data for model input.
        
        Args:
            startup_data: Dictionary containing startup metrics
            
        Returns:
            DataFrame with prepared features
        """
        try:
            # Extract all metrics
            company_info = startup_data.get('company_info', {})
            capital_metrics = startup_data.get('capital_metrics', {})
            advantage_metrics = startup_data.get('advantage_metrics', {})
            market_metrics = startup_data.get('market_metrics', {})
            people_metrics = startup_data.get('people_metrics', {})
            
            # Create features dictionary
            features = {}
            
            # Add numerical features with prefixes
            # Capital metrics
            for key, value in capital_metrics.items():
                features[f'capital_{key}'] = value
                
            # Advantage metrics
            for key, value in advantage_metrics.items():
                features[f'advantage_{key}'] = value
                
            # Market metrics
            for key, value in market_metrics.items():
                features[f'market_{key}'] = value
                
            # People metrics
            for key, value in people_metrics.items():
                features[f'people_{key}'] = value
            
            # Handle categorical features with one-hot encoding
            # Industry
            industry = company_info.get('industry', 'Unknown')
            possible_industries = ['SaaS', 'FinTech', 'HealthTech', 'EdTech', 'ConsumerTech', 'DeepTech', 'CleanTech']
            for ind in possible_industries:
                features[f'industry_{ind}'] = 1 if industry == ind else 0
                
            # Stage
            stage = company_info.get('stage', 'Unknown')
            possible_stages = ['pre-seed', 'seed', 'series-a', 'series-b', 'series-c', 'growth']
            for stg in possible_stages:
                features[f'stage_{stg}'] = 1 if stage == stg else 0
                
            # Business model
            business_model = company_info.get('business_model', 'Unknown')
            possible_models = ['B2B', 'B2C', 'B2B2C', 'SaaS', 'Marketplace', 'Consumer', 'Enterprise']
            for mdl in possible_models:
                features[f'business_model_{mdl}'] = 1 if business_model == mdl else 0
            
            # Create DataFrame
            features_df = pd.DataFrame([features])
            
            # If model is trained and we have feature columns, ensure all expected columns are present
            if self.is_trained and self.feature_columns:
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0  # Default value for missing features
                
                # Ensure only the expected columns are included and in the right order
                features_df = features_df[self.feature_columns]
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return empty DataFrame if feature preparation fails
            return pd.DataFrame(columns=self.feature_columns if self.feature_columns else [])
    
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data for model training until real data is available.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic training data
        """
        np.random.seed(42)  # For reproducibility
        
        # Define industry types and their distributions
        industries = ['SaaS', 'FinTech', 'HealthTech', 'ConsumerTech', 'DeepTech', 'CleanTech', 'EdTech']
        industry_probabilities = [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]
        
        # Define startup stages and their distributions
        stages = ['pre-seed', 'seed', 'series-a', 'series-b', 'series-c', 'growth']
        stage_probabilities = [0.15, 0.25, 0.25, 0.15, 0.1, 0.1]
        
        # Define business models and their distributions
        business_models = ['B2B', 'B2C', 'B2B2C', 'Marketplace', 'SaaS', 'Enterprise', 'Consumer']
        business_model_probabilities = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
        
        # Generate synthetic data
        data = []
        for _ in range(n_samples):
            sample = {}
            
            # Generate categorical features
            sample['industry'] = np.random.choice(industries, p=industry_probabilities)
            sample['stage'] = np.random.choice(stages, p=stage_probabilities)
            sample['business_model'] = np.random.choice(business_models, p=business_model_probabilities)
            
            # Generate metrics
            # Capital metrics (with appropriate ranges)
            sample['capital_burn_rate'] = np.random.uniform(50000, 1000000)
            sample['capital_runway_months'] = np.random.uniform(3, 36)
            sample['capital_ltv_cac_ratio'] = np.random.uniform(0.5, 7)
            sample['capital_gross_margin'] = np.random.uniform(0.1, 0.9)
            sample['capital_mrr'] = np.random.uniform(5000, 1000000)
            sample['capital_cac_payback_months'] = np.random.uniform(2, 36)
            
            # Advantage metrics
            sample['advantage_product_uniqueness'] = np.random.uniform(0.1, 1.0)
            sample['advantage_tech_innovation_score'] = np.random.uniform(0.1, 1.0)
            sample['advantage_network_effects'] = np.random.uniform(0.0, 1.0)
            sample['advantage_product_maturity'] = np.random.uniform(0.1, 1.0)
            sample['advantage_uptime_percentage'] = np.random.uniform(0.8, 0.9999)
            sample['advantage_algorithm_uniqueness'] = np.random.uniform(0.1, 1.0)
            sample['advantage_moat_score'] = np.random.uniform(0.1, 1.0)
            
            # Market metrics
            sample['market_tam_size'] = np.random.uniform(1000000, 10000000000)
            sample['market_monthly_active_users'] = np.random.uniform(100, 1000000)
            sample['market_user_growth_rate'] = np.random.uniform(-0.1, 0.5)
            sample['market_churn_rate'] = np.random.uniform(0.01, 0.3)
            sample['market_net_revenue_retention'] = np.random.uniform(0.7, 1.3)
            sample['market_daily_active_users'] = np.random.uniform(50, 500000)
            sample['market_conversion_rate'] = np.random.uniform(0.01, 0.25)
            
            # People metrics
            sample['people_founder_experience'] = np.random.uniform(0.1, 1.0)
            sample['people_tech_talent_ratio'] = np.random.uniform(0.1, 0.8)
            sample['people_team_completeness'] = np.random.uniform(0.3, 1.0)
            sample['people_team_size'] = np.random.uniform(2, 200)
            sample['people_founder_exits'] = np.random.uniform(0, 3)
            sample['people_management_satisfaction'] = np.random.uniform(0.3, 0.95)
            sample['people_founder_network_reach'] = np.random.uniform(0.1, 1.0)
            
            # Generate target variables with some correlation to features
            # First, calculate base scores for each dimension
            capital_base = (
                0.2 * (sample['capital_runway_months'] / 36) +
                0.3 * np.minimum(sample['capital_ltv_cac_ratio'] / 4, 1.0) +
                0.3 * sample['capital_gross_margin'] +
                0.2 * (1 - np.minimum(sample['capital_cac_payback_months'] / 24, 1.0))
            )
            
            advantage_base = (
                0.25 * sample['advantage_product_uniqueness'] +
                0.25 * sample['advantage_tech_innovation_score'] +
                0.25 * sample['advantage_network_effects'] +
                0.25 * sample['advantage_moat_score']
            )
            
            market_base = (
                0.2 * np.minimum(sample['market_tam_size'] / 1000000000, 1.0) +
                0.3 * np.minimum(sample['market_user_growth_rate'] / 0.3, 1.0) +
                0.3 * (1 - np.minimum(sample['market_churn_rate'] / 0.2, 1.0)) +
                0.2 * np.minimum(sample['market_net_revenue_retention'] / 1.2, 1.0)
            )
            
            people_base = (
                0.3 * sample['people_founder_experience'] +
                0.2 * sample['people_tech_talent_ratio'] +
                0.3 * sample['people_team_completeness'] +
                0.2 * np.minimum(sample['people_founder_exits'] / 2, 1.0)
            )
            
            # Add some noise to make it more realistic
            sample['target_capital_score'] = np.clip(capital_base + np.random.normal(0, 0.1), 0, 1)
            sample['target_advantage_score'] = np.clip(advantage_base + np.random.normal(0, 0.1), 0, 1)
            sample['target_market_score'] = np.clip(market_base + np.random.normal(0, 0.1), 0, 1)
            sample['target_people_score'] = np.clip(people_base + np.random.normal(0, 0.1), 0, 1)
            
            # Overall score is weighted average
            sample['target_overall_score'] = (
                0.3 * sample['target_capital_score'] +
                0.2 * sample['target_advantage_score'] +
                0.25 * sample['target_market_score'] +
                0.25 * sample['target_people_score']
            )
            
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    def train(self, data: Optional[pd.DataFrame] = None, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost models for each dimension of the CAMP framework.
        
        Args:
            data: Training data DataFrame, required when using real data
            test_size: Proportion of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Check if input data is provided
            if data is None or len(data) == 0:
                logger.error("No training data provided")
                return {
                    'status': 'error',
                    'message': 'No training data provided'
                }
            
            logger.info(f"Training models with {len(data)} records")
            
            # Define target variables for each dimension
            target_mapping = {
                'overall': 'overall_success',
                'capital': 'capital_success',
                'advantage': 'advantage_success',
                'market': 'market_success',
                'people': 'people_success'
            }
            
            # Handle categorical variables
            for col in ['industry', 'stage', 'business_model']:
                if col in data.columns:
                    # One-hot encode categorical variables
                    dummies = pd.get_dummies(data[col], prefix=col)
                    data = pd.concat([data, dummies], axis=1)
                    data.drop(col, axis=1, inplace=True)
            
            # Separate features and targets
            target_columns = list(target_mapping.values())
            feature_cols = [col for col in data.columns if col not in target_columns]
            
            logger.info(f"Feature columns: {len(feature_cols)}")
            
            # Store feature columns for later prediction
            self.feature_columns = feature_cols
            
            # Initialize scaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(data[feature_cols])
            
            # Training metrics for result
            training_metrics = {}
            
            # Train model for each dimension
            for dimension, target_col in target_mapping.items():
                if target_col not in data.columns:
                    logger.warning(f"Target column {target_col} not found in data, using fallback")
                    # Create a synthetic target
                    data[target_col] = data.iloc[:, :5].mean(axis=1)
                
                y = data[target_col]
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Convert to DMatrix format
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Set XGBoost parameters
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0
                }
                
                # Train model
                logger.info(f"Training {dimension} model...")
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # Store model
                self.models[dimension] = model
                
                # Evaluate model
                val_preds = model.predict(dval)
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                logger.info(f"Finished training {dimension} model. RMSE: {rmse:.4f}")
                
                # Get feature importance
                importance = model.get_score(importance_type='gain')
                
                # Store metrics
                training_metrics[dimension] = {
                    'rmse': float(rmse),
                    'feature_importance': importance
                }
            
            # Save models
            self.save_models()
            self.is_trained = True
            
            return {
                'status': 'success',
                'metrics': training_metrics,
                'features': len(feature_cols),
                'samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict(self, startup_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Make predictions for a startup using the trained models.
        
        Args:
            startup_data: Dictionary containing startup metrics
            
        Returns:
            Dictionary with predicted scores for each dimension
        """
        try:
            # Check if models are trained
            if not self.is_trained:
                logger.warning("Models not trained yet - prediction not possible")
                return self._generate_fallback_predictions()
            
            # Prepare features
            features_df = self.prepare_features(startup_data)
            
            if features_df.empty:
                logger.error("Feature preparation failed, returning fallback predictions")
                return self._generate_fallback_predictions()
            
            # Scale features
            try:
                features_scaled = self.scaler.transform(features_df)
            except Exception as e:
                logger.error(f"Error scaling features: {str(e)}")
                # Try with a more robust approach
                features_array = features_df.values.astype(float)
                features_scaled = features_array
            
            # Make predictions for each dimension
            predictions = {}
            for dimension, model in self.models.items():
                if model is not None:
                    try:
                        # Convert to DMatrix
                        dpredict = xgb.DMatrix(features_scaled)
                        
                        # Make prediction
                        pred = model.predict(dpredict)[0]
                        
                        # Ensure prediction is in valid range
                        pred = np.clip(pred, 0, 1)
                        
                        # Store prediction
                        predictions[f'{dimension}_score'] = float(pred)
                    except Exception as dim_error:
                        logger.error(f"Error predicting {dimension} score: {str(dim_error)}")
                        predictions[f'{dimension}_score'] = 0.5
                else:
                    logger.error(f"Model for {dimension} is not trained")
                    predictions[f'{dimension}_score'] = 0.5
            
            logger.info(f"Successfully predicted scores: {predictions}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            # Return fallback predictions
            return self._generate_fallback_predictions()
    
    def _generate_fallback_predictions(self) -> Dict[str, float]:
        """Generate fallback predictions when model prediction fails."""
        return {
            'overall_score': 0.5,
            'capital_score': 0.5,
            'advantage_score': 0.5,
            'market_score': 0.5,
            'people_score': 0.5
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = CAMPModel()
    
    # Train model with synthetic data
    training_result = model.train()
    print(f"Training result: {training_result}")
    
    # Test prediction with sample data
    sample_data = {
        'company_info': {
            'industry': 'SaaS',
            'stage': 'seed',
            'business_model': 'B2B'
        },
        'capital_metrics': {
            'burn_rate': 120000,
            'runway_months': 18,
            'ltv_cac_ratio': 3.5,
            'gross_margin': 0.75,
            'cac_payback_months': 10
        },
        'advantage_metrics': {
            'product_uniqueness': 0.8,
            'tech_innovation_score': 0.7,
            'network_effects': 0.6,
            'moat_score': 0.75
        },
        'market_metrics': {
            'tam_size': 5000000000,
            'user_growth_rate': 0.2,
            'churn_rate': 0.05,
            'net_revenue_retention': 1.15
        },
        'people_metrics': {
            'founder_experience': 0.8,
            'tech_talent_ratio': 0.6,
            'team_completeness': 0.7,
            'founder_exits': 1
        }
    }
    
    predictions = model.predict(sample_data)
    print(f"Predictions: {predictions}")
