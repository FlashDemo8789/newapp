"""
Ensemble Model Loader

This module provides tools for building and using ensemble models that combine
multiple underlying models, potentially from different frameworks (LightGBM, PyTorch,
transformer-based models). It supports weighted ensembles, stacking, and boosting.

Features:
- Combine models from different frameworks
- Support for weighted averaging, stacking, and boosting
- Feature importance tracking across ensemble
- Easy serialization and loading
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Callable, Optional, Tuple
import logging
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import lightgbm
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. LightGBM models will not be supported in ensembles.")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch models will not be supported in ensembles.")

# Try to import transformers
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Transformer models will not be supported.")

@dataclass
class ModelInfo:
    """Information about a model in an ensemble"""
    name: str
    model_type: str
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(**data)

class BaseModel:
    """Base class for all models in ensemble"""
    
    def __init__(self, name: str):
        """
        Initialize base model
        
        Args:
            name: Model name
        """
        self.name = name
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement predict")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}

class LightGBMModel(BaseModel):
    """Wrapper for LightGBM models"""
    
    def __init__(self, name: str, model: Any, feature_names: List[str] = None):
        """
        Initialize LightGBM model
        
        Args:
            name: Model name
            model: LightGBM model object
            feature_names: Names of features (if not provided in model)
        """
        super().__init__(name)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")
        
        self.model = model
        self.feature_names = feature_names or getattr(model, 'feature_name_', None)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Get feature importance from model
        importance = self.model.feature_importance(importance_type='gain')
        
        # Convert to dictionary with feature names
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

class PyTorchModel(BaseModel):
    """Wrapper for PyTorch models"""
    
    def __init__(
        self,
        name: str,
        model: Any,
        preprocessor: Callable = None,
        feature_names: List[str] = None
    ):
        """
        Initialize PyTorch model
        
        Args:
            name: Model name
            model: PyTorch model object
            preprocessor: Function to preprocess inputs
            feature_names: Names of features
        """
        super().__init__(name)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Ensure model is in evaluation mode
        self.model.eval()
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Preprocess if needed
        if self.preprocessor:
            X = self.preprocessor(X)
        
        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Run inference without gradients
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Convert output tensor to numpy
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()
        elif isinstance(outputs, (tuple, list)) and outputs:
            # Use first output for multi-output models
            return outputs[0].cpu().numpy()
        else:
            raise ValueError(f"Unsupported PyTorch model output type: {type(outputs)}")

class TransformerModel(BaseModel):
    """Wrapper for transformer-based models"""
    
    def __init__(
        self,
        name: str,
        model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        max_length: int = 512
    ):
        """
        Initialize transformer model
        
        Args:
            name: Model name
            model_name_or_path: Hugging Face model name or path
            tokenizer_name_or_path: Tokenizer name or path (defaults to model_name_or_path)
            max_length: Maximum sequence length
        """
        super().__init__(name)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is not available")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path)
        self.max_length = max_length
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Get embeddings for text inputs
        
        Args:
            texts: List of text strings or pandas Series
            
        Returns:
            Array of embeddings
        """
        # Convert pandas Series to list
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

class EnsembleModel:
    """
    Ensemble model that combines multiple underlying models
    
    This class supports several ensemble methods:
    - Weighted averaging
    - Stacking
    - Boosting
    """
    
    def __init__(self, name: str, ensemble_method: str = "weighted_average"):
        """
        Initialize ensemble model
        
        Args:
            name: Model name
            ensemble_method: Ensemble method (weighted_average, stacking, boosting)
        """
        self.name = name
        self.ensemble_method = ensemble_method
        self.models: List[Tuple[BaseModel, float]] = []  # (model, weight) pairs
        self.meta_model = None  # Used for stacking
        self.model_infos: List[ModelInfo] = []
    
    def add_model(self, model: BaseModel, weight: float = 1.0, metadata: Dict[str, Any] = None):
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
            weight: Weight in the ensemble (used for weighted averaging)
            metadata: Additional model metadata
        """
        self.models.append((model, weight))
        
        # Store model info
        model_info = ModelInfo(
            name=model.name,
            model_type=model.__class__.__name__,
            weight=weight,
            metadata=metadata or {},
            feature_importance=model.get_feature_importance()
        )
        self.model_infos.append(model_info)
    
    def set_meta_model(self, model: Any):
        """
        Set meta model for stacking
        
        Args:
            model: Meta model that combines base model outputs
        """
        self.meta_model = model
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the ensemble
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if self.ensemble_method == "weighted_average":
            return self._predict_weighted_average(X)
        elif self.ensemble_method == "stacking":
            return self._predict_stacking(X)
        elif self.ensemble_method == "boosting":
            return self._predict_boosting(X)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
    def _predict_weighted_average(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using weighted averaging
        
        Args:
            X: Input features
            
        Returns:
            Weighted average predictions
        """
        predictions = []
        weights = []
        
        # Get predictions from each model
        for model, weight in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.error(f"Error in model {model.name}: {e}")
                # Skip this model
        
        if not predictions:
            raise ValueError("All models failed to predict")
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Compute weighted average
        stacked_preds = np.stack(predictions, axis=0)
        return np.average(stacked_preds, axis=0, weights=weights)
    
    def _predict_stacking(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using stacking
        
        Args:
            X: Input features
            
        Returns:
            Stacked predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta model not set for stacking")
        
        # Get predictions from base models
        base_predictions = []
        for model, _ in self.models:
            try:
                pred = model.predict(X)
                
                # Ensure predictions are 2D
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                
                base_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error in base model {model.name}: {e}")
                # Skip this model
        
        if not base_predictions:
            raise ValueError("All base models failed to predict")
        
        # Concatenate base predictions
        meta_features = np.hstack(base_predictions)
        
        # Use meta model to make final predictions
        return self.meta_model.predict(meta_features)
    
    def _predict_boosting(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using boosting
        
        Args:
            X: Input features
            
        Returns:
            Boosted predictions
        """
        predictions = None
        
        # Apply models sequentially, each one improving on the last
        for model, weight in self.models:
            try:
                pred = model.predict(X)
                
                if predictions is None:
                    predictions = weight * pred
                else:
                    predictions += weight * pred
            except Exception as e:
                logger.error(f"Error in model {model.name}: {e}")
                # Skip this model
        
        if predictions is None:
            raise ValueError("All models failed to predict")
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance across the ensemble
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Aggregate feature importance across models
        importance_dict = {}
        
        for model, weight in self.models:
            model_importance = model.get_feature_importance()
            
            for feature, importance in model_importance.items():
                if feature not in importance_dict:
                    importance_dict[feature] = 0.0
                importance_dict[feature] += weight * importance
        
        return importance_dict
    
    def save(self, filepath: str):
        """
        Save ensemble model to file
        
        Args:
            filepath: Path to save the model
        """
        # Create dictionary representation
        model_data = {
            'name': self.name,
            'ensemble_method': self.ensemble_method,
            'model_infos': [info.to_dict() for info in self.model_infos]
        }
        
        # Save model data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save individual models
        model_dir = os.path.splitext(filepath)[0] + '_models'
        os.makedirs(model_dir, exist_ok=True)
        
        for i, (model, _) in enumerate(self.models):
            model_path = os.path.join(model_dir, f"{model.name}.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.error(f"Error saving model {model.name}: {e}")
        
        # Save meta model if it exists
        if self.meta_model is not None:
            meta_model_path = os.path.join(model_dir, "meta_model.pkl")
            try:
                with open(meta_model_path, 'wb') as f:
                    pickle.dump(self.meta_model, f)
            except Exception as e:
                logger.error(f"Error saving meta model: {e}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load ensemble model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ensemble model
        """
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create ensemble
        ensemble = cls(
            name=model_data['name'],
            ensemble_method=model_data['ensemble_method']
        )
        
        # Load model infos
        ensemble.model_infos = [
            ModelInfo.from_dict(info) for info in model_data['model_infos']
        ]
        
        # Load individual models
        model_dir = os.path.splitext(filepath)[0] + '_models'
        
        for info in ensemble.model_infos:
            model_path = os.path.join(model_dir, f"{info.name}.pkl")
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                ensemble.models.append((model, info.weight))
            except Exception as e:
                logger.error(f"Error loading model {info.name}: {e}")
        
        # Load meta model if it exists
        meta_model_path = os.path.join(model_dir, "meta_model.pkl")
        if os.path.exists(meta_model_path):
            try:
                with open(meta_model_path, 'rb') as f:
                    ensemble.meta_model = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading meta model: {e}")
        
        return ensemble

def create_lightgbm_model(
    name: str,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    params: Dict[str, Any] = None,
    num_boost_round: int = 100,
    feature_names: List[str] = None
) -> LightGBMModel:
    """
    Create and train a LightGBM model
    
    Args:
        name: Model name
        X_train: Training features
        y_train: Training labels
        params: LightGBM parameters
        num_boost_round: Number of boosting rounds
        feature_names: Names of features
        
    Returns:
        Trained LightGBM model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not available")
    
    # Default parameters
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    }
    
    # Merge with user parameters
    params = {**default_params, **(params or {})}
    
    # Create training dataset
    if isinstance(X_train, pd.DataFrame):
        feature_names = feature_names or X_train.columns.tolist()
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    else:
        dtrain = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    lgb_model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    
    # Create wrapper
    return LightGBMModel(name, lgb_model, feature_names)

def create_ensemble(
    name: str,
    models: List[BaseModel],
    weights: List[float] = None,
    ensemble_method: str = "weighted_average"
) -> EnsembleModel:
    """
    Create an ensemble from existing models
    
    Args:
        name: Ensemble name
        models: List of models
        weights: Model weights (defaults to equal weights)
        ensemble_method: Ensemble method
        
    Returns:
        Ensemble model
    """
    ensemble = EnsembleModel(name, ensemble_method)
    
    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(models)
    
    # Add models to ensemble
    for model, weight in zip(models, weights):
        ensemble.add_model(model, weight)
    
    return ensemble 