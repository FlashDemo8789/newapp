"""
Feature Store

This module provides a feature store implementation for consistent feature engineering
across different models and analysis runs. It enables feature reuse, versioning,
and reproducible data pipelines.

Features:
- Feature registration and versioning
- Feature transformations and pipelines
- On-demand and batch feature computation
- Feature caching for improved performance
"""

import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    """Definition of a feature in the feature store"""
    name: str
    description: str
    data_type: str  # int, float, category, array, etc.
    owner: str  # Team/person responsible for this feature
    created_at: datetime
    updated_at: datetime
    version: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to string for JSON serialization
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create a FeatureDefinition from a dictionary"""
        # Convert string back to datetime
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

class FeatureTransformer:
    """Base class for feature transformations"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'FeatureTransformer':
        """Fit the transformer on data"""
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data"""
        if not self._fitted:
            raise RuntimeError(f"Transformer {self.name} needs to be fitted before transform")
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        """Call interface for easy usage"""
        return self.transform(X)

class FeatureStore:
    """
    Feature store for managing and serving features
    
    Features:
    - Register new features and their definitions
    - Build and register feature transformations
    - Serve features on-demand or in batch mode
    - Cache frequently used features
    """
    
    def __init__(self, storage_dir: str = "./feature_store"):
        """
        Initialize the feature store
        
        Args:
            storage_dir: Directory where features and definitions will be stored
        """
        self.storage_dir = storage_dir
        self.registry_file = os.path.join(storage_dir, "feature_registry.json")
        self.features: Dict[str, FeatureDefinition] = {}
        self.transformers: Dict[str, FeatureTransformer] = {}
        self.feature_cache: Dict[str, Any] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing registry if it exists
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the feature registry from disk if it exists"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Populate features dictionary
                for feature_name, feature_data in data.get('features', {}).items():
                    self.features[feature_name] = FeatureDefinition.from_dict(feature_data)
                
                logger.info(f"Loaded feature registry with {len(self.features)} features")
            except Exception as e:
                logger.error(f"Error loading feature registry: {e}")
                # Initialize with empty registry
                self.features = {}
    
    def _save_registry(self) -> None:
        """Save the feature registry to disk"""
        try:
            # Convert features to serializable format
            features_dict = {
                name: feature.to_dict()
                for name, feature in self.features.items()
            }
            
            data = {
                'features': features_dict,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Feature registry saved to {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving feature registry: {e}")
    
    def register_feature(
        self,
        name: str,
        description: str,
        data_type: str,
        owner: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> FeatureDefinition:
        """
        Register a new feature or update an existing one
        
        Args:
            name: Unique name for the feature
            description: Description of what the feature represents
            data_type: Data type of the feature (e.g., int, float, category)
            owner: Team/person responsible for this feature
            tags: List of tags for categorization
            metadata: Additional metadata about the feature
            
        Returns:
            The registered feature definition
        """
        now = datetime.now()
        
        # Check if feature exists already
        if name in self.features:
            # Update existing feature
            feature = self.features[name]
            feature.description = description
            feature.data_type = data_type
            feature.owner = owner
            feature.updated_at = now
            feature.version += 1
            feature.tags = tags or feature.tags
            feature.metadata = metadata or feature.metadata
        else:
            # Create new feature definition
            feature = FeatureDefinition(
                name=name,
                description=description,
                data_type=data_type,
                owner=owner,
                created_at=now,
                updated_at=now,
                version=1,
                tags=tags or [],
                metadata=metadata or {}
            )
            self.features[name] = feature
        
        # Save the updated registry
        self._save_registry()
        
        logger.info(f"Registered feature {name} (version {feature.version})")
        return feature
    
    def register_transformer(self, transformer: FeatureTransformer) -> None:
        """
        Register a feature transformer
        
        Args:
            transformer: The transformer to register
        """
        self.transformers[transformer.name] = transformer
        logger.info(f"Registered transformer {transformer.name}")
    
    def transform_features(
        self,
        data: pd.DataFrame,
        transformer_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply transformers to input data
        
        Args:
            data: Input DataFrame
            transformer_names: Names of transformers to apply (if None, applies all)
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        # Determine which transformers to use
        if transformer_names is None:
            # Use all registered transformers
            transformers_to_use = list(self.transformers.values())
        else:
            # Use only the specified transformers
            transformers_to_use = [
                transformer for name, transformer in self.transformers.items()
                if name in transformer_names
            ]
        
        # Apply transformers in sequence
        for transformer in transformers_to_use:
            try:
                result = transformer.transform(result)
            except Exception as e:
                logger.error(f"Error applying transformer {transformer.name}: {e}")
                # Continue with other transformers
        
        return result
    
    def get_feature_vector(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        drop_na: bool = False
    ) -> pd.DataFrame:
        """
        Extract a specific set of features from the data
        
        Args:
            data: Input DataFrame
            feature_names: Names of features to extract
            drop_na: Whether to drop rows with NA values
            
        Returns:
            DataFrame with requested features
        """
        # Get only the requested features
        result = data[feature_names].copy()
        
        # Handle NA values
        if drop_na:
            result = result.dropna()
        
        return result
    
    def compute_feature(
        self,
        name: str,
        data: pd.DataFrame,
        function: Callable[[pd.DataFrame], pd.Series],
        use_cache: bool = True
    ) -> pd.Series:
        """
        Compute a feature using the provided function
        
        Args:
            name: Name of the feature to compute
            data: Input data for computation
            function: Function that computes the feature
            use_cache: Whether to use/update cache
            
        Returns:
            The computed feature values
        """
        # Generate a cache key
        if use_cache:
            # Use hash of data to create a cache key
            data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
            cache_key = f"{name}_{data_hash}"
            
            # Check if in cache
            if cache_key in self.feature_cache:
                logger.debug(f"Using cached feature {name}")
                return self.feature_cache[cache_key]
        
        # Compute the feature
        try:
            feature_values = function(data)
            
            # Update cache if needed
            if use_cache:
                self.feature_cache[cache_key] = feature_values
            
            return feature_values
        except Exception as e:
            logger.error(f"Error computing feature {name}: {e}")
            return pd.Series(np.nan, index=data.index, name=name)
    
    def clear_cache(self) -> None:
        """Clear the feature cache"""
        self.feature_cache.clear()
        logger.info("Feature cache cleared")
    
    def list_features(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List all features, optionally filtered by tags
        
        Args:
            tags: Filter features by these tags (if provided)
            
        Returns:
            List of feature definitions
        """
        if tags is None:
            # Return all features
            return [
                feature.to_dict() for feature in self.features.values()
            ]
        else:
            # Filter by tags
            return [
                feature.to_dict() for feature in self.features.values()
                if any(tag in feature.tags for tag in tags)
            ]
    
    def save_feature_data(self, name: str, data: pd.Series) -> str:
        """
        Save feature data to disk
        
        Args:
            name: Name of the feature
            data: Feature data to save
            
        Returns:
            Path where data was saved
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        # Create directory for feature data
        feature_dir = os.path.join(self.storage_dir, 'data', name)
        os.makedirs(feature_dir, exist_ok=True)
        
        # Save with version number
        version = self.features[name].version
        filename = f"{name}_v{version}.pkl"
        filepath = os.path.join(feature_dir, filename)
        
        # Save to disk
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved feature data for {name} v{version}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving feature data for {name}: {e}")
            raise
    
    def load_feature_data(
        self,
        name: str,
        version: int = None
    ) -> pd.Series:
        """
        Load feature data from disk
        
        Args:
            name: Name of the feature
            version: Specific version to load (if None, loads latest)
            
        Returns:
            Feature data
        """
        if name not in self.features:
            raise ValueError(f"Feature {name} not registered")
        
        # Determine version to load
        if version is None:
            version = self.features[name].version
        
        # Construct path
        feature_dir = os.path.join(self.storage_dir, 'data', name)
        filepath = os.path.join(feature_dir, f"{name}_v{version}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature data not found: {filepath}")
        
        # Load from disk
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded feature data for {name} v{version}")
            return data
        except Exception as e:
            logger.error(f"Error loading feature data for {name}: {e}")
            raise 